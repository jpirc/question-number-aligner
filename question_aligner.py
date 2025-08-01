import pandas as pd
import re
import json
import base64
import requests
from typing import Dict, List, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import streamlit as st
import numpy as np
import io

# --- Data Classes ---
@dataclass
class Question:
    """Represents a single, clean question extracted from the survey."""
    number: str
    text: str
    question_type: str

# --- Helper Functions ---
def index_to_excel_col(idx: int) -> str:
    """Converts a zero-based column index to an Excel-style column letter (A, B, ... Z, AA, AB...)."""
    col = ""
    while idx >= 0:
        col = chr(idx % 26 + 65) + col
        idx = idx // 26 - 1
    return col

# --- Core Logic Class ---
class QuestionNumberAligner:
    """Handles the logic for parsing questions using AI and matching them to dataset columns."""

    def __init__(self):
        # Keywords to help classify questions after they've been extracted by the AI.
        self.question_patterns = {
            'multi_response': ['select all that apply', 'check all that apply', 'please select all', 'choose as many'],
            'matrix': ['how much do you agree', 'please rate', 'on a scale', 'how appealing', 'rate the following'],
            'ranking': ['please rank', 'rank the following']
        }
        self.demographic_patterns = {
            'gender': ['gender', 'male', 'female'],
            'age': ['age', 'years old'],
            'income': ['income', 'household'],
            'education': ['education', 'degree'],
            'employment': ['employment', 'work'],
            'ethnicity': ['hispanic', 'latino'],
            'race': ['race', 'racial'],
            'location': ['region', 'state', 'area', 'live']
        }

    def extract_questions_with_gemini(self, pdf_file_bytes: bytes) -> List[Question]:
        """
        Sends a PDF file to the Gemini API to extract questions and returns them as a list of Question objects.
        """
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("GEMINI_API_KEY is not set in your Streamlit secrets. Please add it to run the AI extraction.")
            return []

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        encoded_pdf = base64.b64encode(pdf_file_bytes).decode('utf-8')
        
        prompt = """
        You are an expert data cleaning assistant specializing in survey reports.
        Analyze the provided PDF document. Your task is to identify and extract EVERY numbered question.

        IMPORTANT: Some surveys have repeated questions for different test groups (e.g., HOH BDAY, HOH PUP, DREAMS BOGO).
        Make sure to capture ALL instances, even if the question text is similar.

        Follow these rules strictly:
        1. Extract ALL questions that start with a number followed by a period (e.g., "1.", "12.", "180.")
        2. Include the FULL question text, including any prefixes like "HOH BDAY -" or "DREAMS FLASH -"
        3. Extract questions even if they appear multiple times with different prefixes
        4. Include questions from ALL sections of the document
        5. Clean the text but preserve important identifiers
        6. Return as a valid JSON object with question numbers as keys and full question text as values

        The response must be valid JSON with this exact format:
        {
          "1": "To which gender identity do you most identify? Select one.",
          "12": "HOH BDAY - How much do you like this ad?",
          "46": "HOH PUP - How much do you like this ad?",
          "180": "Do you consider yourself Hispanic/Latino or not Hispanic/Latino?"
        }

        Extract EVERY question - there should be many questions, potentially over 100 in a full survey.
        """
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "application/pdf", "data": encoded_pdf}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        try:
            with st.spinner("Calling Gemini API..."):
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
                response.raise_for_status()
                
            response_json = response.json()
            
            with st.expander("🔍 API Response Details (for debugging)"):
                st.json(response_json)
            
            candidate = response_json.get('candidates', [{}])[0]
            content_part = candidate.get('content', {}).get('parts', [{}])[0]
            extracted_json_text = content_part.get('text', '{}')
            
            with st.expander("📝 Extracted Questions JSON"):
                st.code(extracted_json_text, language='json')
            
            question_dict = json.loads(extracted_json_text)
            
            st.info(f"📊 Extracted {len(question_dict)} questions from the PDF")
            
            questions = [Question(number=str(num), text=text, question_type=self._determine_question_type(text)) 
                         for num, text in question_dict.items()]
            return sorted(questions, key=lambda q: int(q.number))

        except requests.exceptions.RequestException as e:
            st.error(f"Network error calling Gemini API: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            st.error(f"Error parsing response from Gemini API: {e}")
            if 'extracted_json_text' in locals():
                st.error("Extracted text that failed to parse:")
                st.code(extracted_json_text)
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            import traceback
            st.error(traceback.format_exc())
            return []

    def _determine_question_type(self, text: str) -> str:
        text_lower = text.lower()
        for q_type, patterns in self.question_patterns.items():
            if any(p in text_lower for p in patterns):
                return q_type.replace('_response', '')
        return 'single'

    def clean_column_name(self, col_name: str) -> str:
        cleaned = re.sub(r'[^\w\s-]', ' ', col_name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.lower()

    def calculate_similarity(self, str1: str, str2: str) -> float:
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def match_columns_to_questions(self, df: pd.DataFrame, questions: List[Question]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        A more robust, two-pass matching algorithm using "anchoring" and "interpolation".
        """
        columns = list(df.columns)
        mapping = {col: 'UNMATCHED' for col in columns}
        confidence_scores = {col: 0.0 for col in columns}
        
        sorted_questions = sorted(questions, key=lambda q: int(q.number))
        q_lookup = {q.number: q for q in sorted_questions}
        
        # --- Pass 1: High-Confidence Anchoring ---
        # Find near-perfect matches to act as reliable guideposts.
        anchors = []
        for i, col in enumerate(columns):
            clean_col = self.clean_column_name(col)
            for j, q in enumerate(sorted_questions):
                score = self.calculate_similarity(clean_col, q.text)
                if score > 0.95: # Very high confidence for an anchor
                    anchors.append({'col_idx': i, 'q_idx': j, 'q_num': q.number, 'score': score})
                    break # Assume first great match is the one

        # --- Pass 2: Interpolation between Anchors ---
        # Add start and end points to process the whole list
        proc_anchors = [{'col_idx': -1, 'q_idx': -1}] + anchors + [{'col_idx': len(columns), 'q_idx': len(sorted_questions)}]

        for i in range(len(proc_anchors) - 1):
            start_anchor = proc_anchors[i]
            end_anchor = proc_anchors[i+1]

            # Define the segment of columns and questions to work on
            col_segment = columns[start_anchor['col_idx']+1 : end_anchor['col_idx']]
            q_segment = sorted_questions[start_anchor['q_idx']+1 : end_anchor['q_idx']]

            if not col_segment or not q_segment:
                continue

            # --- Sub-Pass 2a: Multi-Select Grouping within the segment ---
            col_cursor = 0
            while col_cursor < len(col_segment):
                col = col_segment[col_cursor]
                base_question = col.split(' - ')[0].strip()
                
                # Check if this column is part of a multi-select group
                if ' - ' in col and len(base_question) > 10:
                    group_cols = [c for c in col_segment if c.startswith(base_question + ' - ')]
                    if len(group_cols) > 1:
                        # Find the best match for this group within the question segment
                        best_q, best_score = None, 0.0
                        for q in q_segment:
                            if q.question_type != 'multi': continue
                            score = self.calculate_similarity(self.clean_column_name(base_question), q.text)
                            if score > best_score:
                                best_score, best_q = score, q
                        
                        if best_q and best_score > 0.70:
                            for group_col in group_cols:
                                mapping[group_col] = best_q.number
                                confidence_scores[group_col] = best_score
                        
                        # Advance the cursor past all columns in this group
                        col_cursor += len(group_cols)
                        continue
                
                # If not a group, process as a single column
                col_cursor += 1
        
        # --- Sub-Pass 2b: Single Column Matching within segments ---
        q_cursor = 0
        for i, col in enumerate(columns):
            if mapping[col] != 'UNMATCHED': continue # Skip already-matched multi-selects

            clean_col = self.clean_column_name(col)
            best_q, best_score, best_q_idx = None, 0.0, -1
            
            # Define a smart search window based on our current question cursor
            search_window = sorted_questions[q_cursor : min(q_cursor + 20, len(sorted_questions))]
            
            for j, q in enumerate(search_window):
                score = self.calculate_similarity(clean_col, q.text)
                proximity_bonus = max(0, 1 - (j / 20)) * 0.2
                score += proximity_bonus
                if score > best_score:
                    best_score, best_q, best_q_idx = score, q, q_cursor + j
            
            if best_q and best_score > 0.70:
                mapping[col] = best_q.number
                confidence_scores[col] = best_score
                q_cursor = best_q_idx + 1 # Self-correct by advancing the cursor
        
        return mapping, confidence_scores


    def apply_numbering_to_dataset(self, df: pd.DataFrame, mapping: Dict[str, str], prefix_format: str, include_unmatched: bool = True) -> pd.DataFrame:
        df_numbered = df.copy()
        
        if not include_unmatched:
            matched_cols = [col for col in df.columns if mapping.get(col, 'UNMATCHED') != 'UNMATCHED']
            df_numbered = df_numbered[matched_cols]
        
        new_columns = {}
        for col in df_numbered.columns:
            q_num = mapping.get(col)
            if q_num and q_num != 'UNMATCHED' and q_num.isdigit():
                new_columns[col] = f"{prefix_format.format(num=q_num)}{col}"
            else:
                new_columns[col] = col
                
        df_numbered.rename(columns=new_columns, inplace=True)
        return df_numbered

# --- Streamlit UI ---
def create_streamlit_app():
    st.set_page_config(page_title="AI-Powered Question Aligner", layout="wide")
    
    st.title("📊 AI-Powered Question Number Alignment Tool")
    st.markdown("This tool uses AI to read a survey PDF, extract the questions, and match them to your dataset columns.")
    st.markdown("---")

    if 'mapping' not in st.session_state: st.session_state.mapping = None
    if 'confidence_scores' not in st.session_state: st.session_state.confidence_scores = None
    if 'df' not in st.session_state: st.session_state.df = None
    if 'questions' not in st.session_state: st.session_state.questions = []
    
    aligner = QuestionNumberAligner()

    st.subheader("Step 1: Upload Your Files")
    col1, col2 = st.columns(2)
    with col1:
        data_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    with col2:
        pdf_file = st.file_uploader("Upload Survey PDF", type=['pdf'])

    if data_file:
        try:
            st.session_state.df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
            st.success(f"✅ Dataset loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns.")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.session_state.df = None

    with st.expander("📂 Load Previous Mapping (Optional)"):
        uploaded_mapping = st.file_uploader("Load a saved mapping file", type=['json'])
        if uploaded_mapping:
            try:
                loaded_mapping = json.loads(uploaded_mapping.read())
                st.session_state.mapping = loaded_mapping
                st.success("✅ Previous mapping loaded successfully!")
            except Exception as e:
                st.error(f"Error loading mapping: {e}")

    st.markdown("---")
    st.subheader("Step 2: Extract Questions from PDF")
    
    tab1, tab2 = st.tabs(["🤖 AI Extraction", "✏️ Manual Entry"])
    
    with tab1:
        if pdf_file:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🤖 Extract Questions with AI", type="primary", use_container_width=True):
                    with st.spinner("AI is reading and analyzing your PDF... This may take a moment."):
                        pdf_bytes = pdf_file.getvalue()
                        st.session_state.questions = aligner.extract_questions_with_gemini(pdf_bytes)
                    if st.session_state.questions:
                        st.success(f"✅ AI successfully extracted {len(st.session_state.questions)} questions.")
                    else:
                        st.error("AI extraction failed. Please check the error messages above or try manual entry.")
            with col2:
                if st.button("🔄 Retry Extraction", use_container_width=True):
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.info("Upload a PDF to enable AI question extraction.")
    
    with tab2:
        st.markdown("##### Paste Questions (JSON or Text Format)")
        st.info("If AI extraction fails or misses questions, paste them here in JSON format or as numbered list.")
        
        manual_questions = st.text_area(
            "Enter questions",
            height=300,
            placeholder='''Option 1 - JSON format:
{
  "1": "To which gender identity do you most identify? Select one.",
  "12": "HOH BDAY - How much do you like this ad?",
  "180": "Do you consider yourself Hispanic/Latino?"
}

Option 2 - Simple list:
1. To which gender identity do you most identify? Select one.
12. HOH BDAY - How much do you like this ad?
180. Do you consider yourself Hispanic/Latino?''',
            key="manual_questions"
        )
        
        if st.button("📝 Process Manual Questions", use_container_width=True):
            if manual_questions.strip():
                try:
                    if manual_questions.strip().startswith('{'):
                        question_dict = json.loads(manual_questions)
                        st.session_state.questions = [
                            Question(number=str(num), text=text, question_type=aligner._determine_question_type(text))
                            for num, text in question_dict.items()
                        ]
                    else:
                        questions = []
                        for line in manual_questions.strip().split('\n'):
                            match = re.match(r'^(\d+)\.\s+(.+)', line.strip())
                            if match:
                                num, text = match.groups()
                                questions.append(Question(
                                    number=num,
                                    text=text.strip(),
                                    question_type=aligner._determine_question_type(text)
                                ))
                        st.session_state.questions = questions
                    
                    if st.session_state.questions:
                        st.success(f"✅ Successfully processed {len(st.session_state.questions)} questions.")
                    else:
                        st.error("No valid questions found. Check your format.")
                        
                except json.JSONDecodeError as e:
                    st.error(f"JSON parsing error: {e}")
                except Exception as e:
                    st.error(f"Error processing questions: {e}")

    if st.session_state.questions:
        with st.expander(f"📋 Review Extracted Questions ({len(st.session_state.questions)} total)"):
            questions_df = pd.DataFrame([
                {"Number": q.number, "Text": q.text, "Type": q.question_type}
                for q in sorted(st.session_state.questions, key=lambda x: int(x.number))
            ])
            st.dataframe(questions_df, use_container_width=True, height=400)
            
            st.markdown("##### JSON Format (for reuse):")
            question_dict = {q.number: q.text for q in st.session_state.questions}
            st.code(json.dumps(question_dict, indent=2), language='json')

    if st.session_state.df is not None and st.session_state.questions:
        st.markdown("---")
        st.subheader("Step 3: Run Alignment")
        if st.button("🚀 Run Automatic Alignment", use_container_width=True):
            with st.spinner("Analyzing and matching columns..."):
                mapping, confidence_scores = aligner.match_columns_to_questions(st.session_state.df, st.session_state.questions)
                st.session_state.mapping = mapping
                st.session_state.confidence_scores = confidence_scores
    
    if st.session_state.mapping:
        st.markdown("---")
        st.subheader("📋 Alignment Results")
        
        question_lookup = {q.number: q.text for q in st.session_state.questions}
        
        total_cols = len(st.session_state.df.columns)
        matched_cols = sum(1 for v in st.session_state.mapping.values() if v != 'UNMATCHED')
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Columns", total_cols)
        with col2: st.metric("Matched", matched_cols, f"{matched_cols/total_cols*100:.1f}%" if total_cols > 0 else "0.0%")
        with col3: st.metric("Unmatched", total_cols - matched_cols)
        
        st.markdown("##### Review and Edit Mappings")
        st.info("💡 Tip: Review the AI's matches. You can manually enter a question number in the 'Assigned #' column for any row.")
        
        review_data = []
        for i, col in enumerate(st.session_state.df.columns):
            q_num = st.session_state.mapping.get(col, 'UNMATCHED')
            confidence = st.session_state.confidence_scores.get(col, 0.0)
            
            matched_text = f"Q{q_num}. {question_lookup.get(q_num, 'Text not found')}" if q_num != 'UNMATCHED' else 'N/A - Unmatched'
            
            review_data.append({
                'Col': index_to_excel_col(i),
                'Column Name': col,
                'Matched Question Text': matched_text,
                'Confidence': confidence,
                'Assigned #': q_num
            })
        
        review_df = pd.DataFrame(review_data)
        
        edited_df = st.data_editor(
            review_df,
            column_config={
                "Col": st.column_config.TextColumn("Col", width="small", disabled=True),
                "Column Name": st.column_config.TextColumn("Column Name", width="large", disabled=True),
                "Matched Question Text": st.column_config.TextColumn("Matched Question Text", width="large", disabled=True),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.0%"),
                "Assigned #": st.column_config.TextColumn(
                    "Assigned #",
                    help="Manually enter a question number here (e.g., '7', '12', '180')."
                )
            },
            use_container_width=True,
            hide_index=True,
            height=400,
            key="editor"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("✅ Confirm Changes", type="primary", use_container_width=True):
                for idx, row in edited_df.iterrows():
                    # Update the main mapping with any manual changes from the editor
                    st.session_state.mapping[row['Column Name']] = str(row['Assigned #'])
                st.success("✅ Changes confirmed and mapping updated!")
                
        st.markdown("---")
        st.subheader("📥 Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            prefix_format = st.selectbox(
                "Question Number Format",
                ["Q{num}. ", "{num}. ", "#{num}. ", "Question {num}: "],
                help="Choose how question numbers are prefixed to column names."
            )
        with col2:
            include_unmatched = st.checkbox("Include unmatched columns", value=True)
        
        final_df = aligner.apply_numbering_to_dataset(
            st.session_state.df, 
            st.session_state.mapping, 
            prefix_format,
            include_unmatched
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False, sheet_name='Numbered Data')
            excel_data = output.getvalue()
            st.download_button(
                label="📑 Download Excel",
                data=excel_data,
                file_name="renumbered_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            csv = final_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name="renumbered_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            mapping_json = json.dumps(st.session_state.mapping, indent=2)
            st.download_button(
                label="💾 Save Mapping",
                data=mapping_json,
                file_name="question_mapping.json",
                mime="application/json",
                use_container_width=True,
                help="Save this mapping to reuse with similar surveys"
            )

if __name__ == "__main__":
    create_streamlit_app()
