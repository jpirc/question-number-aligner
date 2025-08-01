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
    """Handles the logic for parsing and matching questions using the Gemini AI."""

    def __init__(self):
        self.question_patterns = {
            'multi_response': ['select all that apply', 'check all that apply', 'please select all', 'choose as many'],
            'matrix': ['how much do you agree', 'please rate', 'on a scale', 'how appealing', 'rate the following'],
            'ranking': ['please rank', 'rank the following']
        }

    def extract_questions_with_gemini(self, pdf_file_bytes: bytes) -> List[Question]:
        """Sends a PDF file to the Gemini API to extract questions."""
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("GEMINI_API_KEY is not set in your Streamlit secrets.")
            return []

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
        
        encoded_pdf = base64.b64encode(pdf_file_bytes).decode('utf-8')
        
        prompt = """
        You are a highly specialized text extraction tool. Your ONLY function is to scan the provided PDF and find every instance of a line that begins with a number followed by a period.

        Follow these steps:
        1.  Read the entire document page by page.
        2.  Identify every piece of text that matches the pattern `[NUMBER]. [QUESTION TEXT]`.
        3.  For each match, capture the number and the complete question text that follows. The question text ends when you encounter a chart, a data table (often starting with "Answer Count Percent"), or the start of the next numbered question.
        4.  Clean the extracted text to form a single, coherent sentence. Remove any line breaks or formatting noise within the question itself.
        5.  Assemble the results into a single, valid JSON object where the keys are the question numbers (as strings) and the values are the cleaned question text.

        **CRITICAL INSTRUCTION:** Do not be deterred by messy tables or complex layouts. If you see "39. Please rank the following...", you MUST extract it, even if it's surrounded by charts and tables. Your job is to find the numbered question text itself, no matter what.

        Example of a perfect response for different question types:
        {
          "39": "HOH BDAY - Please rank the following factors from most to least appealing to when considering a kitchen renovation.",
          "40": "HOH BDAY - Which one of the two taglines do you like the most for this campaign?",
          "42": "HOH BDAY - The ads mentioned the “Shop Your Way” program. How much do you agree or disagree with the following statements about the “Shop Your Way” program?"
        }

        Find every single question. Do not skip any.
        """
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "application/pdf", "data": encoded_pdf}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        try:
            with st.spinner("Calling Gemini 1.5 Pro to extract questions... This may take a moment."):
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
                response.raise_for_status()
            response_json = response.json()
            
            candidate = response_json.get('candidates', [{}])[0]
            content_part = candidate.get('content', {}).get('parts', [{}])[0]
            extracted_json_text = content_part.get('text', '{}')
            
            question_dict = json.loads(extracted_json_text)
            
            questions = [Question(number=str(num), text=text, question_type=self._determine_question_type(text)) 
                         for num, text in question_dict.items()]
            return sorted(questions, key=lambda q: int(q.number))

        except Exception as e:
            st.error(f"An error occurred during AI Question Extraction: {e}")
            return []

    def _determine_question_type(self, text: str) -> str:
        text_lower = text.lower()
        for q_type, patterns in self.question_patterns.items():
            if any(p in text_lower for p in patterns):
                return q_type.replace('_response', '')
        return 'single'

    def clean_column_name(self, col_name: str) -> str:
        """Cleans and standardizes a column name for better matching."""
        if not isinstance(col_name, str): return ""
        cleaned = re.sub(r'[^\w\s-]', ' ', col_name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.lower()

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculates a similarity ratio between two strings."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def match_columns_to_questions(self, df: pd.DataFrame, questions: List[Question]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        A more robust, two-pass matching algorithm using "anchoring" and "interpolation".
        """
        columns = list(df.columns)
        mapping = {col: 'UNMATCHED' for col in columns}
        confidence_scores = {col: 0.0 for col in columns}
        
        sorted_questions = sorted(questions, key=lambda q: int(q.number))
        
        # --- Pass 1: High-Confidence Anchoring ---
        anchors = []
        used_q_indices = set()
        for i, col in enumerate(columns):
            clean_col = self.clean_column_name(col)
            if ' - ' in col: continue 
            
            for j, q in enumerate(sorted_questions):
                if j in used_q_indices: continue
                score = self.calculate_similarity(clean_col, q.text)
                if score > 0.95:
                    anchors.append({'col_idx': i, 'q_idx': j})
                    used_q_indices.add(j)
                    break 

        # --- Pass 2: Interpolation between Anchors ---
        proc_anchors = [{'col_idx': -1, 'q_idx': -1}] + anchors + [{'col_idx': len(columns), 'q_idx': len(sorted_questions)}]

        for i in range(len(proc_anchors) - 1):
            start_anchor = proc_anchors[i]
            end_anchor = proc_anchors[i+1]

            col_segment_indices = range(start_anchor['col_idx'] + 1, end_anchor['col_idx'])
            q_segment = sorted_questions[start_anchor['q_idx'] + 1 : end_anchor['q_idx']]

            if not col_segment_indices or not q_segment: continue

            processed_in_segment = set()
            for col_idx in col_segment_indices:
                if col_idx in processed_in_segment: continue
                
                col = columns[col_idx]
                base_question = col.split(' - ')[0].strip()
                
                if ' - ' in col and len(base_question) > 10:
                    group_cols_indices = [c_idx for c_idx in col_segment_indices if columns[c_idx].startswith(base_question + ' - ')]
                    
                    if len(group_cols_indices) > 1:
                        best_q, best_score = None, 0.0
                        for q in q_segment:
                            if q.question_type != 'multi': continue
                            score = self.calculate_similarity(self.clean_column_name(base_question), q.text)
                            if score > best_score:
                                best_score, best_q = score, q
                        
                        if best_q and best_score > 0.70:
                            for group_col_idx in group_cols_indices:
                                mapping[columns[group_col_idx]] = best_q.number
                                confidence_scores[columns[group_col_idx]] = best_score
                                processed_in_segment.add(group_col_idx)

        q_cursor = 0
        for i, col in enumerate(columns):
            if mapping[col] != 'UNMATCHED': continue

            clean_col = self.clean_column_name(col)
            best_q, best_score, best_q_idx = None, 0.0, -1
            
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
                q_cursor = best_q_idx + 1
        
        return mapping, confidence_scores

    def apply_numbering_to_dataset(self, df: pd.DataFrame, mapping: Dict[str, str], prefix_format: str, include_unmatched: bool = True) -> pd.DataFrame:
        df_numbered = df.copy()
        
        if not include_unmatched:
            matched_cols = [col for col in df.columns if mapping.get(col, 'UNMATCHED') != 'UNMATCHED']
            df_numbered = df_numbered[matched_cols]
        
        new_columns = {}
        for col in df_numbered.columns:
            q_num = mapping.get(col)
            if q_num and q_num != 'UNMATCHED' and str(q_num).isdigit():
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
                st.session_state.mapping = json.loads(uploaded_mapping.read())
                st.success("✅ Previous mapping loaded successfully!")
            except Exception as e:
                st.error(f"Error loading mapping: {e}")

    st.markdown("---")
    st.subheader("Step 2: Extract Questions from PDF")
    
    if pdf_file:
        if st.button("🤖 Extract Questions with AI", type="primary", use_container_width=True):
            pdf_bytes = pdf_file.getvalue()
            st.session_state.questions = aligner.extract_questions_with_gemini(pdf_bytes)
            if st.session_state.questions:
                st.success(f"✅ AI successfully extracted {len(st.session_state.questions)} questions.")
            else:
                st.error("AI extraction failed.")
    else:
        st.info("Upload a PDF to enable AI question extraction.")

    if st.session_state.questions:
        with st.expander(f"📋 Review Extracted Questions ({len(st.session_state.questions)} total)"):
            questions_df = pd.DataFrame([{"Number": q.number, "Text": q.text} for q in st.session_state.questions])
            st.dataframe(questions_df, use_container_width=True, height=400)

    if st.session_state.df is not None and st.session_state.questions:
        st.markdown("---")
        st.subheader("Step 3: Run Automatic Alignment")
        if st.button("🚀 Run Automatic Alignment", use_container_width=True):
            with st.spinner("Running new 'self-healing' matching algorithm..."):
                st.session_state.mapping, st.session_state.confidence_scores = aligner.match_columns_to_questions(st.session_state.df, st.session_state.questions)
    
    if st.session_state.mapping:
        st.markdown("---")
        st.subheader("📋 Alignment Results")
        
        question_lookup = {q.number: q.text for q in st.session_state.questions}
        
        total_cols = len(st.session_state.df.columns)
        matched_cols = sum(1 for v in st.session_state.mapping.values() if str(v) != 'UNMATCHED')
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Columns", total_cols)
        with col2: st.metric("Matched", matched_cols, f"{matched_cols/total_cols*100:.1f}%" if total_cols > 0 else "0.0%")
        with col3: st.metric("Unmatched", total_cols - matched_cols)
        
        st.markdown("##### Review and Edit Mappings")
        st.info("💡 Tip: Review the AI's matches. You can manually enter a question number in the 'Assigned #' column for any row.")
        
        # Create the DataFrame for editing from the main mapping
        review_data = []
        for i, col in enumerate(st.session_state.df.columns):
            q_num = st.session_state.mapping.get(col, 'UNMATCHED')
            confidence = st.session_state.confidence_scores.get(col, 0.0)
            matched_text = f"Q{q_num}. {question_lookup.get(str(q_num), 'Text not found')}" if str(q_num) != 'UNMATCHED' else 'N/A - Unmatched'
            
            review_data.append({
                'Col': index_to_excel_col(i),
                'Column Name': col,
                'Matched Question Text': matched_text,
                'Confidence': confidence,
                'Assigned #': str(q_num)
            })
        
        review_df = pd.DataFrame(review_data)
        
        # The data editor now works on the full DataFrame
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
                # On confirmation, update the main mapping from the edited DataFrame
                for idx, row in edited_df.iterrows():
                    st.session_state.mapping[row['Column Name']] = str(row['Assigned #'])
                st.success("✅ Changes confirmed and mapping updated!")
                # A single, intentional rerun to refresh the display with confirmed changes
                st.rerun()
                
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
