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
        # Keywords to help classify questions after they've been extracted by the AI.
        self.question_patterns = {
            'multi_response': ['select all that apply', 'check all that apply', 'please select all', 'choose as many'],
            'matrix': ['how much do you agree', 'please rate', 'on a scale', 'how appealing', 'rate the following'],
            'ranking': ['please rank', 'rank the following']
        }

    def extract_questions_with_gemini(self, pdf_file_bytes: bytes) -> List[Question]:
        """
        Sends a PDF file to the Gemini API to extract questions and returns them as a list of Question objects.
        """
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("GEMINI_API_KEY is not set in your Streamlit secrets. Please add it to run the AI extraction.")
            return []

        # Using the more powerful 'pro' model for the complex extraction task.
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

    def match_with_gemini(self, df: pd.DataFrame, questions: List[Question]) -> Dict[str, str]:
        """
        Uses the Gemini API to perform the matching between dataset columns and extracted questions.
        """
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("GEMINI_API_KEY is not set. Cannot perform AI matching.")
            return {}

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"

        question_dict = {q.number: q.text for q in questions}
        column_list = list(df.columns)

        # --- NEW, MORE GENERALIZED AND ROBUST PROMPT ---
        prompt = f"""
        You are an expert survey data analyst. Your task is to align dataset column headers with a list of official survey questions.
        You will be provided with two JSON objects:
        1. `survey_questions`: A dictionary where keys are question numbers and values are the full question text.
        2. `column_headers`: A list of column header strings from the dataset.

        Follow these rules precisely:
        1.  For each `column_header`, find the single best matching `survey_question`.
        2.  **Recognize Multi-Column Questions:** Questions like multi-selects, matrix, or ranking questions are represented by multiple columns in the dataset. These columns usually share the same base text. For example, "Q7. Which do you own? - Car" and "Q7. Which do you own? - Boat" should BOTH map to question "7". You MUST correctly assign the same question number to all parts of a multi-column question.
        3.  **Use Sequential Order as a Guide:** The order of `column_headers` generally follows the order of `survey_questions`. Use this as a strong clue, but be flexible. The survey may have repeating blocks of questions (monadic design).
        4.  **Do Not Force a Match:** If a `column_header` does not clearly correspond to any of the `survey_questions` (e.g., it is a programming note, an open-ended response field not in the PDF, or metadata), you MUST assign it the value "UNMATCHED". It is better to leave a column unmatched than to assign it an incorrect number.
        5.  Your final output must be a single, valid JSON object where the keys are the exact original `column_headers` and the values are the corresponding question numbers (as strings) or the string "UNMATCHED".

        Here are the inputs:

        **Survey Questions:**
        {json.dumps(question_dict, indent=2)}

        **Column Headers:**
        {json.dumps(column_list, indent=2)}

        Now, provide the final JSON mapping object.
        """

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        try:
            with st.spinner("Calling Gemini 1.5 Pro for intelligent matching..."):
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
                response.raise_for_status()
            
            response_json = response.json()
            
            with st.expander("🔍 AI Matching Response (for debugging)"):
                st.json(response_json)

            candidate = response_json.get('candidates', [{}])[0]
            content_part = candidate.get('content', {}).get('parts', [{}])[0]
            mapping_json_text = content_part.get('text', '{}')

            mapping = json.loads(mapping_json_text)
            return mapping

        except Exception as e:
            st.error(f"An error occurred during AI Matching: {e}")
            return {col: "ERROR" for col in column_list}

    def _determine_question_type(self, text: str) -> str:
        text_lower = text.lower()
        for q_type, patterns in self.question_patterns.items():
            if any(p in text_lower for p in patterns):
                return q_type.replace('_response', '')
        return 'single'

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
        st.subheader("Step 3: Run AI-Powered Alignment")
        if st.button("🚀 Run AI Alignment", use_container_width=True):
            st.session_state.mapping = aligner.match_with_gemini(st.session_state.df, st.session_state.questions)
    
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
            matched_text = f"Q{q_num}. {question_lookup.get(str(q_num), 'Text not found')}" if str(q_num) != 'UNMATCHED' else 'N/A - Unmatched'
            
            review_data.append({
                'Col': index_to_excel_col(i),
                'Column Name': col,
                'Matched Question Text': matched_text,
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
