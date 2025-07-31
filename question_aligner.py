import pandas as pd
import re
import json
import base64
import requests
from typing import Dict, List
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
        # The API key is handled by the environment, so it can be left blank here.
        api_key = ""
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
        
        # Encode the PDF file bytes into base64.
        encoded_pdf = base64.b64encode(pdf_file_bytes).decode('utf-8')
        
        # A detailed prompt telling the AI exactly what to do.
        prompt = """
        You are an expert data cleaning assistant specializing in survey reports.
        Analyze the provided PDF document. Your only task is to identify and extract every numbered question.

        Follow these rules strictly:
        1.  Ignore everything that is not a numbered question. This includes page headers, footers, charts, graphs, data tables, answer choices, and percentages.
        2.  Extract the full, complete question text, even if it spans multiple lines.
        3.  Clean the question text perfectly. Remove any trailing text, answer options, or junk characters.
        4.  Return the output as a single, valid JSON object. The keys should be the question numbers as strings, and the values should be the clean question text.
        
        Example of a perfect response:
        {
          "1": "To which gender identity do you most identify? Select one.",
          "3": "What is your age?",
          "4": "What region of the country do you currently live in?",
          "6": "Which of the following do you currently own? Select al that apply."
        }
        """
        
        # Construct the API request payload.
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "application/pdf",
                                "data": encoded_pdf
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }

        try:
            # Make the API call to Gemini.
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            response_json = response.json()
            
            # Extract the JSON content from the API's response.
            candidate = response_json.get('candidates', [])[0]
            content_part = candidate.get('content', {}).get('parts', [])[0]
            extracted_json_text = content_part.get('text', '{}')
            
            # Parse the JSON string into a Python dictionary.
            question_dict = json.loads(extracted_json_text)
            
            # Convert the dictionary into a list of Question objects.
            questions = []
            for num, text in question_dict.items():
                q_type = self._determine_question_type(text)
                questions.append(Question(number=str(num), text=text, question_type=q_type))
            
            return questions

        except requests.exceptions.RequestException as e:
            st.error(f"Network error calling Gemini API: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            st.error(f"Error parsing response from Gemini API: {e}")
            st.error("The API may have returned an unexpected format. Check the raw response:")
            st.json(response.json())
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return []

    def _determine_question_type(self, text: str) -> str:
        """Classifies a question's type based on keywords in its text."""
        text_lower = text.lower()
        for q_type, patterns in self.question_patterns.items():
            if any(p in text_lower for p in patterns):
                return q_type.replace('_response', '')
        return 'single'

    def clean_column_name(self, col_name: str) -> str:
        """Cleans and standardizes a column name for better matching."""
        cleaned = re.sub(r'[^\w\s-]', ' ', col_name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.lower()

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculates a similarity ratio between two strings."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def find_multi_response_groups(self, columns: List[str]) -> Dict[str, List[str]]:
        """Groups columns that likely belong to the same multi-response question."""
        groups = {}
        used_columns = set()
        for col in columns:
            if col in used_columns:
                continue
            if ' - ' in col:
                base_question = col.split(' - ')[0].strip()
                if len(base_question) < 10: continue
                current_group = [c for c in columns if c.startswith(base_question + ' - ')]
                if len(current_group) > 1:
                    groups[base_question] = current_group
                    used_columns.update(current_group)
        return groups

    def match_columns_to_questions(self, df: pd.DataFrame, questions: List[Question]) -> Dict[str, str]:
        """Main matching algorithm to link dataset columns to the extracted questions."""
        columns = list(df.columns)
        mapping = {}
        used_question_nums = set()

        multi_groups = self.find_multi_response_groups(columns)
        for base_question, group_cols in multi_groups.items():
            clean_base = self.clean_column_name(base_question)
            best_match_q, highest_score = None, 0.0
            for q in questions:
                if q.number in used_question_nums: continue
                score = self.calculate_similarity(clean_base, q.text)
                if q.question_type == 'multi': score += 0.2
                if score > highest_score:
                    highest_score, best_match_q = score, q
            if best_match_q and highest_score > 0.3:
                for col in group_cols:
                    mapping[col] = best_match_q.number
                used_question_nums.add(best_match_q.number)

        for col in columns:
            if col in mapping: continue
            clean_col = self.clean_column_name(col)
            best_match_q, highest_score = None, 0.0
            for q in questions:
                if q.number in used_question_nums: continue
                score = self.calculate_similarity(clean_col, q.text)
                if any(p in clean_col for patterns in self.demographic_patterns.values() for p in patterns) and \
                   any(p in q.text.lower() for patterns in self.demographic_patterns.values() for p in patterns):
                    score += 0.3
                if score > highest_score:
                    highest_score, best_match_q = score, q
            if best_match_q and highest_score > 0.4:
                mapping[col] = best_match_q.number
                if highest_score > 0.7: used_question_nums.add(best_match_q.number)
            else:
                mapping[col] = 'UNMATCHED'
        return mapping

    def apply_numbering_to_dataset(self, df: pd.DataFrame, mapping: Dict[str, str], prefix_format: str) -> pd.DataFrame:
        """Applies the new question numbers to the dataset's column headers."""
        df_numbered = df.copy()
        new_columns = {
            col: f"{prefix_format.format(num=mapping.get(col))}{col}"
            if mapping.get(col) and mapping[col] not in ['UNMATCHED'] and mapping[col].isdigit()
            else col
            for col in df_numbered.columns
        }
        df_numbered.rename(columns=new_columns, inplace=True)
        return df_numbered

# --- Streamlit UI ---
def create_streamlit_app():
    """Initializes and runs the Streamlit user interface."""
    st.set_page_config(page_title="AI-Powered Question Aligner", layout="wide")
    
    st.title("📊 AI-Powered Question Number Alignment Tool")
    st.markdown("This tool uses AI to read a survey PDF, extract the questions, and match them to your dataset columns.")
    st.markdown("---")

    if 'mapping' not in st.session_state:
        st.session_state.mapping = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    
    aligner = QuestionNumberAligner()

    # --- Step 1: Upload Files ---
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

    # --- Step 2: Extract Questions with AI ---
    st.markdown("---")
    st.subheader("Step 2: Extract Questions from PDF")
    if pdf_file:
        if st.button("🤖 Extract Questions with AI", type="primary", use_container_width=True):
            with st.spinner("AI is reading and analyzing your PDF... This may take a moment."):
                pdf_bytes = pdf_file.getvalue()
                st.session_state.questions = aligner.extract_questions_with_gemini(pdf_bytes)
            if st.session_state.questions:
                st.success(f"✅ AI successfully extracted {len(st.session_state.questions)} questions.")
            else:
                st.error("AI extraction failed. Please check the error messages above or try a different PDF.")
    else:
        st.info("Upload a PDF to enable AI question extraction.")

    if st.session_state.questions:
        with st.expander("📋 Review Extracted Questions"):
            st.json({q.number: q.text for q in st.session_state.questions})

    # --- Step 3: Run Alignment & Export ---
    if st.session_state.df is not None and st.session_state.questions:
        st.markdown("---")
        st.subheader("Step 3: Run Alignment & Export")
        if st.button("🚀 Run Automatic Alignment", use_container_width=True):
            with st.spinner("Analyzing and matching columns..."):
                st.session_state.mapping = aligner.match_columns_to_questions(
                    st.session_state.df, 
                    st.session_state.questions
                )
    
    if st.session_state.mapping:
        st.markdown("---")
        st.subheader("📋 Alignment Results")
        
        review_data = []
        for col in st.session_state.df.columns:
            q_num = st.session_state.mapping.get(col, 'UNMATCHED')
            status = '✅ Matched' if str(q_num).isdigit() else '❌ Unmatched'
            review_data.append({'Column Name': col, 'Assigned Question': str(q_num), 'Status': status})
        
        review_df = pd.DataFrame(review_data).set_index('Column Name')
        
        st.markdown("##### Review and Edit Mappings")
        edited_df = st.data_editor(review_df, use_container_width=True, height=400)

        for col_name, row in edited_df.iterrows():
            st.session_state.mapping[col_name] = row['Assigned Question']

        st.markdown("##### Export Your Renumbered Dataset")
        prefix_format = st.selectbox(
            "Question Number Format",
            ["Q{num}. ", "{num}. ", "#{num}. ", "Question {num}: "],
            help="Choose how question numbers are prefixed to column names."
        )
        
        final_df = aligner.apply_numbering_to_dataset(st.session_state.df, st.session_state.mapping, prefix_format)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Numbered Data')
        excel_data = output.getvalue()

        st.download_button(
            label="📑 Download Renumbered Dataset (Excel)",
            data=excel_data,
            file_name="renumbered_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

if __name__ == "__main__":
    create_streamlit_app()
