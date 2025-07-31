import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
import streamlit as st
import numpy as np
import io

# It's good practice to check for the library and guide the user if it's missing.
try:
    import pdfplumber
except ImportError:
    st.error("The 'pdfplumber' library is required to read PDF files. Please install it by running: pip install pdfplumber")
    st.stop()

@dataclass
class Question:
    """Represents a question from the PDF survey overview."""
    number: str
    text: str
    question_type: str  # 'single', 'multi', 'matrix', 'open', 'ranking'
    options: List[str] = None

class QuestionNumberAligner:
    """Handles the core logic of matching dataset columns to survey questions."""
    def __init__(self):
        # Keywords to identify different types of questions from their text.
        self.question_patterns = {
            'multi_response': [
                'select all that apply',
                'check all that apply',
                'please select all',
                'which of the following'
            ],
            'matrix': [
                'how much do you agree',
                'please rate',
                'on a scale',
                'how appealing'
            ],
            'ranking': [
                'please rank',
                'rank the following'
            ]
        }
        
        # Common demographic question keywords to improve matching for that section.
        self.demographic_patterns = {
            'gender': ['gender', 'male', 'female'],
            'age': ['age', 'years old', 'what is your age'],
            'income': ['income', 'household income', 'annual income'],
            'education': ['education', 'highest level', 'degree'],
            'employment': ['employment', 'work', 'occupation'],
            'ethnicity': ['hispanic', 'latino', 'ethnicity'],
            'race': ['race', 'racial', 'black', 'white', 'asian'],
            'location': ['region', 'state', 'area', 'urban', 'suburban', 'zip code']
        }

    def parse_questions_from_text(self, text_content: str) -> List[Question]:
        """
        Parses a list of questions from a block of text.
        This improved regex handles multi-line questions better.
        """
        questions = []
        # Regex to find lines starting with a number and a period.
        # It captures the number and the following text until the next question number.
        pattern = re.compile(r'^(?P<number>\d+)\.\s+(?P<text>.+?)(?=\n\d+\.|\Z)', re.DOTALL | re.MULTILINE)
        
        for match in pattern.finditer(text_content):
            num = match.group('number')
            # Clean up the text: remove newlines and extra spaces
            text = re.sub(r'\s+', ' ', match.group('text')).strip()
            
            # Further cleanup to remove artifacts from PDF tables/scales
            text = re.sub(r'\s+\d+\s+(Very|Somewhat|Neither|Not at all|Strongly)\s?.*', '', text, flags=re.IGNORECASE)
            
            q_type = self._determine_question_type(text)
            questions.append(Question(number=num, text=text, question_type=q_type))
            
        return questions

    def _determine_question_type(self, text: str) -> str:
        """Determines the question type based on keywords."""
        text_lower = text.lower()
        for q_type, patterns in self.question_patterns.items():
            if any(p in text_lower for p in patterns):
                return q_type.replace('_response', '')
        return 'single'

    def clean_column_name(self, col_name: str) -> str:
        """Cleans and standardizes a column name for better matching."""
        # Remove special characters, except for hyphens which can be meaningful.
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

        for i, col in enumerate(columns):
            if col in used_columns:
                continue

            # Common pattern: "Question Text - Option"
            if ' - ' in col:
                base_question = col.split(' - ')[0].strip()
                # Avoid grouping if the base is too short (e.g., "Q1")
                if len(base_question) < 10: continue

                current_group = [col]
                for other_col in columns:
                    if other_col != col and other_col.startswith(base_question + ' - '):
                        current_group.append(other_col)
                
                if len(current_group) > 1:
                    groups[base_question] = current_group
                    used_columns.update(current_group)
        
        return groups

    def match_columns_to_questions(self, df: pd.DataFrame, questions: List[Question], similarity_threshold: float = 0.4) -> Dict[str, str]:
        """
        The main matching algorithm. It maps DataFrame columns to question numbers.
        """
        columns = list(df.columns)
        mapping = {}
        used_question_nums = set()

        # First pass: Handle multi-response groups
        multi_groups = self.find_multi_response_groups(columns)
        for base_question, group_cols in multi_groups.items():
            clean_base = self.clean_column_name(base_question)
            best_match_q = None
            highest_score = 0.0

            for q in questions:
                if q.number in used_question_nums:
                    continue
                
                score = self.calculate_similarity(clean_base, q.text)
                # Boost score for explicit multi-response questions
                if q.question_type == 'multi':
                    score += 0.2
                
                if score > highest_score:
                    highest_score = score
                    best_match_q = q
            
            if best_match_q and highest_score > similarity_threshold:
                for col in group_cols:
                    mapping[col] = best_match_q.number
                used_question_nums.add(best_match_q.number)
                # Remove these columns from the main list to avoid re-processing
                columns = [c for c in columns if c not in group_cols]

        # Second pass: Match remaining individual columns
        for col in columns:
            clean_col = self.clean_column_name(col)
            best_match_q = None
            highest_score = 0.0

            for q in questions:
                if q.number in used_question_nums:
                    continue
                
                score = self.calculate_similarity(clean_col, q.text)
                
                # Boost score for demographic questions based on keywords
                is_demographic = any(p in clean_col for patterns in self.demographic_patterns.values() for p in patterns)
                if is_demographic and any(p in q.text.lower() for patterns in self.demographic_patterns.values() for p in patterns):
                    score += 0.3
                
                if score > highest_score:
                    highest_score = score
                    best_match_q = q
            
            if best_match_q and highest_score > similarity_threshold:
                mapping[col] = best_match_q.number
                # Once a question is used, don't use it again to prevent duplicates.
                used_question_nums.add(best_match_q.number)
            else:
                mapping[col] = 'UNMATCHED'
                
        return mapping

    def apply_numbering_to_dataset(self, df: pd.DataFrame, mapping: Dict[str, str], prefix_format: str) -> pd.DataFrame:
        """Applies the new question numbers to the dataset's column headers."""
        df_numbered = df.copy()
        new_columns = {}
        
        for col in df_numbered.columns:
            q_num = mapping.get(col, 'UNMATCHED')
            if q_num not in ['UNMATCHED', 'META', 'ID', 'SKIP'] and str(q_num).isdigit():
                prefix = prefix_format.format(num=q_num)
                new_columns[col] = f"{prefix}{col}"
            else:
                new_columns[col] = col # Keep original name
                
        df_numbered.rename(columns=new_columns, inplace=True)
        return df_numbered

def create_streamlit_app():
    """Initializes and runs the Streamlit user interface."""
    st.set_page_config(page_title="Question Number Aligner", layout="wide")
    
    st.title("📊 Question Number Alignment Tool")
    st.markdown("Match survey dataset columns to a PDF overview automatically. Upload your files, run the alignment, review the matches, and export your newly numbered dataset.")
    st.markdown("---")

    # Initialize session state variables
    if 'mapping' not in st.session_state:
        st.session_state.mapping = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'questions' not in st.session_state:
        st.session_state.questions = []

    aligner = QuestionNumberAligner()

    # --- Step 1: File Upload ---
    st.subheader("Step 1: Upload Your Files")
    col1, col2 = st.columns(2)
    with col1:
        data_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    with col2:
        pdf_file = st.file_uploader("Upload Survey PDF", type=['pdf'])

    with st.expander("📝 Or, Paste Question List Manually"):
        question_text_manual = st.text_area(
            "Paste questions in 'number. question text' format, one per line:",
            height=200,
            placeholder="1. To which gender identity do you most identify?\n2. What is your age?\n..."
        )

    # Process uploaded files
    if data_file:
        try:
            if data_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(data_file)
            else:
                st.session_state.df = pd.read_excel(data_file)
            st.success(f"✅ Dataset loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns.")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.session_state.df = None

    pdf_text_content = ""
    if pdf_file:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                pdf_text_content = "\n".join(page.extract_text() for page in pdf.pages)
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    
    # Use manual text if provided, otherwise use PDF text
    final_question_text = question_text_manual if question_text_manual else pdf_text_content
    if final_question_text:
        st.session_state.questions = aligner.parse_questions_from_text(final_question_text)
        if st.session_state.questions:
            st.success(f"✅ Parsed {len(st.session_state.questions)} questions.")
            with st.expander("📋 Review Extracted Questions"):
                st.json({q.number: q.text for q in st.session_state.questions})
        else:
            st.warning("Could not automatically parse questions. Please check the format or paste them manually.")

    # --- Step 2: Run Alignment ---
    st.markdown("---")
    st.subheader("Step 2: Run Alignment")
    if st.session_state.df is not None and st.session_state.questions:
        if st.button("🚀 Run Automatic Alignment", type="primary", use_container_width=True):
            with st.spinner("Analyzing and matching columns... This may take a moment."):
                st.session_state.mapping = aligner.match_columns_to_questions(
                    st.session_state.df, 
                    st.session_state.questions
                )
    else:
        st.info("Please upload a dataset and provide questions to enable alignment.")

    # --- Step 3: Review and Export ---
    if st.session_state.mapping:
        st.markdown("---")
        st.subheader("Step 3: Review, Edit, and Export")

        # Create a DataFrame for reviewing the mappings
        review_data = []
        for col in st.session_state.df.columns:
            q_num = st.session_state.mapping.get(col, 'UNMATCHED')
            
            status = '⚪ Manual'
            if q_num == 'UNMATCHED':
                status = '❌ Unmatched'
            elif str(q_num).isdigit():
                status = '✅ Matched'
            
            review_data.append({
                'Column Name': col,
                'Assigned Question': str(q_num), # Ensure it's a string for the editor
                'Status': status
            })
        
        review_df = pd.DataFrame(review_data).set_index('Column Name')

        # Display summary metrics
        total_cols = len(review_df)
        matched_count = (review_df['Status'] == '✅ Matched').sum()
        unmatched_count = (review_df['Status'] == '❌ Unmatched').sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Columns", total_cols)
        col2.metric("Automatically Matched", f"{matched_count}", f"{matched_count/total_cols*100:.1f}%")
        col3.metric("Unmatched", f"{unmatched_count}")

        # Editable table for corrections
        st.markdown("##### Review and Edit Mappings")
        st.info("You can edit the 'Assigned Question' column below. Change numbers or use labels like 'META' or 'SKIP'.")
        
        edited_df = st.data_editor(
            review_df,
            column_config={
                "Assigned Question": st.column_config.TextColumn(help="Edit question numbers or add labels (e.g., 'META', 'SKIP')."),
                "Status": st.column_config.TextColumn(help="Indicates the current match status.", disabled=True)
            },
            use_container_width=True,
            height=400
        )

        # Update the main session state mapping from the editor's changes
        for col_name, row in edited_df.iterrows():
            st.session_state.mapping[col_name] = row['Assigned Question']

        # Export options
        st.markdown("##### Export Your Renumbered Dataset")
        
        export_cols = st.columns(2)
        with export_cols[0]:
            prefix_format = st.selectbox(
                "Question Number Format",
                ["Q{num}. ", "{num}. ", "#{num}. ", "Question {num}: ", "Q{num} - ", "{num}) "],
                help="Choose how question numbers are prefixed to column names."
            )
        with export_cols[1]:
            include_unmatched = st.checkbox("Include unmatched columns in export", value=True)

        # Generate the final DataFrame for download
        final_df = aligner.apply_numbering_to_dataset(
            st.session_state.df, 
            st.session_state.mapping, 
            prefix_format=prefix_format
        )
        
        # Filter out unmatched columns if requested
        if not include_unmatched:
            cols_to_keep = [
                col for original_col, col in zip(st.session_state.df.columns, final_df.columns)
                if st.session_state.mapping.get(original_col, 'UNMATCHED') != 'UNMATCHED'
            ]
            final_df = final_df[cols_to_keep]

        # Prepare data for download buttons
        csv_data = final_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Numbered Data')
        excel_data = output.getvalue()

        dl_cols = st.columns(2)
        with dl_cols[0]:
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name="renumbered_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl_cols[1]:
            st.download_button(
                label="📑 Download as Excel",
                data=excel_data,
                file_name="renumbered_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    create_streamlit_app()
