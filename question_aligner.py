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
    """Represents a question from the PDF"""
    number: str
    text: str
    question_type: str  # 'single', 'multi', 'matrix', 'open'
    options: List[str] = None

class QuestionNumberAligner:
    def __init__(self):
        self.question_patterns = {
            'multi_response': [
                'select all that apply',
                'check all that apply',
                'please select all'
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
        
        # Common demographic questions for reference
        self.demographic_patterns = {
            'gender': ['gender', 'male', 'female'],
            'age': ['age', 'years old', 'what is your age'],
            'income': ['income', 'household income', 'annual income'],
            'education': ['education', 'highest level', 'degree'],
            'employment': ['employment', 'work', 'occupation'],
            'ethnicity': ['hispanic', 'latino', 'ethnicity'],
            'race': ['race', 'racial', 'black', 'white', 'asian'],
            'location': ['region', 'state', 'area', 'urban', 'suburban']
        }
        
    def parse_pdf_questions(self, pdf_text: str) -> List[Question]:
        """
        Parse questions from PDF text with improved logic to handle multi-line questions
        and clean up common PDF extraction artifacts.
        """
        questions = []

        # This regex splits the text into blocks, with each block starting with a question number.
        # It uses a positive lookahead `(?=...)` to split the text *before* the pattern, keeping the delimiter.
        question_blocks = re.split(r'(?=\n^\d+\.\s)', pdf_text, flags=re.MULTILINE)

        for block in question_blocks:
            block = block.strip()
            if not block:
                continue

            # This regex captures the question number and all the text that follows within the block.
            match = re.match(r'^(\d+)\.\s+(.*)', block, re.DOTALL)
            if match:
                num = match.group(1)
                # Join multi-line text into a single line and clean up extra whitespace.
                text = re.sub(r'\s+', ' ', match.group(2)).strip()

                # --- Text Cleaning Logic ---

                # 1. Remove rating scales that are often incorrectly appended to question text.
                #    e.g., "...following statements? 5. Strongly 4. Somewhat..."
                text = re.sub(r'\s+\d\.\s+(Strongly|Somewhat|Neither|Not at all).+', '', text, flags=re.IGNORECASE)

                # 2. If a question mark exists, truncate the string after it. This is a strong
                #    indicator of the end of the actual question, removing trailing junk.
                if '?' in text:
                    text = text.split('?')[0] + '?'

                # 3. Handle duplicated text, a common PDF extraction artifact where the question
                #    is repeated. e.g., "Question text? 30. Question text?"
                mid_string_q_num = f' {num}. '
                if mid_string_q_num in text:
                    # If the question number appears again mid-string, split on it and keep the first part.
                    text = text.split(mid_string_q_num)[0]

                q_type = self._determine_question_type(text)
                questions.append(Question(num, text.strip(), q_type))
        
        return questions
    
    def _determine_question_type(self, text: str) -> str:
        """Determine question type from text"""
        text_lower = text.lower()
        
        for pattern in self.question_patterns['multi_response']:
            if pattern in text_lower:
                return 'multi'
                
        for pattern in self.question_patterns['matrix']:
            if pattern in text_lower:
                return 'matrix'
                
        for pattern in self.question_patterns['ranking']:
            if pattern in text_lower:
                return 'ranking'
                
        return 'single'
    
    def clean_column_name(self, col_name: str) -> str:
        """Clean and standardize column names"""
        # Remove special characters and extra spaces
        cleaned = re.sub(r'[^\w\s-]', ' ', col_name)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.lower()
    
    def calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def find_multi_response_groups(self, columns: List[str]) -> Dict[str, List[str]]:
        """Group columns that belong to the same multi-response question"""
        groups = {}
        used_columns = set()
        
        for i, col in enumerate(columns):
            if col in used_columns:
                continue
                
            # Pattern 1: "Question text - Option"
            if ' - ' in col:
                base_question = col.split(' - ')[0].strip()
                group = [col]
                used_columns.add(col)
                
                # Find other columns with same base
                for j, other_col in enumerate(columns):
                    if other_col != col and other_col.startswith(base_question + ' - '):
                        group.append(other_col)
                        used_columns.add(other_col)
                
                if len(group) > 1:
                    groups[base_question] = group
                    
            # Pattern 2: Look for "Select all that apply" even without dash
            elif 'select all that apply' in col.lower():
                # Extract the base question up to "Select all that apply"
                base_match = re.search(r'^(.*?select all that apply\.?)', col, re.IGNORECASE)
                if base_match:
                    base_question = base_match.group(1).strip()
                    group = [col]
                    used_columns.add(col)
                    
                    # Find other columns with same base pattern
                    for j, other_col in enumerate(columns):
                        if other_col != col and other_col.lower().startswith(base_question.lower()):
                            group.append(other_col)
                            used_columns.add(other_col)
                    
                    if len(group) > 1:
                        groups[base_question] = group
                        
        return groups
    
    def match_columns_to_questions(self, df: pd.DataFrame, questions: List[Question]) -> Dict[str, str]:
        """Main matching algorithm based on the original logic."""
        columns = list(df.columns)
        mapping = {}
        used_questions = set()
        
        # First pass: Find multi-response groups
        multi_groups = self.find_multi_response_groups(columns)
        
        # Sort groups by length (process larger groups first)
        sorted_groups = sorted(multi_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Process multi-response groups first
        for base, group_cols in sorted_groups:
            best_score = 0
            best_match = None
            
            clean_base = self.clean_column_name(base)
            
            # Find the best matching question
            for q in questions:
                if q.number not in used_questions:
                    score = self.calculate_similarity(clean_base, q.text)
                    
                    # Boost score if question has "select all that apply"
                    if 'select all' in q.text.lower() and 'select all' in base.lower():
                        score += 0.4
                    
                    if score > best_score:
                        best_score = score
                        best_match = q.number
            
            # Assign same question number to all columns in group
            if best_match and best_score > 0.3:  # Lower threshold for groups
                for group_col in group_cols:
                    mapping[group_col] = best_match
                used_questions.add(best_match)
        
        # Second pass: Match remaining columns
        for col in columns:
            if col in mapping:
                continue
            
            best_score = 0
            best_match = None
            clean_col = self.clean_column_name(col)
            
            for q in questions:
                if q.number not in used_questions:
                    score = self.calculate_similarity(clean_col, q.text)
                    
                    # Boost score for demographic questions
                    try:
                        if int(q.number) >= 180:  # Demographic section
                            for demo_type, patterns in self.demographic_patterns.items():
                                if any(p in clean_col for p in patterns):
                                    score += 0.3
                                    break
                    except ValueError:
                        pass # q.number might not be a digit
                    
                    if score > best_score:
                        best_score = score
                        best_match = q.number
            
            if best_match and best_score > 0.4:  # Threshold
                mapping[col] = best_match
                # Only mark as used if high confidence
                if best_score > 0.7:
                    used_questions.add(best_match)
            else:
                mapping[col] = 'UNMATCHED'
                
        return mapping
    
    def apply_numbering_to_dataset(self, df: pd.DataFrame, mapping: Dict[str, str], prefix_format: str = "Q{num}. ") -> pd.DataFrame:
        """Apply the question numbers to the dataset with customizable prefix format"""
        df_numbered = df.copy()
        new_columns = {}
        for col in df_numbered.columns:
            q_num = mapping.get(col)
            if q_num and q_num != 'UNMATCHED' and q_num.isdigit():
                prefix = prefix_format.format(num=q_num)
                new_columns[col] = f"{prefix}{col}"
            else:
                # For non-numeric or unmatched, keep original name
                new_columns[col] = col
        
        df_numbered.rename(columns=new_columns, inplace=True)
        return df_numbered

def create_streamlit_app():
    """Initializes and runs the Streamlit user interface."""
    st.set_page_config(page_title="Question Number Aligner", layout="wide")
    
    st.title("📊 Question Number Alignment Tool")
    st.markdown("---")
    
    # Initialize session state
    if 'mapping' not in st.session_state:
        st.session_state.mapping = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    
    aligner = QuestionNumberAligner()

    # --- UI Section ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Dataset (CSV/Excel)")
        data_file = st.file_uploader("Choose your dataset file", type=['csv', 'xlsx'])
        
    with col2:
        st.subheader("📄 Upload PDF Overview")
        pdf_file = st.file_uploader("Choose your PDF file", type=['pdf'])
        st.info("Note: For demo purposes, you can also paste the question list below")
        
    with st.expander("📝 Or paste question list here (for testing)"):
        question_text_manual = st.text_area(
            "Paste questions in format: 'number. question text'",
            height=200,
            placeholder="1. To which gender identity do you most identify?\n2. What is your age?\n..."
        )
        
    # --- Processing Section ---
    if data_file:
        try:
            if data_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(data_file)
            else:
                st.session_state.df = pd.read_excel(data_file)
            st.success(f"✅ Dataset loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.session_state.df = None

    question_text_to_parse = ""
    if pdf_file:
        try:
            # Use the full text from the PDF for the new parser
            with pdfplumber.open(pdf_file) as pdf:
                full_pdf_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            
            if full_pdf_text:
                st.session_state.questions = aligner.parse_pdf_questions(full_pdf_text)
                st.success(f"✅ Extracted {len(st.session_state.questions)} questions from PDF using improved parser.")
                # Show the cleaned, extracted questions for review
                with st.expander("📋 Review extracted questions"):
                    st.json({q.number: q.text for q in st.session_state.questions})
            st.subheader("Raw Extracted PDF Text")
            st.text_area("Copy this text", full_pdf_text, height=300)
            else:
                st.warning("Could not extract any text from the PDF.")
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
    
    elif question_text_manual:
        st.session_state.questions = aligner.parse_pdf_questions(question_text_manual)
        st.success(f"✅ Parsed {len(st.session_state.questions)} questions")
    
    # --- Alignment and Results Section ---
    if st.session_state.df is not None and st.session_state.questions:
        if st.button("🚀 Run Automatic Alignment", type="primary"):
            with st.spinner("Analyzing and matching columns..."):
                st.session_state.mapping = aligner.match_columns_to_questions(
                    st.session_state.df, 
                    st.session_state.questions
                )
                
    if st.session_state.mapping:
        st.markdown("---")
        st.subheader("📋 Alignment Results")
        
        # Create review dataframe
        review_data = []
        for col in st.session_state.df.columns:
            q_num = st.session_state.mapping.get(col, 'UNMATCHED')
            status = '✏️ Manual'
            if q_num == 'UNMATCHED':
                status = '❌ Unmatched'
            elif str(q_num).isdigit():
                status = '✅ Matched'
            elif q_num in ['META', 'ID', 'SKIP']:
                status = '⚪ Metadata/Skip'

            review_data.append({
                'Column Name': col,
                'Assigned Question': str(q_num),
                'Status': status
            })
        
        review_df = pd.DataFrame(review_data)
        
        # Summary metrics
        total_cols = len(review_df)
        matched = (review_df['Status'] == '✅ Matched').sum()
        unmatched = (review_df['Status'] == '❌ Unmatched').sum()
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Columns", total_cols)
        m_col2.metric("Matched", matched, f"{matched/total_cols*100:.1f}%" if total_cols > 0 else "0.0%")
        m_col3.metric("Unmatched", unmatched)
            
        # Filter options
        filter_option = st.radio("Show:", ["All", "Matched only", "Unmatched only"], horizontal=True)
        
        display_df = review_df
        if filter_option == "Matched only":
            display_df = review_df[review_df['Status'] == '✅ Matched']
        elif filter_option == "Unmatched only":
            display_df = review_df[review_df['Status'] == '❌ Unmatched']
        
        # Editable dataframe
        st.markdown("### Review and Edit Mappings")
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Column Name": st.column_config.TextColumn("Column Name", width="large", disabled=True),
                "Assigned Question": st.column_config.TextColumn("Assigned Question", help="Edit question numbers here"),
                "Status": st.column_config.TextColumn("Status", disabled=True)
            },
            use_container_width=True,
            num_rows="dynamic"
        )
        
        if st.button("✏️ Apply Manual Corrections"):
            # Create a mapping from the original index to the new values
            update_map = {row['Column Name']: row['Assigned Question'] for index, row in edited_df.iterrows()}
            # Update the main session state mapping
            for col_name, new_q_num in update_map.items():
                st.session_state.mapping[col_name] = new_q_num
            st.success("✅ Manual corrections applied!")
            st.rerun()
            
        # --- Export Section ---
        st.markdown("---")
        st.subheader("💾 Export Options")
        
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            prefix_format = st.selectbox(
                "Question Number Format",
                ["Q{num}. ", "{num}. ", "#{num}. ", "Question {num}: ", "Q{num} - ", "{num}) "],
                help="Choose how question numbers appear in the export"
            )
            st.caption(f"Preview: {prefix_format.format(num='1')}Your question text here")

        with e_col2:
            include_unmatched = st.checkbox("Include unmatched columns in export", value=True)
        
        # Generate Numbered Dataset
        numbered_df = aligner.apply_numbering_to_dataset(
            st.session_state.df, 
            st.session_state.mapping, 
            prefix_format=prefix_format
        )
        
        if not include_unmatched:
            # Filter out columns that were not successfully matched
            cols_to_keep = [
                col for original_col, col in zip(st.session_state.df.columns, numbered_df.columns)
                if st.session_state.mapping.get(original_col, 'UNMATCHED') not in ['UNMATCHED']
            ]
            numbered_df = numbered_df[cols_to_keep]

        # Prepare for download
        csv_data = numbered_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            numbered_df.to_excel(writer, index=False, sheet_name='Numbered Data')
        excel_data = output.getvalue()

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📊 Download Numbered Dataset (CSV)",
                data=csv_data,
                file_name="numbered_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl_col2:
            st.download_button(
                label="📑 Download Numbered Dataset (Excel)",
                data=excel_data,
                file_name="numbered_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

if __name__ == "__main__":
    create_streamlit_app()
