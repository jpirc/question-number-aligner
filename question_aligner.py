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
                'how appealing',
                'how important' # Added common matrix pattern
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
        """Parse questions from PDF text."""
        questions = []
        lines = pdf_text.split('\n')
        current_question = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match = re.match(r'^(\d+)\.\s*(.+)', line) # Adjusted regex for optional space

            if match:
                # If there was a previous question, add it
                if current_question:
                    questions.append(current_question)

                num, text_start = match.groups()
                full_text = text_start
                q_type = self._determine_question_type(full_text)

                # Look ahead to capture multi-line questions and avoid answer options
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()

                    # Check if the next line is a potential answer option (e.g., starts with a number/letter and period/parenthesis)
                    # or the start of a new question.
                    is_potential_option = re.match(r'^(\d+|\w)\.?\s+', next_line) or re.match(r'^(\d+)\.\s*(.+)', next_line)

                    # Add the next line if it doesn't look like an option or new question start
                    if not is_potential_option and next_line:
                        full_text += " " + next_line
                        j += 1
                    else:
                        break # Stop if next line is a potential option or new question

                current_question = Question(num, full_text.strip(), q_type)
                i = j # Continue scanning from where the lookahead stopped
            else:
                i += 1 # Move to the next line if no question number is found

        # Add the last question found
        if current_question:
            questions.append(current_question)

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
            # This pattern is less reliable for grouping without a clear delimiter like " - "
            # Consider if this pattern should still be used for grouping or just type detection
            # For now, keep it but be mindful of potential mis-groupings.
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
                # Do NOT mark as used yet, as multi-response questions might also have a summary column
                # used_questions.add(best_match) # Removed this line

        # Second pass: Match remaining columns
        for col in columns:
            if col in mapping:
                # If already mapped by a group, check if it's a single column that could also match the full question text well
                # This handles cases where there's a summary column for a multi-select question
                best_score = self.calculate_similarity(self.clean_column_name(col), questions[int(mapping[col])-1].text) # Calculate score with the already mapped question
                if best_score > 0.7: # High confidence re-match
                     used_questions.add(mapping[col]) # Now mark as used if confident
                continue

            best_score = 0
            best_match = None
            clean_col = self.clean_column_name(col)

            for q in questions:
                if q.number not in used_questions:
                    score = self.calculate_similarity(clean_col, q.text)

                    # Boost score for demographic questions - still apply but maybe refine threshold
                    try:
                        # Check if the question number is relatively high, indicating later in the survey
                        if int(q.number) > len(questions) * 0.8: # Assuming demographics are often at the end
                            for demo_type, patterns in self.demographic_patterns.items():
                                if any(p in clean_col for p in patterns):
                                    score += 0.3
                                    break
                    except ValueError:
                        pass # q.number might not be a digit or questions list might be empty

                    if score > best_score:
                        best_score = score
                        best_match = q.number

            # Apply mapping and mark as used only if confident or it's the best match among candidates
            if best_match and best_score > 0.4:  # Threshold
                mapping[col] = best_match
                # Only mark as used if high confidence (e.g., above 0.7) or if this is a definitive single match for a question
                # Refined logic needed here to handle cases where multiple columns could potentially match one question
                # For simplicity, keep the high confidence rule for now.
                if best_score > 0.7:
                    used_questions.add(best_match)
            else:
                mapping[col] = 'UNMATCHED'

        # Final pass to potentially re-assign UNMATCHED based on lower threshold or specific patterns if needed
        # This could involve checking unmatched columns against unused questions with a lower similarity score,
        # or applying specific rules for known problematic patterns.
        # For now, the current logic should improve upon the previous version.

        return mapping


    def apply_numbering_to_dataset(self, df: pd.DataFrame, mapping: Dict[str, str], prefix_format: str = "Q{num}. ") -> pd.DataFrame:
        """Apply the question numbers to the dataset with customizable prefix format"""
        df_numbered = df.copy()
        new_columns = {}
        for col in df_numbered.columns:
            q_num = mapping.get(col)
            if q_num and q_num != 'UNMATCHED' and str(q_num).isdigit(): # Ensure q_num is a digit
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
            with pdfplumber.open(pdf_file) as pdf:
                pdf_text_content = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

            # Filter lines that start with a number followed by a dot and space/text
            question_lines = [line.strip() for line in pdf_text_content.split('\n') if re.match(r'^\d+\.\s*', line.strip())]

            if question_lines:
                question_text_to_parse = '\n'.join(question_lines)
                st.session_state.questions = aligner.parse_pdf_questions(question_text_to_parse)
                st.success(f"✅ Extracted {len(st.session_state.questions)} questions from PDF")
                with st.expander("📋 Review extracted questions"):
                    # Display extracted questions with numbers and text
                    for q in st.session_state.questions:
                         st.text(f"{q.number}. {q.text}")
            else:
                st.warning("Could not find numbered questions in PDF starting with 'Number. Text'. Please use manual input below.")
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
