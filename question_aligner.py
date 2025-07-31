import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
import streamlit as st
import numpy as np

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
        """Parse questions from PDF text"""
        questions = []
        
        # This is a simplified parser - in reality you'd parse the actual PDF
        # For now, I'll create a pattern that matches the structure you showed
        lines = pdf_text.split('\n')
        current_question = None
        
        for line in lines:
            # Match question number and text
            match = re.match(r'^(\d+)\.\s+(.+)', line)
            if match:
                if current_question:
                    questions.append(current_question)
                
                num, text = match.groups()
                q_type = self._determine_question_type(text)
                current_question = Question(num, text, q_type)
                
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
                
            # Look for patterns indicating multi-response
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
        """Main matching algorithm"""
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
            
            # Clean the base question for matching
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
                print(f"Mapped group '{base}' ({len(group_cols)} items) to Q{best_match}")
        
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
                    if int(q.number) >= 180:  # Demographic section
                        for demo_type, patterns in self.demographic_patterns.items():
                            if any(p in clean_col for p in patterns):
                                score += 0.3
                                break
                    
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
        # Create new column names with question numbers
        new_columns = []
        
        for col in df.columns:
            if col in mapping and mapping[col] != 'UNMATCHED':
                # Format the question number using the specified format
                if mapping[col].isdigit():
                    prefix = prefix_format.format(num=mapping[col])
                    new_col = f"{prefix}{col}"
                else:
                    # For non-numeric mappings (META, ID, etc), don't add prefix
                    new_col = col
            else:
                # Keep original name for unmatched columns
                new_col = col
                
            new_columns.append(new_col)
            
        df_numbered = df.copy()
        df_numbered.columns = new_columns
        
        return df_numbered

# Streamlit UI
def create_streamlit_app():
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
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Dataset (CSV/Excel)")
        data_file = st.file_uploader("Choose your dataset file", type=['csv', 'xlsx'])
        
    with col2:
        st.subheader("📄 Upload PDF Overview")
        pdf_file = st.file_uploader("Choose your PDF file", type=['pdf'])
        st.info("Note: For demo purposes, you can also paste the question list below")
        
    # Manual question input (for testing)
    with st.expander("📝 Or paste question list here (for testing)"):
        question_text = st.text_area(
            "Paste questions in format: 'number. question text'",
            height=200,
            placeholder="1. To which gender identity do you most identify?\n2. What is your age?\n..."
        )
        
    # Process files
    if data_file:
        # Load dataset
        if data_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(data_file)
        else:
            st.session_state.df = pd.read_excel(data_file)
            
        st.success(f"✅ Dataset loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns")
        
    # Initialize aligner
    aligner = QuestionNumberAligner()
    
    # Parse questions (from PDF or manual input)
    if pdf_file:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text() + "\n"
                
                # Extract questions that start with numbers
                import re
                question_lines = []
                for line in pdf_text.split('\n'):
                    # Match lines that start with a number followed by a period
                    if re.match(r'^\d+\.\s+', line.strip()):
                        question_lines.append(line.strip())
                
                if question_lines:
                    question_text = '\n'.join(question_lines)
                    st.session_state.questions = aligner.parse_pdf_questions(question_text)
                    st.success(f"✅ Extracted {len(st.session_state.questions)} questions from PDF")
                    
                    # Show extracted questions for review
                    with st.expander("📋 Review extracted questions"):
                        st.text(question_text)
                else:
                    st.warning("Could not find numbered questions in PDF. Please use manual input below.")
                    
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            st.info("Please install pdfplumber: pip install pdfplumber")
    
    elif question_text:
        st.session_state.questions = aligner.parse_pdf_questions(question_text)
        st.success(f"✅ Parsed {len(st.session_state.questions)} questions")
    
    # Run alignment
    if st.session_state.df is not None and st.session_state.questions:
        if st.button("🚀 Run Automatic Alignment", type="primary"):
            with st.spinner("Analyzing and matching columns..."):
                st.session_state.mapping = aligner.match_columns_to_questions(
                    st.session_state.df, 
                    st.session_state.questions
                )
                
    # Display results
    if st.session_state.mapping:
        st.markdown("---")
        st.subheader("📋 Alignment Results")
        
        # Create review dataframe - MAINTAIN ORIGINAL ORDER
        review_data = []
        # Use the original DataFrame columns to maintain order
        for col in st.session_state.df.columns:
            if col in st.session_state.mapping:
                q_num = st.session_state.mapping[col]
                # Determine status based on current mapping
                if q_num == 'UNMATCHED':
                    status = '❌ Unmatched'
                elif q_num in ['META', 'ID', 'SKIP']:
                    status = '⚪ Metadata/Skip'
                elif q_num.isdigit():
                    status = '✅ Matched'
                else:
                    status = '✏️ Manual'
                    
                review_data.append({
                    'Column Name': col,  # Use original column name without modifications
                    'Assigned Question': q_num,
                    'Status': status
                })
            
        review_df = pd.DataFrame(review_data)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_cols = len(review_df)
            st.metric("Total Columns", total_cols)
        with col2:
            matched = len(review_df[review_df['Status'] == '✅ Matched'])
            st.metric("Matched", matched, f"{matched/total_cols*100:.1f}%")
        with col3:
            unmatched = len(review_df[review_df['Status'] == '❌ Unmatched'])
            st.metric("Unmatched", unmatched)
            
        # Filter options
        filter_option = st.radio(
            "Show:",
            ["All", "Matched only", "Unmatched only"],
            horizontal=True
        )
        
        if filter_option == "Matched only":
            display_df = review_df[review_df['Status'] == '✅ Matched']
        elif filter_option == "Unmatched only":
            display_df = review_df[review_df['Status'] == '❌ Unmatched']
        else:
            display_df = review_df
            
        # Editable dataframe for manual corrections
        st.markdown("### Review and Edit Mappings")
        
        # Add helper button for multi-response questions
        if st.button("🔧 Fix Multi-Response Questions"):
            # Find potential multi-response groups that were split
            for col, q_num in st.session_state.mapping.items():
                if ' - ' in col and 'select all that apply' in col.lower():
                    base = col.split(' - ')[0].strip()
                    # Find all columns with same base
                    for other_col in st.session_state.mapping:
                        if other_col.startswith(base + ' - ') and st.session_state.mapping[other_col] != q_num:
                            st.session_state.mapping[other_col] = q_num
            st.success("✅ Multi-response questions regrouped!")
            st.rerun()
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Assigned Question": st.column_config.TextColumn(
                    "Assigned Question",
                    help="Edit question numbers here",
                    width="small",
                )
            },
            num_rows="fixed",
            use_container_width=True
        )
        
        # Apply changes button
        if st.button("✏️ Apply Manual Corrections"):
            # Update mapping with edits
            for _, row in edited_df.iterrows():
                st.session_state.mapping[row['Column Name']] = row['Assigned Question']
            st.success("✅ Manual corrections applied!")
            
            # Refresh the display to show updated statuses
            st.rerun()
            
        # Export options
        st.markdown("---")
        st.subheader("💾 Export Options")
        
        # Add prefix format selector
        col1, col2 = st.columns(2)
        with col1:
            prefix_format = st.selectbox(
                "Question Number Format",
                ["Q{num}. ", "{num}. ", "#{num}. ", "Question {num}: ", "Q{num} - ", "{num}) "],
                help="Choose how question numbers appear in the export"
            )
            
            # Show preview
            st.caption(f"Preview: {prefix_format.format(num='1')}Your question text here")
        
        with col2:
            # Add checkbox for including/excluding unmatched columns
            include_unmatched = st.checkbox(
                "Include unmatched columns in export", 
                value=True,
                help="Uncheck to exclude columns that couldn't be matched to questions"
            )
        
        # Custom prefix option
        use_custom = st.checkbox("Use custom format")
        if use_custom:
            custom_format = st.text_input(
                "Custom format (use {num} for the question number)",
                value="Q{num}. ",
                help="Examples: 'Q{num}. ' or 'Question {num}: ' or '{num}) '"
            )
            prefix_format = custom_format
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Numbered Dataset"):
                # Apply numbering with custom prefix
                numbered_df = aligner.apply_numbering_to_dataset(
                    st.session_state.df, 
                    st.session_state.mapping,
                    prefix_format=prefix_format
                )
                
                # Filter out unmatched columns if requested
                if not include_unmatched:
                    # Get columns that have valid question numbers
                    matched_cols = [col for col in numbered_df.columns 
                                  if not any(unmatch in col for unmatch in ['UNMATCHED', 'META', 'ID'])]
                    numbered_df = numbered_df[matched_cols]
                
                # Offer both CSV and Excel options
                st.markdown("##### Download Format:")
                
                # Excel download
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    numbered_df.to_excel(writer, sheet_name='Sheet1', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📑 Download as Excel (.xlsx)",
                    data=excel_data,
                    file_name="numbered_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # CSV download with proper encoding
                csv = numbered_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📄 Download as CSV (UTF-8)",
                    data=csv,
                    file_name="numbered_dataset.csv",
                    mime="text/csv"
                )
                
        with col2:
            if st.button("📄 Export Mapping"):
                mapping_df = pd.DataFrame(
                    list(st.session_state.mapping.items()),
                    columns=['Original Column', 'Question Number']
                )
                
                # Offer both CSV and Excel options for mapping too
                st.markdown("##### Download Format:")
                
                # Excel download
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    mapping_df.to_excel(writer, sheet_name='Mapping', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📑 Download Mapping as Excel",
                    data=excel_data,
                    file_name="column_mapping.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # CSV with proper encoding
                csv = mapping_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📄 Download Mapping as CSV",
                    data=csv,
                    file_name="column_mapping.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    # Run with: streamlit run question_aligner.py
    create_streamlit_app()                num, text = match.groups()
                q_type = self._determine_question_type(text)
                current_question = Question(num, text, q_type)
                
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
                
            # Look for patterns indicating multi-response
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
        """Main matching algorithm"""
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
            
            # Clean the base question for matching
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
                print(f"Mapped group '{base}' ({len(group_cols)} items) to Q{best_match}")
        
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
                    if int(q.number) >= 180:  # Demographic section
                        for demo_type, patterns in self.demographic_patterns.items():
                            if any(p in clean_col for p in patterns):
                                score += 0.3
                                break
                    
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
        # Create new column names with question numbers
        new_columns = []
        
        for col in df.columns:
            if col in mapping and mapping[col] != 'UNMATCHED':
                # Format the question number using the specified format
                if mapping[col].isdigit():
                    prefix = prefix_format.format(num=mapping[col])
                    new_col = f"{prefix}{col}"
                else:
                    # For non-numeric mappings (META, ID, etc), don't add prefix
                    new_col = col
            else:
                # Keep original name for unmatched columns
                new_col = col
                
            new_columns.append(new_col)
            
        df_numbered = df.copy()
        df_numbered.columns = new_columns
        
        return df_numbered

# Streamlit UI
def create_streamlit_app():
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
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Dataset (CSV/Excel)")
        data_file = st.file_uploader("Choose your dataset file", type=['csv', 'xlsx'])
        
    with col2:
        st.subheader("📄 Upload PDF Overview")
        pdf_file = st.file_uploader("Choose your PDF file", type=['pdf'])
        st.info("Note: For demo purposes, you can also paste the question list below")
        
    # Manual question input (for testing)
    with st.expander("📝 Or paste question list here (for testing)"):
        question_text = st.text_area(
            "Paste questions in format: 'number. question text'",
            height=200,
            placeholder="1. To which gender identity do you most identify?\n2. What is your age?\n..."
        )
        
    # Process files
    if data_file:
        # Load dataset
        if data_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(data_file)
        else:
            st.session_state.df = pd.read_excel(data_file)
            
        st.success(f"✅ Dataset loaded: {len(st.session_state.df)} rows, {len(st.session_state.df.columns)} columns")
        
    # Initialize aligner
    aligner = QuestionNumberAligner()
    
    # Parse questions (from PDF or manual input)
    if pdf_file:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                pdf_text = ""
                for page in pdf.pages:
                    pdf_text += page.extract_text() + "\n"
                
                # Extract questions that start with numbers
                import re
                question_lines = []
                for line in pdf_text.split('\n'):
                    # Match lines that start with a number followed by a period
                    if re.match(r'^\d+\.\s+', line.strip()):
                        question_lines.append(line.strip())
                
                if question_lines:
                    question_text = '\n'.join(question_lines)
                    st.session_state.questions = aligner.parse_pdf_questions(question_text)
                    st.success(f"✅ Extracted {len(st.session_state.questions)} questions from PDF")
                    
                    # Show extracted questions for review
                    with st.expander("📋 Review extracted questions"):
                        st.text(question_text)
                else:
                    st.warning("Could not find numbered questions in PDF. Please use manual input below.")
                    
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            st.info("Please install pdfplumber: pip install pdfplumber")
    
    elif question_text:
        st.session_state.questions = aligner.parse_pdf_questions(question_text)
        st.success(f"✅ Parsed {len(st.session_state.questions)} questions")
    
    # Run alignment
    if st.session_state.df is not None and st.session_state.questions:
        if st.button("🚀 Run Automatic Alignment", type="primary"):
            with st.spinner("Analyzing and matching columns..."):
                st.session_state.mapping = aligner.match_columns_to_questions(
                    st.session_state.df, 
                    st.session_state.questions
                )
                
    # Display results
    if st.session_state.mapping:
        st.markdown("---")
        st.subheader("📋 Alignment Results")
        
        # Create review dataframe
        review_data = []
        for col, q_num in st.session_state.mapping.items():
            # Determine status based on current mapping
            if q_num == 'UNMATCHED':
                status = '❌ Unmatched'
            elif q_num in ['META', 'ID', 'SKIP']:
                status = '⚪ Metadata/Skip'
            elif q_num.isdigit():
                status = '✅ Matched'
            else:
                status = '✏️ Manual'
                
            review_data.append({
                'Column Name': col,
                'Assigned Question': q_num,
                'Status': status
            })
            
        review_df = pd.DataFrame(review_data)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_cols = len(review_df)
            st.metric("Total Columns", total_cols)
        with col2:
            matched = len(review_df[review_df['Status'] == '✅ Matched'])
            st.metric("Matched", matched, f"{matched/total_cols*100:.1f}%")
        with col3:
            unmatched = len(review_df[review_df['Status'] == '❌ Unmatched'])
            st.metric("Unmatched", unmatched)
            
        # Filter options
        filter_option = st.radio(
            "Show:",
            ["All", "Matched only", "Unmatched only"],
            horizontal=True
        )
        
        if filter_option == "Matched only":
            display_df = review_df[review_df['Status'] == '✅ Matched']
        elif filter_option == "Unmatched only":
            display_df = review_df[review_df['Status'] == '❌ Unmatched']
        else:
            display_df = review_df
            
        # Editable dataframe for manual corrections
        st.markdown("### Review and Edit Mappings")
        
        # Add helper button for multi-response questions
        if st.button("🔧 Fix Multi-Response Questions"):
            # Find potential multi-response groups that were split
            for col, q_num in st.session_state.mapping.items():
                if ' - ' in col and 'select all that apply' in col.lower():
                    base = col.split(' - ')[0].strip()
                    # Find all columns with same base
                    for other_col in st.session_state.mapping:
                        if other_col.startswith(base + ' - ') and st.session_state.mapping[other_col] != q_num:
                            st.session_state.mapping[other_col] = q_num
            st.success("✅ Multi-response questions regrouped!")
            st.rerun()
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "Assigned Question": st.column_config.TextColumn(
                    "Assigned Question",
                    help="Edit question numbers here",
                    width="small",
                )
            },
            num_rows="fixed",
            use_container_width=True
        )
        
        # Apply changes button
        if st.button("✏️ Apply Manual Corrections"):
            # Update mapping with edits
            for _, row in edited_df.iterrows():
                st.session_state.mapping[row['Column Name']] = row['Assigned Question']
            st.success("✅ Manual corrections applied!")
            
            # Refresh the display to show updated statuses
            st.rerun()
            
        # Export options
        st.markdown("---")
        st.subheader("💾 Export Options")
        
        # Add prefix format selector
        col1, col2 = st.columns(2)
        with col1:
            prefix_format = st.selectbox(
                "Question Number Format",
                ["Q{num}. ", "{num}. ", "#{num}. ", "Question {num}: ", "Q{num} - ", "{num}) "],
                help="Choose how question numbers appear in the export"
            )
            
            # Show preview
            st.caption(f"Preview: {prefix_format.format(num='1')}Your question text here")
        
        with col2:
            # Add checkbox for including/excluding unmatched columns
            include_unmatched = st.checkbox(
                "Include unmatched columns in export", 
                value=True,
                help="Uncheck to exclude columns that couldn't be matched to questions"
            )
        
        # Custom prefix option
        use_custom = st.checkbox("Use custom format")
        if use_custom:
            custom_format = st.text_input(
                "Custom format (use {num} for the question number)",
                value="Q{num}. ",
                help="Examples: 'Q{num}. ' or 'Question {num}: ' or '{num}) '"
            )
            prefix_format = custom_format
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Numbered Dataset"):
                # Apply numbering with custom prefix
                numbered_df = aligner.apply_numbering_to_dataset(
                    st.session_state.df, 
                    st.session_state.mapping,
                    prefix_format=prefix_format
                )
                
                # Filter out unmatched columns if requested
                if not include_unmatched:
                    # Get columns that have valid question numbers
                    matched_cols = [col for col in numbered_df.columns 
                                  if not any(unmatch in col for unmatch in ['UNMATCHED', 'META', 'ID'])]
                    numbered_df = numbered_df[matched_cols]
                
                # Offer both CSV and Excel options
                st.markdown("##### Download Format:")
                
                # Excel download
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    numbered_df.to_excel(writer, sheet_name='Sheet1', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📑 Download as Excel (.xlsx)",
                    data=excel_data,
                    file_name="numbered_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # CSV download with proper encoding
                csv = numbered_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📄 Download as CSV (UTF-8)",
                    data=csv,
                    file_name="numbered_dataset.csv",
                    mime="text/csv"
                )
                
        with col2:
            if st.button("📄 Export Mapping"):
                mapping_df = pd.DataFrame(
                    list(st.session_state.mapping.items()),
                    columns=['Original Column', 'Question Number']
                )
                
                # Offer both CSV and Excel options for mapping too
                st.markdown("##### Download Format:")
                
                # Excel download
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    mapping_df.to_excel(writer, sheet_name='Mapping', index=False)
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📑 Download Mapping as Excel",
                    data=excel_data,
                    file_name="column_mapping.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # CSV with proper encoding
                csv = mapping_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📄 Download Mapping as CSV",
                    data=csv,
                    file_name="column_mapping.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    # Run with: streamlit run question_aligner.py
    create_streamlit_app()
