import pandas as pd
import re
import json
import base64
import requests
from typing import Dict, List, Any
from dataclasses import dataclass
import streamlit as st
import io
import os
import time

# --- Data Classes ---
@dataclass
class Question:
    """Represents a single, clean question extracted from the survey."""
    number: str
    text: str

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

    def _get_api_key(self) -> str:
        """Retrieves the Gemini API key from Streamlit secrets."""
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("GEMINI_API_KEY is not set in your Streamlit secrets. Please add it to your secrets.toml file.")
            st.stop()
        return api_key

    def calculate_optimal_batch_size(self, 
                                   total_columns: int, 
                                   total_questions: int,
                                   column_complexity: float) -> int:
        """
        Dynamically calculate optimal batch size based on multiple factors.
        
        Args:
            total_columns: Total number of columns to process
            total_questions: Total number of questions in the survey
            column_complexity: Average length/complexity of column names
        
        Returns:
            Optimal batch size
        """
        # Base calculations
        base_size = 100
        
        # Adjust based on total columns (fewer API calls for large datasets)
        if total_columns > 1000:
            size_multiplier = 2.0
        elif total_columns > 500:
            size_multiplier = 1.5
        elif total_columns < 100:
            size_multiplier = 0.5  # Smaller batches for small datasets
        else:
            size_multiplier = 1.0
        
        # Adjust based on question count (larger context needs smaller batches)
        if total_questions > 200:
            question_factor = 0.7
        elif total_questions > 100:
            question_factor = 0.85
        else:
            question_factor = 1.0
        
        # Adjust based on column name complexity
        # Longer column names = more tokens = smaller batches
        if column_complexity > 100:  # Very long column names
            complexity_factor = 0.6
        elif column_complexity > 50:
            complexity_factor = 0.8
        else:
            complexity_factor = 1.0
        
        # Calculate final batch size
        optimal_size = int(base_size * size_multiplier * question_factor * complexity_factor)
        
        # Enforce limits based on Gemini's token constraints
        # Gemini 1.5 Pro has ~2M token context, but we want to stay well below
        min_size = 25
        max_size = 300  # Conservative max to ensure we don't hit token limits
        
        return max(min_size, min(optimal_size, max_size))
    
    def estimate_column_complexity(self, column_headers: List[str]) -> float:
        """Calculate average complexity of column names"""
        if not column_headers:
            return 0
        
        total_length = sum(len(col) for col in column_headers)
        avg_length = total_length / len(column_headers)
        
        # Check for complex patterns
        complex_patterns = sum(1 for col in column_headers if ' - ' in col or ':' in col)
        complexity_ratio = complex_patterns / len(column_headers)
        
        # Combined complexity score
        return avg_length * (1 + complexity_ratio)

    def extract_questions_with_gemini(self, pdf_file_bytes: bytes) -> List[Question]:
        """
        Sends a PDF file to the Gemini API to extract a clean list of numbered questions.
        """
        api_key = self._get_api_key()
        # Updated to use the correct model name
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"
        
        encoded_pdf = base64.b64encode(pdf_file_bytes).decode('utf-8')
        
        prompt = """
        You are a highly specialized text extraction tool. Your ONLY function is to scan the provided PDF and find every instance of a line that begins with a question number (e.g., "1.", "Q2.", "3.").
        Follow these steps:
        1. Read the entire document page by page.
        2. Identify every piece of text that appears to be a numbered survey question.
        3. For each match, capture the question number and the complete question text that follows. The question text ends when you encounter a chart, a data table, or the start of the next numbered question.
        4. Clean the extracted text to form a single, coherent sentence. Remove any line breaks or formatting noise.
        5. Assemble the results into a single, valid JSON object. The object should contain one key, "questions", which holds an array of objects. Each object in the array must have two keys: "question_number" (as a string) and "question_text" (the cleaned text).
        Example of a perfect response:
        {
          "questions": [
            { "question_number": "39", "question_text": "HOH BDAY - Please rank the following factors from most to least appealing to when considering a kitchen renovation." },
            { "question_number": "40", "question_text": "HOH BDAY - Which one of the two taglines do you like the most for this campaign?" }
          ]
        }
        Find every single question. Do not skip any. The output must be only the JSON object.
        """
        
        # Add instruction to output JSON in the prompt
        json_prompt = prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no additional text or formatting."
        
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": json_prompt}, 
                {"inlineData": {"mimeType": "application/pdf", "data": encoded_pdf}}
            ]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json"
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }

        try:
            with st.spinner("Calling Gemini 1.5 Pro to extract questions... This may take a moment."):
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
                
                # If we get a 404, try alternative model names
                if response.status_code == 404:
                    st.warning("Trying alternative model endpoint...")
                    # Try without version suffix
                    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
                    response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
                
                response.raise_for_status()
                
            response_json = response.json()
            
            if not response_json.get('candidates'):
                st.error("AI Question Extraction Failed: The API returned no candidates. This might be due to a content policy violation or an API error.")
                st.json(response_json)
                return []

            candidate = response_json.get('candidates', [{}])[0]
            content_part = candidate.get('content', {}).get('parts', [{}])[0]
            extracted_json_text = content_part.get('text', '{}')
            
            # Clean up the response if needed
            extracted_json_text = extracted_json_text.strip()
            if extracted_json_text.startswith('```json'):
                extracted_json_text = extracted_json_text[7:]
            if extracted_json_text.endswith('```'):
                extracted_json_text = extracted_json_text[:-3]
            
            question_data = json.loads(extracted_json_text)
            
            questions = [Question(number=str(q['question_number']), text=q['question_text']) 
                         for q in question_data.get('questions', [])]
            
            return sorted(questions, key=lambda q: int(re.search(r'\d+', q.number).group() if re.search(r'\d+', q.number) else 0))

        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response status code: {e.response.status_code}")
                st.error(f"Response text: {e.response.text[:500]}...")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            st.error(f"Error parsing AI response for question extraction. The response may be malformed. Error: {e}")
            if 'extracted_json_text' in locals():
                st.text_area("Extracted JSON Text", value=extracted_json_text, height=200)
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred during AI Question Extraction: {e}")
            return []

    def _match_batch_with_gemini(self, questions_window: List[Question], column_batch: Dict[int, str], last_q_num: int) -> str:
        """
        Processes a single batch of columns with sequential context.
        Returns a simple pipe-delimited string.
        """
        api_key = self._get_api_key()
        # Updated to use the correct model name
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"

        compact_questions = "\n".join([f"{q.number}: {q.text}" for q in questions_window])
        formatted_columns = "\n".join([f'{idx}: "{header}"' for idx, header in column_batch.items()])

        prompt = f"""
        You are an expert market research data analyst. Your task is to map a batch of dataset columns to a list of survey questions, maintaining a strict numerical sequence.

        **CRITICAL INSTRUCTIONS:**
        1.  **SEQUENTIAL CONTEXT IS MANDATORY:** In the previous batch, the last assigned question number was **{last_q_num}**. You MUST continue the numbering from there. Do not jump backwards to earlier question numbers, even if the text seems to match. This is likely a monadic study with repeating question blocks.
        2.  **Accuracy is Mandatory:** If not highly confident, assign "UNMATCHED".
        3.  **Handle Multi-Part Questions:** If multiple columns belong to the same question (e.g., "Q65 - Option A", "Q65 - Option B"), assign the SAME question number to all of them.
        4.  **Output Format:** Your response MUST be plain text. Each line MUST follow this exact format: `column_index|assigned_question|reasoning`. Use the pipe `|` character as a delimiter. Do not output JSON or any other format.
        5.  **Assigned Question VALUE:** For the "assigned_question" part, you MUST return ONLY the question number (e.g., "63", "64a") or the literal string "UNMATCHED".

        Example Response:
        150|63|Follows sequence from last batch.
        151|64|Matches question about social media usage.
        152|64|Option for question 64.

        ---
        **INPUT DATA**
        1. Available Survey Questions (Number: Text):
        {compact_questions}

        2. Batch of Dataset Column Headers to Map (Index: Header):
        {formatted_columns}
        ---
        **YOUR TASK**
        Analyze the inputs and generate the pipe-delimited text output mapping ALL column indices in the provided batch, respecting the sequential context.
        """

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 4096,
                "responseMimeType": "text/plain"
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        }
        
        try:
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
            
            # If we get a 404, try alternative model names
            if response.status_code == 404:
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
            
            response.raise_for_status()
            response_json = response.json()

            if not response_json.get('candidates'):
                st.error(f"AI Matching Failed for a batch: The API returned no candidates.")
                return ""

            candidate = response_json.get('candidates', [{}])[0]
            content_part = candidate.get('content', {}).get('parts', [{}])[0]
            return content_part.get('text', '').strip()

        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred during a batch match: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response text: {e.response.text[:500]}...")
            return ""
        except Exception as e:
            st.error(f"An unexpected error occurred during a batch match: {e}")
            return ""

    def match_with_gemini_in_batches(self, questions: List[Question], column_headers: List[str], batch_size: int = None) -> Dict[str, Dict[str, str]]:
        """
        Orchestrates the stateful, sequential matching process with a focused question window.
        Now with dynamic batch sizing support.
        """
        # Calculate optimal batch size if not provided
        if batch_size is None:
            complexity = self.estimate_column_complexity(column_headers)
            batch_size = self.calculate_optimal_batch_size(
                len(column_headers), 
                len(questions), 
                complexity
            )
            st.info(f"🎯 Using dynamic batch size: {batch_size} columns per API call")
        
        full_mapping = {}
        last_matched_q_num = 0
        
        # Create a lookup dictionary from question number to Question object
        # This handles non-integer question numbers like 'Q1' or '1a' by extracting the first integer
        question_lookup = {}
        for q in questions:
            match = re.search(r'\d+', q.number)
            if match:
                question_lookup[int(match.group())] = q
        
        # A sorted list of the integer question numbers
        question_indices = sorted(question_lookup.keys())

        num_batches = (len(column_headers) + batch_size - 1) // batch_size
        progress_bar = st.progress(0, text="Starting sequential batch processing...")

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            batch_headers = column_headers[start_index:end_index]
            
            column_batch_dict = {start_index + j: header for j, header in enumerate(batch_headers)}

            # --- Focused Question Window Logic ---
            # Find the index in our sorted list where we should start looking for questions
            window_start_index = 0
            for idx, q_num in enumerate(question_indices):
                # Start the window from the last matched question number
                if q_num > last_matched_q_num:
                    window_start_index = idx
                    break
            
            # Define the end of the window to keep the prompt focused
            window_end_index = min(window_start_index + 40, len(question_indices)) # Look at the next 40 questions
            
            # Slice the list of question numbers to get our window
            questions_in_window_indices = question_indices[window_start_index:window_end_index]
            
            # Get the actual Question objects for the window
            questions_window = [question_lookup[q_num] for q_num in questions_in_window_indices]

            progress_text = f"Processing batch {i+1}/{num_batches} (Context: Start from Q{last_matched_q_num+1}, Batch size: {batch_size})"
            progress_bar.progress((i) / num_batches, text=progress_text)
            
            response_string = self._match_batch_with_gemini(questions_window, column_batch_dict, last_matched_q_num)
            
            if not response_string:
                st.error(f"Batch {i+1} failed: Received an empty response from the AI. Stopping process.")
                progress_bar.empty()
                return {}

            max_q_in_batch = last_matched_q_num
            for line in response_string.split('\n'):
                parts = line.strip().split('|')
                if len(parts) == 3:
                    col_idx, assigned_q_str, reasoning = parts
                    full_mapping[col_idx] = {
                        "assigned_question": assigned_q_str,
                        "reasoning": reasoning
                    }
                    match = re.search(r'\d+', assigned_q_str)
                    if match:
                        current_q_num = int(match.group())
                        if current_q_num > max_q_in_batch:
                            max_q_in_batch = current_q_num
                else:
                    st.warning(f"Skipping malformed line in batch {i+1}: '{line}'")
            
            # Only update if the AI found a new higher number
            if max_q_in_batch > last_matched_q_num:
                last_matched_q_num = max_q_in_batch
            
            time.sleep(1)

        progress_bar.progress(1.0, text="Batch processing complete!")
        time.sleep(1)
        progress_bar.empty()
        return full_mapping

    def apply_numbering_to_dataset(self, df: pd.DataFrame, final_mapping_df: pd.DataFrame, prefix_format: str, include_unmatched: bool) -> pd.DataFrame:
        """Applies the final numbering to the dataset based on the review table."""
        if not isinstance(final_mapping_df, pd.DataFrame) or not {'Assigned #', 'Column Name'}.issubset(final_mapping_df.columns):
            st.error("Could not apply numbering. The mapping table is missing required columns. Please click 'Confirm All Changes' after editing.")
            return pd.DataFrame() 

        df_numbered = df.copy()
        
        mapping = pd.Series(final_mapping_df['Assigned #'].values, index=final_mapping_df['Column Name']).to_dict()

        if not include_unmatched:
            matched_cols = [col for col, q_num in mapping.items() if str(q_num) != 'UNMATCHED']
            df_numbered = df_numbered[matched_cols]
        
        new_columns = {}
        for col in df_numbered.columns:
            q_num = mapping.get(col)
            if q_num and q_num != 'UNMATCHED' and re.search(r'\d', str(q_num)):
                num_only = re.search(r'\d+', str(q_num)).group()
                new_columns[col] = f"{prefix_format.format(num=num_only)}{col}"
            else:
                new_columns[col] = col
                
        df_numbered.rename(columns=new_columns, inplace=True)
        return df_numbered

# --- Streamlit UI ---
def create_streamlit_app():
    st.set_page_config(page_title="AI Survey Renumbering Tool", layout="wide")
    
    st.title("📊 AI-Powered Survey Renumbering Tool")
    st.markdown("Automate survey column renumbering by matching a dataset (CSV/Excel) with a survey report (PDF). Now with support for monadic studies!")
    st.markdown("---")

    if 'review_df' not in st.session_state: st.session_state.review_df = None
    if 'original_df' not in st.session_state: st.session_state.original_df = None
    if 'questions' not in st.session_state: st.session_state.questions = []
    if 'loaded_data_file_name' not in st.session_state: st.session_state.loaded_data_file_name = ""
    
    aligner = QuestionNumberAligner()

    # Sidebar for advanced settings
    with st.sidebar:
        st.subheader("⚙️ Advanced Settings")
        
        # Batch size options
        batch_option = st.radio(
            "Batch Size Strategy:",
            ["Automatic (Recommended)", "Manual"],
            help="Automatic mode optimizes based on your data"
        )
        
        if batch_option == "Manual":
            manual_batch_size = st.slider(
                "Columns per batch:",
                min_value=25,
                max_value=300,
                value=100,
                step=25,
                help="Larger batches = fewer API calls but higher latency"
            )
        else:
            manual_batch_size = None

    st.subheader("Step 1: Upload Your Files")
    col1, col2 = st.columns(2)
    with col1:
        data_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    with col2:
        pdf_file = st.file_uploader("Upload Survey Report (PDF)", type=['pdf'])

    if data_file and data_file.name != st.session_state.loaded_data_file_name:
        try:
            df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
            st.session_state.original_df = df
            st.session_state.loaded_data_file_name = data_file.name
            st.session_state.review_df = None
            st.session_state.questions = []
            st.success(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.session_state.original_df = None

    st.markdown("---")
    
    if st.session_state.original_df is not None and pdf_file:
        if st.button("🚀 Start Full Renumbering Process", type="primary", use_container_width=True):
            st.session_state.questions = []
            st.session_state.review_df = None
            
            pdf_bytes = pdf_file.getvalue()
            questions = aligner.extract_questions_with_gemini(pdf_bytes)
            st.session_state.questions = questions

            if questions:
                st.success(f"✅ AI successfully extracted {len(questions)} questions.")
                
                column_headers = st.session_state.original_df.columns.tolist()
                
                # Use the selected batch size (automatic or manual)
                match_results_dict = aligner.match_with_gemini_in_batches(
                    questions, 
                    column_headers,
                    batch_size=manual_batch_size  # Will be None for automatic mode
                )

                if match_results_dict:
                    reconstructed_list = []
                    for i, header in enumerate(column_headers):
                        mapping_info = match_results_dict.get(str(i), {
                            "assigned_question": "UNMATCHED",
                            "reasoning": "AI did not provide a mapping for this index."
                        })
                        
                        reconstructed_list.append({
                            "column_index": i,
                            "original_header": header,
                            "assigned_question": mapping_info.get("assigned_question", "UNMATCHED"),
                            "reasoning": mapping_info.get("reasoning", "N/A")
                        })
                    
                    review_df = pd.DataFrame(reconstructed_list)
                    review_df['Col'] = [index_to_excel_col(i) for i in review_df['column_index']]
                    review_df.rename(columns={
                        'original_header': 'Column Name',
                        'assigned_question': 'Assigned #',
                        'reasoning': 'AI Reasoning'
                    }, inplace=True)
                    st.session_state.review_df = review_df[['Col', 'Column Name', 'Assigned #', 'AI Reasoning']]
                    st.success("✅ AI matching complete!")
                else:
                    st.error("AI batch matching failed to produce results. Please check the errors above.")
                    st.session_state.review_df = None
            else:
                st.error("AI question extraction failed. Cannot proceed with matching.")

    if st.session_state.questions:
        with st.expander(f"📋 Review Extracted Questions ({len(st.session_state.questions)} total)"):
            questions_df = pd.DataFrame([q.__dict__ for q in st.session_state.questions])
            st.dataframe(questions_df, use_container_width=True, height=300)

    if st.session_state.review_df is not None:
        st.markdown("---")
        st.subheader("Step 2: Review and Manually Edit Matches")
        
        edited_df = st.data_editor(
            st.session_state.review_df,
            column_config={
                "Col": st.column_config.TextColumn("Col", width="small", disabled=True),
                "Column Name": st.column_config.TextColumn("Column Name", width="large", disabled=True),
                "Assigned #": st.column_config.TextColumn("Assigned # (Editable)"),
                "AI Reasoning": st.column_config.TextColumn("AI Reasoning", width="large", disabled=True)
            },
            use_container_width=True,
            hide_index=True,
            height=500,
            key="editor"
        )
        
        if st.button("✅ Confirm All Changes", type="primary"):
            st.session_state.review_df = edited_df
            st.toast("Changes confirmed and saved!", icon="👍")
            st.rerun()

        st.markdown("---")
        st.subheader("Step 3: Download Renumbered File")
        st.warning("⚠️ Please click '✅ Confirm All Changes' above to apply your edits before downloading.")
        
        col1, col2 = st.columns(2)
        with col1:
            prefix_format = st.selectbox(
                "Question Number Format",
                ["Q{num}_", "Q{num} - ", "{num}. ", "Question {num}: "],
                help="Choose how question numbers are prefixed to column names."
            )
        with col2:
            include_unmatched = st.checkbox("Include unmatched columns in final file", value=True)
        
        final_df = aligner.apply_numbering_to_dataset(
            st.session_state.original_df, 
            st.session_state.review_df,
            prefix_format,
            include_unmatched
        )
        
        if not final_df.empty:
            st.markdown("##### Preview of Renumbered Data")
            st.dataframe(final_df.head())
            
            base_filename = os.path.splitext(st.session_state.loaded_data_file_name)[0]
            renumbered_excel_filename = f"{base_filename}_renumbered.xlsx"
            renumbered_csv_filename = f"{base_filename}_renumbered.csv"

            c1, c2 = st.columns(2)
            with c1:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='Numbered Data')
                excel_data = output.getvalue()
                st.download_button(
                    label="📑 Download as Excel",
                    data=excel_data,
                    file_name=renumbered_excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with c2:
                csv = final_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=renumbered_csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    create_streamlit_app()
