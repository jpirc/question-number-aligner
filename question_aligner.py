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

    def test_api_connection(self) -> Dict[str, Any]:
        """Test the API connection with the recommended stable endpoint."""
        api_key = self._get_api_key()
        
        # UPDATED: Testing with the Gemini 2.0 endpoint
        endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"
        
        test_payload = {
            "contents": [{"parts": [{"text": "Hello, please respond with 'API is working'"}]}]
        }
        
        results = {}
        working_endpoint = None
        
        url = f"{endpoint}?key={api_key}"
        try:
            response = requests.post(url, json=test_payload, headers={'Content-Type': 'application/json'}, timeout=10)
            if response.status_code == 200:
                results[endpoint] = {"status": "✅ SUCCESS", "response": response.json()}
                working_endpoint = endpoint
            else:
                results[endpoint] = {"status": f"❌ Error {response.status_code}", "error": response.text[:200]}
        except Exception as e:
            results[endpoint] = {"status": "❌ Connection Failed", "error": str(e)}
        
        return {"results": results, "working_endpoint": working_endpoint}

    def calculate_optimal_batch_size(self, 
                                   total_columns: int, 
                                   total_questions: int,
                                   column_complexity: float) -> int:
        """
        Dynamically calculate optimal batch size based on multiple factors.
        """
        base_size = 100
        if total_columns > 1000: size_multiplier = 2.0
        elif total_columns > 500: size_multiplier = 1.5
        elif total_columns < 100: size_multiplier = 0.5
        else: size_multiplier = 1.0
        
        if total_questions > 200: question_factor = 0.7
        elif total_questions > 100: question_factor = 0.85
        else: question_factor = 1.0
        
        if column_complexity > 100: complexity_factor = 0.6
        elif column_complexity > 50: complexity_factor = 0.8
        else: complexity_factor = 1.0
        
        optimal_size = int(base_size * size_multiplier * question_factor * complexity_factor)
        min_size = 25
        max_size = 300
        return max(min_size, min(optimal_size, max_size))
    
    def estimate_column_complexity(self, column_headers: List[str]) -> float:
        """Calculate average complexity of column names"""
        if not column_headers: return 0
        total_length = sum(len(col) for col in column_headers)
        avg_length = total_length / len(column_headers)
        complex_patterns = sum(1 for col in column_headers if ' - ' in col or ':' in col)
        complexity_ratio = complex_patterns / len(column_headers)
        return avg_length * (1 + complexity_ratio)

    def extract_questions_with_gemini(self, pdf_file_bytes: bytes) -> List[Question]:
        """
        Sends a PDF file to the Gemini API to extract a clean list of numbered questions.
        """
        api_key = self._get_api_key()
        
        # UPDATED: Using the stable Gemini 2.0 endpoint
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent?key={api_key}"
        
        encoded_pdf = base64.b64encode(pdf_file_bytes).decode('utf-8')
        
        prompt = """
        You are a highly specialized text extraction tool. Your ONLY function is to scan the provided PDF and find every instance of a line that begins with a question number (e.g., "1.", "Q2.", "3.").
        Follow these steps:
        1. Read the entire document page by page.
        2. Identify every piece of text that appears to be a numbered survey question.
        3. For each match, capture the question number and the complete question text that follows. The question text ends when you encounter a chart, a data table, or the start of the next numbered question.
        4. Clean the extracted text to form a single, coherent sentence. Remove any line breaks or formatting noise.
        5. Return the results as a valid JSON object with this structure:
        {
          "questions": [
            { "question_number": "1", "question_text": "Question text here" },
            { "question_number": "2", "question_text": "Another question" }
          ]
        }
        Find every single question. Do not skip any.
        """
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt}, 
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": encoded_pdf
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 8192
            }
        }

        try:
            with st.spinner("Calling Gemini API to extract questions... This may take a moment."):
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
                
                if response.status_code != 200:
                    st.error(f"API Error {response.status_code}: {response.text[:500]}")
                    return []
                
            response_json = response.json()
            
            if not response_json.get('candidates'):
                st.error("AI Question Extraction Failed: The API returned no candidates.")
                st.json(response_json) # Show the full error from the API
                return []

            candidate = response_json['candidates'][0]
            content = candidate['content']
            parts = content['parts']
            extracted_text = parts[0]['text']
            
            json_start = extracted_text.find('{')
            json_end = extracted_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = extracted_text[json_start:json_end]
            else:
                json_str = extracted_text
            
            json_str = re.sub(r'^```json\s*|\s*```$', '', json_str.strip())
            
            question_data = json.loads(json_str)
            questions = [Question(number=str(q['question_number']), text=q['question_text']) 
                         for q in question_data.get('questions', [])]
            
            return sorted(questions, key=lambda q: int(re.search(r'\d+', q.number).group() if re.search(r'\d+', q.number) else 0))
                
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            st.error(f"Error parsing AI response: {e}")
            st.text_area("Raw response:", extracted_text if 'extracted_text' in locals() else str(response_json), height=200)
            return []
        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred: {e}")
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return []

    def _match_batch_with_gemini(self, questions_window: List[Question], column_batch: Dict[int, str], last_q_num: int) -> str:
        """
        Processes a single batch of columns with sequential context.
        """
        api_key = self._get_api_key()
        
        # UPDATED: Using the stable Gemini 2.0 endpoint
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent?key={api_key}"

        compact_questions = "\n".join([f"{q.number}: {q.text}" for q in questions_window])
        formatted_columns = "\n".join([f'{idx}: "{header}"' for idx, header in column_batch.items()])

        prompt = f"""
        You are an expert market research data analyst. Your task is to map dataset columns to survey questions.
        CRITICAL INSTRUCTIONS:
        1. The last assigned question number was {last_q_num}. Continue from there - do not go backwards.
        2. If not confident, assign "UNMATCHED".
        3. If multiple columns belong to the same question, use the SAME question number.
        4. Output format: column_index|assigned_question|reasoning

        Available Questions:
        {compact_questions}

        Column Headers to Map:
        {formatted_columns}

        Provide your response as plain text with one mapping per line.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": { "temperature": 0.1, "maxOutputTokens": 4096 }
        }
        
        try:
            response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
            if response.status_code != 200:
                st.error(f"Batch API Error {response.status_code}: {response.text[:200]}")
                return ""
            response_json = response.json()
            if not response_json.get('candidates'):
                st.error("AI Matching Failed for a batch: No candidates returned.")
                return ""
            return response_json['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            st.error(f"Error in batch match: {e}")
            return ""

    def match_with_gemini_in_batches(self, questions: List[Question], column_headers: List[str], batch_size: int = None) -> Dict[str, Dict[str, str]]:
        """
        Orchestrates the stateful, sequential matching process with a focused question window.
        """
        if batch_size is None:
            complexity = self.estimate_column_complexity(column_headers)
            batch_size = self.calculate_optimal_batch_size(len(column_headers), len(questions), complexity)
            st.info(f"🎯 Using dynamic batch size: {batch_size} columns per API call")
        
        full_mapping = {}
        last_matched_q_num = 0
        
        question_lookup = {}
        for q in questions:
            match = re.search(r'\d+', q.number)
            if match:
                question_lookup[int(match.group())] = q
        
        question_indices = sorted(question_lookup.keys())

        num_batches = (len(column_headers) + batch_size - 1) // batch_size
        progress_bar = st.progress(0, text="Starting sequential batch processing...")

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            batch_headers = column_headers[start_index:end_index]
            column_batch_dict = {start_index + j: header for j, header in enumerate(batch_headers)}

            window_start_index = 0
            for idx, q_num in enumerate(question_indices):
                if q_num > last_matched_q_num:
                    window_start_index = idx
                    break
            
            window_end_index = min(window_start_index + 40, len(question_indices))
            questions_in_window_indices = question_indices[window_start_index:window_end_index]
            questions_window = [question_lookup[q_num] for q_num in questions_in_window_indices]

            progress_text = f"Processing batch {i+1}/{num_batches} (Context: Start from Q{last_matched_q_num+1}, Batch size: {batch_size})"
            progress_bar.progress((i) / num_batches, text=progress_text)
            
            response_string = self._match_batch_with_gemini(questions_window, column_batch_dict, last_matched_q_num)
            
            if not response_string:
                st.error(f"Batch {i+1} failed. Stopping process.")
                progress_bar.empty()
                return {}

            max_q_in_batch = last_matched_q_num
            for line in response_string.split('\n'):
                parts = line.strip().split('|')
                if len(parts) == 3:
                    col_idx, assigned_q_str, reasoning = parts
                    full_mapping[col_idx] = {"assigned_question": assigned_q_str, "reasoning": reasoning}
                    match = re.search(r'\d+', assigned_q_str)
                    if match:
                        current_q_num = int(match.group())
                        if current_q_num > max_q_in_batch:
                            max_q_in_batch = current_q_num
                elif line.strip():
                    st.warning(f"Skipping malformed line in batch {i+1}: '{line}'")
            
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
    st.markdown("Automate survey column renumbering by matching a dataset (CSV/Excel) with a survey report (PDF).")
    st.markdown("---")

    if 'review_df' not in st.session_state: st.session_state.review_df = None
    if 'original_df' not in st.session_state: st.session_state.original_df = None
    if 'questions' not in st.session_state: st.session_state.questions = []
    if 'loaded_data_file_name' not in st.session_state: st.session_state.loaded_data_file_name = ""
    
    aligner = QuestionNumberAligner()

    with st.sidebar:
        st.subheader("⚙️ Advanced Settings")
        
        if st.button("🔧 Test API Connection"):
            with st.spinner("Testing API endpoint..."):
                test_results = aligner.test_api_connection()
            st.subheader("API Test Results")
            if test_results['working_endpoint']:
                st.success(f"✅ Found working endpoint!")
                st.code(test_results['working_endpoint'], language='text')
            else:
                st.error("❌ API connection failed.")
                st.json(test_results)
        
        st.divider()
        
        batch_option = st.radio("Batch Size Strategy:", ["Automatic (Recommended)", "Manual"], help="Automatic mode optimizes based on your data")
        manual_batch_size = None
        if batch_option == "Manual":
            manual_batch_size = st.slider("Columns per batch:", min_value=25, max_value=300, value=100, step=25)

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
                match_results_dict = aligner.match_with_gemini_in_batches(questions, column_headers, batch_size=manual_batch_size)

                if match_results_dict:
                    reconstructed_list = []
                    for i, header in enumerate(column_headers):
                        mapping_info = match_results_dict.get(str(i), {"assigned_question": "UNMATCHED", "reasoning": "AI did not provide a mapping."})
                        reconstructed_list.append({
                            "column_index": i,
                            "original_header": header,
                            "assigned_question": mapping_info.get("assigned_question", "UNMATCHED"),
                            "reasoning": mapping_info.get("reasoning", "N/A")
                        })
                    
                    review_df = pd.DataFrame(reconstructed_list)
                    review_df['Col'] = [index_to_excel_col(i) for i in review_df['column_index']]
                    review_df.rename(columns={'original_header': 'Column Name', 'assigned_question': 'Assigned #', 'reasoning': 'AI Reasoning'}, inplace=True)
                    st.session_state.review_df = review_df[['Col', 'Column Name', 'Assigned #', 'AI Reasoning']]
                    st.success("✅ AI matching complete!")
                else:
                    st.error("AI batch matching failed. Please check the errors above.")
            else:
                st.error("AI question extraction failed. Cannot proceed.")

    if st.session_state.questions:
        with st.expander(f"📋 Review Extracted Questions ({len(st.session_state.questions)} total)"):
            st.dataframe(pd.DataFrame([q.__dict__ for q in st.session_state.questions]), use_container_width=True, height=300)

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
            use_container_width=True, hide_index=True, height=500, key="editor"
        )
        
        if st.button("✅ Confirm All Changes", type="primary"):
            st.session_state.review_df = edited_df
            st.toast("Changes confirmed!", icon="👍")
            st.rerun()

        st.markdown("---")
        st.subheader("Step 3: Download Renumbered File")
        
        col1, col2 = st.columns(2)
        with col1:
            prefix_format = st.selectbox("Question Number Format", ["Q{num}_", "Q{num} - ", "{num}. ", "Question {num}: "])
        with col2:
            include_unmatched = st.checkbox("Include unmatched columns", value=True)
        
        final_df = aligner.apply_numbering_to_dataset(st.session_state.original_df, st.session_state.review_df, prefix_format, include_unmatched)
        
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
                st.download_button("📑 Download as Excel", output.getvalue(), renumbered_excel_filename, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            
            with c2:
                st.download_button("📄 Download as CSV", final_df.to_csv(index=False, encoding='utf-8-sig'), renumbered_csv_filename, "text/csv", use_container_width=True)

if __name__ == "__main__":
    create_streamlit_app()
