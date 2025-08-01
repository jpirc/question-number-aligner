import pandas as pd
import re
import json
import base64
import requests
from typing import Dict, List, Any
from dataclasses import dataclass
import streamlit as st
import io

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

    def extract_questions_with_gemini(self, pdf_file_bytes: bytes) -> List[Question]:
        """
        Sends a PDF file to the Gemini API to extract a clean list of numbered questions.
        """
        api_key = self._get_api_key()
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
        
        encoded_pdf = base64.b64encode(pdf_file_bytes).decode('utf-8')
        
        prompt = """
        You are a highly specialized text extraction tool. Your ONLY function is to scan the provided PDF and find every instance of a line that begins with a question number (e.g., "1.", "Q2.", "3.").

        Follow these steps:
        1. Read the entire document page by page.
        2. Identify every piece of text that appears to be a numbered survey question.
        3. For each match, capture the question number and the complete question text that follows. The question text ends when you encounter a chart, a data table, or the start of the next numbered question.
        4. Clean the extracted text to form a single, coherent sentence. Remove any line breaks or formatting noise.
        5. Assemble the results into a single, valid JSON object. The object should contain one key, "questions", which holds an array of objects. Each object in the array must have two keys: "question_number" (as a string) and "question_text" (the cleaned text).

        **CRITICAL INSTRUCTION:** Do not be deterred by messy tables or complex layouts. If you see "39. Please rank the following...", you MUST extract it. Your job is to find the numbered question text itself, no matter what.

        Example of a perfect response:
        {
          "questions": [
            {
              "question_number": "39",
              "question_text": "HOH BDAY - Please rank the following factors from most to least appealing to when considering a kitchen renovation."
            },
            {
              "question_number": "40",
              "question_text": "HOH BDAY - Which one of the two taglines do you like the most for this campaign?"
            }
          ]
        }

        Find every single question. Do not skip any. The output must be only the JSON object.
        """
        
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": "application/pdf", "data": encoded_pdf}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        try:
            with st.spinner("Calling Gemini 1.5 Pro to extract questions... This may take a moment."):
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
                response.raise_for_status()
            response_json = response.json()
            
            candidate = response_json.get('candidates', [{}])[0]
            content_part = candidate.get('content', {}).get('parts', [{}])[0]
            extracted_json_text = content_part.get('text', '{}')
            
            question_data = json.loads(extracted_json_text)
            
            questions = [Question(number=str(q['question_number']), text=q['question_text']) 
                         for q in question_data.get('questions', [])]
            
            # Sort questions numerically
            return sorted(questions, key=lambda q: int(re.search(r'\d+', q.number).group() if re.search(r'\d+', q.number) else 0))

        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred: {e}")
            return []
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            st.error(f"Error parsing AI response for question extraction. The response may be malformed. Error: {e}")
            st.text_area("Raw AI Response", value=response.text if 'response' in locals() else "No response object.", height=200)
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred during AI Question Extraction: {e}")
            return []

    def match_with_gemini(self, questions: List[Question], column_headers: List[str]) -> List[Dict[str, Any]]:
        """
        Uses Gemini 1.5 Pro with a specialized "expert analyst" prompt to match columns to questions.
        Returns an empty list on failure.
        """
        api_key = self._get_api_key()
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"

        questions_json_str = json.dumps([q.__dict__ for q in questions], indent=2)
        formatted_columns = "\n".join([f'{i}: "{header}"' for i, header in enumerate(column_headers)])

        prompt = f"""
        You are an expert market research data analyst. Your task is to map a list of raw dataset column headers to a structured list of survey questions.

        **CRITICAL INSTRUCTIONS:**
        1.  **High Accuracy Over Completeness:** It is MANDATORY to be accurate. If you are not highly confident in a match, you MUST assign "UNMATCHED". It is better to leave a column unmatched than to assign it an incorrect question number.
        2.  **Strict Order Preservation:** The final output MUST be an array of objects that corresponds exactly to the original order of the input column headers, from index 0 to {len(column_headers) - 1}. Do not reorder the columns.
        3.  **Handle Complex Questions:**
            * **Multi-Select/Matrix:** Multiple columns often belong to a single question (e.g., "Q5 - Red", "Q5 - Blue", "Q5 - Green"). Assign the SAME question number (e.g., "Q5") to ALL columns that are part of that group. Use the base text of the column header before the hyphen or delimiter to identify the group.
            * **Open-Ends & "Other, Specify":** Columns like "Q5_other_specify" or "Q6_comment" belong to their parent question. Assign them the parent question number (e.g., "Q5" or "Q6").
        4.  **No Forced Matches:** Do NOT match internal system variables, metadata, or programming notes (e.g., 'record_id', 'start_time', 'end_time', 'device_type', 'user_ip', 'weight'). Assign these "UNMATCHED".
        5.  **Output Format:** Your entire response must be ONLY a single JSON object. Do not include any text, code block markers, or explanations outside of the JSON. The JSON object must have a single key, "column_mapping", which is an array of objects. Each object in the array represents a column and must have these four keys: "column_index" (integer), "original_header" (string), "assigned_question" (string, either the question number or "UNMATCHED"), and "reasoning" (a brief string explaining your decision).

        ---
        **INPUT DATA**
        **1. Survey Questions (JSON Format):**
        {questions_json_str}
        **2. Dataset Column Headers (Index: "Header"):**
        {formatted_columns}
        ---
        **YOUR TASK**
        Analyze the two inputs and generate the JSON output mapping every single column header from the list provided, following all instructions precisely.
        """

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        
        extracted_json_text = ""
        try:
            with st.spinner("AI is matching questions to columns... This may take a moment."):
                response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
                response.raise_for_status()
            response_json = response.json()

            candidate = response_json.get('candidates', [{}])[0]
            content_part = candidate.get('content', {}).get('parts', [{}])[0]
            extracted_json_text = content_part.get('text', '{}').strip()
            
            if not extracted_json_text.startswith('{') or not extracted_json_text.endswith('}'):
                st.error("Error: AI response for column matching was truncated or not valid JSON. This can happen with very large surveys.")
                st.text_area("Invalid AI Response Snippet", value=extracted_json_text[:1000], height=150)
                return []

            match_results = json.loads(extracted_json_text)
            return match_results.get('column_mapping', [])

        except requests.exceptions.RequestException as e:
            st.error(f"A network error occurred during AI matching: {e}")
            return []
        except json.JSONDecodeError as e:
            st.error(f"Error parsing AI response for column matching. The response was not valid JSON. Error: {e}")
            st.text_area("Raw AI Response", value=extracted_json_text, height=200)
            return []
        except (KeyError, IndexError) as e:
            st.error(f"Error processing AI response structure. It might be missing expected keys. Error: {e}")
            st.text_area("Raw AI Response", value=response.text if 'response' in locals() else "No response object.", height=200)
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred during AI Matching: {e}")
            return []

    def apply_numbering_to_dataset(self, df: pd.DataFrame, final_mapping_df: pd.DataFrame, prefix_format: str, include_unmatched: bool) -> pd.DataFrame:
        """Applies the final numbering to the dataset based on the review table."""
        # Defensive check to prevent crashes from failed AI calls
        if not isinstance(final_mapping_df, pd.DataFrame) or not {'Assigned #', 'Column Name'}.issubset(final_mapping_df.columns):
            st.error("Could not apply numbering. The mapping table is missing required columns. Please click 'Confirm All Changes' after editing.")
            return pd.DataFrame() # Return empty dataframe to avoid crashing downstream

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
    
    aligner = QuestionNumberAligner()

    st.subheader("Step 1: Upload Your Files")
    col1, col2 = st.columns(2)
    with col1:
        data_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    with col2:
        pdf_file = st.file_uploader("Upload Survey Report (PDF)", type=['pdf'])

    # Load data file into state immediately
    if data_file and (st.session_state.original_df is None or data_file.name != st.session_state.get('loaded_data_file_name')):
        try:
            df = pd.read_csv(data_file) if data_file.name.endswith('.csv') else pd.read_excel(data_file)
            st.session_state.original_df = df
            st.session_state.loaded_data_file_name = data_file.name
            # Reset downstream state if a new file is uploaded
            st.session_state.review_df = None
            st.session_state.questions = []
            st.success(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")
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
                match_results = aligner.match_with_gemini(questions, column_headers)

                if match_results:
                    review_df = pd.DataFrame(match_results)
                    review_df['Col'] = [index_to_excel_col(i) for i in review_df['column_index']]
                    review_df.rename(columns={
                        'original_header': 'Column Name',
                        'assigned_question': 'Assigned #',
                        'reasoning': 'AI Reasoning'
                    }, inplace=True)
                    st.session_state.review_df = review_df[['Col', 'Column Name', 'Assigned #', 'AI Reasoning']]
                    st.success("✅ AI matching complete!")
                else:
                    st.error("AI matching failed to produce results. Please check the errors above. You may need to try again or use a smaller/simpler survey.")
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
        
        # The download logic now uses the canonical st.session_state.review_df
        final_df = aligner.apply_numbering_to_dataset(
            st.session_state.original_df, 
            st.session_state.review_df,
            prefix_format,
            include_unmatched
        )
        
        if not final_df.empty:
            st.markdown("##### Preview of Renumbered Data")
            st.dataframe(final_df.head())
            
            c1, c2 = st.columns(2)
            with c1:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='Numbered Data')
                excel_data = output.getvalue()
                st.download_button(
                    label="📑 Download as Excel",
                    data=excel_data,
                    file_name="renumbered_dataset.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with c2:
                csv = final_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name="renumbered_dataset.csv",
                    mime="text/csv",
                    use_container_width=True
                )

if __name__ == "__main__":
    create_streamlit_app()
