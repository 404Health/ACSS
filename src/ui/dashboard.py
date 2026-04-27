import sys
import os
import tempfile
# Force Streamlit to recognize the main 'acss-core' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
from src.agents.workflow import build_workflow
from src.audio.transcriber import process_clinical_audio

# 1. Setup the Web Page
st.set_page_config(page_title="ACSS Dashboard", layout="wide")
st.title("🛡️ Autonomous Clinical Scribe System")
st.markdown("### Human-in-the-Loop Validation Interface")

# 2. Initialize the AI Pipeline
@st.cache_resource
def get_pipeline():
    return build_workflow()

app = get_pipeline()

# 3. Build the Sidebar (For Input)
st.sidebar.header("1. Input Clinical Audio")
st.sidebar.info("Upload a patient dictation or consultation audio file to begin.")

# Add the Audio File Uploader
uploaded_audio = st.sidebar.file_uploader("Upload Audio (.wav, .mp3)", type=["wav", "mp3", "m4a"])

transcript_input = ""

# Process the audio if uploaded
if uploaded_audio is not None:
    with st.sidebar.status("🎙️ Processing Audio with WhisperX...", expanded=True) as status:
        # Streamlit uploads are stored in memory. We need to save it to a temp file for WhisperX.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_audio.getvalue())
            tmp_path = tmp_file.name
            
        st.write("Transcribing audio locally...")
        # Run our custom transcriber function
        transcript_input = process_clinical_audio(tmp_path)
        status.update(label="✅ Audio Transcribed!", state="complete", expanded=False)
        
        # Clean up the temp file
        os.remove(tmp_path)

# Show the text box so the doctor can review/edit the raw transcript before running the Scribe
st.sidebar.markdown("### Raw Transcript")
transcript_input = st.sidebar.text_area(
    "Review and edit if necessary:",
    value=transcript_input,
    height=200
)

# 4. Run the Pipeline from the UI
if st.sidebar.button("Run Multi-Agent Pipeline", type="primary"):
    if not transcript_input:
        st.sidebar.error("Please upload audio or type a transcript first.")
    else:
        with st.spinner("🤖 Waking up Qwen models and running LangGraph Pipeline..."):
            initial_state = {
                "transcript": transcript_input,
                "extracted_facts": {},
                "soap_note": "",
                "validation_flags": [],
                "final_approved": False
            }
            
            final_state = app.invoke(initial_state)
            st.session_state["result"] = final_state

# 5. Build the Main Interface (For Review)
if "result" in st.session_state:
    state = st.session_state["result"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Drafted SOAP Note")
        edited_note = st.text_area(
            "Review and Edit (Required before EHR sync):", 
            value=state["soap_note"], 
            height=300
        )
        
        if st.button("✅ Approve & Send to EHR", type="primary"):
            st.success("Note Approved! Data successfully locked and synced to mocked EHR.")
            st.balloons()
            
    with col2:
        st.subheader("🛡️ Validator Audit & Codes")
        for flag in state["validation_flags"]:
            if "Error" in flag or "RED FLAG" in flag:
                st.error(flag)
            else:
                st.info(flag)
                
        st.markdown("---")
        st.subheader("JSON Extracted Facts")
        st.json(state["extracted_facts"])