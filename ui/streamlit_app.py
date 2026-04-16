#streamlit_app.py
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from agents.transcription_agent import transcription_agent
from agents.routing_agent import routing_agent
from utils.agent_graph import build_graph
from utils.check_audio import is_audio_file
import zipfile
import io

graph = build_graph()
st.set_page_config(page_title="Call Analyzer", layout="wide")

st.title("📞 AI Call Center Assistant")
tab1, tab2, tab3, tab4 = st.tabs(["Call Analyzer","Langraph Agent Workflow","Execution Trace", "Call Recommendations"])

with tab1:
    sample_dir_path = os.path.join("data", "sample_transcripts")
    st.subheader("Download Sample Audio Dataset")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
      for file in os.listdir(sample_dir_path):
        file_path = os.path.join(sample_dir_path, file)
        zip_file.write(file_path, arcname=file)

    zip_buffer.seek(0)

    st.download_button(
      label="Download Sample Audio Files",
      data=zip_buffer,
      file_name="call_analyzer_sample_audio_dataset.zip",
      mime="application/zip"
    )    

    uploaded_file = st.file_uploader("Upload Call Audio", type=["wav", "mp3", "m4a"])
    
    if uploaded_file:
        if not is_audio_file(uploaded_file):
            st.error("❌ Please upload a valid audio file")
            st.stop()

        st.success("✅ Valid audio file uploaded")

        try:
            with st.spinner("Analyzing..."):
                with open("temp_audio.wav", "wb") as f:
                    f.write(uploaded_file.read())
                st.session_state["result"] = graph.invoke({
	            "audio_path": "temp_audio.wav"
                })
                result = st.session_state["result"]
        except Exception as e:
            st.warning("=== Please upload a valid audio file. Check error details below ====")
            st.exception(e)
            st.stop()

        st.subheader("Transcript")
        st.text_area("Call Transcript", result["transcript"], height=250)

        col1, col2 = st.columns(2)

        with col1:
            #rsummary = json.loads(result["summary"])
            rsummary = result["summary"]
            st.subheader("Summary")
            st.write(rsummary["summary"])
        
            st.subheader("Key Issue")
            st.write(rsummary["key_issue"])

            st.subheader("Resolution")
            st.write(rsummary["resolution"])

            st.subheader("Sentiment")
            st.write(rsummary["sentiment"])

        with col2:
            #rqa_score = json.loads(result["qa_score"])
            rqa_score = result["qa_score"]
            st.subheader("Quality Scores (1-10)")
            st.metric("Empathy", rqa_score["empathy"])
            st.metric("Professionalism", rqa_score["professionalism"])
            st.metric("Resolution", rqa_score["resolution"])
            st.metric("Tone", rqa_score["tone"])

            st.subheader("Action Items")
            for aitems in rsummary["action_items"]:
                #st.write(rsummary["action_items"])
                st.markdown(f"- {aitems}")


        st.subheader("Tags")
        for tag in rsummary["tags"]:
            st.markdown(f"- {tag}")
        
with tab2:
    st.subheader("LangGraph Agent Workflow")
    st.markdown("""
        ### Agent Pipeline

        1. Intake Agent validates input  
        2. Transcription Agent converts audio to text  
        3. Summarization Agent extracts key insights  
        4. QA Agent evaluates service quality  
        5. Routing Agent handles retries and fallback
    """)
    st.image(graph.get_graph().draw_mermaid_png())

with tab3:
    st.subheader("Execution trace")
    rtrace = st.session_state.get("result")
    if rtrace is not None:
        rtrace_logs = rtrace.get("trace", [])
        st.code("\n".join(rtrace_logs))

with tab4:
    st.subheader("Recommendations for Improvement")
    rec_result = st.session_state.get("result")
    if rec_result and rec_result.get("recommendation"):
        st.json(rec_result["recommendation"])
        if rec_result.get("improved_transcript"):
            st.text_area("Improved transcript", rec_result["improved_transcript"], height=250)
    else:
        st.write("Recommendation is not available. It is possible that call resolution is within approved limits ( > 5).")
