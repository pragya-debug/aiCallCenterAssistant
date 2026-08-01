import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import tempfile
import zipfile
import streamlit as st

from utils.agent_graph import build_graph
from utils.check_audio import is_audio_file


graph = build_graph()
st.set_page_config(page_title="Call Analyzer", layout="wide")

st.title("📞 AI Call Center Assistant")
tab1, tab2, tab3, tab4 = st.tabs([
    "Call Analyzer",
    "Langraph Agent Workflow",
    "Execution Details", 
    "Call Recommendations"
])

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

    uploaded_file = st.file_uploader(
        "Upload Call Audio", 
        type=["wav", "mp3", "m4a"]
    )

    if uploaded_file:
        if not is_audio_file(uploaded_file):
            st.error("❌ Please upload a valid audio file")
            st.stop()

        st.success("✅ Valid audio file uploaded")

        # Write to temp file — unique name, cleaned up after use
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            with st.spinner("Analyzing call..."):
                st.session_state["result"] = graph.invoke({
	            "audio_path": tmp_path
                })
                result = st.session_state["result"]

        except Exception as e:
            st.error("❌ Analysis failed. Please try again with a valid audio file.")
            st.exception(e)
            st.stop()

        finally:
            # Always clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


        st.subheader("Transcript")
        st.text_area("Call Transcript", result["transcript"], height=250)

        col1, col2 = st.columns(2)

        with col1:
            rsummary = result["summary"]
            st.subheader("Summary")
            st.write(rsummary["summary"])
        
            st.subheader("Key Issue")
            st.write(rsummary["key_issue"])

            st.subheader("Resolution")
            st.write(rsummary["resolution"])

            st.subheader("Sentiment")
            st.write(rsummary["sentiment"])

            st.subheader("Tags")
            for tag in rsummary["tags"]:
                st.markdown(f"- {tag}")

        with col2:
            rqa_score = result["qa_score"]
            st.subheader("Quality Scores (1-10)")
            st.metric("Empathy", rqa_score["empathy"])
            st.metric("Professionalism", rqa_score["professionalism"])
            st.metric("Resolution", rqa_score["resolution"])
            st.metric("Tone", rqa_score["tone"])

            st.subheader("Action Items")
            for aitems in rsummary["action_items"]:
                st.markdown(f"- {aitems}")


        
with tab2:
    st.subheader("LangGraph Agent Workflow")
    st.markdown("""
        ### Agent Pipeline

        1. Intake Agent - validates input  
        2. Transcription Agent - converts audio to text  
        3. Summarization Agent - extracts key insights  
        4. QA Agent - evaluates service quality  
        5. Routing Agent - handles retries and fallback
        6. Recommendation Agent - handles recommendations
        7. Evaluation Agent - handles evaluations of the pipeline.
    """)
    st.image(graph.get_graph().draw_mermaid_png())

with tab3:
    st.subheader("Execution trace")
    rtrace = st.session_state.get("result")
    if rtrace is not None:
        rtrace_logs = rtrace.get("trace", [])
        st.code("\n".join(rtrace_logs))

    st.subheader("Evaluation Results")
    if rtrace is not None:
        eval_pass_rate = rtrace.get("eval_pass_rate")
        eval_report = rtrace.get("eval_report")
        
        if eval_report:
            # Pass rate metric
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Pass Rate",
                    f"{eval_pass_rate:.0%}" if eval_pass_rate is not None else "N/A"
                )
            with col2:
                st.metric("Passed", rtrace.get("eval_passed", 0))
            with col3:
                st.metric("Total", rtrace.get("eval_total", 5))
            
            st.divider()
            
            # Per dimension results
            for result in eval_report.get("results", []):
                icon = "✅" if result["passed"] else "❌"
                st.write(f"{icon} **{result['test_name']}** — {result['details']}")
        else:
            st.info("No evaluation results available")

with tab4:
    st.subheader("Recommendations for Improvement")
    rec_result = st.session_state.get("result")
    if rec_result and rec_result.get("recommendation"):
        rec = rec_result["recommendation"]

        st.markdown("#### Improvement Areas")
        for area in rec.get("improvement_areas", []):
            st.markdown(f"- {area}")

        st.markdown("#### Suggested Phrases")
        for phrase in rec.get("suggested_phrases", []):
            st.markdown(f"- {phrase}")

        st.markdown("#### Overall Advice")
        st.write(rec.get("overall_advice", ""))

        if rec_result.get("improved_transcript"):
            st.markdown("#### Improved Transcript")
            st.text_area(
                "Improved Transcript",
                rec_result["improved_transcript"],
                height=250
            )
    else:
        st.write("Recommendation is not available. It is possible that call resolution is within approved limits ( > 5).")
