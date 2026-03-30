#routing_agent.py

def routing_agent(state):

    transcript = state.get("transcript")
    summary = state.get("summary")
    qa_score = state.get("qa_score")

    if transcript is None or len(transcript) < 10:
        return "retry_transcription"

    if summary is None or len(summary) < 20:
        return "retry_summary"

    if qa_score is None:
        return "retry_qa"

    return "complete"
