from utils.logger import log_step
from utils.callstate import CallState

MAX_STEPS = 5

def routing_agent(state: CallState) -> str:
    """
    Controls conditional flow between pipeline agents based on state.
    Handles error-driven retries and routes to recommendation or end
    based on QA resolution score.

    Args:
        state: Pipeline state containing error flags, qa_score,
               recommendation, and retry_count

    Returns:
        String name of next agent to run or '__end__' to terminate

    Raises:
        None — returns '__end__' on max retries exceeded
    """
    log_step("routing_agent", {"status": "starting"})

    state["retry_count"] = state.get("retry_count", 0) + 1
   
    # STOP infinite loops
    if state["retry_count"] > MAX_STEPS:
        log_step("routing_agent", {"status": "max steps exceeded", 
                                    "retry_count": state["retry_count"]})
        state["error"] = "max_steps_exceeded"
        return "__end__"

    #centralized error driven routing, cleaner and scalable
    if state.get("error") == "bad_transcript":
        log_step("routing_agent", {"status": "retrying", "agent": "transcription_agent"})
        return "transcription_agent"

    if state.get("error") == "bad_summary":
        log_step("routing_agent", {"status": "retrying", "agent": "summarization_agent"})
        return "summarization_agent"

    if state.get("error") == "bad_qa":
        log_step("routing_agent", {"status": "retrying", "agent": "qa_agent"})
        return "qa_agent"

    qa_score = state.get("qa_score", {}).get("resolution", 0)

    if qa_score <= 5 and not state.get("recommendation"):
        log_step("routing_agent", {"status": "routing to recommendation",
                                    "qa_score": qa_score})
        return "recommendation_agent"

    # All checks passed — pipeline complete
    log_step("routing_agent", {"status": "complete", "qa_score": qa_score})
 
    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("routing_agent done")

    return "__end__"
