from evaluate import run_evaluation_suite
from utils.callstate import CallState
from utils.logger import log_step

def evaluation_agent(state: CallState) -> CallState:
    """
    Final pipeline agent — runs multi-dimension evaluation
    framework against all pipeline outputs.
    Stores results in CallState for UI display.

    Args:
        state: Pipeline state containing call data such as qa score, transcript etc

    Returns:
        Updated state with evaluation result populated

    Raises:
        Exception: If evaluation generation or storage fails
    """
    log_step("evaluation_agent", {"status": "starting evaluation of pipeline outputs"})

    try:
        # Extract qa_score from state — it is a dict
        # resolution is on 0-10 scale — convert to 0-1 for eval framework
        qa_score_data = state.get("qa_score", {})
        resolution_score = qa_score_data.get("resolution", 0)
        qa_score_normalized = resolution_score / 10.0
        
        # Prepare pipeline output for evaluation
        pipeline_output = {
            "transcript": state.get("transcript", ""),
            "summary": str(state.get("summary", "")),
            "qa_score": qa_score_normalized,
            "next_agent": state.get("next", ""),
            "recommendation": str(state.get("recommendation", ""))
        }
        
        # Run all five evaluations
        report = run_evaluation_suite(pipeline_output)
        
        # Store results in state
        state["eval_report"] = report
        state["eval_pass_rate"] = report["pass_rate"]
        state["eval_passed"] = report["passed"]
        state["eval_total"] = report["total"]
        
        log_step("evaluation_agent", {
            "status": "complete",
            "passed": report["passed"],
            "total": report["total"],
            "pass_rate": report["pass_rate"]
        })
        
    except Exception as e:
        log_step("evaluation_agent", {"status": "failed", "error": str(e)})
        state["eval_report"] = None
        state["eval_pass_rate"] = 0.0
        state["eval_passed"] = 0
        state["eval_total"] = 5

    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("evaluation_agent done")
    return state
