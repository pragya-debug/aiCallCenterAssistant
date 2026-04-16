#routing_agent.py
from utils.logger import log_step
MAX_STEPS = 5

def routing_agent(state):

   state["retry_count"] = state.get("retry_count", 0) + 1
   # STOP infinite loops
   if state["retry_count"] > MAX_STEPS:
        state["error"] = "max_steps_exceeded"
        return "__end__"

   #centralized error driven routing, cleaner and scalable
   if state.get("error") == "bad_transcript":
        log_step("error bad_transcript")
        return "transcription_agent"

   if state.get("error") == "bad_summary":
        log_step("error bad_summary")
        return "summarization_agent"

   if state.get("error") == "bad_qa":
        log_step("error bad_qa")
        return "qa_agent"

   qa_score = state.get("qa_score", {}).get("resolution", 0)

   # if resolution score is <= 50% (ie 5/10) then ask for recommendation
   if qa_score <= 5 and not state.get("recommendation"):
        return "recommendation_agent"

   state["trace"].append("complete")
   return "__end__"
