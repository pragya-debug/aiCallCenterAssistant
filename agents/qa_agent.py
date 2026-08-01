from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from utils.callstate import CallState
from utils.logger import log_step
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QASchema(BaseModel):
    """Validates and enforces QA scoring structure from GPT-4o response."""
    empathy: int
    professionalism: int
    resolution: int
    tone: int

def qa_agent(state: CallState) -> CallState:
    """
    Evaluates call transcript against QA rubric using GPT-4o.
    Scores call on empathy, professionalism, resolution and tone.

    Args:
        state: Pipeline state containing transcript to evaluate

    Returns:
        Updated state with qa_score dict populated with scores 1-10
        per dimension. Sets error field if scoring fails.

    Raises:
        Exception: If LLM call or response parsing fails
    """
    log_step("qa_agent", {"status": "starting"})
    transcript = state.get("transcript", "")
    rubric = """
    Score (1-10) the call using the following rubric:

    - empathy: Did the agent acknowledge customer feelings and show understanding?
    - professionalism: Was the agent polite, respectful, and appropriate?
    - resolution: Did the agent solve the issue or provide a clear next step?
    - tone: Was the agent's tone calm, positive, and helpful?

    Scoring guidelines:
    1-3 = Poor
    4-6 = Average
    7-8 = Good
    9-10 = Excellent
    """

    prompt = f"""
    {rubric}

    Transcript:
    {transcript}

    Return JSON only:
    - empathy
    - professionalism
    - resolution
    - tone
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw_qaoutput = response.choices[0].message.content
        # converts raw_qaoutput from str -> dict
        if not raw_qaoutput:
            log_step("qa_agent", {"status": "empty response"})
            state["qa_score"] = {}
        else:
            # Parse and validate response against QASchema
            qa_dict = json.loads(raw_qaoutput)
            qa_out = QASchema(**qa_dict).model_dump()
            state["qa_score"] = qa_out
            log_step("qa_agent", {"status": "complete", "qa_score": qa_out})
    except Exception as e:
        log_step("qa_agent", {"status": "failed", "error": str(e)})
        state["error"] = "bad_qa"
        state["qa_score"] = {}

    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("qa_agent done")
    return state
