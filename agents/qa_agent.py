# quality_score_agent.py
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.callstate import CallState
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QASchema(BaseModel):
    empathy: int
    professionalism: int
    resolution: int
    tone: int

def qa_agent(state: CallState):
    transcript = state["transcript"]
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

    Return JSON:
    - empathy
    - professionalism
    - resolution
    - tone
    
    Return JSON only.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw_qaoutput = response.choices[0].message.content
    try:
        # converts raw_qaoutput from str -> dict
        if not raw_qaoutput:
            state["qa_score"] = {}
        else:
            qa_dict = json.loads(raw_qaoutput)
            qa_out = QASchema(**qa_dict).dict()
            state["qa_score"] = qa_out
    except Exception as e:
        print ("Exception: ", e)
        state["error"] = "bad_qa"

    state["trace"].append("qa_done")
    return state
