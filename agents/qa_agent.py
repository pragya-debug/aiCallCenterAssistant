# quality_score_agent.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.callstate import CallState

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def qa_agent(state: CallState):
    transcript = state["transcript"]
    rubric = """
Score (1-10) the call using rubric:

Empathy
Professionalism
Resolution
Tone
"""
    prompt = f"""
{rubric}

Transcript:
{transcript}

Return JSON only.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    state["qa_score"] = response.choices[0].message.content
    return state
