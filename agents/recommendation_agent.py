import json
import os
from utils.callstate import CallState
from pydantic import BaseModel, field_validator
from openai import OpenAI
from typing import List

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RecSchema(BaseModel):
    # enforces structured json format and avoids downstream crashes
    improvement_areas: List[str]
    suggested_phrases: List[str]
    overall_advice: str

    @field_validator("suggested_phrases", mode="before")
    def normalize_suggested_phrases(cls, v):
        # Ensure model returns a list
        if isinstance(v, dict):
            return list(v.values())
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("improvement_areas", mode="before")
    def normalize_improvement_areas(cls, v):
        if isinstance(v, dict):
            return list(v.values())
        if isinstance(v, str):
            return [v]
        return v


def recommendation_agent(state:CallState):

    transcript = state.get("transcript", "")
    qa_score = state.get("qa_score", {}).get("resolution", 0)

    prompt = f"""
    The following call transcript received a QA (resolution) score of {qa_score}/10.

    Provide recommendations to improve:
    - agent behavior
    - communication clarity
    - customer satisfaction

    Transcript:
    {transcript}

    Return JSON with:
    - improvement_areas must be a list of words
    - suggested_phrases must be a list of phrases
    - overall_advice
    Return JSON only.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw_output = response.choices[0].message.content

    # validate output per RecSchema
    try:
        rec_dict = json.loads(raw_output)
        valid_rec = RecSchema(**rec_dict).dict()
        state["recommendation"] = valid_rec
        state["improved_transcript"] = improved_transcript(valid_rec, transcript)
    except Exception as e:
        print ("bad recommendation error: ", e)
        state["error"] = "bad_recommendation"

    state["trace"].append("recommendation_done")
    return state


def improved_transcript(valid_rec, transcript):
    prompt = f"""
    Improve the following call transcript using the recommendations.

    Transcript:
    {transcript}

    Recommendations:
    {valid_rec}

    Rewrite the conversation to:
    - sound more empathetic
    - improve clarity
    - improve customer satisfaction

    Return only improved transcript as plain text.
    """
    improved_transcript = ""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )

    raw_output = response.choices[0].message.content
    if raw_output is not None:
        improved_transcript = str(raw_output)
        print ("Improved transcript generated successfully.")
    else:
        improved_transcript = "No improved transcript generated."
        print ("No Improved transcript generated.")

    return improved_transcript
