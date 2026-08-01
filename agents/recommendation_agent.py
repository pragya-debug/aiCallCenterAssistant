from dotenv import load_dotenv
load_dotenv()
import json
from openai import OpenAI
import os
from pydantic import BaseModel, field_validator
from typing import List
from utils.callstate import CallState
from utils.logger import log_step

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RecSchema(BaseModel):
    """Enforces structured json format and avoids downstream crashes."""
    improvement_areas: List[str]
    suggested_phrases: List[str]
    overall_advice: str

    @field_validator("suggested_phrases", mode="before")
    def normalize_suggested_phrases(cls, v):
        """Ensure suggested_phrases is always a list."""
        if isinstance(v, dict):
            return list(v.values())
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("improvement_areas", mode="before")
    def normalize_improvement_areas(cls, v):
        """Ensure improvement_areas is always a list."""
        if isinstance(v, dict):
            return list(v.values())
        if isinstance(v, str):
            return [v]
        return v


def recommendation_agent(state: CallState) -> CallState:
    """
    Generates coaching recommendations for low scoring calls.
    Uses GPT-4o to analyze transcript and provide improvement areas,
    suggested phrases, and overall advice. Also generates an improved
    transcript applying the recommendations.

    Args:
        state: Pipeline state containing transcript and qa_score

    Returns:
        Updated state with recommendation and improved_transcript populated

    Raises:
        Exception: If LLM call or response parsing fails
    """
    log_step("recommendation_agent", {"status": "starting"})

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

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content

        # validate output against RecSchema
        rec_dict = json.loads(raw_output)
        valid_rec = RecSchema(**rec_dict).model_dump()
        state["recommendation"] = valid_rec
        state["improved_transcript"] = generate_improved_transcript(
	    valid_rec, transcript)
        log_step("recommendation_agent", {"status": "complete"})

    except Exception as e:
        log_step("recommendation_agent", {"status":"failed", "error": str(e)})
        state["error"] = "bad_recommendation"

    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("recommendation_agent done")
    return state


def generate_improved_transcript(valid_rec: dict, transcript: str) -> str:
    """
    Generates an improved version of the call transcript
    applying the coaching recommendations.

    Args:
        valid_rec: Validated recommendation dictionary containing
                   improvement_areas, suggested_phrases, overall_advice
        transcript: Original call transcript

    Returns:
        Improved transcript as plain text string

    Raises:
        Exception: If LLM call fails
    """
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )

        raw_output = response.choices[0].message.content

        if raw_output:
            log_step("generate_improved_transcript", {"status": "complete"})
            return str(raw_output)
        else:
            log_step("generate_improved_transcript", {"status": "empty response"})
            return "No improved transcript generated."

    except Exception as e:
        log_step("generate_improved_transcript", {"status": "failed", "error": str(e)})
        return "No improved transcript generated."
