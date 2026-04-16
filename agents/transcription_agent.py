# transcription_agent.py
# Use Whisper API.”

import whisper
from utils.callstate import CallState
from utils.validation import validate_transcript

model = whisper.load_model("base")

def transcription_agent(state: CallState):

    if state.get("transcript"):
        validate_step(state)
        return state

    audio_path = state["audio_path"]
    result = model.transcribe(audio_path)
    state["transcript"] = result["text"]
    validate_step(state)

    return state

def validate_step(state):
    # prevents bad transcripts from flowing down
    transcript = state.get("transcript")
    if not validate_transcript(transcript):
        state["error"] = "bad_transcript"
        # TODO: check if this is needed
        state["transcript"] = None
    else:
        state["transcript"] = transcript

    state["trace"].append("transcription_done")
