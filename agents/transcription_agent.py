# transcription_agent.py
# Use Whisper API.”

import whisper
from utils.callstate import CallState

model = whisper.load_model("base")

def transcription_agent(state: CallState):

    if state.get("transcript"):
        return state

    audio_path = state["audio_path"]

    result = model.transcribe(audio_path)

    state["transcript"] = result["text"]

    return state
