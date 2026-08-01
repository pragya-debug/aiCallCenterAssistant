import whisper
from utils.callstate import CallState
from utils.logger import log_step
from utils.validation import validate_transcript

# Load Whisper model once at module level — avoids reloading on every call
model = whisper.load_model("base")

def transcription_agent(state: CallState) -> CallState:
    """
    Transcribes audio file to text using OpenAI Whisper.
    Skips transcription if transcript already exists in state.
    Validates transcript quality before passing downstream.

    Args:
        state: Pipeline state containing audio_path to transcribe

    Returns:
        Updated state with transcript populated.
        Sets error field to 'bad_transcript' if validation fails.

    Raises:
        Exception: If Whisper transcription fails
    """
    log_step("transcription_agent", {"status": "starting"})
    if state.get("transcript"):
        log_step("transcription_agent", {"status": "transcript exists — validating"})
        validate_transcription(state)
        return state

    try:
        audio_path = state.get("audio_path", "")
        log_step("transcription_agent", {"status": "transcribing", "audio_path": audio_path})
        result = model.transcribe(audio_path)
        state["transcript"] = result["text"]
        log_step("transcription_agent", {
            "status": "transcription complete",
            "length": len(state["transcript"])
        })
    except Exception as e:
        log_step("transcription_agent", {"status": "failed", "error": str(e)})
        state["error"] = "bad_transcript"
        state["transcript"] = None
        # Update trace and return early
        if "trace" not in state or state["trace"] is None:
            state["trace"] = []
        state["trace"].append("transcription_agent done")
        return state

    validate_transcription(state)
    return state

def validate_transcription(state: CallState) -> None:
    """
    Validates transcript quality and sets error flag if invalid.
    Prevents bad transcripts from flowing downstream.

    Args:
        state: Pipeline state containing transcript to validate

    Returns:
        None — updates state in place
    """
    transcript = state.get("transcript")
    if not validate_transcript(transcript):
        log_step("validate_transcription", {"status": "failed — bad transcript"})
        state["error"] = "bad_transcript"
        state["transcript"] = None
    else:
        log_step("validate_transcription", {"status": "passed"})
        state["transcript"] = transcript

    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("transcription_agent done")
