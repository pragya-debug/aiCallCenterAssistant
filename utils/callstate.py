from typing import TypedDict, Optional, Dict

class CallState(TypedDict, total=False):
    audio_path: str
    transcript: Optional[str]
    summary: Optional[Dict]
    qa_score: Optional[Dict]
