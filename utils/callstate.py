from typing import TypedDict, Optional, Dict

class CallState(TypedDict, total=False):
    audio_path: str
    transcript: Optional[str]
    summary: Optional[Dict]
    qa_score: Optional[Dict]
    recommendation: Optional[Dict]
    improved_transcript: str

    error: Optional[str]
    retry_count: int
    trace: list[str]
    next: str

    # Evaluation outputs
    eval_report: Optional[Dict]
    eval_pass_rate: Optional[float]
    eval_passed: Optional[int]
    eval_total: Optional[int]
