from typing import TypedDict, Optional, Dict

class CallState(TypedDict, total=False):
    """
    Typed dictionary representing shared state across all pipeline agents.
    Uses total=False so all fields are optional — agents populate
    fields incrementally as pipeline progresses.
    """
    # Pipeline input
    audio_path: str

    # Agent outputs
    transcript: Optional[str]
    summary: Optional[Dict]
    qa_score: Optional[Dict]
    recommendation: Optional[Dict]
    improved_transcript: str

    # Flow control
    error: Optional[str]
    retry_count: int
    trace: list[str]
    next: str

    # Evaluation outputs
    eval_report: Optional[Dict]
    eval_pass_rate: Optional[float]
    eval_passed: Optional[int]
    eval_total: Optional[int]
