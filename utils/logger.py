"""
Centralized logging utility for AI CallSense pipeline.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

def log_step(step: str, data: dict = None) -> None:
    """
    Logs a pipeline step with optional data payload.

    Args:
        step: Name of the agent or step being logged
        data: Optional dictionary of additional context

    Returns:
        None
    """
    logging.info({"step": step, "data": data})
