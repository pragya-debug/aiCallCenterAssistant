from utils.validation import validate_input

def intake_agent(input_data):
    """
    Entry point into pipeline, validates input file received from UI.
    Ensures valid audio file path is sent downstream.

    Args:
        input_data: Dictionary containing audio_path from UI

    Returns:
        Updated state with audio_path, metadata, and trace initialized

    Raises:
        ValueError: If input from UI is not a valid audio file
    """
    validation_result = validate_input(input_data)

    if not validation_result["valid"]:
        error_msg = validation_result.get("error", "Invalid input")
        raise ValueError(f"Input validation failed: {error_msg}")

    return {
        "audio_path": input_data.get("audio_path"),
        "metadata": validation_result["metadata"],
        "trace": ["intake_agent done"],
        "retry_count": 0
    }
