# intake_agent.py
# validates input format

from utils.validation import validate_input

def intake_agent(input_data):

    validation_result = validate_input(input_data)

    if not validation_result["valid"]:
        raise ValueError("Invalid input")

    return {
        "audio_path": input_data.get("audio_path"),
        "transcript": input_data.get("transcript"),
        "metadata": validation_result["metadata"],
        "trace": ["intake done"],
        "retry_count": 0
    }
