#validation.py
def validate_input(data):

    if "audio_path" not in data and "transcript" not in data:
        return {"valid": False}

    metadata = {
        "source": "audio" if "audio_path" in data else "text"
    }

    return {
        "valid": True,
        "metadata": metadata
    }
