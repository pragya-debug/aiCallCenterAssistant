#validation.py
def validate_input(data):
    # ensures valid audio file path or transcript is available as input data

    if "audio_path" not in data and "transcript" not in data:
        return {"valid": False}

    metadata = {
        "source": "audio" if "audio_path" in data else "text"
    }

    return {
        "valid": True,
        "metadata": metadata
    }

def validate_transcript(text):
    # prevents bad transcripts from flowing downstream
    return text and len(text) > 20
