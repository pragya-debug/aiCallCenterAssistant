import os
MIN_TRANSCRIPT_LENGTH = 20

def validate_input(data):
    """
    Validates pipeline input data.
    Ensures audio file path exists and is not empty.
    
    Args:
        data: Dictionary containing audio_path
        
    Returns:
        Dictionary with valid boolean and metadata
    """
    # Check audio_path key exists
    if "audio_path" not in data:
        return {
            "valid": False,
            "error": "No audio_path provided"
        }
    
    audio_path = data["audio_path"]
    
    # Check file exists on disk
    if not os.path.exists(audio_path):
        return {
            "valid": False,
            "error": f"Audio file not found: {audio_path}"
        }
    
    # Check file is not empty
    if os.path.getsize(audio_path) == 0:
        return {
            "valid": False,
            "error": "Audio file is empty"
        }
    
    return {
        "valid": True,
        "metadata": {"source": "audio"}
    }


def validate_transcript(text: str) -> bool:
    """
    Validates transcript meets minimum quality requirements.
    Prevents bad transcripts from flowing downstream.

    Args:
        text: Transcript text to validate

    Returns:
        True if transcript is valid, False otherwise
    """
    return bool(text and len(text) > MIN_TRANSCRIPT_LENGTH)
