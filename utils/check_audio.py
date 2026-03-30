import mimetypes

ALLOWED_EXTENSIONS = ["wav", "mp3", "m4a", "ogg", "flac"]
ALLOWED_AUDIO_TYPES = [
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/x-m4a",
    "audio/m4a",
    "audio/mp4"
]

def is_audio_file(file):

    extension = file.name.split(".")[-1].lower()
    mime_type, _ = mimetypes.guess_type(file.name)
    print ("Detected MIME:", mime_type)

    return extension in ALLOWED_EXTENSIONS or mime_type in ALLOWED_MIME_TYPES
