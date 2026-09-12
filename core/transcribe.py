from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from .text_translator import translator

def get_transcript(video_link: str, translate: bool = False) -> str:
    video_link = video_link.strip()
    if not video_link:
        raise ValueError("Please enter a YouTube URL.")

    parsed_url = urlparse(video_link)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    if parsed_url.netloc in {"youtu.be", "www.youtu.be"}:
        video_id = parsed_url.path.strip("/").split("/")[0]
    elif "/shorts/" in parsed_url.path:
        video_id = parsed_url.path.split("/shorts/", 1)[1].split("/", 1)[0]

    if not video_id:
        raise ValueError("Enter a valid YouTube URL, such as https://www.youtube.com/watch?v=VIDEO_ID.")

    ytt_api = YouTubeTranscriptApi()

    transcribe_chunk = ytt_api.fetch(video_id, languages=['hi', 'en'])
    transcript = " ".join(chunk.text for chunk in transcribe_chunk)
    if translate is True:
        transcript = translator(transcript)

    return transcript
