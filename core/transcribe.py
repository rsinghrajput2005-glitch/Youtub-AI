import logging
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

from .text_translator import translator
from .whisper_fallback import WhisperFallbackError, transcribe_youtube_audio


logger = logging.getLogger(__name__)

TRANSCRIPT_API_ERRORS = (
    RequestBlocked,
    IpBlocked,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

TRANSCRIPT_ERROR_MESSAGE = (
    "We couldn't retrieve a transcript for this video. YouTube may be blocking "
    "automated requests, or the video may not have accessible audio/captions."
)


class TranscriptError(RuntimeError):
    """Raised when captions and the audio transcription fallback both fail."""


def _fetch_transcript(video_link: str, video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()

    try:
        transcribe_chunk = ytt_api.fetch(video_id, languages=["hi", "en"])
        transcript = " ".join(chunk.text for chunk in transcribe_chunk).strip()
        if transcript:
            return transcript
        logger.warning("YouTube returned an empty transcript for %s", video_id)
    except TRANSCRIPT_API_ERRORS as error:
        logger.warning(
            "YouTube captions unavailable for %s (%s); trying Whisper fallback",
            video_id,
            type(error).__name__,
        )
    except Exception:
        logger.exception(
            "Unexpected youtube-transcript-api failure for %s; trying fallback",
            video_id,
        )

    try:
        return transcribe_youtube_audio(video_link)
    except WhisperFallbackError as error:
        logger.error("Transcript fallback failed for %s: %s", video_id, error)
        raise TranscriptError(TRANSCRIPT_ERROR_MESSAGE) from error


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

    transcript = _fetch_transcript(video_link, video_id)
    if translate:
        transcript = translator(transcript)

    return transcript
