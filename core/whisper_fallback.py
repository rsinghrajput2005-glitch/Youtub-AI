import logging
import os
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import whisper
from yt_dlp import YoutubeDL


logger = logging.getLogger(__name__)


class WhisperFallbackError(RuntimeError):
    """Raised when audio download or Whisper transcription cannot complete."""


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    return whisper.load_model(model_name)


def transcribe_youtube_audio(video_link: str) -> str:
    model_name = os.getenv("WHISPER_MODEL", "base").strip() or "base"

    try:
        with TemporaryDirectory(prefix="youtube-ai-") as temp_dir:
            output_template = str(Path(temp_dir) / "audio.%(ext)s")
            options = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
            }

            with YoutubeDL(options) as downloader:
                downloader.download([video_link])

            audio_files = [
                path for path in Path(temp_dir).glob("audio.*") if path.is_file()
            ]
            if not audio_files:
                raise WhisperFallbackError("No audio file was downloaded.")

            model = _load_model(model_name)
            result = model.transcribe(
                str(audio_files[0]),
                language=None,
                fp16=False,
            )
            transcript = (result.get("text") or "").strip()
            if not transcript:
                raise WhisperFallbackError("Whisper returned an empty transcript.")

            return transcript
    except WhisperFallbackError:
        raise
    except Exception as error:
        logger.exception("Whisper fallback failed for YouTube URL")
        raise WhisperFallbackError(
            "Audio download or Whisper transcription failed."
        ) from error