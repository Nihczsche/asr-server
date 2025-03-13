from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from asr_server.api_models import TranscriptionSegment, TranscriptionWord
from asr_server.text_utils import Transcription

if TYPE_CHECKING:
    from faster_whisper import transcribe

    from asr_server.audio import Audio
    from asr_server.triton_manager import NeMoClient

logger = logging.getLogger(__name__)


class NeMoASR:
    def __init__(
        self,
        nemo: NeMoClient,
        **kwargs,
    ) -> None:
        self.nemo = nemo
        self.transcribe_opts = kwargs

    def _transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,
    ) -> tuple[Transcription, transcribe.TranscriptionInfo]:
        start = time.perf_counter()
        segments, transcription_info = self.nemo.transcribe(
            audio.data,
            initial_prompt=prompt,
            word_timestamps=True,
            **self.transcribe_opts,
        )
        segments = TranscriptionSegment.from_faster_whisper_segments(segments)
        words = TranscriptionWord.from_segments(segments)
        for word in words:
            word.offset(audio.start)
        transcription = Transcription(words)
        end = time.perf_counter()
        logger.info(
            f"Transcribed {audio} in {end - start:.2f} seconds. Prompt: {prompt}. Transcription: {transcription.text}"
        )
        return (transcription, transcription_info)

    async def transcribe(
        self,
        audio: Audio,
        prompt: str | None = None,
    ) -> tuple[Transcription, transcribe.TranscriptionInfo]:
        """Wrapper around _transcribe so it can be used in async context."""
        # is this the optimal way to execute a blocking call in an async context?
        # TODO: verify performance when running inference on a CPU
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._transcribe,
            audio,
            prompt,
        )
