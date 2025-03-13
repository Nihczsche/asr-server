from __future__ import annotations

from collections import OrderedDict
import contextlib
from dataclasses import asdict
import gc
import json
import logging
import socket
import threading
import time
from typing import TYPE_CHECKING, BinaryIO

import nemo.collections.asr as nemo_asr
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import Segment, TranscriptionInfo, TranscriptionOptions, VadOptions
import numpy as np
from pytriton.client import ModelClient

# from pytriton.decorators import batch
from pytriton.client.utils import create_client_from_url, wait_for_server_ready
from pytriton.constants import CREATE_TRITON_CLIENT_TIMEOUT_S, DEFAULT_TRITON_STARTUP_TIMEOUT_S
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from tritonclient.grpc import InferenceServerException

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pytriton.proxy.types import Request

    from asr_server.config import (
        NeMoConfig,
    )

logger = logging.getLogger(__name__)

# TODO: enable concurrent model downloads


class NeMoClient:
    def __init__(
        self,
        model_id: str,
        whisper: nemo_asr.models.ASRModel,
        triton: Triton,
    ) -> None:
        self.model_id = model_id
        self.sampling_rate = 16000
        self.client = ModelClient(triton._url, self.model_id)  # noqa: SLF001

    def transcribe(
        self,
        audio: str | BinaryIO | np.ndarray,
        **kwargs,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]:
        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=self.sampling_rate)

        config = np.array(
            [bytes(json.dumps(kwargs), "utf-8")],
            dtype=np.object_,
        )

        inputs = {
            "audio": audio,
            "config": config,
        }

        result_dict = self.client.infer_sample(**inputs)
        segments = result_dict["segments"][0]
        segments = json.loads(segments)
        segments = [Segment(**segment) for segment in segments["seg_list"]]

        info = result_dict["ts_info"][0]
        info = json.loads(info)
        info = TranscriptionInfo(**info)

        return segments, info


class SelfDisposingNeMoInferFn:
    def __init__(
        self,
        model_id: str,
        nemo_config: NeMoConfig,
        triton: Triton,
        *,
        on_unload: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.nemo_config = nemo_config
        self.on_unload = on_unload

        self.ref_count: int = 0
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.nemo: nemo_asr.models.ASRModel | None = None

        self.triton = triton
        self.triton_model_id = self.model_id.replace("/", ".")
        self.model_version = 1
        self.infer_fn: Callable | None = None

    def unload(self) -> None:
        with self.rlock:
            if self.nemo is None:
                raise ValueError(f"Model {self.model_id} is not loaded. {self.ref_count=}")
            if self.ref_count > 0:
                raise ValueError(f"Model {self.model_id} is still in use. {self.ref_count=}")
            if self.expire_timer:
                self.expire_timer.cancel()

            # HACK: PyTriton does not have an unbind method
            with contextlib.closing(
                create_client_from_url(self.triton._url, network_timeout_s=CREATE_TRITON_CLIENT_TIMEOUT_S)  # noqa: SLF001
            ) as client:
                server_live = False
                with contextlib.suppress(socket.timeout, OSError, InferenceServerException):
                    server_live = client.is_server_live()

                triton_model_tuple = (self.triton_model_id.lower(), self.model_version)
                model = self.triton._model_manager._models.get(triton_model_tuple)  # noqa: SLF001
                if model is not None:
                    model.clean()
                    if server_live:
                        client.unload_model(model.model_name)
                    del self.triton._model_manager._models[triton_model_tuple]  # noqa: SLF001

                if server_live:
                    # after unload there is a short period of time when server is not ready
                    wait_for_server_ready(client, timeout_s=DEFAULT_TRITON_STARTUP_TIMEOUT_S)

            self.infer_fn = None
            self.nemo = None
            # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992

            gc.collect()
            logger.info(f"Model {self.model_id} unloaded")
            if self.on_unload is not None:
                self.on_unload(self.model_id)

    def _load(self) -> None:
        with self.rlock:
            assert self.nemo is None
            logger.debug(f"Loading model {self.model_id}")
            start = time.perf_counter()
            # self.nemo = WhisperModel(
            #     self.model_id,
            #     device=self.whisper_config.inference_device,
            #     device_index=self.whisper_config.device_index,
            #     compute_type=self.whisper_config.compute_type,
            #     cpu_threads=self.whisper_config.cpu_threads,
            #     num_workers=self.whisper_config.num_workers,
            # )
            self.nemo = nemo_asr.models.ASRModel.from_pretrained(
                self.model_id,
            )

            print(f"Type: {type(self.nemo)}")

            def infer_fn(requests: list[Request]) -> list[dict]:
                responses = []
                for request in requests:
                    in_0 = request["audio"]
                    in_1 = request["config"]

                    config_json = in_1[0]
                    config: dict = json.loads(config_json)  # TODO

                    output = self.nemo.transcribe(in_0, timestamps=True)

                    if isinstance(
                        self.nemo, nemo_asr.models.EncDecHybridRNNTCTCBPEModel | nemo_asr.models.EncDecRNNTBPEModel
                    ):
                        hyp_list = output[0]
                    else:
                        hyp_list = output

                    seg_list = []
                    duration = 0
                    for id_val, segment in enumerate(hyp_list[0].timestep["segment"]):
                        duration += segment["end"] - segment["start"]
                        seg = Segment(
                            id=id_val,
                            seek=0,
                            start=segment["start"],
                            end=segment["end"],
                            text=segment["segment"],
                            tokens=[],
                            avg_logprob=0.0,
                            compression_ratio=1.0,
                            no_speech_prob=0.5,
                            temperature=0.0,
                            words=None,
                        )
                        if hasattr(segment, "_asdict"):
                            seg_list.append(seg._asdict())
                        else:
                            seg_list.append(asdict(seg))

                    segments = {
                        "seg_list": seg_list,
                    }

                    transcription_info = TranscriptionInfo(
                        language="tbd",
                        language_probability=0.99,
                        duration=duration,
                        duration_after_vad=duration,
                        all_language_probs=None,
                        transcription_options=TranscriptionOptions(
                            beam_size=config.get("beam_size"),
                            best_of=config.get("best_of", 5),
                            patience=config.get("patience", 1),
                            length_penalty=config.get("length_penalty", 1),
                            repetition_penalty=config.get("repetition_penalty", 1),
                            no_repeat_ngram_size=config.get("no_repeat_ngram_size", 0),
                            log_prob_threshold=config.get("log_prob_threshold", -1.0),
                            no_speech_threshold=config.get("no_speech_threshold", 0.6),
                            compression_ratio_threshold=config.get("compression_ratio_threshold", 2.4),
                            temperatures=0.0,
                            initial_prompt="",
                            prefix=None,
                            suppress_blank=True,
                            suppress_tokens=None,
                            prepend_punctuations="",
                            append_punctuations="",
                            max_new_tokens=None,
                            hotwords=None,
                            word_timestamps=False,
                            hallucination_silence_threshold=None,
                            condition_on_previous_text=False,
                            clip_timestamps=[],
                            prompt_reset_on_temperature=0.5,
                            multilingual=False,
                            without_timestamps=True,
                            max_initial_timestamp=0.0,
                        ),
                        vad_options=VadOptions(),
                    )

                    if hasattr(transcription_info, "_asdict"):
                        ts_info = transcription_info._asdict()
                    else:
                        ts_info = asdict(transcription_info)

                    responses.append(
                        {
                            "segments": np.array(
                                [bytes(json.dumps(segments), "utf-8")],
                                dtype=np.object_,
                            ),
                            "ts_info": np.array(
                                [bytes(json.dumps(ts_info), "utf-8")],
                                dtype=np.object_,
                            ),
                        },
                    )

                return responses

            self.infer_fn = infer_fn
            self.triton.bind(
                model_name=self.triton_model_id,
                model_version=self.model_version,
                infer_func=self.infer_fn,
                inputs=[
                    Tensor(name="audio", dtype=np.float32, shape=(-1,)),
                    Tensor(name="config", dtype=object, shape=(1,)),
                ],
                outputs=[
                    Tensor(name="segments", dtype=object, shape=(1,)),
                    Tensor(name="ts_info", dtype=object, shape=(1,)),
                ],
                config=ModelConfig(
                    batching=False,
                ),
            )

            logger.info(f"Model {self.model_id} binded to Triton in {time.perf_counter() - start:.2f}s")

    def _increment_ref(self) -> None:
        with self.rlock:
            self.ref_count += 1
            if self.expire_timer:
                logger.debug(f"Model was set to expire in {self.expire_timer.interval}s, cancelling")
                self.expire_timer.cancel()
            logger.debug(f"Incremented ref count for {self.model_id}, {self.ref_count=}")

    def _decrement_ref(self) -> None:
        with self.rlock:
            self.ref_count -= 1
            logger.debug(f"Decremented ref count for {self.model_id}, {self.ref_count=}")
            if self.ref_count <= 0:
                if self.nemo_config.ttl > 0:
                    logger.info(f"Model {self.model_id} is idle, scheduling offload in {self.nemo_config.ttl}s")
                    self.expire_timer = threading.Timer(self.nemo_config.ttl, self.unload)
                    self.expire_timer.start()
                elif self.nemo_config.ttl == 0:
                    logger.info(f"Model {self.model_id} is idle, unloading immediately")
                    self.unload()
                else:
                    logger.info(f"Model {self.model_id} is idle, not unloading")

    def __enter__(self) -> NeMoClient:
        with self.rlock:
            if self.infer_fn is None:
                self._load()
            self._increment_ref()
            assert self.infer_fn is not None
            return NeMoClient(self.triton_model_id, self.nemo, self.triton)

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        self._decrement_ref()


class TritonManager:
    def __init__(self, nemo_config: NeMoConfig) -> None:
        self._triton = Triton(
            config=TritonConfig(
                http_port=8001,
                grpc_port=8002,
                metrics_port=8003,
            )
        )
        self._triton.run()
        self.nemo_config = nemo_config
        self.loaded_infer_fns: OrderedDict[str, SelfDisposingNeMoInferFn] = OrderedDict()
        self._lock = threading.Lock()

    def _handle_model_unload(self, model_name: str) -> None:
        if model_name in self.loaded_infer_fns:
            del self.loaded_infer_fns[model_name]

    def unload_model(self, model_name: str) -> None:
        with self._lock:
            model = self.loaded_infer_fns.get(model_name)
            if model is None:
                raise KeyError(f"Model {model_name} not found")
            self.loaded_infer_fns[model_name].unload()

    def load_model(self, model_name: str) -> SelfDisposingNeMoInferFn:
        with self._lock:
            if model_name in self.loaded_infer_fns:
                logger.debug(f"{model_name} model already loaded")
                return self.loaded_infer_fns[model_name]
            self.loaded_infer_fns[model_name] = SelfDisposingNeMoInferFn(
                model_name,
                self.nemo_config,
                self._triton,
                on_unload=self._handle_model_unload,
            )

            return self.loaded_infer_fns[model_name]
