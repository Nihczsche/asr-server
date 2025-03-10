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

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.transcribe import Segment, TranscriptionInfo
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
        WhisperConfig,
    )

logger = logging.getLogger(__name__)

# TODO: enable concurrent model downloads


class WhisperClient:
    def __init__(
        self,
        model_id: str,
        whisper: WhisperModel,
        triton: Triton,
    ) -> None:
        self.model_id = model_id
        self.sampling_rate = whisper.feature_extractor.sampling_rate
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


class SelfDisposingWhisperInferFn:
    def __init__(
        self,
        model_id: str,
        whisper_config: WhisperConfig,
        triton: Triton,
        *,
        on_unload: Callable[[str], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.whisper_config = whisper_config
        self.on_unload = on_unload

        self.ref_count: int = 0
        self.rlock = threading.RLock()
        self.expire_timer: threading.Timer | None = None
        self.whisper: WhisperModel | None = None

        self.triton = triton
        self.triton_model_id = self.model_id.replace("/", ".")
        self.model_version = 1
        self.infer_fn: Callable | None = None

    def unload(self) -> None:
        with self.rlock:
            if self.whisper is None:
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
            self.whisper = None
            # WARN: ~300 MB of memory will still be held by the model. See https://github.com/SYSTRAN/faster-whisper/issues/992

            gc.collect()
            logger.info(f"Model {self.model_id} unloaded")
            if self.on_unload is not None:
                self.on_unload(self.model_id)

    def _load(self) -> None:
        with self.rlock:
            assert self.whisper is None
            logger.debug(f"Loading model {self.model_id}")
            start = time.perf_counter()
            self.whisper = WhisperModel(
                self.model_id,
                device=self.whisper_config.inference_device,
                device_index=self.whisper_config.device_index,
                compute_type=self.whisper_config.compute_type,
                cpu_threads=self.whisper_config.cpu_threads,
                num_workers=self.whisper_config.num_workers,
            )

            def infer_fn(requests: list[Request]) -> list[dict]:
                responses = []
                for request in requests:
                    in_0 = request["audio"]
                    in_1 = request["config"]

                    config_json = in_1[0]
                    config: dict = json.loads(config_json)

                    segments, transcription_info = self.whisper.transcribe(in_0, **config)

                    seg_list = []
                    for segment in segments:
                        if hasattr(segment, "_asdict"):
                            seg_list.append(segment._asdict())
                        else:
                            seg_list.append(asdict(segment))
                    segments = {
                        "seg_list": seg_list,
                    }

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
                if self.whisper_config.ttl > 0:
                    logger.info(f"Model {self.model_id} is idle, scheduling offload in {self.whisper_config.ttl}s")
                    self.expire_timer = threading.Timer(self.whisper_config.ttl, self.unload)
                    self.expire_timer.start()
                elif self.whisper_config.ttl == 0:
                    logger.info(f"Model {self.model_id} is idle, unloading immediately")
                    self.unload()
                else:
                    logger.info(f"Model {self.model_id} is idle, not unloading")

    def __enter__(self) -> WhisperClient:
        with self.rlock:
            if self.infer_fn is None:
                self._load()
            self._increment_ref()
            assert self.infer_fn is not None
            return WhisperClient(self.triton_model_id, self.whisper, self.triton)

    def __exit__(self, *_args) -> None:  # noqa: ANN002
        self._decrement_ref()


class TritonManager:
    def __init__(self, whisper_config: WhisperConfig) -> None:
        self._triton = Triton(
            config=TritonConfig(
                http_port=8001,
                grpc_port=8002,
                metrics_port=8003,
            )
        )
        self._triton.run()
        self.whisper_config = whisper_config
        self.loaded_infer_fns: OrderedDict[str, SelfDisposingWhisperInferFn] = OrderedDict()
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

    def load_model(self, model_name: str) -> SelfDisposingWhisperInferFn:
        with self._lock:
            if model_name in self.loaded_infer_fns:
                logger.debug(f"{model_name} model already loaded")
                return self.loaded_infer_fns[model_name]
            self.loaded_infer_fns[model_name] = SelfDisposingWhisperInferFn(
                model_name,
                self.whisper_config,
                self._triton,
                on_unload=self._handle_model_unload,
            )

            return self.loaded_infer_fns[model_name]
