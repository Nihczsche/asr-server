from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from asr_server.config import Config
from asr_server.triton_manager import TritonManager

config = Config()
triton_manager = TritonManager(config.whisper)

@lru_cache
def get_config() -> Config:
    return config

ConfigDependency = Annotated[Config, Depends(get_config)]

@lru_cache
def get_triton_manager() -> TritonManager:
    return triton_manager

TritonManagerDependency = Annotated[TritonManager, Depends(get_triton_manager)]
