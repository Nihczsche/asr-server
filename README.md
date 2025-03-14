# Faster Whisper Server
`asr-server` is an OpenAI API-compatible transcription server which uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) as its backend.
Features:
- GPU and CPU support.
- Easily deployable using Docker.
- **Configurable through environment variables (see [config.py](./src/asr_server/config.py))**.
- OpenAI API compatible.
- Streaming support (transcription is sent via [SSE](https://en.wikipedia.org/wiki/Server-sent_events) as the audio is transcribed. You don't need to wait for the audio to fully be transcribed before receiving it).
- Live transcription support (audio is sent via websocket as it's generated).
- Dynamic model loading / offloading. Just specify which model you want to use in the request and it will be loaded automatically. It will then be unloaded after a period of inactivity.

Please create an issue if you find a bug, have a question, or a feature suggestion.

## OpenAI API Compatibility ++
See [OpenAI API reference](https://platform.openai.com/docs/api-reference/audio) for more information.
- Audio file transcription via `POST /v1/audio/transcriptions` endpoint.
    - Unlike OpenAI's API, `asr-server` also supports streaming transcriptions(and translations). This is useful for when you want to process large audio files and would rather receive the transcription in chunks as they are processed rather than waiting for the whole file to be transcribed. It works similarly to chat messages when chatting with LLMs.
- Audio file translation via `POST /v1/audio/translations` endpoint.
-  Live audio transcription via `WS /v1/audio/transcriptions` endpoint.
    - LocalAgreement2 ([paper](https://aclanthology.org/2023.ijcnlp-demo.3.pdf) | [original implementation](https://github.com/ufal/whisper_streaming)) algorithm is used for live transcription.
    - Only transcription of a single channel, 16000 sample rate, raw, 16-bit little-endian audio is supported.

## Quick Start
[Hugging Face Space](https://huggingface.co/spaces/Iatalking/fast-whisper-server)

![image](https://github.com/fedirz/asr-server/assets/76551385/6d215c52-ded5-41d2-89a5-03a6fd113aa0)

Using Docker
```bash
docker run --gpus=all --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/asr-server:latest-cuda
# or
docker run --publish 8000:8000 --volume ~/.cache/huggingface:/root/.cache/huggingface fedirz/asr-server:latest-cpu
```
Using Docker Compose
```bash
curl -sO https://raw.githubusercontent.com/fedirz/asr-server/master/compose.yaml
docker compose up --detach asr-server-cuda
# or
docker compose up --detach asr-server-cpu
```

Using Kubernetes: [tutorial](https://substratus.ai/blog/deploying-faster-whisper-on-k8s)

## Usage
If you are looking for a step-by-step walkthrough, check out [this](https://www.youtube.com/watch?app=desktop&v=vSN-oAl6LVs) YouTube video.

### OpenAI API CLI
```bash
export OPENAI_API_KEY="cant-be-empty"
export OPENAI_BASE_URL=http://localhost:8000/v1/
```
```bash
openai api audio.transcriptions.create -m nvidia/parakeet-rnnt-1.1b -f audio.wav --response-format text

openai api audio.translations.create -m nvidia/parakeet-rnnt-1.1b -f audio.wav --response-format verbose_json
```
### OpenAI API Python SDK
```python
from openai import OpenAI

client = OpenAI(api_key="cant-be-empty", base_url="http://localhost:8000/v1/")

audio_file = open("audio.wav", "rb")
transcript = client.audio.transcriptions.create(
    model="nvidia/parakeet-rnnt-1.1b", file=audio_file
)
print(transcript.text)
```

### cURL
```bash
# If `model` isn't specified, the default model is used
curl http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav"
curl http://localhost:8000/v1/audio/transcriptions -F "file=@audio.mp3"
curl http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "stream=true"
curl http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "model=nvidia/parakeet-rnnt-1.1b"
# It's recommended that you always specify the language as that will reduce the transcription time
curl http://localhost:8000/v1/audio/transcriptions -F "file=@audio.wav" -F "language=en"

curl http://localhost:8000/v1/audio/translations -F "file=@audio.wav"
```

### Live Transcription (using Web Socket)
From [live-audio](./examples/live-audio) example

https://github.com/fedirz/asr-server/assets/76551385/e334c124-af61-41d4-839c-874be150598f

[websocat](https://github.com/vi/websocat?tab=readme-ov-file#installation) installation is required.
Live transcribing audio data from a microphone.
```bash
ffmpeg -loglevel quiet -f alsa -i default -ac 1 -ar 16000 -f s16le - | websocat --binary ws://localhost:8000/v1/audio/transcriptions
```
