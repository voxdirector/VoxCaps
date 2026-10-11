"""Serial Chat Completions transport for audio annotation and text refinement."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen


def response_text(response) -> str:
    if "text/event-stream" not in response.headers.get("Content-Type", ""):
        result = json.load(response)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("invalid_chat_response") from None
        if choice.get("finish_reason") == "length":
            raise ValueError("annotation_output_truncated")
        if not isinstance(content, str):
            raise ValueError("annotation_response_requires_text_content")
        return content

    parts, data = [], []

    def consume_event():
        if not data:
            return False
        value = "\n".join(data)
        data.clear()
        if value == "[DONE]":
            return True
        event = json.loads(value)
        if "error" in event:
            raise RuntimeError("annotation_stream_error")
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason") == "length":
                raise ValueError("annotation_output_truncated")
            content = choice.get("delta", {}).get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("annotation_response_requires_text_content")
                parts.append(content)
        return False

    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif not line and consume_event():
            break
    consume_event()
    if not parts:
        raise ValueError("annotation_response_has_no_text")
    return "".join(parts)


def request_json(
    messages: list[dict],
    *,
    endpoint: str,
    model: str,
    key_env: str = "DASHSCOPE_API_KEY",
    auth_scheme: str = "bearer",
    timeout: float = 360,
    max_tokens: int = 4096,
    stream: bool = True,
    modalities: list[str] | None = None,
) -> dict:
    url = urlsplit(endpoint)
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.username
        or url.password
        or url.fragment
        or "WORKSPACE_ID" in url.netloc
    ):
        raise ValueError("annotation_endpoint_is_not_configured")
    if set(parse_qs(url.query)) - {"api-version"}:
        raise ValueError("only_api_version_query_is_allowed")
    if not model.strip():
        raise ValueError("model_is_required")
    key = os.environ.get(key_env)
    if not key:
        raise ValueError("annotation_api_key_environment_variable_is_missing")
    if auth_scheme not in {"bearer", "azure"}:
        raise ValueError("unsupported_auth_scheme")
    if not isinstance(stream, bool):
        raise ValueError("stream_must_be_boolean")
    if modalities is not None and modalities != ["text"]:
        raise ValueError("annotation_requires_text_output_only")
    headers = {"Content-Type": "application/json"}
    headers["Authorization" if auth_scheme == "bearer" else "api-key"] = (
        "Bearer " + key if auth_scheme == "bearer" else key
    )
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "modalities": ["text"],
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            content = response_text(response)
    except HTTPError as error:
        raise RuntimeError(f"annotation_http_error:{error.code}") from None
    except URLError:
        raise RuntimeError("annotation_connection_failed") from None
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    annotation = json.loads(text)
    if not isinstance(annotation, dict):
        raise ValueError("annotation_response_requires_json_object")
    return annotation


def audio_content(audio: Path, media_format: str = "input_audio") -> dict:
    if audio.suffix.lower() != ".wav" or audio.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("annotation_requires_wav_up_to_32_mib")
    encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
    data_url = "data:audio/wav;base64," + encoded
    if media_format in {"video_url", "audio_url"}:
        return {"type": media_format, media_format: {"url": data_url}}
    if media_format in {"input_audio", "input_audio_raw"}:
        return {
            "type": "input_audio",
            "input_audio": {
                "data": encoded if media_format == "input_audio_raw" else data_url,
                "format": "wav",
            },
        }
    raise ValueError("unsupported_audio_transport")


def request_annotation(
    audio: Path, system: str, user: str, *, media_format: str = "input_audio", **kwargs
) -> dict:
    return request_json(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [{"type": "text", "text": user}, audio_content(audio, media_format)],
            },
        ],
        **kwargs,
    )


def request_text_json(system: str, text: str, **kwargs) -> dict:
    return request_json(
        [{"role": "system", "content": system}, {"role": "user", "content": text}], **kwargs
    )
