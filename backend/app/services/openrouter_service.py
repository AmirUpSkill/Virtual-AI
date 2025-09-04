import logging
import re
import base64
from typing import Optional, Any, Dict, List

import anyio
import requests
from openai import OpenAI, NotFoundError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageGenerationError(Exception):
    """Raised when the upstream returns a user/actionable error (e.g., bad prompt, policy)."""


class UpstreamUnavailable(Exception):
    """Raised when upstream is temporarily unavailable (rate limit, 5xx). Safe to retry."""


def _build_messages(prompt: str, initial_url: Optional[str], reference_url: Optional[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if initial_url:
        content.append({"type": "image_url", "image_url": {"url": initial_url}})
    if reference_url:
        content.append({"type": "image_url", "image_url": {"url": reference_url}})
    return [{"role": "user", "content": content}]


def _extract_url_from_text(text: str) -> Optional[str]:
    match = re.search(r"https?://\S+", text)
    return match.group(0) if match else None


def _decode_b64(data_b64: str) -> bytes:
    try:
        return base64.b64decode(data_b64)
    except Exception as exc:
        raise ImageGenerationError("Failed to decode base64 image from upstream") from exc


def _fetch_bytes_from_url(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        raise UpstreamUnavailable(f"Upstream URL fetch failed: {resp.status_code}")
    if resp.status_code >= 400:
        raise ImageGenerationError(f"Failed to fetch image URL: {resp.status_code}")
    return resp.content


def _deep_find_image_payload(obj: Any) -> Optional[bytes]:
    """Search arbitrarily nested dict/list for a b64 image or image_url."""
    try:
        if isinstance(obj, dict):
            if obj.get("b64_json"):
                return _decode_b64(obj["b64_json"])  # type: ignore[arg-type]
            if obj.get("image_url"):
                url_container = obj["image_url"]
                if isinstance(url_container, dict) and url_container.get("url"):
                    return _fetch_bytes_from_url(url_container["url"])  # type: ignore[arg-type]
            for v in obj.values():
                found = _deep_find_image_payload(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = _deep_find_image_payload(v)
                if found:
                    return found
    except Exception:
        return None
    return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_bytes_from_part(part: Any) -> Optional[bytes]:
    ptype = _get(part, "type")
    if ptype in {"image_url", "output_image"}:
        url_container = _get(part, "image_url")
        url = None
        if isinstance(url_container, dict):
            url = url_container.get("url")
        elif url_container is not None:
            url = getattr(url_container, "url", None)
        if url:
            return _fetch_bytes_from_url(url)

    b64_data = _get(part, "b64_json") or _get(part, "b64")
    if b64_data:
        return _decode_b64(b64_data)
    return None


def _extract_bytes_from_parts(parts: List[Any]) -> Optional[bytes]:
    for part in parts:
        try:
            data = _extract_bytes_from_part(part)
            if data:
                return data
        except Exception:
            continue
    return None


def _responses_fallback(
    client: OpenAI,
    model: str,
    prompt: str,
    initial_url: Optional[str],
    reference_url: Optional[str],
    extra_body: Optional[Dict[str, Any]],
) -> Optional[bytes]:
    try:
        resp2 = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        *([{ "type": "input_image", "image_url": {"url": initial_url}}] if initial_url else []),
                        *([{ "type": "input_image", "image_url": {"url": reference_url}}] if reference_url else []),
                    ],
                }
            ],
            extra_body=extra_body or {},
        )
    except NotFoundError as exc:
        logger.warning("Model not available via responses API: %s", exc)
        raise ImageGenerationError("Model not available for image output") from exc
    except Exception as exc:
        logger.warning("OpenRouter responses API call failed: %s", exc)
        raise UpstreamUnavailable("OpenRouter responses call failed") from exc

    try:
        data2 = resp2.model_dump()  # type: ignore[attr-defined]
    except Exception:
        data2 = getattr(resp2, "__dict__", None)

    return _deep_find_image_payload(data2) if data2 is not None else None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=8),
    retry=retry_if_exception_type(UpstreamUnavailable),
)
async def generate_image(
    *,
    prompt: str,
    initial_url: Optional[str] = None,
    reference_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "google/gemini-2.5-flash-image-preview:free",
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Call OpenRouter (OpenAI-compatible) to generate an edited image.

    Returns raw image bytes (PNG/JPEG) or raises a domain error.
    """
    key = api_key or settings.OPENROUTER_API_KEY
    messages = _build_messages(prompt=prompt, initial_url=initial_url, reference_url=reference_url)

    def _sync_request_and_extract_bytes() -> bytes:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        headers: Dict[str, str] = {}
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                extra_headers=headers or None,
                extra_body=extra_body or {},
            )
        except NotFoundError as exc:
            logger.warning("Model not found or not available: %s", exc)
            raise ImageGenerationError("Model not available. Check model name or your access.") from exc
        except Exception as exc:
            logger.warning("OpenRouter request failed: %s", exc)
            raise UpstreamUnavailable("OpenRouter request failed") from exc

        # Extract content
        try:
            content = getattr(resp.choices[0].message, "content", None)  # type: ignore[attr-defined]
        except Exception as exc:
            raise ImageGenerationError("Malformed response from upstream: missing message content") from exc

        if isinstance(content, str):
            url = _extract_url_from_text(content)
            if not url:
                raise ImageGenerationError("Upstream returned text content without an image URL")
            return _fetch_bytes_from_url(url)

        if isinstance(content, list):
            found = _extract_bytes_from_parts(content)
            if found:
                return found

        # Fallback to responses API
        found = _responses_fallback(
            client=client,
            model=model,
            prompt=prompt,
            initial_url=initial_url,
            reference_url=reference_url,
            extra_body=extra_body,
        )
        if found:
            return found

        raise ImageGenerationError("Upstream did not include an image payload")

    # Run the blocking HTTP work in a worker thread to keep the loop responsive
    return await anyio.to_thread.run_sync(_sync_request_and_extract_bytes)
