import asyncio
import logging
import os
import secrets
from typing import Optional

# Ensure project root (backend/) is on sys.path when running as a script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import anyio
import requests

from app.core.config import settings
from app.db.session import AsyncSessionLocal, async_engine
from app.db.base import Base
from app.api.v1.schemas.generate import JobCreate
from app.crud import crud_job
from app.db.models import JobStatus
from app.services import openrouter_service, storage_service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke")


DEFAULT_INITIAL_URL = (
    "https://i.pinimg.com/736x/fd/80/8e/fd808e5c2377c94bc21b5453dfccfc33.jpg"
)
DEFAULT_REFERENCE_URL = (
    "https://i.pinimg.com/1200x/a8/a1/fb/a8a1fb968a89178036f36a32bb1ee27a.jpg"
)
DEFAULT_PROMPT = (
    "Make the first model wear the outfit from the second image. Maintain identity and realism."
)


def sniff_content_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "application/octet-stream"


async def _download_bytes(url: str, timeout: int = 60) -> bytes:
    def _get() -> bytes:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content

    return await anyio.to_thread.run_sync(_get)


async def main():
    # Ensure DB schema exists
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Ensure required env values are present
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured. Set it in backend/.env")

    initial_url = os.getenv("SMOKE_INITIAL_URL", DEFAULT_INITIAL_URL)
    reference_url = os.getenv("SMOKE_REFERENCE_URL", DEFAULT_REFERENCE_URL)
    prompt = os.getenv("SMOKE_PROMPT", DEFAULT_PROMPT)

    logger.info("Using initial_url=%s", initial_url)
    logger.info("Using reference_url=%s", reference_url)
    model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-image-preview")
    logger.info("Using model=%s", model)

    # Create a DB session
    async with AsyncSessionLocal() as db:
        # Prepare MinIO keys
        run_id = secrets.token_hex(8)
        initial_key = f"smoke/{run_id}/initial.jpg"
        reference_key = f"smoke/{run_id}/reference.jpg"
        generated_key = f"smoke/{run_id}/generated.jpg"

        # Create job row as PENDING
        job = await crud_job.create_job(
            db,
            job_in=JobCreate(
                prompt=prompt,
                initial_image_key=initial_key,
                reference_image_key=reference_key,
            ),
        )
        logger.info("Created job uuid=%s status=%s", job.uuid, job.status)

        # Upload initial + reference images into MinIO (for parity with real flow)
        try:
            initial_bytes = await _download_bytes(initial_url)
            ref_bytes = await _download_bytes(reference_url)

            await storage_service.upload_from_bytes(
                key=initial_key, data=initial_bytes, content_type=sniff_content_type(initial_bytes)
            )
            await storage_service.upload_from_bytes(
                key=reference_key, data=ref_bytes, content_type=sniff_content_type(ref_bytes)
            )
        except Exception as exc:
            logger.exception("Failed uploading input images to MinIO: %s", exc)
            await crud_job.update_job_status(db, job_uuid=job.uuid, new_status=JobStatus.FAILED)
            raise

        # Call OpenRouter to generate the edited image
        try:
            gen_bytes = await openrouter_service.generate_image(
                prompt=prompt,
                initial_url=initial_url,
                reference_url=reference_url,
                http_referer=os.getenv("HTTP_REFERER"),
                x_title=os.getenv("X_TITLE"),
                model=model,
                extra_body={"max_tokens": 1024}
            )
        except openrouter_service.ImageGenerationError as e:
            logger.error("Generation error: %s", e)
            await crud_job.update_job_status(db, job_uuid=job.uuid, new_status=JobStatus.FAILED)
            raise
        except openrouter_service.UpstreamUnavailable as e:
            logger.error("Upstream unavailable: %s", e)
            await crud_job.update_job_status(db, job_uuid=job.uuid, new_status=JobStatus.FAILED)
            raise

        # Upload the generated image to MinIO
        await storage_service.upload_from_bytes(
            key=generated_key, data=gen_bytes, content_type=sniff_content_type(gen_bytes)
        )

        # Mark job COMPLETED and attach generated key
        await crud_job.update_job_status(
            db, job_uuid=job.uuid, new_status=JobStatus.COMPLETED, generated_key=generated_key
        )

        # Produce a signed URL for manual verification
        url = await storage_service.get_signed_url(key=generated_key, expires=3600)
        logger.info("SMOKE SUCCESS -> job_uuid=%s\nSigned URL (1h): %s", job.uuid, url)
        print(f"JOB_UUID={job.uuid}")
        print(f"SIGNED_URL={url}")


if __name__ == "__main__":
    asyncio.run(main())
