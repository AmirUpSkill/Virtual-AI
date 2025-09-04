import logging
from functools import lru_cache

import anyio
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.core.config import settings

logger = logging.getLogger(__name__)

_BUCKET_CHECKED = False


@lru_cache(maxsize=1)
def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY_ID,
        aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def _ensure_bucket_sync():
    client = _get_client()
    bucket = settings.S3_BUCKET_NAME
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code == 404:
            client.create_bucket(Bucket=bucket)
        else:
            raise


async def ensure_bucket_exists() -> None:
    global _BUCKET_CHECKED
    if _BUCKET_CHECKED:
        return
    await anyio.to_thread.run_sync(_ensure_bucket_sync)
    _BUCKET_CHECKED = True


async def upload_from_bytes(*, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to the configured S3-compatible bucket under the provided key."""
    await ensure_bucket_exists()

    def _put_object():
        _get_client().put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    await anyio.to_thread.run_sync(_put_object)
    logger.info("Uploaded object to S3: key=%s bucket=%s", key, settings.S3_BUCKET_NAME)
    return key


async def get_signed_url(*, key: str, expires: int = 3600) -> str:
    """Generate a time-limited signed URL for reading an object."""
    await ensure_bucket_exists()

    def _sign() -> str:
        return _get_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.S3_BUCKET_NAME, "Key": key},
            ExpiresIn=expires,
        )

    return await anyio.to_thread.run_sync(_sign)
