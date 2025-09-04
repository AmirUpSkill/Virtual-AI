import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models import GenerationJob
from app.api.v1.schemas.generate import JobCreate, JobUpdate

async def create_job(db: AsyncSession, *, job_in: JobCreate) -> GenerationJob:
    """
    Create a new generation job in the database.
    """
    # Create a SQLAlchemy model instance from the Pydantic schema
    db_job = GenerationJob(**job_in.model_dump())

    # Add the instance to the session and commit
    db.add(db_job)
    await db.commit()

    # Refresh the instance to get the data back from the DB (like the generated UUID)
    await db.refresh(db_job)
    return db_job

async def get_job_by_uuid(db: AsyncSession, *, job_uuid: uuid.UUID) -> GenerationJob | None:
    """
    Retrieve a generation job by its UUID.
    """
    # Create a select statement
    statement = select(GenerationJob).where(GenerationJob.uuid == job_uuid)
    
    # Execute the statement and get the first result
    result = await db.execute(statement)
    return result.scalars().first()

async def update_job(db: AsyncSession, *, db_job: GenerationJob, job_in: JobUpdate) -> GenerationJob:
    """
    Update an existing generation job.
    """
    # Get the update data from the Pydantic model
    update_data = job_in.model_dump(exclude_unset=True)

    # Update the SQLAlchemy model's attributes
    for field, value in update_data.items():
        setattr(db_job, field, value)

    # Add and commit the changes
    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)
    return db_job

async def delete_job(db: AsyncSession, *, job_uuid: uuid.UUID) -> GenerationJob | None:
    """
    Delete a generation job by its UUID.
    """
    # First, get the job
    job_to_delete = await get_job_by_uuid(db, job_uuid=job_uuid)

    if job_to_delete:
        # If it exists, delete and commit
        await db.delete(job_to_delete)
        await db.commit()
    
    return job_to_delete


# Convenience helper to keep service layer thin
from app.db.models import JobStatus

async def update_job_status(
    db: AsyncSession,
    *,
    job_uuid: uuid.UUID,
    new_status: JobStatus,
    generated_key: str | None = None,
) -> GenerationJob | None:
    """
    Update a job's status (and optionally its generated image key) by UUID.

    Returns the updated job or None if not found.
    """
    job = await get_job_by_uuid(db, job_uuid=job_uuid)
    if not job:
        return None

    job.status = new_status
    if generated_key is not None:
        job.generated_image_key = generated_key

    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job
