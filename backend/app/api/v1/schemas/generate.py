import uuid
from pydantic import BaseModel, Field
from app.db.models import JobStatus

# --- Base Schema  : A Shared Properties --- 
class JobBase(BaseModel):
    prompt: str = Field(
        ...,
        description="Natural language instruction for the desired edit",
        examples=["A photo of a woman wearing a blue bag"],
    )
    initial_image_key: str = Field(
        ...,
        description="Object storage key for the initial image in S3/MinIO",
        examples=["user1/job1/initial.jpg"],
    )
    reference_image_key: str | None = Field(
        None,
        description="Object storage key for the optional reference image",
        examples=["user1/job1/reference.jpg"],
    )
# --- Create Schema --- 
class JobCreate(JobBase):
    pass 
# --- Update Schema --- 
class JobUpdate(BaseModel):
    status: JobStatus | None = None 
    generated_image_key: str | None = None
# --- Response Schema --- 
class Job(JobBase):
    uuid: uuid.UUID
    status: JobStatus 
    generated_image_key: str | None = None
    class Config:
        from_attributes = True 
        