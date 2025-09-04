import uuid
import enum
from sqlalchemy import Column, Integer, String, DateTime, func, Enum
from sqlalchemy.dialects.postgresql import UUID
from .base import Base

class JobStatus(str , enum.Enum):
    """
        Enum for the status of a generation Job .
    """
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class GenerationJob(Base):
    """
        Model for the generation jobs table.
    """
    __tablename__ = "generation_jobs"

    # --- Primary Key --- 
    id = Column(Integer , primary_key=True , index=True)
    # --- Public unique indentifier --- 
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True, nullable=False)
    # --- Job Details --- 
    prompt = Column(String, nullable=False)
    status = Column(Enum(JobStatus) , default=JobStatus.PENDING)
    # --- MinIO object storage keys --- 
    initial_image_key = Column(String , nullable=False)
    reference_image_key = Column(String , nullable=True)
    generated_image_key = Column(String, nullable=True)
    # --- Timestamps --- 
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<GenerationJob(uuid='{self.uuid}', status='{self.status}')>"