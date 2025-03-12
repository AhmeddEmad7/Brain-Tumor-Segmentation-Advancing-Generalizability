from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base
from app.core.database import Base
class NiftiModel(Base):
    __tablename__ = 'nifti_files'
    
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)
    file_path = Column(String, index=True)
    file_size = Column(String, index=True)
    subject = Column(String, index=True)    # New column for subject
    session = Column(String, index=True)    # New column for session
    modality = Column(String, index=True)
    uploaded_at = Column(TIMESTAMP, server_default=func.now())
    data_meta  = Column(JSON)
    is_deleted = Column(Boolean, default=False)