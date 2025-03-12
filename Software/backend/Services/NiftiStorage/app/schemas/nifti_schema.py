from pydantic  import BaseModel 
from typing import Optional, Any
from datetime  import datetime

class NiftiFileBase(BaseModel):
    id: Optional[int]
    file_name: str
    file_path: str
    file_size: int
    uploaded_at:Optional [datetime]
    is_deleted: bool = False
    subject: str       # New field for subject
    session: str       # New field for session
    modality: str
    data_meta : Optional[Any]
    
class NiftiFileCreate(NiftiFileBase):
    pass

class NiftiFile(NiftiFileBase):
    id: int
    uploaded_at: datetime

    class Config:
        orm_mode = True