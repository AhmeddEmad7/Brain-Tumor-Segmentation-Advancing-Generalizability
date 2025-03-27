from pydantic  import BaseModel 
from typing import Optional, Any
from datetime  import datetime

class NiftiFileBase(BaseModel):
    file_name: str
    file_path: str
    file_size: int
    subject: str
    session: str
    modality: str
    data_meta: Optional[Any] = None
    is_deleted: bool = False

    
class NiftiFileCreate(NiftiFileBase):
    pass
from pydantic import BaseModel 
from typing import Optional, Any
from datetime import datetime

class NiftiFileBase(BaseModel):
    file_name: str
    file_path: str
    # If your SQLAlchemy column is a String, either convert to int here or change the SQLA column type.
    file_size: int  
    subject: str
    session: str
    modality: str
    # Provide a default value (or None) so it isn’t required
    data_meta: Optional[Any] = {}
    is_deleted: bool = False

# This is used when creating a new record.
class NiftiFileCreate(NiftiFileBase):
    pass

# This model is used when reading records from the DB.
class NiftiFile(NiftiFileBase):
    id: int
    uploaded_at: datetime

    class Config:
        orm_mode = True

class NiftiFile(NiftiFileBase):
    id: int
    uploaded_at: datetime

    class Config:
        from_attributes = True
