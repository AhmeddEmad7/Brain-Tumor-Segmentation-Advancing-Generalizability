from typing import List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import date
T = TypeVar("T")

class ReportSchema(BaseModel):
    studyId: str
    content: str

    class Config:
        orm_mode = True


class RequestReport(ReportSchema):
    pass

class ReportHeader(BaseModel):
    patientId: str
    patientName: str
    studyDate: str
    modality: str

class PDFRequest(BaseModel):
    studyId: str
    header:ReportHeader
    content: str 
    

class Response(BaseModel, Generic[T]):
    code: T
    status: str
    message: str
    result: Optional[T]
    
class ReportResponse(BaseModel):
    studyId: str
    content: str