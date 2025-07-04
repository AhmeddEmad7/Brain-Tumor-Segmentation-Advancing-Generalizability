from sqlalchemy.orm import Session
from app.models.nifti_model import NiftiModel
from app.schemas.nifti_schema import NiftiFileCreate

def create_nifti_file(db: Session, nifti: NiftiFileCreate):
    db_nifti = NiftiModel(**nifti.dict())
    db.add(db_nifti)
    db.commit()
    db.refresh(db_nifti)
    return db_nifti

def get_all_nifti_files(db: Session, skip: int = 0, limit: int = 100):
    return db.query(NiftiModel).offset(skip).limit(limit).all()

def get_nifti_file_by_id(db: Session, file_id: int):
    # Assuming you have a model called NiftiFile and you are using SQLAlchemy
    return db.query(NiftiModel).filter(NiftiModel.id == file_id).first()
