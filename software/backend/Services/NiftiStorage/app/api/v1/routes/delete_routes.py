from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends, Form
from app.utils import BDISModalityType
from app.core.config import settings
from pathlib import Path
from app.utils import store_helpers as utils
from app.core.database import get_db
from app.schemas.nifti_schema import NiftiFileCreate
from app.api.v1.nifti_crud import create_nifti_file , get_nifti_file_by_id
from decouple import config
from app.models.nifti_model import NiftiModel
from sqlalchemy.orm import Session
import csv
import json
import os
import shutil

delete_router = APIRouter()
PROJECT_DIR = Path(__file__).resolve().parents[4]
uploads_folder_name = config('UPLOADS_FOLDER')
UPLOAD_DIR = PROJECT_DIR / uploads_folder_name

@delete_router.delete("/db/files/{file_id}")
async def delete_nifti_file(file_id: int, db: Session = Depends(get_db)):
    print('PROJECT_DIR',PROJECT_DIR)
    nifti_file = get_nifti_file_by_id(db, file_id)
    if not nifti_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with id {file_id} not found in the database"
        )
    
    # Build the absolute file path.
    absolute_file_path = PROJECT_DIR / Path(nifti_file.file_path)
    print('Absolute file path:', absolute_file_path)
    

    # Delete the sidecar JSON file if it exists.
    sidecar_path = os.path.splitext(absolute_file_path)[0] + ".json"
    if os.path.exists(sidecar_path):
        os.remove(sidecar_path)
        print("Deleted sidecar JSON:", sidecar_path)
    
    # Compute the session folder by removing the last two segments.
    # For: storage/sub-01/sub-ses-01/anat/sub-01_ses-01_T1w.nii.gz, we want:
    # storage/sub-01/sub-ses-01
    session_folder = absolute_file_path.parent.parent
    print("Session folder to delete:", session_folder)
    
    try:
        if os.path.exists(session_folder):
            shutil.rmtree(session_folder)
            print(f"Deleted session folder: {session_folder}")
        else:
            print("Session folder does not exist:", session_folder)
    except Exception as e:
        print(f"Could not remove session folder {session_folder}: {e}")
    
    # Delete the record from the database.
    db.delete(nifti_file)
    db.commit()
    
    return {"status": "success", "message": "File and session folder deleted successfully", "record_id": file_id}
