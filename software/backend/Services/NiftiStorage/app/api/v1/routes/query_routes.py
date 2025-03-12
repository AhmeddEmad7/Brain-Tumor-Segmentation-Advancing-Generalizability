import os
from fastapi import APIRouter, HTTPException, status, Depends
from app.utils import query_helpers as utils
from pathlib import Path
from app.core.config import settings
from app.utils import BDISModalityType
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.v1.nifti_crud import get_all_nifti_files
from app.models.nifti_model import NiftiModel
# User Route Entry Point
query_router = APIRouter()
UPLOAD_DIR = Path(settings.UPLOADS_DIR)

@query_router.get("/db/files")
async def get_all_files(db: Session = Depends(get_db)):
    """
    Return all file records with subject, session, modality, file name, and file path.
    """
    nifti_files = get_all_nifti_files(db)
    if not nifti_files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files found in the database")
    
    files_data = [
        {
            "subject": file.subject,
            "session": file.session,
            "modality": file.modality,
            "file_name": file.file_name,
            "file_path": file.file_path
        }
        for file in nifti_files
    ]
    return {"files": files_data}
@query_router.get("/db/subject/{subject_id}/session/{session_id}/modality/{modality_type}")
async def get_specific_file_paths(
    subject_id: str, 
    session_id: str, 
    modality_type: BDISModalityType, 
    db: Session = Depends(get_db)
):
    """
    Return all file paths for a specific subject, session, and modality.
    """
    files = db.query(NiftiModel).filter(
        NiftiModel.subject == subject_id,
        NiftiModel.session == session_id,
        NiftiModel.modality == modality_type
    ).all()
    if not files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No files found for the given subject, session, and modality")
    files_data = [
        {
            "file_name": file.file_name,
            "file_path": file.file_path,
            "subject" : file.subject,
            "session" : file.session,
            "modality" : file.modality
        }
        for file in files
    ]
    return {"files": files_data}
@query_router.get("/")
async def get_all_files():
    # Get all the files in the upload directory
    projects = os.listdir(UPLOAD_DIR)
    projects_object = {}

    for project in projects:
        projects_object[project] = {}
        project_files = utils.get_files_in_project(project, UPLOAD_DIR)
        projects_object[project] = project_files

    return {
        "files": projects_object
    }


@query_router.get("/project/{project_id}")
async def get_project_files(project_id: str):
    # check if the project exists
    if not os.path.exists(UPLOAD_DIR / project_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Project does not exist")

    # Get all the files in the upload directory
    project_files = utils.get_files_in_project(project_id, UPLOAD_DIR)

    return {
        "files": project_files
    }


@query_router.get("/project/{project_id}/sub/{sub_id}")
async def get_subs_files(project_id: str, sub_id: str):
    # check if the project exists
    if not os.path.exists(UPLOAD_DIR / project_id / f'sub-{sub_id}'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Subject does not exist")

    # Get all the files in the upload directory
    project_files = utils.get_files_in_project(project_id, UPLOAD_DIR)

    return {
        "files": project_files[f'sub-{sub_id}']
    }


@query_router.get("/project/{project_id}/sub/{sub_id}/modality/{modality_id}")
async def get_modality_files(project_id: str, sub_id: str, modality_id: BDISModalityType):

    # check if the modality exists in the subject
    if not os.path.exists(UPLOAD_DIR / project_id / f'sub-{sub_id}' / modality_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The modality does not exist in this subject")

    # Get all the files in the upload directory
    project_files = utils.get_files_in_project(project_id, UPLOAD_DIR)

    return {
        "files": project_files[f'sub-{sub_id}'][modality_id]
    }

