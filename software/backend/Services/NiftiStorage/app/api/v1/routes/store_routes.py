from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends, Form
from app.utils import BDISModalityType
from app.core.config import settings
from pathlib import Path
from app.utils import store_helpers as utils
from app.core.database import get_db
from app.schemas.nifti_schema import NiftiFileCreate
from app.api.v1.nifti_crud import create_nifti_file
from decouple import config
from app.models.nifti_model import NiftiModel
from sqlalchemy.orm import Session
import csv
import json
import os

store_router = APIRouter()
PROJECT_DIR = Path(__file__).resolve().parents[3]

# Get the folder name from the environment variable, e.g., "storage"
uploads_folder_name = config('UPLOADS_FOLDER')
UPLOAD_DIR = PROJECT_DIR / uploads_folder_name
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Define the participants.tsv file (for subject-level metadata)
PARTICIPANTS_TSV = UPLOAD_DIR / "participants.tsv"

def update_participants_tsv(subject: str, age: str = None, sex: str = None):
    headers = ["participant_id", "age", "sex"]
    file_exists = PARTICIPANTS_TSV.exists()
    rows = []
    if file_exists:
        with open(PARTICIPANTS_TSV, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(row)
    if not any(row["participant_id"] == subject for row in rows):
        new_row = {"participant_id": subject, "age": age if age else "", "sex": sex if sex else ""}
        with open(PARTICIPANTS_TSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_row)

def decode_bytes(obj):
    if isinstance(obj, dict):
        return {k: decode_bytes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_bytes(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    else:
        return obj

@store_router.post("/")
async def upload_nifti_file(
    file: UploadFile = File(...),
    file_type: BDISModalityType = 'anat',
    subject_num: int = Form(...),
    session_num: int = Form(...),
    subject_age: str = Form(None),
    subject_sex: str = Form(None),
    db: Session = Depends(get_db)
):
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file was uploaded")
    if not utils.check_is_nifti(file):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type, we only accept NIfTI files"
        )

    # Convert numeric inputs to BIDS format
    subject = f"sub-{subject_num:02d}"
    session = f"ses-{session_num:02d}"

    # Update participants.tsv if subject is new
    update_participants_tsv(subject, subject_age, subject_sex)

    # Validate modality
    if file_type not in [member.value for member in BDISModalityType]:
        raise ValueError("Invalid modality type")

    # Determine scan label and file extension
    scan_label = "T1w" if file_type == "anat" else file_type
    if file.filename.endswith('.nii.gz'):
        ext = '.nii.gz'
    elif file.filename.endswith('.nii'):
        ext = '.nii.gz'
    else:
        ext = '.nii.gz'

    # Construct new file name and storage path (folder structure: UPLOAD_DIR/sub-XX/ses-XX/<modality>)
    new_file_name = f"{subject}_{session}_{scan_label}{ext}"
    case_path = utils.get_path(subject, session, file_type)
    if not os.path.exists(case_path):
        os.makedirs(case_path)
    file_path = os.path.join(case_path, new_file_name)
    print('file_path in nifti', file_path)

    # Check if the file already exists in the database
    niftifile = db.query(NiftiModel).filter(NiftiModel.file_path == file_path).first()    
    if niftifile is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File already exists")
    relative_file_path = os.path.relpath(file_path)
    relative_file_path = Path(relative_file_path).as_posix() 
    # Wrap file save and DB operations in a try/except block
    try:
        # Save the NIfTI file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        file_size = os.path.getsize(file_path)
        
        # Extract scan-specific metadata and save as a sidecar JSON file
        scan_metadata = utils.extract_nifti_metadata(file_path)
        sidecar_path = os.path.splitext(file_path)[0] + ".json"
        decoded_metadata = decode_bytes(scan_metadata)
        with open(sidecar_path, "w") as f:
            json.dump(decoded_metadata, f, indent=4)
        
        # Build the database record
        # Ensure that fields such as id, uploaded_at, and data_meta are optional in your NiftiFileCreate model,
        # or provide default values here.
        nifti_data = NiftiFileCreate(
            file_name=new_file_name,
            file_path=f"{relative_file_path}",              
            file_size=file_size,
            modality=file_type,
            subject=subject,
            session=session
        )
        
        # Create the record in the database
        record = create_nifti_file(db=db, nifti=nifti_data)
    
    except Exception as e:
        # Clean up: remove file and sidecar JSON if an error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        # Optionally, remove the folder if it's empty
        if os.path.exists(case_path) and not os.listdir(case_path):
            os.rmdir(case_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
    return {
        "status": "success",
        "message": "File uploaded successfully",
        "subject": subject,
        "session": session,
        "file path":record.file_path,
        "record_id": record.id,
    }
