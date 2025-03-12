from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_v1_router
from app.core.config import settings
from pathlib import Path
from app.core.database import Base, engine
import uvicorn

# ✅ Create the FastAPI app first
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1"
)

# ✅ Create tables in the database
Base.metadata.create_all(bind=engine)
print("Tables created successfully!")

# ✅ Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Define the directory where uploaded files will be saved
UPLOAD_DIR = Path(settings.UPLOADS_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

@app.get("/")
async def read_root():
    """
    Root endpoint for the API
    """
    return {
        "Welcome to": "Nifti Storage Service",
        "Made by": "Ibrahim Mohamed",
        "At": "Systems and Biomedical Engineering Department, Cairo University",
        "For": "Graduation Project Thesis",
    }

# ✅ Include the API V1 routes
app.include_router(api_v1_router)

# ✅ Run the app (only if script is run directly)
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # ✅ Change to "main:app"
        host="0.0.0.0",
        port=7070,
        reload=True
    )
