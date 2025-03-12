from pydantic import BaseSettings, AnyHttpUrl
from decouple import config
import os 
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.environ.get('DATABASE_URL')

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    UPLOADS_DIR: str = config('UPLOADS_FOLDER')
    PROJECT_NAME: str = 'Nifti Storage Service'
    BACKEND_CORS_ORIGINS: str = '*'
    Database_URls: str = DATABASE_URL
    

    class Config:
        case_sensitive = True


settings = Settings()
