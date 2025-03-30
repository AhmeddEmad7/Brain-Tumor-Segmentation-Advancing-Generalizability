from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
engine = create_engine(settings.Database_URls) 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

print("Database URL:", settings.Database_URls)
with engine.connect() as connection:
    print("Database connected successfully!")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()