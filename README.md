# Brain-Tumor-Segmentation-Advancing-Generalizability
#  Medical Imaging Platform – Setup Guide

This platform consists of multiple services such as `ApiGateway`,`Frontend`, `Reporting`, `Inference`, `NiftiStorage`, `Orthanc`, `RabbitMQ`, and `Redis`.

---

## 📦Backend and Services Overview

```
software/
├── frontend
└── backend
    ├── Services
    │   ├── Inference
    |   |   ├── segmentation
    |   │   └── motion_artifact_correction
    │   ├── NiftiStorage
    │   ├── Orthanc
    │   ├── RabbitMQ
    │   ├── Redis
    │   └── Reporting
    └── ApiGateway

```
## 🌐 Running Frontend and Backend (Software Folder)

### ▶️ Frontend

```bash
cd software/frontend
npm install
npm run dev
```

### ▶️ Backend

```bash
cd software/backend
```

---

### 🚪 Run ApiGateway

```bash
cd backend/ApiGateway
python -m venv venv
source venv/bin/activate        # or activate for Windows
pip install -r requirements.txt
python manage.py runserver
```

---
### 📍 Inference Subservices

Each AI service like `segmentation` or `motion_artifact_correction` uses RabbitMQ for message communication.

```bash
# Example for Segmentation or Motion Artifact Correction
cd Services/Inference/<subservice>

python -m venv venv
source venv/bin/activate        # For Linux/macOS
venv\Scripts\activate           # For Windows

pip install -r requirements.txt

# Start the service (waiting for RabbitMQ messages)
python connection.py
```

---
## 🛠️ Running Services Locally

> For **each service**, follow the steps below (except ApiGateway which has a special command to run):

### 🔁 Generic Steps

```bash
# 1. Navigate to the service
cd Services/<ServiceName>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # For Linux/macOS
venv\Scripts\activate           # For Windows

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the service
python main.py
```

---

## 🐳 Running Services with Docker

> Make sure Docker and Docker Compose are installed.

### 🧱 Step 1: Build Docker Images

```bash
docker compose build 
```

### ▶️ Step 2: Run Services Using Docker Compose

```bash
docker compose up
```

> This command builds and runs all services together as defined in your `docker-compose.yml`.

---

## ⚙️ Environment Configuration

Each service should include:
- `.env` file for environment variables.
- `requirements.txt` for dependencies.
- `Dockerfile` for Docker build.

---
