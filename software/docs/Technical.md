# 🧠 Brain Tumor Segmentation Platform – Technical Documentation

## 1. 🧾 Platform Overview

An advanced, microservices-based medical imaging platform designed to support AI-powered brain tumor analysis through DICOM/NIfTI visualization, automated segmentation, real-time interaction, and clinical reporting.

## 2. 🛠️ Technology Stack

### 💻 Frontend

* **React 18**, **TypeScript**, **Vite**
* Medical Viewer: **Cornerstone3D**, **VTK.js**

### ⚙️ Backend Services

* **API Gateway**: Django
* **Inference Service**: Django
* **Reporting Service**: FastAPI
* **NIfTI Storage Service**: FastAPI
* **Orthanc PACS**: DICOM storage, retrieval, and web services

### 🧱 Infrastructure

* **Containerization**: Docker, Docker Compose
* **Database**: PostgreSQL
* **Message Queue**: RabbitMQ
* **Cache**: Redis

## 3. 🔗 Service Communication and Integration

### 🌐 API Gateway (Django)

* Acts as the single point of contact for the frontend
* Forwards requests to microservices: segmentation, reporting, nifti storage
* Handles authentication, routing, and permission validation

### 🧠 Inference Service (Django)

* Consumes DICOM/NIfTI images via RabbitMQ
* Applies segmentation model (ONNX/PyTorch-based)
* Outputs segmentation mask (DICOM-SEG, NIfTI)
* Publishes metadata to Redis for reporting and visualization

### 📝 Reporting Service (FastAPI)

* Accepts structured content and metadata
* Generates editable, LLM-assisted PDF reports
* Supports clinical sections: Findings, Impression, Diagnosis, Recommendations
* Integrates screenshot and measurement metadata

### 📦 NIfTI Storage Service (FastAPI)

* BIDS-compliant file handling
* Exposes REST endpoints to upload, retrieve, and organize `.nii`/`.nii.gz` files
* Converts and links to segmentation and viewer pipelines

### 🗂️ Orthanc PACS (DICOM Web)

* Handles storage/retrieval of DICOM & DICOM-SEG
* Supports WADO-RS, QIDO-RS, STOW-RS for web-based integration
* Provides anonymization and caching mechanisms

### 🧩 Redis & RabbitMQ

* Redis: Stores transient AI results and session state
* RabbitMQ: Handles async task distribution for inference and processing jobs

## 4. 🔄 Data Flow Example

1. 📤 User uploads DICOM study via API Gateway
2. 🗃️ API Gateway registers study in PostgreSQL and Orthanc
3. 📬 Segmentation task sent via RabbitMQ to Inference Service
4. 🧠 Inference returns mask → stored in NIfTI/Orthanc → metadata pushed to Redis
5. 📑 Reporting service fetches metadata and auto-generates clinical report
6. 📥 PDF report returned to frontend via API Gateway

## 5. ⚙️ Deployment and Configuration

### 🧪 Environment Setup

* Each service runs in its own Docker container
* Environment variables for ports, secrets, and endpoints are defined in `.env` files
* Docker Compose ensures service orchestration and volume sharing

### 🔐 Security

* Role-based access via JWT tokens
* Secure endpoints with HTTPS (via reverse proxy or gateway)
* API keys for inter-service communication
* HIPAA-aligned architecture with encryption for sensitive data

### 📈 Monitoring

* Healthcheck endpoints in FastAPI/Django services
* Container logs centralized via Docker logging drivers
* Planned integration with Prometheus + Grafana

## 6. 🧪 Development Notes

* Code follows MVC/service-based separation
* APIs are RESTful with OpenAPI documentation (FastAPI)
* Frontend dynamically queries volume and segmentation data

## 7. 🚀 Future Enhancements

* Move to Kubernetes for scalable deployments
* Add audit logs and real-time admin dashboard
* Git-based CI/CD planned for automatic builds
