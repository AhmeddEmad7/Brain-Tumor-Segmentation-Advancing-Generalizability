# Brain Tumor Segmentation Platform - Technical Documentation

## 1. Project Overview

### Purpose

An advanced medical imaging platform specifically designed for brain tumor analysis, featuring:

* 3D DICOM/NIfTI visualization
* AI-powered multi-class segmentation
* Automated analysis and reporting
* Real-time collaborative viewing

### Technology Stack

* **Frontend**: React 18, TypeScript, Vite
* **Medical Imaging**: Cornerstone3D, VTK.js
* **AI Integration**: RabbitMQ, Redis
* **DICOM Services**: Orthanc PACS Dicom Web
* **Containerization**: Docker, Microservices
* **Database**: PostgreSQL
* **API Gateway**: Django
*  **Inference** : Django
* **Reporting** : FastAPI
* **Nifti Storage** : FastAPI


## 2. Medical Image Import and Compatibility

### Supported Formats

* **DICOM**

  * Single/Multi-frame images
  * Enhanced DICOM
  * DICOM-SEG
  
* **NIfTI**

  * `.nii` and `.nii.gz`
  * Custom NIfTI storage service
* **Additional Formats**

  * JPEG/PNG (for screenshots/exports)

### Processing Pipeline

* Automated DICOM metadata extraction
* NIfTI header parsing


## 3. Viewing Capabilities

### 2D Viewing

* Multi-planar slice navigation
* Window/level presets
* Synchronized viewing
* Custom overlay support

### MPR Features

* Real-time MPR reconstruction
* Orthogonal and oblique views
* Synchronized crosshair navigation
* Custom plane orientation

### Volume Rendering

* 3D volume visualization
* Custom transfer functions
* GPU-accelerated rendering

## 4. Segmentation Features

### Manual Tools

* Brush tool with size adjustment
* Polygon/Rectangle ROI
* 3D spherical brush
* Smart CT/MR segmentation

### AI-Powered Segmentation

* Automated tumor detection
* Multi-class segmentation
* Real-time inference

### Data Management

* DICOM-SEG export/import
* NIfTI mask support
* JSON metadata
* Statistics calculation

## 5. AI Integrations

### Segmentation Service

* Deep learning models for tumor detection
* Multi-class classification
* Real-time inference via RabbitMQ
* Model versioning and management

### Motion Correction

* Automated artifact detection
* Real-time correction
* Quality assessment

## 6. Image Navigation

### Interaction Tools

* Mouse-based navigation
* Keyboard shortcuts
* Custom tool bindings

### Performance Features
* Memory management
* Load balancing

## 7. Measurement Tools

### Available Tools

* Length measurement
* Area calculation
* Volume estimation
* Angle measurement
* Statistical analysis

### Data Management

* Export capabilities
* Report integration

## 8. PACS Integration

### DICOM Web Services

* WADO-RS/QIDO-RS/STOW-RS
* C-STORE/C-FIND/C-MOVE
* Custom Orthanc plugins
* Offline caching

### Security Features

* DICOM anonymization
* Access control
* Data encryption

## 9. 3D Reconstruction

### Volume Rendering

* Custom transfer functions
* Real-time updates
* Surface rendering

## 10. Reporting

### Features

* Structured reporting templates
* AI-assisted report generation
* Measurement integration
* Image screenshots
* AI findings inclusion

### Export Formats

* PDF reports
* Custom templates

## 11. Authentication & Access Control

### Security Features

* Role-based access
* Session management

## 12. Additional Features

### System Integration

* Microservices architecture
* Redis caching
* Message queuing
* API gateway

### UI Components

* Customizable layouts
* Tool panels
* Study browser
* Real-time notifications

## 13. Environment and Configuration

### Infrastructure

* Docker Compose setup
* Microservices orchestration
* Database management
* Cache configuration

### Security

* API key management
* SSL/TLS configuration
* HIPAA compliance
* Data encryption

### Deployment

* Environment variables
* Service discovery
* Load balancing
* Monitoring setup
