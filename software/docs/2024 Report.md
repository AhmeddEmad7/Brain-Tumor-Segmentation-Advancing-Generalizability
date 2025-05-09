# MMM.AI Software Team 2024 Year Report

## Feature List:

### ✅ Completed

* **Medical Image Import and Compatibility**

  * DICOM Uploads
  * NIfTI Uploads
  * JPEG/PNG Uploads (Screenshot & Export Support)

* **Viewing**

  * DICOM Viewing (2D and 3D)
  * NIfTI Viewer Support
  * MPR Support (Orthogonal + Oblique)

* **Segmentation**

  * Manual Segmentation Tools (Brush, ROI, 3D Tools)
  * Multi-class AI Segmentation (Real-time)
  * Upload/Download DICOM-SEG & NIfTI Masks

* **Image Navigation**

  * Zoom
  * Pan
  * Window Level Presets and Adjustments
  * Synchronized Viewports

* **Measurements**

  * Advanced Annotation Tools: Length, Area, Angle, Arrows, Bidirectional
  * Save and Upload Measurements
  * Integrated into Reporting

* **PACS Integration**

  * Full Orthanc PACS Integration
  * DICOM Web Support (WADO/QIDO/STOW)
  * Offline Image Caching

* **3D Visualization and Reconstruction**

  * GPU-accelerated Volume Rendering
  * Transfer Function Support
  * Surface Rendering using VTK.js

* **Reporting**

  * Structured Report Builder
  * Screenshot Insertion
  * Measurement Summary Inclusion
  * **LLM-based Report Generation**

    * Automated Findings, Impression, Diagnosis, and Recommendation Sections
    * Integration with AI models for clinical report assistance
  * Snapshot Capture for Report Inclusion
  * Editable Report Interface (Real-time)
  ### Export Formats
  
  * PDF reports
  * Custom templates

* **NIfTI Storage Service**

    * Dedicated BIDS-compliant NIfTI archival and retrieval system
    * REST API support for file serving

* **NIfTI Processing Service**
    * Supports normalization, metadata parsing, and conversion workflows
    * Integrated with AI and viewer pipelines


* **Authentication & Security**

  * JWT Login System
  * Role-based Access Control (RBAC)
  * Audit Logging System

* **Platform Infrastructure**

  * Docker-based Microservices
  * RabbitMQ Message Queues
  * Redis Caching Layer
  * PostgreSQL Data Store


### 🔄 Future Roadmap

* Segmentation Editing and Versioning
* Collaborative Viewing (Multi-user Sync)
* AI Quality Assessment Tools (Artifact Detection)
* Synthesis Page (Patient Summary + AI Findings)
* User Preferences and Layout Saving
* Add Support for DICOM RT and PET SUV Tools
* Deploy on Cloud with Auto-Scaling (Kubernetes)
* Open API for External App Integration
* Monitoring & Health Dashboards

## ⚠️ Challenges Faced

* Handling segmentation consistency across formats
* Dependency upgrades (Cornerstone3D, VTK.js, Orthanc)
* AI model performance tuning and optimization
* Contributing to a large modular system and understanding the codebase before making progress
* Lack of documentation across open-source imaging and AI libraries

