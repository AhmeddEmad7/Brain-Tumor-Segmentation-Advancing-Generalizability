from fastapi import APIRouter, Depends, Path, HTTPException
from sqlalchemy.orm import Session
from app.config import SessionLocal
import app.crud
from app.schemas import RequestReport, Response ,ReportResponse, PDFRequest
from reportlab.pdfgen import canvas
import base64
from io import BytesIO
from reportlab.lib.utils import ImageReader
from textwrap import wrap
import os 
import redis
import json
from datetime import date
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_CENTER, TA_LEFT

router = APIRouter()
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_db = int(os.getenv('REDIS_DB', 0))
PDF_OUTPUT_DIR = os.getenv('PDF_OUTPUT_DIR', 'pdfs')
os.makedirs(PDF_OUTPUT_DIR, exist_ok=True)
client_redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/")
async def create_report(report: RequestReport, db: Session = Depends(get_db)):
    _report = app.crud.create_report(db, report)
    if not _report:
        return Response(
            code=404, status="Fail", message="Report not created", result=None
        ).model_dump(exclude_none=True)

    return Response(
        code=200,
        status="Success",
        message="Report created successfully",
        result=_report,
    ).model_dump(exclude_none=True)

@router.post("/generate-pdf")
async def generate_pdf(req: PDFRequest, db: Session = Depends(get_db)):
    try:
        blocks = json.loads(req.content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid report content JSON")

    os.makedirs("pdfs", exist_ok=True)
    timestamp = date.today().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{req.studyId}_{timestamp}.pdf"
    out_path = os.path.join(PDF_OUTPUT_DIR, filename)

    c = canvas.Canvas(out_path)
    y = 800
    page_width = 595  # A4 width in points
    margin = 50
    content_width = page_width - (2 * margin)

    # Create styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Preliminary',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name='Header',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceBefore= 200,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name='Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        spaceAfter=20
    ))
    styles.add(ParagraphStyle(
        name='BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        spaceAfter=20
    ))

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Patient {req.header.patientId} – {req.header.modality} Report")
    y -= 30
    c.setFont("Helvetica", 12)
    for label, val in [
        ("Name", req.header.patientName),
        ("Study Date", req.header.studyDate),
    ]:
        c.drawString(margin, y, f"{label}: {val}")
        y -= 20
    y -= 30

    # Body
    preliminary_notice_found = False
    for block in blocks:
        block_type = block.get("type")
        text = ""

        # Handle images
        if block_type == "images" or (block_type == "p" and any(child.get("type") == "img" for child in block.get("children", []))):
            images = block.get("data") or [child.get("url") for child in block.get("children", []) if child.get("type") == "img"]

            image_width = 120
            image_height = 170
            padding = 10
            x = margin
            max_x = page_width - image_width - margin

            for img_b64 in images:
                if img_b64.startswith("data:"):
                    img_b64 = img_b64.split(",", 1)[1]
                try:
                    img_bytes = base64.b64decode(img_b64)
                    img = ImageReader(BytesIO(img_bytes))

                    if x > max_x:
                        x = margin
                        y -= (image_height + padding)

                    if y - image_height < margin:
                        c.showPage()
                        y = 800

                    c.drawImage(img, x, y - image_height, width=image_width, height=image_height)
                    x += image_width + padding

                except Exception as e:
                    print("Image decode failed:", e)

            y -= (image_height + 20)
            continue

        # Headings
        if block_type == "h2":
            text = " ".join(child.get("text", "") for child in block.get("children", []))
            p = Paragraph(text, styles['Header'])
            w, h = p.wrap(content_width, y)
            if y - h < margin:
                c.showPage()
                y = 800
            p.drawOn(c, margin, y - h)
            y -= h + 10
            continue

        # Preliminary Notice (h3)
        if block_type == "h3":
            # Get the text from children
            text = " ".join(child.get("text", "") for child in block.get("children", []))
            
            # Position at the start of image section with top margin
            y = 800 - image_height - 40  # Added more top margin
            
            # Create a style for the preliminary notice
            preliminary_style = ParagraphStyle(
                'Preliminary',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=10,
                spaceAfter=10
            )
            
            # Create paragraph with width limit
            p = Paragraph(text, preliminary_style)
            w, h = p.wrap(content_width, y)
            if y - h < margin:
                c.showPage()
                y = 800
            p.drawOn(c, margin, y - h)
            y -= h + 10  # Add space after the notice
            
            preliminary_notice_found = True
            continue

        # Paragraphs
        if block_type == "p":
            text = " ".join(child.get("text", "") for child in block.get("children", []))
            style = styles['BodyBold'] if preliminary_notice_found else styles['Body']
            p = Paragraph(text, style)
            w, h = p.wrap(content_width, y)
            if y - h < margin:
                c.showPage()
                y = 800
            p.drawOn(c, margin, y - h)
            y -= h + 5

        if y < margin:
            c.showPage()
            y = 800

    c.save()

    return Response(
        code=200,
        status="Success",
        message="PDF saved successfully",
        result={"filename": filename, "path": out_path}
    ).model_dump(exclude_none=True)
    
@router.get("/redis/{study_id}")
async def get_report_from_redis(study_id: str):
    key = f"report_{study_id}"
    report = client_redis.get(key)
    print(f"Redis key: {key}")
    print(f"Redis report: {report}")
    print('report.decode()',report.decode())
    if report:
        return {"content": report.decode()}
    else:
        raise HTTPException(status_code=404, detail="Report not found in Redis")
    
@router.get("/")
async def get_reports(db: Session = Depends(get_db)):
    _reports = app.crud.get_reports(db)
    if not _reports:
        return Response(
            code=200, status="Success", message="No reports found", result=None
        ).model_dump(exclude_none=True)

    return Response(
        code=200,
        status="Success",
        message="Reports retrieved successfully",
        result=_reports,
    ).model_dump(exclude_none=True)


@router.get("/{study_id}")
async def get_reports(
    study_id: str,
    db: Session = Depends(get_db),
):
    _reports = app.crud.get_reports_by_study_id(db, study_id)
    if not _reports:
        return Response(
            code=404,
            status="Fail",
            message="No report found with this ID",
            result=None,
        ).model_dump(exclude_none=True)

    return Response(
        code=200,
        status="Success",
        message="Report retrieved successfully",
        result=_reports,
    ).model_dump(exclude_none=True)


@router.put("/{report_id}")
async def update_report(
    report: RequestReport,
    report_id: int = Path(..., title="The ID of the report to update"),
    db: Session = Depends(get_db),
):
    _report = app.crud.update_report(db, report_id, report)
    if not _report:
        return Response(
            code=200,
            status="Success",
            message="No report found with this ID",
            result=None,
        ).model_dump(exclude_none=True)

    return Response(
        code=200,
        status="Success",
        message="Report updated successfully",
        result=_report,
    ).model_dump(exclude_none=True)


@router.delete("/{report_id}/study/{study_id}")
async def delete_report(
    report_id: int = Path(..., title="The ID of the report to delete"),
    study_id: str = Path(..., title="The ID of the study to which the report belongs"),
    db: Session = Depends(get_db),
):

    if not app.crud.delete_report(db, report_id, study_id):
        return Response(
            code=200,
            status="Success",
            message="No report found with this ID",
            result=None,
        ).model_dump(exclude_none=True)

    return Response(
        code=200, status="Success", message="Report deleted successfully", result=None
    ).model_dump(exclude_none=True)


@router.delete("/")
async def delete_reports(db: Session = Depends(get_db)):

    if not app.crud.delete_reports(db):
        return Response(
            code=200, status="Success", message="No reports found", result=None
        ).model_dump(exclude_none=True)

    return Response(
        code=200,
        status="Success",
        message="All reports deleted successfully",
        result=None,
    ).model_dump(exclude_none=True)
