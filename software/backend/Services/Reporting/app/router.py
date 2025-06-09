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

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Patient {req.header.patientId} – {req.header.modality} Report")
    y -= 20
    c.setFont("Helvetica", 12)
    for label, val in [
        ("Name", req.header.patientName),
        # ("Birthdate", req.header.patientBirthDate.isoformat()),
        ("Study Date", req.header.studyDate),
    ]:
        c.drawString(50, y, f"{label}: {val}")
        y -= 15
    y -= 20

    # Body
    for block in blocks:
        block_type = block.get("type")
        text = ""

        # 👇 Handle images inside either `type: images` or `p > img`
        if block_type == "images" or (block_type == "p" and any(child.get("type") == "img" for child in block.get("children", []))):
            images = block.get("data") or [child.get("url") for child in block.get("children", []) if child.get("type") == "img"]

            image_width = 120
            image_height = 170
            padding = 10
            x = 50
            max_x = page_width - image_width - 50

            for img_b64 in images:
                if img_b64.startswith("data:"):
                    img_b64 = img_b64.split(",", 1)[1]
                try:
                    img_bytes = base64.b64decode(img_b64)
                    img = ImageReader(BytesIO(img_bytes))

                    if x > max_x:
                        x = 50
                        y -= (image_height + padding)

                    if y - image_height < 50:
                        c.showPage()
                        y = 800

                    c.drawImage(img, x, y - image_height, width=image_width, height=image_height)
                    x += image_width + padding

                except Exception as e:
                    print("Image decode failed:", e)

            y -= (image_height + 15)
            continue

        # 👇 Headings
        if block_type == "h2":
            text = " ".join(child.get("text", "") for child in block.get("children", []))
            wrapped_lines = wrap(text, width=55)  # Adjust width for heading lines

            c.setFont("Helvetica-Bold", 13)
            for line in wrapped_lines:
                
                if y < 60:
                    c.showPage()
                    y = 800
                    c.setFont("Helvetica-Bold", 13)  # Reset font after page break

                c.drawString(50, y, line)
                y -= 18

            y -= 5  # spacing after heading
            continue

        # 👇 Paragraphs with word wrapping
        if block_type == "p":
            text = " ".join(child.get("text", "") for child in block.get("children", []))
            c.setFont("Helvetica", 11)
            max_line_width = 95  # max characters before wrapping

            for line in text.split("\n"):
                wrapped_lines = wrap(line.strip(), max_line_width)
                for wrapped in wrapped_lines:
                    c.drawString(50, y, wrapped)
                    y -= 15

            y -= 5

        if y < 50:
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
