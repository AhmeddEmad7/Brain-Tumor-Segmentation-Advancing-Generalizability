import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from nibabel.orientations import aff2axcodes

def interpret_anatomical_location(centroid, mid_x, mid_y_anterior, mid_y_posterior, mid_z_inferior, mid_z_superior, orientation):
    x, y, z = centroid

    # Hemisphere determination (based on x-axis)
    if orientation[0] == "L":  # R→L
        hemisphere = "left" if x > mid_x else "right"
    else:  # L→R
        hemisphere = "right" if x > mid_x else "left"

    # Anterior/posterior region (based on y-axis)
    if orientation[1] == "P":  # A→P
        if y < mid_y_anterior:
            region = "anterior"
        elif y < mid_y_posterior:
            region = "central anterior-posterior"
        else:
            region = "posterior"
    else:  # P→A
        if y > mid_y_anterior:
            region = "anterior"
        elif y > mid_y_posterior:
            region = "central anterior-posterior"
        else:
            region = "posterior"

    # Inferior/superior height (based on z-axis)
    if orientation[2] == "S":  # I→S 
        if z > mid_z_superior:
            height = "superior"
        elif z > mid_z_inferior:
            height = "central superior-inferior"
        else:
            height = "inferior"
    else:  # S→I
        if z < mid_z_superior:
            height = "superior"
        elif z < mid_z_inferior:
            height = "central superior-inferior"
        else:
            height = "inferior"

    # Determine lobe based on region and height
    if region == "anterior":
        lobe = "frontal lobe" if height in ("superior", "central superior-inferior") else "temporal lobe"
    elif region == "posterior":
        lobe = "parietal lobe" if height in ("superior", "central superior-inferior") else "occipital lobe"
    else: # central anterior-posterior
        lobe = "temporal lobe" if height == "inferior" else "parietal lobe"

    if "central" in region and "central" in height:
        return f"An abnormal mass is identified centered along both the anteroposterior and craniocaudal axes, most likely centered in the {hemisphere} {lobe}.\n"
    elif "central" in region:
        return f"An abnormal mass is identified in the {height} part of the brain, centered along the anteroposterior axis, and most likely located in the {hemisphere} {lobe}.\n"
    elif "central" in height:
        return f"An abnormal mass is identified in the {region} part of the brain, centered along the craniocaudal axis, and most likely located in the {hemisphere} {lobe}.\n"
    else:
        return f"An abnormal mass is identified in the {region} and {height} part of the brain, most likely centered in the {hemisphere} {lobe}.\n"

def extract_tumor_features(brain_vol, segmentation_mask, mask_channels, metadata, voxel_size=(1,1,1)):
    voxel_volume = np.prod(voxel_size)
    orientation = aff2axcodes(metadata['affine'])

    # Mask out background (only consider voxels within brain region)
    brain_mask = brain_vol > 0
    total_voxels = np.sum(brain_mask)
    
    whole_tumor_vol = np.sum(mask_channels[1]) * voxel_volume / 1000  # Convert to cc
    tumor_core_vol = np.sum(mask_channels[2]) * voxel_volume / 1000
    enhancing_vol = np.sum(mask_channels[3]) * voxel_volume / 1000
    
    unique, counts = np.unique(segmentation_mask, return_counts=True)
    class_counts = dict(zip(unique, counts))
    necrosis_vol = (class_counts.get(1, 0) * voxel_volume) / 1000
    edema_vol = (class_counts.get(2, 0) * voxel_volume) / 1000
    
    whole_tumor_percentage = (whole_tumor_vol * 1000 / total_voxels) * 100
    necrosis_percentage = (necrosis_vol / whole_tumor_vol) * 100 if whole_tumor_vol > 0 else 0
    edema_percentage = (edema_vol / whole_tumor_vol) * 100 if whole_tumor_vol > 0 else 0
    enhancing_percentage = (enhancing_vol / whole_tumor_vol) * 100 if whole_tumor_vol > 0 else 0
    
    coords = np.argwhere(segmentation_mask > 0)
    centroid = tuple(np.mean(coords, axis=0).astype(int)) if len(coords) > 0 else (0, 0, 0)
    print("Tumor Centroid:", centroid)
    
    brain_coords = np.argwhere(brain_mask)
    x_min, y_min, z_min = np.min(brain_coords, axis=0)
    x_max, y_max, z_max = np.max(brain_coords, axis=0) + 1
    
    print(f"Brain bounding box:")
    print(f"  x: {x_min} to {x_max}")
    print(f"  y: {y_min} to {y_max}")
    print(f"  z: {z_min} to {z_max}")
    
    # Define midlines based on actual brain bounds
    mid_x = int((x_min + x_max) / 2)

    if orientation[1] == "P":  # A→P
        mid_y_anterior = int(y_min + (y_max - y_min) * 0.45)
        mid_y_posterior = int(y_min + (y_max - y_min) * 0.55)
    else:  # P→A
        mid_y_posterior = int(y_min + (y_max - y_min) * 0.45)
        mid_y_anterior = int(y_min + (y_max - y_min) * 0.55)

    if orientation[2] == "S":  # I→S
        mid_z_inferior = int(z_min + (z_max - z_min) * 0.45)
        mid_z_superior = int(z_min + (z_max - z_min) * 0.55)
    else:  # S→I
        mid_z_superior = int(z_min + (z_max - z_min) * 0.45)
        mid_z_inferior = int(z_min + (z_max - z_min) * 0.55)
    
    print(f"Calculated midlines!")
    findings = interpret_anatomical_location(
                                centroid, mid_x, mid_y_anterior, mid_y_posterior,
                                mid_z_inferior, mid_z_superior, orientation)
    
    return {
        "whole_tumor_volume": whole_tumor_vol,
        "tumor_core_volume": tumor_core_vol,
        "enhancing_tumor_volume": enhancing_vol,
        "necrosis_volume": necrosis_vol,
        "edema_volume": edema_vol,
        "whole_tumor_percentage": whole_tumor_percentage,
        "necrosis_percentage": necrosis_percentage,
        "edema_percentage": edema_percentage,
        "enhancing_percentage": enhancing_percentage,
        "tumor_centroid": centroid,
        "findings": findings,
    }

def generate_report(data, llm_data):
    return (
        f"Findings:\n"
        f"{data['findings']}\n"
        
        f"Composition analysis:\n"
        f"- Enhancing component: {data['enhancing_percentage']:.2f}% of the whole lesion.\n"
        f"- Necrotic component: {data['necrosis_percentage']:.2f}% of the whole lesion.\n"
        f"- Edema component: {data['edema_percentage']:.2f}% of the whole lesion.\n"
        f"- The total lesion represents {data['whole_tumor_percentage']:.2f}% of the total brain volume.\n\n"
        
        f"Quantitative analysis:\n"
        f"- Tumor core volume (including necrotic and enhancing components): {data['tumor_core_volume']:.2f} cc\n"
        f"- Enhancing tumor volume: {data['enhancing_tumor_volume']:.2f} cc\n"
        f"- Necrotic tumor volume: {data['necrosis_volume']:.2f} cc\n"
        f"- Edema volume: {data['edema_volume']:.2f} cc\n"
        f"- Total lesion volume: {data['whole_tumor_volume']:.2f} cc\n\n"
        
        f"Impression*:\n"
        f"{llm_data['impression']}\n\n"
        
        f"Likely Diagnosis*:\n"
        f"{llm_data['diagnosis']}\n\n"
        
        f"Recommendations*:\n"
        f"{llm_data['recommendations']}\n\n\n"
        
        f"* These sections are preliminary outputs based on automated interpretation of the findings and analysis above, and "
        f"should be thoroughly reviewed and confirmed by a qualified radiologist prior to clinical application."
        )

def generate_pdf(findings, patient, filename="radiology_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles["Heading1"],
        fontSize=24,
        alignment=1,  # 0 = left, 1 = center, 2 = right
        spaceAfter=20
    )

    heading3_style = styles["Normal"]
    body_style = styles["Normal"]
    footnote_style = ParagraphStyle(
        name="FootnoteStyle",
        parent=styles["Normal"],
        fontSize=styles["Normal"].fontSize - 2
    )

    story = []

    story.append(Paragraph("Radiology Report", title_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"<b>Patient:</b> <u>{patient}</u>", heading3_style))
    story.append(Spacer(1, 0.25 * inch))

    for line in findings.split('\n'):
        if line.strip() == "":
            story.append(Spacer(1, 0.2 * inch))
        else:
            line_stripped = line.strip()
            is_header = any(line_stripped.startswith(header) for header in [
                "Findings:", "Composition analysis:", "Quantitative analysis:",
                "Impression*:", "Likely Diagnosis*:", "Recommendations*:"
            ])
            is_footnote = line_stripped.startswith("* These")

            if is_header:
                line = f"<b>{line}</b>"
                story.append(Paragraph(line, body_style))
            elif is_footnote:
                line = f"<b><i>{line}</i></b>"
                story.append(Paragraph(line, footnote_style))
            else:
                story.append(Paragraph(line, body_style))

    doc.build(story)