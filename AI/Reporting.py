import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def interpret_anatomical_location(centroid):
    x, y, z = centroid  # LPS: x=left (0-240mm), y=posterior (0-240mm), z=superior (0-155mm)
    
    hemisphere = "left" if x > 120 else "right"  # Midline at x=120
    
    superior_z = 90  # Above = superior (frontal/parietal)
    inferior_z = 60  # Below = inferior (temporal/occipital)
    
    anterior_y = 90   # Below = anterior (frontal/temporal)
    posterior_y = 160  # Above = posterior (parietal/occipital)
    
    if y < anterior_y:
        region = "anterior"
    elif y < posterior_y:
        region = "middle"
    else:
        region = "posterior"
    
    if z > superior_z:
        height = "superior"
    elif z > inferior_z:
        height = "middle"
    else:
        height = "inferior"
    
    if region == "anterior":
        if height == "superior":
            lobe = "frontal lobe"
        else:
            lobe = "temporal lobe"  # Anterior-inferior = temporal
    elif region == "posterior":
        if height == "superior":
            lobe = "parietal lobe"
        else:
            lobe = "occipital lobe"
    else:  # Middle region
        if height == "superior":
            lobe = "frontal lobe"
        else:
            lobe = "temporal lobe"  # Middle-inferior = temporal
    
    # Insular lobe check (central deep region)
    # if (100 <= x <= 140) and (80 <= y <= 160) and (60 <= z <= 95):
    #     lobe = "insular lobe"

    if (height == region):
        description = f"{region} part of the brain, most likely centered in the {hemisphere} {lobe}"
    else:
        description = f"{region} and {height} part of the brain, most likely centered in the {hemisphere} {lobe}"
    
    return description, hemisphere, lobe

def extract_tumor_features(brain_vol, segmentation_mask, mask_channels, voxel_size=(1,1,1)):
    voxel_volume = np.prod(voxel_size)

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

    anatomical_location, hemisphere, lobe = interpret_anatomical_location(centroid)
    
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
        "anatomical_location": anatomical_location,
        "hemisphere": hemisphere,
        "lobe": lobe
    }

def generate_report(data):
    return (
        f"Findings:\n"
        f"An abnormal mass is identified in the {data['anatomical_location']}. \n\n"
        f"Composition analysis:\n"
        f"- Enhancing component: {data['enhancing_percentage']:.2f}% of the whole lesion.\n"
        f"- Necrotic component: {data['necrosis_percentage']:.2f}% of the whole lesion.\n"
        f"- Edema component: {data['edema_percentage']:.2f}% of the whole lesion.\n"
        f"- The total lesion represents {data['whole_tumor_percentage']:.2f}% of the total brain volume.\n\n"
        f"Quantitative analysis of the tumor segmentation reveals the following:\n"
        f"- Tumor core volume (including necrotic and enhancing components): {data['tumor_core_volume']:.2f} cc\n"
        f"- Enhancing tumor volume: {data['enhancing_tumor_volume']:.2f} cc\n"
        f"- Necrotic tumor volume: {data['necrosis_volume']:.2f} cc\n"
        f"- Edema volume: {data['edema_volume']:.2f} cc\n"
        f"- Total lesion volume: {data['whole_tumor_volume']:.2f} cc\n\n"
        f"Impression:\n"
        f"Findings are consistent with a mass lesion in the {data['hemisphere']} {data['lobe']}. Clinical correlation and further evaluation, including possible biopsy and follow-up imaging, are recommended as per clinical context."
    )

def generate_pdf(findings, patient, filename="radiology_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    story = []

    title_style = styles["Heading1"]
    heading3_style = styles["Heading3"]
    story.append(Paragraph("Radiology Report", title_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f"Patient: {patient}", heading3_style))
    story.append(Spacer(1, 0.25 * inch))

    body_style = styles["Normal"]
    for line in findings.split('\n'):
        if line.strip() == "":
            story.append(Spacer(1, 0.2 * inch))
        else:
            story.append(Paragraph(line, body_style))

    doc.build(story)