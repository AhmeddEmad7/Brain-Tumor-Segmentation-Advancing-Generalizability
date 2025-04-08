import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def interpret_anatomical_location(centroid, mid_x, mid_y_anterior, mid_y_posterior, mid_z_inferior, mid_z_superior):
    x, y, z = centroid # LPS

    hemisphere = "left" if x > mid_x else "right"

    # Determine anetrior and posterior "region"
    if y < mid_y_anterior:
        region = "anterior"
    elif y < mid_y_posterior:
        region = "central anterior-posterior"
    else:
        region = "posterior"

    # Determine superior and inferior "height"
    if z > mid_z_superior:
        height = "superior"
    elif z > mid_z_inferior:
        height = "central superior-inferior"
    else:
        height = "inferior"

    # Determine lobe based on combined region + height
    if region == "anterior":
        if height in ("superior", "central superior-inferior"):
            lobe = "frontal lobe"
        else:
            lobe = "temporal lobe"
            
    elif region == "posterior":
        if height in ("superior", "central superior-inferior"):
            lobe = "parietal lobe"
        else:
            lobe = "occipital lobe"
            
    else:  # region == "central anterior-posterior"
        if height == "inferior":
            lobe = "temporal lobe"
        else: #
            lobe = "frontal lobe"

    if "central" in (region, height):
        central_desc = []
        if region == "central anterior-posterior":
            central_desc.append("central anterior-posterior")
        else:
            central_desc.append(region)

        if height == "central superior-inferior":
            central_desc.append("central superior-inferior")
        else:
            central_desc.append(height)

        joined_desc = " and ".join(central_desc)
        description = f"{joined_desc} part of the brain, most likely centered in the {hemisphere} {lobe}"
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
    
    brain_coords = np.argwhere(brain_mask)
    x_min, y_min, z_min = np.min(brain_coords, axis=0)
    x_max, y_max, z_max = np.max(brain_coords, axis=0) + 1
    
    print(f"Brain bounding box:")
    print(f"  x: {x_min} to {x_max}")
    print(f"  y: {y_min} to {y_max}")
    print(f"  z: {z_min} to {z_max}")
    
    # Define midlines based on actual brain bounds
    mid_x = int((x_min + x_max) / 2)
    mid_y_anterior = int(y_min + (y_max - y_min) * 0.45)
    mid_y_posterior = int(y_min + (y_max - y_min) * 0.55)
    mid_z_inferior = int(z_min + (z_max - z_min) * 0.45)
    mid_z_superior = int(z_min + (z_max - z_min) * 0.55)
    
    print(f"Calculated midlines:")
    print(f"  Mid X (left-right): {mid_x:.2f}")
    print(f"  Mid Y anterior: {mid_y_anterior:.2f}")
    print(f"  Mid Y posterior: {mid_y_posterior:.2f}")
    print(f"  Mid Z inferior: {mid_z_inferior:.2f}")
    print(f"  Mid Z superior: {mid_z_superior:.2f}")

    anatomical_location, hemisphere, lobe = interpret_anatomical_location(
                                                centroid, mid_x, mid_y_anterior,
                                                mid_y_posterior, mid_z_inferior,
                                                mid_z_superior)

    
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