import pathlib

from dotenv import load_dotenv
import os
import dicomweb_client
import subprocess
import json
import SimpleITK
import subprocess
import numpy as np
import pydicom_seg
import time
import nibabel as nib
from pydicom.filereader import dcmread
from scipy.ndimage import zoom
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SegmentationStorage
from src.colors import GENERIC_ANATOMY_COLORS
from .Teacher_Inference.Inference import inference
from src.colored_dicom import convert_array_to_dicom_seg
import redis

load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
studies_dir = os.path.join(os.path.dirname(current_dir), 'studies')

client = dicomweb_client.api.DICOMwebClient(f"{os.getenv('ORTHANC_URL')}/dicom-web")
client_redis = redis.Redis(host="localhost", port=6379, db=0)

if not os.path.exists(studies_dir):
    print(f"Creating directory {studies_dir}")
    os.mkdir(studies_dir)

color_dict = {
    0:(0,0,0), ## background
    1:(255,0,0), ##  
    2:(0,255,0), ## green Edema
    3:(0,0,255) ## blue Enhancing Tumor
}

label_info = [
    {"name": "Label1", "necrosis": "Description of Label1", "color": (255, 0, 0)},
    {"name": "Label2", "edima": "Description of Label2", "color": (0, 255, 0)},
    {"name": "Label3", "description": "Description of Label3", "color": (0, 0, 255)},
    # Add more labels as needed
]


def map_color_to_label(mask, color_dict=color_dict) -> np.ndarray:
    # Create an empty array for the colored overlay with the same shape as the MRI but with an additional dimension for color channels
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    
    for label, color in color_dict.items():
        if label == 0:
            continue  # Skip background
        color = np.array(color)  # Keep color values in [0, 255] range
        # color = np.array(color)/255  # Keep color values in [0, 255] range
        for i in range(mask.shape[0]):  # Process each slice
            indices = (mask[i] == label)
            colored_mask[i][indices] = color
    
    return colored_mask


def overlay_mask(mri, colored_mask, alpha=0.5) -> np.ndarray:
    if mri.ndim == 2:
        # If MRI is 2D, stack it to create an RGB image
        mri_rgb = np.stack([mri]*3, axis=-1)
    elif mri.ndim == 3:
        # If MRI is 3D, add a new axis for RGB channels
        mri_rgb = np.repeat(mri[..., np.newaxis], 3, axis=-1)

    # Normalize MRI to be in the range of 0-255 if necessary
    if np.max(mri_rgb) > 255:
        mri_rgb = (mri_rgb / np.max(mri_rgb)) * 200

    # Convert MRI to uint8
    mri_rgb = mri_rgb.astype(np.uint8)

    # Only blend where the mask has values (non-background)
    mask_indices = np.any(colored_mask != 0, axis=-1)
    blended_image = mri_rgb.copy()

    # Apply blending only to the areas covered by the mask
    blended_image[mask_indices] = (
        (0) * mri_rgb[mask_indices] + alpha * colored_mask[mask_indices]
    ).astype(np.uint8)

    return blended_image


def save_array_nifti(output_array,output_path,i):

    affine = np.eye(4)
    output_nifti = nib.Nifti1Image(output_array, affine=affine)
    output_path = os.path.join(os.path.dirname(output_path), f'mask{i}.nii.gz')
    nib.save(output_nifti, output_path)
    print(f"Saved NIfTI Mask at: {output_path}")
    return output_path
    
    
## me7taga 3omda fe tzbet al axis 
def resize_nifti_to_array(input_file, target_shape, interpolation_order=1) -> np.ndarray:
    # Load NIfTI file
    img = nib.load(input_file)
    data = img.get_fdata()
    # print("data of nifti load", data)
    original_shape = data.shape
    
    # Calculate zoom factors for each dimension
    zoom_factors = [t / o for t, o in zip(target_shape, original_shape)]
    
    # Perform resizing with interpolation
    resized_data = zoom(data, zoom_factors, order=interpolation_order)
    
    return resized_data 


def get_dicom_series(study_uid, series_uid, tag):
    print(f"Retrieving series {series_uid} from study {study_uid} as {tag}...")

    instances = client.retrieve_series(study_instance_uid=study_uid, series_instance_uid=series_uid)
    
    instance_metadata = client.retrieve_instance_metadata(study_instance_uid=study_uid, series_instance_uid=series_uid, sop_instance_uid=instances[0].SOPInstanceUID)
    
    if instance_metadata:
                # Assuming the metadata is returned as a list and the first item contains the desired data
                # global_metadata = instance_metadata
                client_redis.set(f"metadata/{series_uid}", json.dumps(instance_metadata),ex=1800) #30 min
                # print("on redis successfully", instance_metadata)
        # print("Metadata updated in global variable:", instance_metadata)
    
    print(f"Retrieved {len(instances)} instances")

    # save the dicom files in a directory
    current_study_path = os.path.join(studies_dir, study_uid, tag)

    if not os.path.exists(current_study_path):
        print(f"Creating directory {current_study_path}")
        os.makedirs(current_study_path)

    print(f"Saving DICOM files to {current_study_path}...")
    for instance in instances:
        # Define the output file path
        output_path = os.path.join(current_study_path, f"{instance.SOPInstanceUID}.dcm")

        # Save the DICOM file
        instance.save_as(output_path)

    # Convert the DICOM files to NIfTI
    print(f"Converting DICOM files to NIfTI")

    dicom_to_nifti(current_study_path, current_study_path, tag)

    print(f"Series {series_uid} from study {study_uid} retrieved and converted to NIfTI")

    return os.path.join(current_study_path, f"{tag}.nii.gz")


def dicom_to_nifti(dicom_dir, output_dir, file_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Call dcm2niix
    subprocess.run(["dcm2niix", "-z", "y", "-f", f"{file_name}", "-o", output_dir, dicom_dir], check=True)
    
    
def segmentation(t1c_path, t1n_path, t2f_path, t2w_path,output_dir):
    ## get the segmentation mask then save as Nifti file
    prediction_path = inference(t1c_path, t1n_path, t2f_path, t2w_path,output_dir)
    # Mask_path = save_array_nifti(prediction,studies_dir,0)
    print("Mask_path saved as Nifti file")
    return prediction_path

def load_nifti_image(file_path):
    image = SimpleITK.ReadImage(file_path)
    # convert to np array
    image_np = SimpleITK.GetArrayFromImage(image)
    return image_np






def nifti_to_dicom_seg(series_dir, label, label_info, output_file, file_ext="*", use_itk=True) -> str:
    start = time.time()

    label_np = load_nifti_image(label)
    unique_labels = np.unique(label_np.flatten()).astype(np.int_)
    unique_labels = unique_labels[unique_labels != 0]

    info = label_info[0] if label_info and 0 < len(label_info) else {}
    model_name = info.get("model_name", "AIName")

    segment_attributes = []
    for i, idx in enumerate(unique_labels):
        info = label_info[i] if label_info and i < len(label_info) else {}
        name = info.get("name", "unknown")
        description = info.get("description", "Unknown")
        rgb = list(info.get("color", GENERIC_ANATOMY_COLORS.get(name, (255, 0, 0))))[0:3]
        rgb = [int(x) for x in rgb]

        print(f"{i} => {idx} => {name}")

        segment_attribute = info.get(
            "segmentAttribute",
            {
                "labelID": int(idx),
                "SegmentLabel": name,
                "SegmentDescription": description,
                "SegmentAlgorithmType": "AUTOMATIC",
                "SegmentAlgorithmName": "MMMAILABEL",
                "SegmentedPropertyCategoryCodeSequence": {
                    "CodeValue": "123037004",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": "Anatomical Structure",
                },
                "SegmentedPropertyTypeCodeSequence": {
                    "CodeValue": "78961009",
                    "CodingSchemeDesignator": "SCT",
                    "CodeMeaning": name,
                },
                "recommendedDisplayRGBValue": rgb,
            },
        )
        segment_attributes.append(segment_attribute)

    template = {
        "ContentCreatorName": "Reader1",
        "ClinicalTrialSeriesID": "Session1",
        "ClinicalTrialTimePointID": "1",
        "SeriesDescription": model_name,
        "SeriesNumber": "300",
        "InstanceNumber": "1",
        "segmentAttributes": [segment_attributes],
        "ContentLabel": "SEGMENTATION",
        "ContentDescription": "MMM.AI - Image segmentation",
        "ClinicalTrialCoordinatingCenterName": "MMM.AI",
        "BodyPartExamined": "",
    }

    # print(json.dumps(template, indent=2))
    if not segment_attributes:
        print("Missing Attributes/Empty Label provided")
        return ""

    if use_itk:
        output_file = itk_image_to_dicom_seg(label, series_dir, template, output_file)
    else:
        template = pydicom_seg.template.from_dcmqi_metainfo(template)
        writer = pydicom_seg.MultiClassWriter(
            template=template,
            inplane_cropping=False,
            skip_empty_slices=False,
            skip_missing_segment=False,
        )

        # Read source Images
        series_dir = pathlib.Path(series_dir)
        image_files = series_dir.glob(file_ext)
        image_datasets = [dcmread(str(f), stop_before_pixels=True) for f in image_files]
        print(f"Total Source Images: {len(image_datasets)}")

        mask = SimpleITK.ReadImage(label)
        mask = SimpleITK.Cast(mask, SimpleITK.sitkUInt16)

        dcm = writer.write(mask, image_datasets)
        dcm.save_as(output_file)

    print(f"nifti_to_dicom_seg latency : {time.time() - start} (sec)")
    return output_file


def itk_image_to_dicom_seg(label, series_dir, template, output_file) -> str:
    meta_data = output_file + ".json"
    output_file = output_file + ".dcm"

    with open(meta_data, "w") as fp:
        json.dump(template, fp)

    print("meta_data", meta_data)
    print("output_file", output_file)

    # command = "itkimage2segimage" 
    command = r"C:\Users\hazem\dcmqi\build\dcmqi-build\bin\Release\itkimage2segimage"
    args = [
        "--inputImageList",
        label,
        "--inputDICOMDirectory",
        series_dir,
        "--outputDICOM",
        output_file,
        "--inputMetadata",
        meta_data,
    ]
    returncode = run_command(command, args)
    os.unlink(meta_data)
    return output_file



def run_command(command, args=None):
    cmd = [command]

    if args:
        args = [str(a) for a in args]
        cmd.extend(args)

    print("Running Command:: {}".format(" ".join(cmd)))

    process = subprocess.Popen(
        cmd,
        # stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        env=os.environ.copy(),
    )

    while process.poll() is None:
        line = process.stdout.readline()
        line = line.rstrip()
        if line:
            print(line)

    print(f"Return code: {process.returncode}")
    process.stdout.close()
    return process.returncode
