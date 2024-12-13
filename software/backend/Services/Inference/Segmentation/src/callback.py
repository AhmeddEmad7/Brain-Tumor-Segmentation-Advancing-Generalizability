import json
import time
import os
import nibabel as nib
import numpy as np
import dicomweb_client.api
import pydicom
import requests
import redis
from src.processing import get_dicom_series, nifti_to_dicom_seg , segmentation 
from src.colored_dicom import convert_array_to_dicom_seg

current_dir = os.path.dirname(os.path.abspath(__file__))
studies_dir = os.path.join(current_dir, '..', 'studies')


if not os.path.exists(studies_dir):
        print(f"Creating directory {studies_dir}")
        os.makedirs(studies_dir)
        
# client = dicomweb_client.api.DICOMwebClient(f"{os.getenv('ORTHANC_URL')}/dicom-web")
client = dicomweb_client.api.DICOMwebClient(f"http://localhost:8042/dicom-web")

client_redis = redis.Redis(host="localhost", port=6379, db=0)

def segmentation_callback(ch, method, properties, body):
    print(f" [Segmentation] Received {body.decode()}, starting processing...")

    body = json.loads(body.decode())
    study_uid = body['studyInstanceUid']
    sequences = body['sequences']

    # 1. get the series from orthanc
    print('[Segmentation] Loading the volume')

    volume_paths = {}

    for key, value in sequences.items():
        print(f"Key: {key}, Value: {value}")
        volume_path = get_dicom_series(study_uid, value, key)
        volume_paths[key] = volume_path
        print(f"Volume path: {volume_path}")

    # 2. load the volumes
    # t1 = nib.load(volume_paths['t1']).get_fdata()
    # t2 = nib.load(volume_paths['t2']).get_fdata()
    # flair = nib.load(volume_paths['flair']).get_fdata()
    # t1ce = nib.load(volume_paths['t1c']).get_fdata()

    t1 = volume_paths['t1']
    t2 = volume_paths['t2']
    flair = volume_paths['flair']
    t1ce = volume_paths['t1c']

    print('[Segmentation] Volumes loaded')

    # 3. apply the model
    start_time = time.time()

    ###### INSERT THE MODEL FUNCTION HERE ######
    output_dir = os.path.join(studies_dir, study_uid)
    segmentation_mask_nifti_path = segmentation(t1, t2, flair, t1ce,output_dir)
    # segmentation_mask = np.flip(segmentation_mask, axis=1)
    # End the timer
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Segmentation completed in {elapsed_time} seconds")
    
    # save the corrected volume
    print("Saving the segmentation mask As nifti...")
    # nib.save(nib.Nifti1Image(np.array(segmentation_mask), np.eye(4)),
    #          os.path.join(studies_dir, study_uid, 'segmentation.nii.gz'))

    print("Converting to DICOM SEG...")
    print(sequences['t1'])
    metadata = client_redis.get(f"metadata/{sequences['t1']}")
    print("data retreive from redis")
    print("Loaded Metadata:", metadata)
    metadata = json.loads(metadata)
    print("Start making the dicom seg")
    output_file_segmentation = nifti_to_dicom_seg(
        os.path.join(studies_dir, study_uid, 't1'),
        segmentation_mask_nifti_path,
        label_info,
        os.path.join(studies_dir, study_uid, 'segmentation')
    )

    print("Sending to Orthanc...")
    
    dcm_seg_dataset = pydicom.dcmread(os.path.join(studies_dir, study_uid,'segmentation.dcm'))
    print("Store to Orthanc...")
    # send the results to orthanc
    client.store_instances(datasets=[dcm_seg_dataset])
    print("store in orthanc Success")
    ch.basic_ack(delivery_tag=method.delivery_tag)
    
    
label_info = [
    {"name": "Label1", "description": "Necrosis region", "color": (255, 0, 0), "model_name": "Teacher_model_after_epoch_105_v1.0"},
    {"name": "Label2", "description": "Edema region", "color": (0, 255, 0), "model_name": "AIModel_v1.0"},
    {"name": "Label3", "description": "Enhancing Tumor", "color": (0, 0, 255), "model_name": "AIModel_v1.0"},
]
def upload_dicom_series_orthanc(dicom_dir):
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]
    for dicom_file in dicom_files:
        try:
            with open(dicom_file, 'rb') as f:
                files = {'file': (os.path.basename(dicom_file), f, 'application/dicom')}
                response = requests.post("http://localhost:8042/instances", files=files)
                response.raise_for_status()  # raises exception for HTTP errors
            print(f"Uploaded DICOM file: {dicom_file}")
        except Exception as e:
            print(f"Failed to upload {dicom_file} due to {e}")
    print(f"All DICOM files from {dicom_dir} uploaded to Orthanc.")   