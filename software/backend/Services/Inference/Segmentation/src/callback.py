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
import pika

current_dir = os.path.dirname(os.path.abspath(__file__))
studies_dir = os.path.join(current_dir, '..', 'studies')


if not os.path.exists(studies_dir):
        print(f"Creating directory {studies_dir}")
        os.makedirs(studies_dir)
        
client = dicomweb_client.api.DICOMwebClient(f"{os.getenv('ORTHANC_URL')}/dicom-web")

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_db = int(os.getenv('REDIS_DB', 0))

client_redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

def store_finding_in_redis(study_uid: str, finding: str):
    redis_key = f"report_{study_uid}"
    client_redis.set(redis_key, finding)
    print(f"Stored finding for study {study_uid} in Redis under key {redis_key}")

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
        
        
    t1 = volume_paths['t1']
    t2 = volume_paths['t2']
    flair = volume_paths['flair']
    t1ce = volume_paths['t1c']

    print('[Segmentation] Volumes loaded')

    # 3. apply the model
    start_time = time.time()

    ###### INSERT THE MODEL FUNCTION HERE ######
    output_dir = os.path.join(studies_dir, study_uid)
    segmentation_mask_nifti_path,finding = segmentation(t1ce, t1, flair, t2, output_dir)
    
    # send the finding to MQ
    print("Finding:", finding)
    print("Sending reporting message...")
    store_finding_in_redis(study_uid, finding)
    # End the timer
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Segmentation completed in {elapsed_time} seconds")
    
    # save the corrected volume
    print("Saving the segmentation mask As nifti...")

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
    {"name": "Label1", "description": "Necrosis region", "color": (255, 0, 0), "model_name": "AIModel_v1.0"},
    {"name": "Label2", "description": "Edema region", "color": (0, 255, 0), "model_name": "AIModel_v1.0"},
    {"name": "Label3", "description": "Enhancing Tumor", "color": (0, 0, 255), "model_name": "AIModel_v1.0"},
]
