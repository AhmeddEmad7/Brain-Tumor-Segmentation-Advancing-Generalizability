import json
import time
import os
import nibabel as nib
import numpy as np
import requests
import redis
from src.processing import load_model, correct_volume, get_dicom_series
from src.improved_nifti2dicom import convert_nifti_to_dicom 
model = load_model()
current_dir = os.path.dirname(os.path.abspath(__file__))
studies_dir = os.path.join(current_dir, '..', 'studies')

client_redis = redis.Redis(host="localhost", port=6379, db=0)


def motion_correction_callback(ch, method, properties, body):
        print(f" [Motion Artifacts] Received {body.decode()}, starting processing...")
        body = json.loads(body.decode())
        study_uid = body['studyInstanceUid']
        series_uid = body['seriesInstanceUid']

        nifti_path = get_dicom_series(study_uid, series_uid)
        print('nifti_path', nifti_path)
        
        volume = nib.load(nifti_path)
        print("volume:", volume)
        
        print("volume_data", volume.get_fdata())
        ## this will return the voxel intensities which represent the scanned tissue or material densities in medical imaging.
        
        new_volume_data = correct_volume(model, volume.get_fdata())
        new_volume = nib.Nifti1Image(np.array(new_volume_data), volume.affine)
        ## affine is a 4x4 matrix that describes the position of the image array data in physical space.
        print("new_volume data",new_volume.header)
        ## this also will return the metadata 
        
        corrected_nifti_path = os.path.join(os.path.dirname(studies_dir), f'{series_uid}.nii.gz')
        
        nib.save(new_volume, corrected_nifti_path)

        print("Processed volume saved to:", corrected_nifti_path)
        
        print("start conversion to dicom again")
        
        nifti_to_dicom_path = os.path.join(studies_dir, f'volume\{series_uid}')

        # Ensure the directory exists
        os.makedirs(nifti_to_dicom_path, exist_ok=True)    
        
        global_metadata = client_redis.get(f"metadata/{series_uid}")
        print("global_metadata from redis",global_metadata)
        global_metadata = json.loads(global_metadata)
        
        convert_nifti_to_dicom(corrected_nifti_path, nifti_to_dicom_path,global_metadata)
        
        print("uploading to orthanc")
        
        upload_dicom_series_orthanc(nifti_to_dicom_path)

        # 4. send results to orthanc
        # print(f" [Motion Artifacts] Motion correction done in {elapsed_time} seconds!")

        ch.basic_ack(delivery_tag=method.delivery_tag)
    
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