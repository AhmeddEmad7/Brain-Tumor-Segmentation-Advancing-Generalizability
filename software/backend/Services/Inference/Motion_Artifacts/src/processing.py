import keras
import tensorflow.keras.backend as K
from skimage import exposure
from dotenv import load_dotenv
from skimage.transform import resize
import numpy as np
import os
import pydicom
import dicomweb_client
import subprocess
import redis
import json
from src.model_functions import ms_ssim_score, ssim_loss, ssim_score, psnr

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
studies_dir = os.path.join(current_dir, '..', 'studies')
model_path = os.path.join(current_dir, '..', 'models', 'stacked_model.h5')

client = dicomweb_client.api.DICOMwebClient(f"{os.getenv('ORTHANC_URL')}/dicom-web")


redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_db = int(os.getenv('REDIS_DB', 0))

client_redis = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

# global_metadata = None 
def load_model():

    global model_path
    print('loading the model')
    model = keras.models.load_model(
        model_path,
        custom_objects={
            'ms_ssim_score': ms_ssim_score,
            'ssim_loss': ssim_loss,
            'ssim_score': ssim_score,
            'psnr': psnr,
            'K': K
        }
    )

    print('model loaded')

    return model


def sample_reshape(sample):
    # Resize each slice to (256, 256)
    sample_resized = resize(sample, (256, 256), anti_aliasing=True) 
    sample_resized = sample_resized.reshape(1, sample_resized.shape[0], sample_resized.shape[1], 1)  # Adding batch size and channel dimensions
    sample_resized = sample_resized.reshape(1, 256, 256, 1)
    ## size edit according to nifti sample you have to resize
    # sample = sample.reshape(1, sample.shape[0], sample.shape[1], 1)
    
    return sample_resized

def correct_volume(model, volume):
    volume = exposure.rescale_intensity(volume, out_range=(-1.0, 1.0))
    free_volume = []
    for slic in range(1, volume.shape[0] - 1):
        pred_slice = model.predict(
            [sample_reshape(volume[slic - 1]), sample_reshape(volume[slic]), sample_reshape(volume[slic + 1])],
            verbose=0)
        pred_slice = np.clip(pred_slice[0], -1, 1)  # Ensure pred_slice is clipped to the range
        scaled_slice = ((pred_slice) * 2047.5).astype(np.uint16)  # Rescale and convert to uint16
        # Append only the correctly scaled slice
        free_volume.append(scaled_slice)
    return free_volume


def get_dicom_series(study_uid, series_uid):
    # global global_metadata  # Declare the use of the global variable

    try:
        instances = client.retrieve_series(study_instance_uid=study_uid, series_instance_uid=series_uid)
        # metadata = client.retrieve_series_metadata(study_instance_uid=study_uid, series_instance_uid=series_uid)
        instance_metadata = client.retrieve_instance_metadata(study_instance_uid=study_uid, series_instance_uid=series_uid, sop_instance_uid=instances[0].SOPInstanceUID)
        if instance_metadata:
                # Assuming the metadata is returned as a list and the first item contains the desired data
                # global_metadata = instance_metadata
                client_redis.set(f"metadata/{series_uid}", json.dumps(instance_metadata),ex=1800) #30 min
                print("on redis successfully", instance_metadata)
        # print("Metadata updated in global variable:", instance_metadata)
        dicom_dir = os.path.join(studies_dir, series_uid)
        
        os.makedirs(dicom_dir, exist_ok=True)
        
        for instance in instances:
            # Assuming instance is a pydicom.Dataset
            output_path = os.path.join(dicom_dir, f"{instance.SOPInstanceUID}.dcm")
            instance.save_as(output_path)
            
            # print("Type of instance:", type(instance))  # Print the type of each instance
            # print("Instance content:", instance)
            # break
        
        dicom_to_nifti(dicom_dir, studies_dir, series_uid)
        
        print(f"Series {series_uid} from study {study_uid} retrieved and converted to NIfTI")

        return f"{studies_dir}/{series_uid}.nii.gz"
    
    except Exception as e:
        print(f"Error retrieving series {series_uid} from study {study_uid}: {e}")
        return None

def dicom_to_nifti(dicom_dir, output_dir, file_name):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Call dcm2niix
    subprocess.run(["dcm2niix", "-z", "y", "-f", "%j", "-o", output_dir, dicom_dir], check=True)
