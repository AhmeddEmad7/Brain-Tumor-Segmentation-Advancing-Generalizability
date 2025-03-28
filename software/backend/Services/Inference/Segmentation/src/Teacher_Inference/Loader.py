import torch
import numpy as np
import nibabel as nib
from monai.transforms import LoadImage, Compose, NormalizeIntensityd, RandSpatialCropd, RandFlipd, \
                             RandRotate90d, Rand3DElasticd, RandAdjustContrastd, CenterSpatialCropd,\
                             Resized, RandRotated, Spacingd, CropForegroundd, SpatialPadd, AsDiscrete

def resize_with_aspect_ratio(keys, target_size):
    def transform(data):
        for key in keys:
            volume = data[key]
            original_shape = volume.shape[-3:]

            scaling_factor = min(
                target_size[0] / original_shape[0],
                target_size[1] / original_shape[1],
                target_size[2] / original_shape[2]
            )

            # Computing the intermediate size while preserving aspect ratio
            new_shape = [
                int(dim * scaling_factor) for dim in original_shape
            ]

            resize_transform = Resized(keys=[key], spatial_size=new_shape, mode="trilinear" if key == "imgs" else "nearest-exact")
            data = resize_transform(data)

            pad_transform = SpatialPadd(keys=[key], spatial_size=target_size, mode="constant")
            data = pad_transform(data)
        return data

    return transform


def load_sequences_from_paths(t1c_path, t1n_path, t2f_path, t2w_path):   # lazem yekon el path .nii aw .nii.gz     
    loadimage = LoadImage(reader='NibabelReader', image_only=False)
    
    t1c_loader, t1c_metadata = loadimage( t1c_path )
    t1n_loader, t1n_metadata = loadimage( t1n_path )
    t2f_loader, t2f_metadata = loadimage( t2f_path )
    t2w_loader, t2w_metadata = loadimage( t2w_path )

    metadata = [t1c_metadata, t1n_metadata, t2f_metadata, t2w_metadata]

    t1c_tensor = torch.Tensor(t1c_loader).unsqueeze(0)
    t1n_tensor = torch.Tensor(t1n_loader).unsqueeze(0)
    t2f_tensor = torch.Tensor(t2f_loader).unsqueeze(0)
    t2w_tensor = torch.Tensor(t2w_loader).unsqueeze(0)

    concat_tensor = torch.cat( (t1c_tensor, t1n_tensor, t2f_tensor, t2w_tensor), 0 )
    raw_data = {'imgs' : np.array(concat_tensor[:,:,:,:])}
    int_volumes = {'imgs' : torch.from_numpy(raw_data['imgs']).type(torch.IntTensor)}

    processed_data = preprocess_data(raw_data)
    norm_imgs  = np.array(processed_data['imgs'])
    float_volumes = {'imgs' : torch.from_numpy(norm_imgs).type(torch.FloatTensor)}
    
    return float_volumes, int_volumes, metadata

def preprocess_data(data):
    transform = Compose([
            # CropForegroundd(keys=["imgs"], source_key="imgs"),
            NormalizeIntensityd( keys=['imgs'], nonzero=False, channel_wise=True)
        ])

    preprocessed_data = transform(data)
    return preprocessed_data


nifti_output_path1 =    r'D:\Brain-Tumor-Segmentation-Advancing-Generalizability\Software\backend\Services\Inference\Segmentation\studies/t1.nii.gz'
nifti_output_path2 =    r'D:\Brain-Tumor-Segmentation-Advancing-Generalizability\Software\backend\Services\Inference\Segmentation\studies/t1c.nii.gz'
nifti_output_path3 =    r'D:\Brain-Tumor-Segmentation-Advancing-Generalizability\Software\backend\Services\Inference\Segmentation\studies/flair.nii.gz'
nifti_output_path4 =    r'D:\Brain-Tumor-Segmentation-Advancing-Generalizability\Software\backend\Services\Inference\Segmentation\studies/t2.nii.gz'
        