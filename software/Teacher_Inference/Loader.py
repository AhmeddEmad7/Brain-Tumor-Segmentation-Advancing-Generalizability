import torch
import numpy as np
from monai.transforms import LoadImage, Compose, NormalizeIntensityd, RandSpatialCropd, RandFlipd, \
                             RandRotate90d, Rand3DElasticd, RandAdjustContrastd, CenterSpatialCropd,\
                             Resized, RandRotated, Spacingd, CropForegroundd, SpatialPadd, AsDiscrete

def load_sequences_from_paths(t1c_path, t1n_path, t2f_path, t2w_path):   # lazem yekon el path .nii aw .nii.gz     
        loadimage = LoadImage(reader='NibabelReader', image_only=True)
        
        t1c_loader   = loadimage( t1c_path )
        t1n_loader   = loadimage( t1n_path )
        t2f_loader   = loadimage( t2f_path )
        t2w_loader   = loadimage( t2w_path )

        # Make the dimension of channel
        t1c_tensor   = torch.Tensor(t1c_loader).unsqueeze(0)
        t1n_tensor   = torch.Tensor(t1n_loader).unsqueeze(0)
        t2f_tensor   = torch.Tensor(t2f_loader).unsqueeze(0)
        t2w_tensor   = torch.Tensor(t2w_loader).unsqueeze(0)

        concat_tensor = torch.cat( (t1c_tensor, t1n_tensor, t2f_tensor, t2w_tensor), 0 )
        raw_data = { 'imgs'  : np.array(concat_tensor[:,:,:,:])}

        processed_data = preprocess_data(raw_data)
        imgs  = np.array(processed_data['imgs'])

        volume = { 'imgs'  : torch.from_numpy( imgs  ).type( torch.FloatTensor )}
        
        return volume

def preprocess_data(data):
    transform = Compose([
        NormalizeIntensityd( keys=['imgs'], nonzero=False, channel_wise=True ),
        Resized(keys=['imgs'],
                spatial_size=[128,128,128],
                mode=['trilinear']),
        ])

    # Apply the transform
    preprocessed_data = transform(data)
    return preprocessed_data