import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from monai.transforms import LoadImage, Compose, NormalizeIntensityd, AsDiscrete

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
    
    return float_volumes, int_volumes, metadata, t1c_tensor

def preprocess_data(data):
    transform = Compose([
            NormalizeIntensityd( keys=['imgs'], nonzero=False, channel_wise=True)
        ])

    preprocessed_data = transform(data)
    return preprocessed_data