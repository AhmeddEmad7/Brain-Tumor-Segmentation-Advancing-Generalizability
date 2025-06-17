import os
import numpy as np
import torch
import nibabel as nib
from monai.transforms import Compose, LoadImage, NormalizeIntensityd


def load_sequences_from_paths(t1c_path: str, t1w_path: str, t2f_path: str, t2w_path: str) -> tuple[dict, dict, list]:
    """
    Loads 3D image sequences (T1c, T1w, T2f, T2w) from specified NIfTI paths,
    concatenates them, and applies initial preprocessing (normalization).

    Args:
        t1c_path (str): Path to the T1-contrast enhanced NIfTI file.
        t1w_path (str): Path to the T1-weighted NIfTI file.
        t2f_path (str): Path to the T2-FLAIR NIfTI file.
        t2w_path (str): Path to the T2-weighted NIfTI file.

    Returns:
        tuple[dict, dict, list]:
            - float_volumes (dict): Dictionary with preprocessed (normalized) images as a FloatTensor.
            - int_volumes (dict): Dictionary with raw image data as an IntTensor.
            - metadata (list): List of metadata dictionaries for each loaded sequence.
    """
    loadimage = LoadImage(reader='NibabelReader', image_only=False)

    t1c_loader, t1c_metadata = loadimage(t1c_path)
    t1w_loader, t1w_metadata = loadimage(t1w_path)
    t2f_loader, t2f_metadata = loadimage(t2f_path)
    t2w_loader, t2w_metadata = loadimage(t2w_path)

    metadata = [t1c_metadata, t1w_metadata, t2f_metadata, t2w_metadata]

    t1c_tensor = torch.Tensor(t1c_loader).unsqueeze(0)
    t1w_tensor = torch.Tensor(t1w_loader).unsqueeze(0)
    t2f_tensor = torch.Tensor(t2f_loader).unsqueeze(0)
    t2w_tensor = torch.Tensor(t2w_loader).unsqueeze(0)

    concat_tensor = torch.cat((t1c_tensor, t1w_tensor, t2f_tensor, t2w_tensor), 0)

    raw_data_np = np.array(concat_tensor.cpu())
    raw_data = {'imgs': raw_data_np}
    int_volumes = {'imgs': torch.from_numpy(raw_data['imgs']).type(torch.IntTensor)}

    processed_data = _preprocess_data(raw_data)
    norm_imgs = np.array(processed_data['imgs'])
    float_volumes = {'imgs': torch.from_numpy(norm_imgs).type(torch.FloatTensor)}

    return float_volumes, int_volumes, metadata


def _preprocess_data(data: dict) -> dict:
    """
    Applies intensity normalization to the input image data.
    (This is an internal helper function for load_sequences_from_paths)

    Args:
        data (dict): Dictionary containing 'imgs' (numpy array) to be transformed.

    Returns:
        dict: Transformed data dictionary with normalized images.
    """
    transform = Compose([
        NormalizeIntensityd(keys=['imgs'], nonzero=False, channel_wise=True)
    ])
    preprocessed_data = transform(data)
    return preprocessed_data


def save_nifti_volumes(int_volumes: dict, metadata: list, output_dir: str):
    """
    Saves the raw input image volumes as NIfTI files.

    Args:
        int_volumes (dict): Dictionary containing raw image data as an IntTensor.
        metadata (list): List of metadata dictionaries for each loaded sequence, containing 'affine'.
        output_dir (str): Directory where the NIfTI files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    sequence_names = ['t1c', 't1w', 't2f', 't2w']

    for i in range(len(metadata)):
        nifti_image = nib.Nifti1Image(int_volumes['imgs'][i].numpy(), affine=metadata[i]['affine'])
        file_name = f"{sequence_names[i]}.nii.gz"
        file_path = os.path.join(output_dir, file_name)
        nib.save(nifti_image, file_path)
        print(f"Saved raw input volume: {file_path}") 