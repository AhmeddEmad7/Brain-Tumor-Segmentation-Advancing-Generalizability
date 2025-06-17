import os
import nibabel as nib
import numpy as np
import torch
from typing import Sequence
from monai.transforms import Compose, CropForegroundd, LoadImage, NormalizeIntensityd, RandAdjustContrastd, RandFlipd, Resized, SpatialPadd
from torch.utils.data import Dataset


class CustomDataset3D(Dataset):
    def __init__(self, data_dirs: list, patient_lists: list, mode: str):
        """
        Custom PyTorch Dataset for 3D medical brain MRI images.

        Args:
            data_dirs (list): List of base directories where patient data is stored.
            patient_lists (list): List of patient IDs to include in this dataset instance.
            mode (str): Current mode of the dataset (e.g., 'training', 'validation', 'testing').
        """
        self.data_dirs = data_dirs
        self.patient_lists = patient_lists
        self.mode = mode

    @staticmethod
    def resize_with_aspect_ratio(keys: Sequence[str], target_size: Sequence[int]):
        """
        Static method to create a transform that resizes volumes while preserving aspect ratio
        and then pads them to a target size.

        Args:
            keys (Sequence[str]): Keys in the data dictionary to apply the transform to (e.g., "imgs", "masks").
            target_size (Sequence[int]): The desired final spatial size (depth, height, width).

        Returns:
            monai.transforms.Compose: A MONAI Compose transform.
        """
        def transform(data):
            for key in keys:
                volume = data[key]
                original_shape = volume.shape[-3:]

                scaling_factor = min(
                    target_size[0] / original_shape[0],
                    target_size[1] / original_shape[1],
                    target_size[2] / original_shape[2]
                )

                new_shape = [
                    int(dim * scaling_factor) for dim in original_shape
                ]

                resize_transform = Resized(keys=[key], spatial_size=new_shape, mode="trilinear" if key == "imgs" else "nearest-exact")
                data = resize_transform(data)

                pad_transform = SpatialPadd(keys=[key], spatial_size=target_size, mode="constant")
                data = pad_transform(data)
            return data

        return transform

    def preprocess(self, data: dict, mode: str):
        """
        Applies preprocessing transformations based on the dataset mode.

        Args:
            data (dict): Dictionary containing 'imgs' and 'masks' to be transformed.
            mode (str): Current mode ('training', 'validation', 'testing').

        Returns:
            dict: Transformed data dictionary.
        """
        if mode == 'training':
            transform = Compose([
                CropForegroundd(keys=["imgs", "masks"], source_key="imgs"),
                self.resize_with_aspect_ratio(keys=["imgs", "masks"], target_size=[128, 128, 128]),
                NormalizeIntensityd( keys=['imgs'], nonzero=False, channel_wise=True),
                RandFlipd(
                    keys=["imgs", "masks"],
                    prob=0.5,
                    spatial_axis=2, # Flip along depth axis
                ),
                RandAdjustContrastd(
                    keys=["imgs"],
                    prob=0.15,
                    gamma=(0.65, 1.5),
                ),
            ])
        elif mode == 'validation' or mode == 'testing':
            transform = Compose([
                CropForegroundd(keys=["imgs", "masks"], source_key="imgs"),
                self.resize_with_aspect_ratio(keys=["imgs", "masks"], target_size=[128, 128, 128]),
                NormalizeIntensityd( keys=['imgs'], nonzero=False, channel_wise=True)
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'training', 'validation', or 'testing'.")

        augmented_data = transform(data)
        return augmented_data

    def __len__(self) -> int:
        """
        Returns the total number of patients in the dataset.
        """
        return len(self.patient_lists)

    def __getitem__(self, idx: int) -> dict:
        """
        Loads and preprocesses a single patient's data.

        Args:
            idx (int): Index of the patient in the patient_lists.

        Returns:
            dict: A dictionary containing 'imgs' (torch.FloatTensor), 'masks' (torch.FloatTensor),
                  'patient_id' (str), and 'data_type' (str).
        """
        patient_id = self.patient_lists[idx]
        loadimage = LoadImage(reader='NibabelReader', image_only=True)

        data_type = patient_id.split('-')[1]

        if data_type == 'GLI':
            patient_folder_path = os.path.join(self.data_dirs['GLI'], patient_id)
        elif data_type == 'SSA':
            patient_folder_path = os.path.join(self.data_dirs['SSA'], patient_id)
        elif data_type == 'PED':
            patient_folder_path = os.path.join(self.data_dirs['PED'], patient_id)
        elif data_type == 'MEN':
            patient_folder_path = os.path.join(self.data_dirs['MEN'], patient_id)
        elif data_type == 'MET':
            patient_folder_path = os.path.join(self.data_dirs['MET'], patient_id)
        else:
            raise ValueError(f"Unknown data type: {data_type} for patient: {patient_id}")


        def resolve_file_path(folder: str, name: str) -> str:
            """
            Resolves the actual file path for image volumes, handling cases where
            the expected .nii file might be nested in an additional subdirectory.
            """
            file_path = os.path.join(folder, name)
            if os.path.isdir(file_path):
                for root, _, files in os.walk(file_path):
                    for file in files:
                        if file.endswith(".nii"):
                            return os.path.join(root, file)
            return file_path

        t1c_path  = resolve_file_path(patient_folder_path, patient_id + '-t1c.nii')
        t1n_path  = resolve_file_path(patient_folder_path, patient_id + '-t1n.nii')
        t2f_path  = resolve_file_path(patient_folder_path, patient_id + '-t2f.nii')
        t2w_path  = resolve_file_path(patient_folder_path, patient_id + '-t2w.nii')
        seg_path  = os.path.join(patient_folder_path, patient_id + '-seg.nii')

        t1c_loader   = loadimage(t1c_path)
        t1n_loader   = loadimage(t1n_path)
        t2f_loader   = loadimage(t2f_path)
        t2w_loader   = loadimage(t2w_path)
        masks_loader = loadimage(seg_path)

        t1c_tensor   = torch.Tensor(t1c_loader).unsqueeze(0)
        t1n_tensor   = torch.Tensor(t1n_loader).unsqueeze(0)
        t2f_tensor   = torch.Tensor(t2f_loader).unsqueeze(0)
        t2w_tensor   = torch.Tensor(t2w_loader).unsqueeze(0)
        masks_tensor = torch.Tensor(masks_loader).unsqueeze(0)

        concat_tensor = torch.cat( (t1c_tensor, t1n_tensor, t2f_tensor, t2w_tensor, masks_tensor), 0 )
        data = {
            'imgs'  : np.array(concat_tensor[0:4,:,:,:]), # First 4 channels are images
            'masks' : np.array(concat_tensor[4:,:,:,:])  # Last channel is mask
        }

        augmented_imgs_masks = self.preprocess(data, self.mode)
        imgs  = np.array(augmented_imgs_masks['imgs'])
        masks = np.array(augmented_imgs_masks['masks'])

        return {
            'imgs'       : torch.from_numpy(imgs).type(torch.FloatTensor),
            'masks'      : torch.from_numpy(masks).type(torch.FloatTensor),
            'patient_id' : patient_id,
            'data_type'  : data_type
        } 