from AI_Inference_Pipeline_Updated.DynUNet import DynUNet
from AI_Inference_Pipeline_Updated.Loader import load_sequences_from_paths
import torch
import torch.nn.functional as F
import numpy as np
from monai.transforms import AsDiscrete
from pathlib import Path
import nibabel as nib
import os

def load_model(model_path):
    model_path = Path(model_path)
    model = DynUNet( spatial_dims=3, in_channels=4, out_channels=4, deep_supervision=False)       
    if (model_path).is_file():
        print(f"Found model: {model_path}")
        ckpt = torch.load(model_path, map_location='cuda', weights_only=True) #map_location='cuda' de momken t3mlak moshkla bs sebha law zabta
        model.load_state_dict(ckpt['teacher_model'])
        print(f"Loaded model: {model_path}")
    
    return model

def generate_prediction_mask(pred, num_classes=4):
    pred_probs = F.softmax(pred, dim=1)
    pred_discrete = AsDiscrete(argmax=True, dim=1)(pred_probs)
    new_volume = pred_discrete.squeeze(0).squeeze(0)

    return new_volume.float()
    
def save_nifti_volumes(int_volumes, metadata, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sequence_names = ['t1c', 't1n', 't2f', 't2w']
    
    for i in range(len(metadata)):
        nifti_image = nib.Nifti1Image(int_volumes['imgs'][i].numpy(), affine=metadata[i]['affine'])
        file_name = f"{sequence_names[i]}.nii.gz"
        file_path = os.path.join(output_dir, file_name)
        nib.save(nifti_image, file_path)
        print(f"Saved: {file_path}")

def inference(t1c_path, t1n_path, t2f_path, t2w_path, output_dir):
    input, int_volumes, metadata = load_sequences_from_paths(t1c_path, t1n_path, t2f_path, t2w_path)
    save_nifti_volumes(int_volumes, metadata, output_dir)
    
    input['imgs'] = input['imgs'].unsqueeze(0)
    print(input['imgs'].shape)
    
    model = load_model('/kaggle/input/gliomateacheroriginallabels/Teacher_model_after_epoch_100_trainLoss_0.3842_valLoss_0.1886.pth')
    model.eval()
    
    output = model(input['imgs'])
    prediction = generate_prediction_mask(output['pred'])
    print(prediction.shape)

    # Saving prediction
    nifti_pred = nib.Nifti1Image(prediction.numpy(), affine=metadata[0]['affine'])
    nib.save(nifti_pred, os.path.join(output_dir, f"pred.nii.gz"))

    return np.array(prediction)

## Example usage:

# t1c_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t1c.nii/00000116_brain_t1ce.nii'
# t1n_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t1n.nii/00000116_brain_t1.nii'
# t2f_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t2f.nii/00000116_brain_flair.nii'
# t2w_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t2w.nii/00000116_brain_t2.nii'

# # seg_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-seg.nii'
# output_dir = '/kaggle/working/output'
# prediction = inference(t1c_path, t1n_path, t2f_path, t2w_path, output_dir)