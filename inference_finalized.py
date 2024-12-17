import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.transforms import LoadImage, Compose, NormalizeIntensityd, AsDiscrete
from pathlib import Path
import os

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

def load_model(model_path):
    model_path = Path(model_path)
    model = DynUNet( spatial_dims=3, in_channels=4, out_channels=4, deep_supervision=False)       
    if (model_path).is_file():
        print(f"Found model: {model_path}")
        ckpt = torch.load(model_path, map_location='cuda', weights_only=True) #map_location='cuda' de momken t3mlak moshkla bs sebha law zabta
        model.load_state_dict(ckpt['teacher_model'])
        print(f"Loaded model: {model_path}")
    
    return model

def generate_prediction_mask(pred):
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
    
    input['imgs'] = input['imgs'].unsqueeze(0).to('cuda')
    print("Input to model shape:", input['imgs'].shape)
    
    model = load_model('/kaggle/input/gliomateacheroldlabelsbgincluded/Teacher_model_after_epoch_100_trainLoss_0.2821_valLoss_0.1429.pth')
    model = model.to('cuda')
    model.eval()

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input['imgs'],
            roi_size=(128, 128, 128),
            sw_batch_size=4,
            predictor=model,
            overlap=0.25,
            # mode='gaussian'
        )
    
    # output = model(input['imgs'])
    prediction = generate_prediction_mask(output['pred'])
    print("Prediction shape:", prediction.shape)

    # Saving prediction
    nifti_pred = nib.Nifti1Image(prediction.cpu().numpy(), affine=metadata[0]['affine'])
    nifti_pred.header.set_intent('label', name='Label Map')
    
    # Save the NIfTI file and _label left as it is
    nib.save(nifti_pred, os.path.join(output_dir, f"prediction_label.nii.gz")) # Do not change the name

    return np.array(prediction.cpu())


# Doing inference here
t1c_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t1c.nii/00000116_brain_t1ce.nii'
t1n_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t1n.nii/00000116_brain_t1.nii'
t2f_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t2f.nii/00000116_brain_flair.nii'
t2w_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t2w.nii/00000116_brain_t2.nii'
# seg_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-seg.nii'

output_dir = '/kaggle/working/output'
prediction = inference(t1c_path, t1n_path, t2f_path, t2w_path, output_dir)
print('Inference done!')