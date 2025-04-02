import torch
import nibabel as nib
from AI.DynUNet import DynUNet
from Loader import load_sequences_from_paths
from monai.inferers import sliding_window_inference
from Reporting import extract_tumor_features, generate_report, generate_pdf
from pathlib import Path
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

def generate_prediction_mask(pred):
    output_probs = (torch.sigmoid(pred) > 0.5)
    _, _, H, W, D = output_probs.shape

    output = output_probs[0] # Get the only element in the batch (first one)
    seg_mask = torch.zeros((H, W, D))

    seg_mask[torch.where(output[1, ...] == 1)] = 2  # WT --> ED
    seg_mask[torch.where(output[2, ...] == 1)] = 1  # TC --> NCR
    seg_mask[torch.where(output[3, ...] == 1)] = 3  # ET --> ET

    return seg_mask.float(), output.float()
    
def save_nifti_volumes(int_volumes, metadata, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sequence_names = ['t1c', 't1n', 't2f', 't2w']
    
    for i in range(len(metadata)):
        nifti_image = nib.Nifti1Image(int_volumes['imgs'][i].numpy(), affine=metadata[i]['affine'])
        file_name = f"{sequence_names[i]}.nii.gz"
        file_path = os.path.join(output_dir, file_name)
        nib.save(nifti_image, file_path)
        print(f"Saved: {file_path}")

def inference(t1c_path, t1n_path, t2f_path, t2w_path, output_dir, model_path):
    input_data, int_volumes, metadata, brain_volume = load_sequences_from_paths(t1c_path, t1n_path, t2f_path, t2w_path)
    save_nifti_volumes(int_volumes, metadata, output_dir)
    
    input_data['imgs'] = input_data['imgs'].unsqueeze(0).to('cuda')
    print("Input to model shape:", input_data['imgs'].shape)

    model = load_model(model_path)
    model = model.to('cuda')
    model.eval()

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_data['imgs'],
            roi_size=(128, 128, 128),
            sw_batch_size=2,
            predictor=model,
            overlap=0.25,
            mode='gaussian'
        )

    prediction, mask_channels = generate_prediction_mask(output['pred'])
    prediction, mask_channels, brain_volume = prediction.cpu().numpy(), mask_channels.cpu().numpy(), brain_volume[0].cpu().numpy()
    print("Prediction shape:", prediction.shape)

    # Saving prediction
    nifti_pred = nib.Nifti1Image(prediction, affine=metadata[0]['affine'])
    nifti_pred.header.set_intent('label', name='Label Map')
    nib.save(nifti_pred, os.path.join(output_dir, f"prediction_label.nii.gz"))

    tumor_features = extract_tumor_features(brain_volume, prediction, mask_channels)
    findings = generate_report(tumor_features)

    # Generating a PDF IF NEEDED #

    # patient = metadata[0]['filename_or_obj'].split('/')[5]
    # generate_pdf(findings, patient)
    # print("\n==== Generated Report as PDF file ====\n")

    return prediction, findings


# # Doing inference here
# t1c_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t1c.nii/00000116_brain_t1ce.nii'
# t1n_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t1n.nii/00000116_brain_t1.nii'
# t2f_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t2f.nii/00000116_brain_flair.nii'
# t2w_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-t2w.nii/00000116_brain_t2.nii'
# model_path = '/kaggle/input/gliomateachernewlabels/Teacher_model_after_epoch_100_trainLoss_0.5972_valLoss_0.3019.pth'
# # seg_path = '/kaggle/input/bratsglioma/Training/BraTS-GLI-00006-000/BraTS-GLI-00006-000-seg.nii'

# output_dir = '/kaggle/working/output'
# prediction, findings = inference(t1c_path, t1n_path, t2f_path, t2w_path, output_dir, model_path)
# print('Inference done!')