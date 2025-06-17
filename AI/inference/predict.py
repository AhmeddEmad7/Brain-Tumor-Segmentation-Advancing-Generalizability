import torch
import numpy as np
from pathlib import Path
from monai.inferers import sliding_window_inference
from models.dyn_unet import DynUNet
from inference.preprocess import load_sequences_from_paths, save_nifti_volumes
from inference.postprocess import generate_prediction_mask, save_prediction_mask


def load_inference_model(model_path: str, device: str = 'cuda') -> DynUNet:
    """
    Loads a pre-trained DynUNet model for inference.

    Args:
        model_path (str): Path to the saved model checkpoint.
        device (str): The device to load the model onto (e.g., 'cuda' or 'cpu').

    Returns:
        DynUNet: The loaded model in evaluation mode.
    """
    model_path = Path(model_path)
    model = DynUNet( spatial_dims=3, in_channels=4, out_channels=4, deep_supervision=False)

    if model_path.is_file():
        print(f"Found model checkpoint: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['student_model']) # Change to teacher_model if needed
        print(f"Loaded model weights from: {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = model.to(device)
    model.eval()
    return model


def predict(
    t1c_path: str,
    t1w_path: str,
    t2f_path: str,
    t2w_path: str,
    output_dir: str,
    model_path: str,
    device: str = 'cuda',
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    mode: str = 'gaussian'
) -> np.ndarray:
    """
    Executes the full inference pipeline: loads data, preprocesses, runs model inference,
    post-processes predictions, and saves results.

    Args:
        t1c_path (str): Path to T1-contrast enhanced NIfTI file.
        t1w_path (str): Path to T1-native NIfTI file.
        t2f_path (str): Path to T2-FLAIR NIfTI file.
        t2w_path (str): Path to T2-weighted NIfTI file.
        output_dir (str): Directory to save raw input volumes and the final prediction mask.
        model_path (str): Path to the pre-trained model checkpoint.

    Returns:
        np.ndarray: The final processed prediction mask as a NumPy array.
    """
    print("\n--- Starting Inference Process ---")
    float_volumes, int_volumes, metadata = load_sequences_from_paths(
        t1c_path, t1w_path, t2f_path, t2w_path
    )

    save_nifti_volumes(int_volumes, metadata, output_dir)

    input_data_tensor = float_volumes['imgs'].unsqueeze(0).to(device)
    print("Input to model shape:", input_data_tensor.shape)

    model = load_inference_model(model_path, device)
    print(f"Model loaded onto {device}.")

    with torch.no_grad():
        print("Running sliding window inference...")
        output_logits_dict = sliding_window_inference(
            inputs=input_data_tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode=mode,
            device=device,
            progress=True
        )
        pred_logits = output_logits_dict['pred']

    print("Sliding window inference complete.")
    print("Raw prediction logits shape:", pred_logits.shape)

    prediction_mask_tensor = generate_prediction_mask(pred_logits)
    print("Final prediction mask shape:", prediction_mask_tensor.shape)

    save_prediction_mask(prediction_mask_tensor, metadata, output_dir)
    print(f"Prediction mask saved at {output_dir}")

    print("--- Inference Process Complete ---")
    return np.array(prediction_mask_tensor.cpu())