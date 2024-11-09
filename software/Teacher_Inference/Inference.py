from Teacher_Inference.DynUNet import DynUNet
from Teacher_Inference.Loader import load_sequences_from_paths
import torch
import torch.nn.functional as F
import numpy as np
from monai.transforms import AsDiscrete
from pathlib import Path

def load_model(model_path): # hot el path lel model el .pth hena
    model_path = Path(model_path)
    model = DynUNet( spatial_dims=3, in_channels=4, out_channels=4, deep_supervision=False)       
    if (model_path).is_file():
        print(f"Found model: {model_path}")
        ckpt = torch.load(model_path, map_location='cuda', weights_only=True) #map_location='cuda' de momken t3mlak moshkla bs sebha law zabta
        model.load_state_dict(ckpt['teacher_model'])
        print(f"Loaded model: {model_path}")
    
    return model

def post_process(pred, num_classes=4): #Bx4xHxWxD
    pred_probs = F.softmax(pred, dim=1)  #Bx4xHxWxD
    pred_discrete = AsDiscrete(argmax=True, dim=1)(pred_probs)  #Bx1xHxWxD

    new_volume = pred_discrete.squeeze(0).squeeze(0) #HxWxD

    return new_volume.float()

def inference(t1c_path, t1n_path, t2f_path, t2w_path):
    input = load_sequences_from_paths(t1c_path, t1n_path, t2f_path, t2w_path)
    model = load_model('hot_el_path_lel_model_el_pth_hena.pth')
    model.eval()
    output = model(input['imgs'])
    prediction = post_process(output['pred'])

    return np.array(prediction) # Aw (prediction) bas law 3ayzo tensor
