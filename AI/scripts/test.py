import pandas as pd
import torch
from models.dyn_unet import DynUNet
from testing.test_pipeline import test_net
from data.loaders import prepare_data_loaders
from config.default_config import get_args

args = get_args()

if __name__ == "__main__": 
    if args.mode == 'test':
        print("Running in TEST mode...")
        _, _, testLoader = prepare_data_loaders(args)

        student_model = DynUNet(spatial_dims=3, in_channels=4, out_channels=4, deep_supervision=False).to('cuda')
        if (args.student_model_path).is_file():
            print(f"Found model: {args.student_model_path}")
            ckpt = torch.load(args.student_model_path, map_location='cuda', weights_only=True)
            student_model.load_state_dict(ckpt['student_model'])
            print(f"Loaded model: {args.student_model_path}")
            
            print("Starting test pipeline...")
            total_metrics = test_net(student_model, testLoader)
            total_metrics_df = pd.DataFrame(total_metrics)
            
            args.output_dir.mkdir(parents=True, exist_ok=True)
            
            output_csv_path = args.output_dir / 'total_metrics.csv'
            total_metrics_df.to_csv(output_csv_path, index=False)
            print(f"Test metrics saved to {output_csv_path}")
        else:
            print(f"Error: Student model not found at {args.student_model_path}. Please check the path or download the model.")
    else:
        print(f"To run tests, execute: python scripts/test.py test [other_args]")
        print(f"Current mode selected is '{args.mode}'. Testing logic will not be executed.")