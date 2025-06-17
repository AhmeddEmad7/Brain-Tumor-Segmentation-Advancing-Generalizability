import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from config.default_config import get_args
from inference.predict import predict

if __name__ == "__main__":  
    args = get_args()
    if args.mode == 'inference':
        print("Starting INFERENCE process...")
        
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Inference results will be saved to: {args.output_dir}")

        try:
            mask = predict(
                t1c_path=args.t1c_path,
                t1w_path=args.t1w_path,
                t2f_path=args.t2f_path,
                t2w_path=args.t2w_path,
                output_dir=args.output_dir,
                model_path=args.inference_model_path,
                device=args.device,
                roi_size=[128, 128, 128], # Fixed roi size for student model
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
                mode=args.sw_mode
            )
            print("Inference process completed successfully!")
        except FileNotFoundError as e:
            print(f"Error during inference: {e}")
            print("Please ensure all input NIfTI files and the model checkpoint exist at the specified paths.")
        except Exception as e:
            print(f"An unexpected error occurred during inference: {e}")
    else:
        print(f"Current mode selected is '{args.mode}'. Inference logic will not be executed.")
        print("To run inference, please use: python scripts/infer.py inference --t1c_path /path/to/T1C.nii.gz [other_inference_args]")
