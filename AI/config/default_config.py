import argparse
from pathlib import Path

def get_args():
    """
    Parses command-line arguments for training, testing, and inference configuration.

    Returns:
        argparse.Namespace: An object containing all configuration arguments.
    """
    parser = argparse.ArgumentParser(description="Configuration for training and testing.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--workers', type=int, default=2,
                        help='Number of data loading workers.')
    parser.add_argument('--train_batch_size', type=int, default=5,
                        help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=2,
                        help='Batch size for validation.')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='Batch size for testing.')
    parser.add_argument('--base_data_dir', type=str, default='datasets/raw/',
                        help='Base directory where raw datasets for different tumor types are located.')

    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operation mode (train, test, or inference)')

    # --- Training sub-parser ---
    train_parser = subparsers.add_parser('train', help='Arguments for training mode',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('--epochs', type=int, default=2, required=True,
                              help='Number of training epochs.')
    train_parser.add_argument('--learning_rate', type=float, default=1e-3,
                              help='Initial learning rate.')
    train_parser.add_argument('--weight_decay', type=float, default=1e-5,
                              help='Weight decay for optimizer.')
    train_parser.add_argument('--gli_teacher_model_path', type=str,
                              default='checkpoints/teacher_models/GLI_Teacher_Checkpoint.pth',
                              help='Path to the GLI teacher model checkpoint (.pth file).')
    train_parser.add_argument('--ssa_teacher_model_path', type=str,
                              default='checkpoints/teacher_models/SSA_Teacher_Checkpoint.pth',
                              help='Path to the SSA teacher model checkpoint (.pth file).')
    train_parser.add_argument('--ped_teacher_model_path', type=str,
                              default='checkpoints/teacher_models/PED_Teacher_Checkpoint.pth',
                              help='Path to the PED teacher model checkpoint (.pth file).')
    train_parser.add_argument('--men_teacher_model_path', type=str,
                              default='checkpoints/teacher_models/MEN_Teacher_Checkpoint.pth',
                              help='Path to the MEN teacher model checkpoint (.pth file).')
    train_parser.add_argument('--met_teacher_model_path', type=str,
                              default='checkpoints/teacher_models/MET_Teacher_Checkpoint.pth',
                              help='Path to the MET teacher model checkpoint (.pth file).')
    train_parser.add_argument('--in_checkpoint_dir', type=str, default='checkpoints/student_model/',
                              help='Input directory for student model checkpoints (for resuming training).')
    train_parser.add_argument('--out_checkpoint_dir', type=str, default='checkpoints/student_model/',
                              help='Output directory for saving trained student model checkpoints.')


    # --- Testing sub-parser ---
    test_parser = subparsers.add_parser('test', help='Arguments for testing mode',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    test_parser.add_argument('--student_model_path', type=str, required=True,
                             help='Path to the student model checkpoint (.pth file) to be loaded for testing.')
    test_parser.add_argument('--output_dir', type=str, default='testing/results/',
                             help='Directory to save testing df results.')

    # --- Inference sub-parser ---
    inference_parser = subparsers.add_parser('inference', help='Arguments for inference mode',
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inference_parser.add_argument('--inference_model_path', type=str, required=True,
                                  help='Path to the model checkpoint (.pth file) to be loaded for inference (can be student or teacher).')
    inference_parser.add_argument('--device', type=str, default='cuda',
                                  help='Device to run inference on (e.g., "cuda" or "cpu").')
    inference_parser.add_argument('--sw_batch_size', type=int, default=4,
                                  help='Batch size for sliding window inference.')
    inference_parser.add_argument('--overlap', type=float, default=0.25,
                                  help='Overlap percentage for sliding window inference (0.0 to 1.0).')
    inference_parser.add_argument('--sw_mode', type=str, default='gaussian', choices=['gaussian', 'constant'],
                                  help='Blending mode for sliding window inference (gaussian or constant).')
    inference_parser.add_argument('--t1c_path', type=str, required=True,
                                  help='Path to the T1-contrast enhanced NIfTI file for inference.')
    inference_parser.add_argument('--t1w_path', type=str, required=True,
                                  help='Path to the T1-native NIfTI file for inference.')
    inference_parser.add_argument('--t2f_path', type=str, required=True,
                                  help='Path to the T2-FLAIR NIfTI file for inference.')
    inference_parser.add_argument('--t2w_path', type=str, required=True,
                                  help='Path to the T2-weighted NIfTI file for inference.')
    inference_parser.add_argument('--output_dir', type=str, default='inference/outputs/',
                                  help='Directory to save inference outputs (raw volumes and prediction mask).')


    args = parser.parse_args()

    args.data_dirs = {
        "GLI": Path(args.base_data_dir) / "GLI_data",
        "SSA": Path(args.base_data_dir) / "SSA_data",
        "PED": Path(args.base_data_dir) / "PED_data",
        "MEN": Path(args.base_data_dir) / "MEN_data",
        "MET": Path(args.base_data_dir) / "MET_data"
    }

    if args.mode == 'train':
        args.in_checkpoint_dir = Path(args.in_checkpoint_dir)
        args.out_checkpoint_dir = Path(args.out_checkpoint_dir)
        args.teacher_model_paths = {
            'GLI': Path(args.gli_teacher_model_path),
            'SSA': Path(args.ssa_teacher_model_path),
            'PED': Path(args.ped_teacher_model_path),
            'MEN': Path(args.men_teacher_model_path),
            'MET': Path(args.met_teacher_model_path)
        }
        
    elif args.mode == 'test':
        args.student_model_path = Path(args.student_model_path)
        args.output_dir = Path(args.output_dir)

    else: # mode == inference
        args.inference_model_path = Path(args.inference_model_path)
        args.t1c_path = Path(args.t1c_path)
        args.t1w_path = Path(args.t1w_path)
        args.t2f_path = Path(args.t2f_path)
        args.t2w_path = Path(args.t2w_path)
        args.output_dir = Path(args.output_dir)

    return args