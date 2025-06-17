import os
import random
from itertools import chain
from torch.utils.data import DataLoader
from data.make_dataset import CustomDataset3D
import argparse

def stratify_batches(dataset_lists: list, batch_size: int = 5) -> list:
    """
    Combines patient lists from multiple datasets for training in a round-robin fashion
    to ensure balanced representation across batches for training.

    Args:
        dataset_lists (list): A list of lists, where each inner list contains patient IDs
                              for a specific tumor type.
        batch_size (int): The desired batch size. This function expects batch_size
                          to be equal to the number of datasets to ensure one sample
                          from each dataset per effective "step" in the combined list.

    Returns:
        list: A stratified list of patient IDs, effectively interleaved.
    """
    max_len = max(len(dataset) for dataset in dataset_lists)

    if batch_size != len(dataset_lists):
        raise ValueError(
            "Batch size must equal the number of datasets for this function to "
            "interleave correctly (e.g., one sample from each type per batch)."
        )

    combined_patient_ids = []

    for i in range(0, max_len, batch_size):
        for j in range(batch_size):
            index = (i + j) % max_len
            batch_segment = [dataset[index % len(dataset)] for dataset in dataset_lists]
            combined_patient_ids.extend(batch_segment)

    return combined_patient_ids


def prepare_data_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepares training, validation, and test data loaders based on provided arguments.

    Args:
        args (argparse.Namespace): An object containing configuration parameters such as:
                                   'data_dirs', 'train_batch_size', 'val_batch_size',
                                   'test_batch_size', 'workers'.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: trainLoader, valLoader, testLoader
    """
    train_datasets_patients, val_datasets_patients, test_datasets_patients = [], [], []
    split_ratio = {'training': 0.71, 'validation': 0.09, 'testing': 0.2}

    print("Splitting datasets and preparing patient lists...")
    for i, data_dir in enumerate(args.data_dirs.values()):
        patient_lists_all = os.listdir(data_dir)
        patient_lists_all.sort()
        total_patients = len(patient_lists_all)

        random.seed(5)
        random.shuffle(patient_lists_all)

        train_split_idx = int(split_ratio['training'] * total_patients)
        val_split_idx = int(split_ratio['validation'] * total_patients)

        train_patient_lists = patient_lists_all[:train_split_idx]
        val_patient_lists = patient_lists_all[train_split_idx : train_split_idx + val_split_idx]
        test_patient_lists = patient_lists_all[train_split_idx + val_split_idx :]

        train_patient_lists.sort()
        val_patient_lists.sort()
        test_patient_lists.sort()

        train_datasets_patients.append(train_patient_lists)
        val_datasets_patients.append(val_patient_lists)
        test_datasets_patients.append(test_patient_lists)

        data_dir_name = data_dir.name
        print(f'Number of training samples in {data_dir_name} Dataset: {len(train_patient_lists)}')
        print(f'Number of validation samples in {data_dir_name} Dataset: {len(val_patient_lists)}')
        print(f'Number of testing samples in {data_dir_name} Dataset: {len(test_patient_lists)} ')

    combined_train_patient_ids = stratify_batches(train_datasets_patients, batch_size=args.train_batch_size)
    combined_val_patient_ids = list(chain.from_iterable(val_datasets_patients))
    combined_test_patient_ids = list(chain.from_iterable(test_datasets_patients))

    print(f'Total combined training samples: {len(combined_train_patient_ids)}')
    print(f'Total combined validation samples: {len(combined_val_patient_ids)}')
    print(f'Total combined testing samples: {len(combined_test_patient_ids)}')

    train_dataset = CustomDataset3D(args.data_dirs, combined_train_patient_ids, mode='training')
    val_dataset = CustomDataset3D(args.data_dirs, combined_val_patient_ids, mode='validation')
    test_dataset = CustomDataset3D(args.data_dirs, combined_test_patient_ids, mode='testing')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.workers,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.workers,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.workers,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=False
    )

    return train_loader, val_loader, test_loader 