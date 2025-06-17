import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def extract_layers_features(model, loader, model_path):
    """
    Extracts features from the model's layers.
    """
    torch.manual_seed(0)
    model.eval()
    n_test_batches = len(loader)

    logits_outputs = []
    ultimate_decoder_outputs = []
    penultimate_decoder_outputs = []
    bottleneck_features = []
    labels = []

    layer_to_list = {
        'logits_feature_map': logits_outputs,
        'ultimate_decoder_feature_map': ultimate_decoder_outputs,
        'penultimate_decoder_feature_map': penultimate_decoder_outputs,
        'bottleneck_feature_map': bottleneck_features,
    }

    with tqdm(total=n_test_batches, desc=f'Extracting features using model: {model_path}', unit='batch', leave=False) as pbar:
        with torch.no_grad():
            for step, y in enumerate(loader):
                y['imgs'], y['masks'] = y['imgs'].to('cuda'), y['masks'].to('cuda')
                
                with torch.amp.autocast('cuda'):
                    output = model(y['imgs'])

                    for layer_name in layer_to_list.keys():
                        if layer_name not in output:
                            raise ValueError(f"Layer '{layer_name}' not found in model output.")

                        feature_map = output[layer_name].cpu().numpy()
                        feature_map_gap = np.mean(feature_map, axis=(2, 3, 4))
                        layer_to_list[layer_name].append(feature_map_gap)

                    labels.append(y['data_type'][0])

                pbar.update(1)

    logits_outputs = np.concatenate(logits_outputs, axis=0)
    ultimate_decoder_outputs = np.concatenate(ultimate_decoder_outputs, axis=0)
    penultimate_decoder_outputs = np.concatenate(penultimate_decoder_outputs, axis=0)
    bottleneck_features = np.concatenate(bottleneck_features, axis=0)

    print(f"Logits Outputs shape: {logits_outputs.shape}")
    print(f"Ultimate Decoder Outputs shape: {ultimate_decoder_outputs.shape}")
    print(f"Penultimate Decoder Outputs shape: {penultimate_decoder_outputs.shape}")
    print(f"Bottleneck Features shape: {bottleneck_features.shape}")

    return {'logits': logits_outputs,
            'ultimate_decoder': ultimate_decoder_outputs,
            'penultimate_decoder': penultimate_decoder_outputs,
            'bottleneck': bottleneck_features,
            'labels': labels}
    
def tsne_population_wise_analysis(data_dict, color_palette = {'GLI': 'red', 'SSA': 'blue', 'PED': 'green', 'MEN': 'purple', "MET": 'orange'}, figsize=(12, 10)):
    """
    Performs t-SNE analysis on the population-wise features.
    """
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
    # scaler = StandardScaler()

    features_dict = {key: value for i, (key, value) in enumerate(data_dict.items()) if key != 'labels'}
    labels = data_dict['labels']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.ravel()
    
    for i, (name, features) in enumerate(features_dict.items()):
        ax = axes[i]
        # features_flattened = scaler.fit_transform(features)
        features_tsne = tsne.fit_transform(features)
        
        sns.scatterplot(
            x=features_tsne[:, 0], y=features_tsne[:, 1],
            hue=labels,
            palette=color_palette,
            alpha=0.8,
            ax=ax
        )
        ax.set_title(f't-SNE Plot of {name}')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend(title='Labels', loc='best')
        
    plt.tight_layout()
    plt.show()

def get_region_stats(logits):
    """
    Calculates statistics for each tumor region.
    """
    tumor_regions = logits[1:4]  # WT, TC, ET channels
    features_list = []
    for region in tumor_regions:
        region_values = region.flatten()
        features = [
            np.mean(region_values),
            np.median(region_values),
            np.percentile(region_values, 75) - np.percentile(region_values, 25),  # IQR
            np.mean(np.abs(region_values - np.median(region_values))),  # MAD (Mean Absolute Deviation)
        ]
        features_list.append(features)

    return features_list  # List of 3 lists (regions), each one carries 4 numbers for each region

def extract_region_features(model, loader, model_path):
    """
    Extracts features from the model's regions.
    """
    torch.manual_seed(0)
    model.eval()
    n_test_batches = len(loader)

    features_all = []
    labels_all = []

    with tqdm(total=n_test_batches, desc=f'Extracting features per region using model: {model_path}', unit='batch', leave=False) as pbar:
        with torch.no_grad():
            for step, y in enumerate(loader):
                y['imgs'], y['masks'] = y['imgs'].to('cuda'), y['masks'].to('cuda')
                
                with torch.amp.autocast('cuda'):
                    output = model(y['imgs'])
                    logits = output['logits_feature_map'].cpu().numpy()
                    WT_features, TC_features, ET_features = get_region_stats(logits.squeeze(0))

                    features_all.append(WT_features)
                    labels_all.append(1)  # WT label
            
                    features_all.append(TC_features)
                    labels_all.append(2)  # TC label
            
                    features_all.append(ET_features)
                    labels_all.append(3)  # ET label

                pbar.update(1)

    features_all = np.array(features_all)
    labels_all = np.array(labels_all)

    print(f"All features shape: {features_all.shape}")
    print(f"Regions labels shape: {labels_all.shape}")

    return {'features': features_all,
            'labels': labels_all}

def tsne_region_wise_analysis(data_dict, population, ax):
    """
    Performs t-SNE analysis on the region-wise features.
    """
    features = data_dict['features']
    labels = data_dict['labels']
    
    tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
    tsne_result = tsne.fit_transform(features)

    scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.set_title(f"t-SNE for {population} Population")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(handles=scatter.legend_elements()[0], labels=['WT', 'TC', 'ET'])


######### 1. Example Usage for Population Wise Analysis #########

## dyn_unet.py ## -> return dict of features from different layers
# return { 'logits_feature_map': output1,
#          'ultimate_decoder_feature_map' : x12,
#          'penultimate_decoder_feature_map' : x11,
#          'bottleneck_feature_map' : x6}

# def prepare_test_loader(args):
#     test_datasets = []
#     split_ratio = {'training': 0.71, 'validation': 0.09, 'testing': 0.2}
    
#     for i, data_dir in enumerate(args['data_dirs']):
#         patient_lists = os.listdir( data_dir )
#         patient_lists.sort()
#         total_patients = len(patient_lists)

#         random.seed(5)
#         random.shuffle(patient_lists)
    
#         train_split = int(split_ratio['training'] * total_patients)
#         val_split = int(split_ratio['validation'] * total_patients)    
#         test_patient_lists = patient_lists[train_split + val_split :]

#         test_patient_lists.sort()
#         print(test_patient_lists)
        
#         test_datasets.append(test_patient_lists)
        
#     combined_testDataset = list(chain.from_iterable(test_datasets))
#     print(f'Number of combined testing samples without aggregation', len(combined_testDataset))
    
#     testDataset = CustomDataset3D( args['data_dirs'], combined_testDataset, mode='testing')
    
#     testLoader = DataLoader(
#         testDataset, batch_size=args['test_batch_size'], num_workers=args['workers'], prefetch_factor=2,
#         pin_memory=True, shuffle=False)

#     return testLoader

# args = {
#     'workers': 2,
#     'test_batch_size': 1,
#     'data_dirs': ["/kaggle/input/bratsglioma/Training/", "/kaggle/input/bratsafrica24/", "/kaggle/input/bratsped/Training/", "/kaggle/input/bratsmen/", "/kaggle/input/bratsmet24/"]
# }

# testLoader = prepare_test_loader(args)

# path = Path('/kaggle/input/final_kd_student_model/pytorch/default/2/Final_KD_Student_Model_v2.pth')
# model = load_model(path)

# features = extract_layers_features(model, testLoader, path)
# tsne_population_wise_analysis(features)
######### Example Usage for Population Wise Analysis #########



######### 2. Example Usage for Region Wise Analysis#########
# def prepare_test_loaders(args):
#     # random.seed(5)
#     test_datasets = {}
#     split_ratio = {'training': 0.71, 'validation': 0.09, 'testing': 0.2}
    
#     for i, data_dir in enumerate(args['data_dirs']):
#         patient_lists = os.listdir( data_dir )
#         data_type = patient_lists[0].split('-')[1]
#         patient_lists.sort()
#         total_patients = len(patient_lists)

#         random.seed(5)
#         random.shuffle(patient_lists)
    
#         train_split = int(split_ratio['training'] * total_patients)
#         val_split = int(split_ratio['validation'] * total_patients)
    
#         test_patient_lists = patient_lists[train_split + val_split :]
#         test_patient_lists.sort()

#         print(test_patient_lists)
        
#         print(f'Number of testing samples in {data_dir.split("/")[3]} DataSet: {len(test_patient_lists)} ')
#         test_datasets[data_type] = test_patient_lists
    
#     GLI_testDataset = CustomDataset3D( args['data_dirs'], test_datasets['GLI'], mode='testing')
#     SSA_testDataset = CustomDataset3D( args['data_dirs'], test_datasets['SSA'], mode='testing')
#     PED_testDataset = CustomDataset3D( args['data_dirs'], test_datasets['PED'], mode='testing')
#     MEN_testDataset = CustomDataset3D( args['data_dirs'], test_datasets['MEN'], mode='testing')
#     MET_testDataset = CustomDataset3D( args['data_dirs'], test_datasets['MET'], mode='testing')
    
#     GLI_testLoader = DataLoader(
#         GLI_testDataset, batch_size=args['test_batch_size'], num_workers=args['workers'], prefetch_factor=2,
#         pin_memory=True, shuffle=False)

#     SSA_testLoader = DataLoader(
#         SSA_testDataset, batch_size=args['test_batch_size'], num_workers=args['workers'], prefetch_factor=2,
#         pin_memory=True, shuffle=False)

#     PED_testLoader = DataLoader(
#         PED_testDataset, batch_size=args['test_batch_size'], num_workers=args['workers'], prefetch_factor=2,
#         pin_memory=True, shuffle=False)

#     MEN_testLoader = DataLoader(
#         MEN_testDataset, batch_size=args['test_batch_size'], num_workers=args['workers'], prefetch_factor=2,
#         pin_memory=True, shuffle=False)

#     MET_testLoader = DataLoader(
#         MET_testDataset, batch_size=args['test_batch_size'], num_workers=args['workers'], prefetch_factor=2,
#         pin_memory=True, shuffle=False)

#     return GLI_testLoader, SSA_testLoader, PED_testLoader, MEN_testLoader, MET_testLoader

# index_to_data = { 0: 'GLI',
#                   1: 'SSA',
#                   2: 'PED',
#                   3: 'MEN',
#                   4: 'MET' }       


# GLI_testLoader, SSA_testLoader, PED_testLoader, MEN_testLoader, MET_testLoader = prepare_test_loaders(args)

# path = Path('/kaggle/input/final_kd_student_model/pytorch/default/2/Final_KD_Student_Model_v2.pth')
# model = load_model(path)

# fig, axes = plt.subplots(1, 5, figsize=(30, 10))
# for i, (loader, ax) in enumerate(zip([GLI_testLoader, SSA_testLoader, PED_testLoader, MEN_testLoader, MET_testLoader], axes)):
#     data = extract_features(model, loader, path)
#     tsne_analysis(data, index_to_data[i], ax)
#     print(f"Finished {index_to_data[i]} dataset!")

# plt.tight_layout()
# plt.show()
######### Example Usage for Region Wise Analysis#########