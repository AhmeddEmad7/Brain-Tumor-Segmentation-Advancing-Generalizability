import torch
from testing.metrics import ComputeMetrics
from tqdm import tqdm

def test_net(model, loader):
    torch.manual_seed(0)
    model.eval()
    n_test_batches = len(loader)

    compute_metrics = ComputeMetrics()
    total_metrics = {"WT": {'dice_score': 0, 'hausdorff_distance': 0},
                     "TC": {'dice_score': 0, 'hausdorff_distance': 0},
                     "ET": {'dice_score': 0, 'hausdorff_distance': 0}}

    with tqdm(total=n_test_batches, desc='Testing', unit='batch', leave=False) as pbar:
        with torch.no_grad():
            for step, y in enumerate(loader):
                y['imgs'], y['masks']= y['imgs'].to('cuda'), y['masks'].to('cuda')
                
                with torch.amp.autocast('cuda'):
                    print(f"--- Now patient: {y['patient_id']} ---")
                    output = model(y['imgs'])
                    wt_metrics, tc_metrics, et_metrics = compute_metrics(output['pred'], y['masks'])
                    
                    total_metrics['WT']['dice_score'] += wt_metrics[0].item()
                    total_metrics['WT']['hausdorff_distance'] += wt_metrics[1].item()

                    total_metrics['TC']['dice_score'] += tc_metrics[0].item()
                    total_metrics['TC']['hausdorff_distance'] += tc_metrics[1].item()

                    total_metrics['ET']['dice_score'] += et_metrics[0].item()
                    total_metrics['ET']['hausdorff_distance'] += et_metrics[1].item()
                                    
                pbar.update(1)

        total_metrics['WT']['dice_score'] /= n_test_batches
        total_metrics['WT']['hausdorff_distance'] /= n_test_batches

        total_metrics['TC']['dice_score'] /= n_test_batches
        total_metrics['TC']['hausdorff_distance'] /= n_test_batches

        total_metrics['ET']['dice_score'] /= n_test_batches
        total_metrics['ET']['hausdorff_distance'] /= n_test_batches


        print("************************************************************************")
        print(f"Average Dice Score for WT: {total_metrics['WT']['dice_score']:.4f}")
        print(f"Average Hausdorff Distance for WT: {total_metrics['WT']['hausdorff_distance']:.4f}")

        print("-----------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------")
                                     
        print(f"Average Dice Score for TC: {total_metrics['TC']['dice_score']:.4f}")
        print(f"Average Hausdorff Distance for TC: {total_metrics['TC']['hausdorff_distance']:.4f}")
                              
        print("-----------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------")
                                     
        print(f"Average Dice Score for ET: {total_metrics['ET']['dice_score']:.4f}")
        print(f"Average Hausdorff Distance for ET: {total_metrics['ET']['hausdorff_distance']:.4f}")
        print("************************************************************************")

        model.train()

    return total_metrics