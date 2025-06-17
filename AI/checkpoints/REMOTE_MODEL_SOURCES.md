# Remote Model Checkpoint Sources

This directory is intended to store trained model checkpoints. However, due to their large size, the actual checkpoint files are **not** committed directly to this Git repository. Instead, they are hosted externally.

Below are the sources for the teacher models used in this project, typically located on Kaggle datasets. When running this project on Kaggle, these paths will be automatically mounted. For local execution, you will need to ensure these files are downloaded and placed into the corresponding local input directories as configured in `config/default_config.py`.

## Teacher Model Checkpoints

| Tumor Type | Dataset Link (URL)                                     | Epoch Checkpoint |
| :--------- | :--------------------------------------------------------- | :--------------- |
| GLI        | `https://www.kaggle.com/datasets/ahmedfb/gliomateachernewlabels` | 100              |
| SSA        | `https://www.kaggle.com/datasets/ahmedkaggle3/africanewlabels` | 67               |
| PED        | `https://www.kaggle.com/datasets/ahmedkaggle3/pednewlabel` | 99               |
| MEN        | `https://www.kaggle.com/datasets/ahmedelzayatmain/meningiomateachernewlabels` | 85               |
| MET        | `https://www.kaggle.com/datasets/mariemmagdy4/met-teacher-new-labels` | 100              |

---

## Student Model Initial Checkpoint

If a pre-trained student model checkpoint is used for resuming training or transfer learning, its expected path should also be specified in `config/default_config.py`.