from train import *
from metadata import get_metadata

wsi_files_with_metadata,filtered_metadata = get_metadata()
train_dataset, val_dataset = create_train_val_datasets(wsi_repo_id,wsi_files_with_metadata,filtered_metadata,0.2,82,32,224,0,True)
num_classes = len(train_dataset.icd10_labels)

run_training_pipeline(train_dataset,val_dataset,num_classes,3)
