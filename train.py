import os
import json
import argparse
import albumentations as A
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from cape_town_segmentation.model import SolarModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from cape_town_segmentation.data_gen import get_training_augmentation, get_validation_augmentation, Dataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,5,6,7'

# Paths
CP_DATA_DIR = "/home/as1233/cape_town_segmentation/data/cal_320"

cp_x_train_dir = os.path.join(CP_DATA_DIR, 'train/images')
cp_y_train_dir = os.path.join(CP_DATA_DIR, 'train/masks')

cp_x_valid_dir = os.path.join(CP_DATA_DIR, 'val/images')
cp_y_valid_dir = os.path.join(CP_DATA_DIR, 'val/masks')

cp_x_test_dir = os.path.join(CP_DATA_DIR, 'test/images')
cp_y_test_dir = os.path.join(CP_DATA_DIR, 'test/masks')

cp_tile_size = 320


# Load datasets
CLASSES = ["solar_panel"]


cp_train_dataset = Dataset(
    cp_x_train_dir,
    cp_y_train_dir,
    augmentation=get_training_augmentation(cp_tile_size),
    classes=CLASSES,
)

cp_valid_dataset = Dataset(
    cp_x_valid_dir,
    cp_y_valid_dir,
    augmentation=get_validation_augmentation(cp_tile_size),
    classes=CLASSES,
)

cp_test_dataset = Dataset(
    cp_x_test_dir,
    cp_y_test_dir,
    augmentation=get_validation_augmentation(cp_tile_size),
    classes=CLASSES,
)

cp_train_loader = DataLoader(cp_train_dataset, batch_size=16, shuffle=True, num_workers=4)
cp_valid_loader = DataLoader(cp_valid_dataset, batch_size=16, shuffle=False, num_workers=4)
cp_test_loader = DataLoader(cp_test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Define constants
EPOCHS = 100
T_MAX = EPOCHS * len(cp_train_loader)
OUT_CLASSES = 1

csv_logger = CSVLogger(save_dir="/home/as1233/cape_town_segmentation/logs/cal_320", name='cal_320')

if config["mode"] == "train":
    model = SolarModel("FPN", "resnext50_32x4d", in_channels=3, out_classes=OUT_CLASSES, T_max=T_MAX)
if config["mode"] == "finetune":
    model = SolarModel.load_from_checkpoint(
        checkpoint_path="/home/as1233/cape_town_segmentation/trained_models/cal_320/epoch=39-valid_dataset_iou=0.9293.ckpt",
        arch='FPN',
        encoder_name='resnext50_32x4d',
        in_channels=3,
        out_classes=OUT_CLASSES,
        T_max = T_MAX
    )

# # Move the model to GPU
# model.to('cuda')


checkpoint_callback = ModelCheckpoint(
    dirpath='/home/as1233/cape_town_segmentation/trained_models/cal_320_new',   
    filename='{epoch}-{valid_dataset_iou:.4f}',  
    monitor='valid_dataset_iou',                
    save_top_k=-1,                       
    mode='min',                         
)


trainer = pl.Trainer(
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    devices=-1,  
    accelerator='gpu',  
    strategy='ddp',  
    precision=16,  
    accumulate_grad_batches=4,  
    logger=csv_logger  
)

# Train the model
trainer.fit(
    model,
    train_dataloaders=cp_train_loader,
    val_dataloaders=cp_valid_loader,
)


# print eval metrics 
model.eval()

valid_metrics = trainer.validate(model, dataloaders=cp_valid_loader, verbose=False)
print(valid_metrics)


# print test metrics
test_metrics = trainer.test(model, dataloaders=cp_test_loader, verbose=False)
print(test_metrics)


# with open("/home/as1233/cape_town_segmentation/results/cp_640.txt", "w") as f:
#     f.write("Validation Metrics:\n")
#     f.write(json.dumps(valid_metrics, indent=4) + "\n\n")  # Save as JSON format for readability

#     f.write("Test Metrics:\n")
#     f.write(json.dumps(test_metrics, indent=4) + "\n")