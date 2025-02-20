import os
import json
import albumentations as A
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from model import SolarModel
from data_gen import get_training_augmentation, get_validation_augmentation, Dataset


with open("config.json", "r") as f:
    CONFIG = json.load(f)

# Set CUDA devices
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG["cuda_visible_devices"]

# Load datasets
CLASSES = ["solar_panel"]

train_dataset = Dataset(
    CONFIG["x_train_dir"], CONFIG["y_train_dir"],
    augmentation=get_training_augmentation(CONFIG["tile_size"]),
    classes=CLASSES
)

valid_dataset = Dataset(
    CONFIG["x_valid_dir"], CONFIG["y_valid_dir"],
    augmentation=get_validation_augmentation(CONFIG["tile_size"]),
    classes=CLASSES
)

test_dataset = Dataset(
    CONFIG["x_test_dir"], CONFIG["y_test_dir"],
    augmentation=get_validation_augmentation(CONFIG["tile_size"]),
    classes=CLASSES
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

# Define model
T_MAX = CONFIG["epochs"] * len(train_loader)

if CONFIG["mode"] == "train":
    model = SolarModel("FPN", "resnext50_32x4d", in_channels=3, out_classes=CONFIG["out_classes"], T_max=T_MAX)
elif CONFIG["mode"] == "finetune":
    model = SolarModel.load_from_checkpoint(
        checkpoint_path=CONFIG["finetune_checkpoint"],
        arch='FPN',
        encoder_name='resnext50_32x4d',
        in_channels=3,
        out_classes=CONFIG["out_classes"],
        T_max=T_MAX
    )

# Callbacks and Trainer
checkpoint_callback = ModelCheckpoint(
    dirpath=CONFIG["trained_models_dir"],
    filename='{epoch}-{valid_dataset_iou:.4f}',
    monitor='valid_dataset_iou',
    save_top_k=-1,
    mode='min'
)

csv_logger = CSVLogger(save_dir=CONFIG["logs_dir"], name="training_logs")

trainer = pl.Trainer(
    max_epochs=CONFIG["epochs"],
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
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

# Evaluate the model
model.eval()
valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
print(valid_metrics)

test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
print(test_metrics)

# Save results
with open(CONFIG["results_file"], "w") as f:
    f.write("Validation Metrics:\n")
    f.write(json.dumps(valid_metrics, indent=4) + "\n\n")
    f.write("Test Metrics:\n")
    f.write(json.dumps(test_metrics, indent=4) + "\n")
