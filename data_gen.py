# code edited from https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/examples/cars%20segmentation%20(camvid).ipynb

# This code defines a custom PyTorch Dataset class for image segmentation.
# It loads image-mask pairs, applies preprocessing (like converting images from BGR to RGB and binarizing the masks),
# and supports data augmentation using the albumentations library. 
# Two augmentation pipelines are defined: strong random transformations for training and simple padding for validation. 
# This setup helps improve model robustness and ensures that images and masks stay properly aligned during training.

import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)

    """

    CLASSES = ["solar_panel"]

    def __init__(
        self,
        images_dir,
        masks_dir,
        classes=None,
        augmentation=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [
            os.path.join(masks_dir, 'm_' + image_id[2:])  # Replace "i_" with "m_"
            for image_id in self.ids
        ]
        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = [255]

        self.augmentation = augmentation

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        # BGR-->RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype("float")
        mask = (mask == 255).astype('float32')  
        mask = np.expand_dims(mask, axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)

    def __len__(self):
        return len(self.ids)
    

# Data augmentation
def get_training_augmentation(tile_size):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.2, rotate_limit=15, shift_limit=0.1, p=1, border_mode=0
        ),
        A.PadIfNeeded(min_height=tile_size, min_width=tile_size, always_apply=True),
        A.RandomCrop(height=tile_size, width=tile_size, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)

# Validation set images augmentation
def get_validation_augmentation(tile_size):
    """Ensure validation images are correctly sized."""
    test_transform = [
        A.PadIfNeeded(min_height=tile_size, min_width=tile_size, always_apply=True),
    ]
    return A.Compose(test_transform)