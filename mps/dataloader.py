from monai.transforms import (
    Compose, NormalizeIntensityD, ResizeD, ToTensorD, CenterSpatialCropd, RandGaussianNoised, 
    RandAdjustContrastD, RandGaussianSmoothD, RandAffined, RandHistogramShiftd, RandCoarseDropoutd
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib
import os
import glob

class MRIDataModule:
    class MRIDataset(Dataset):
        def __init__(self, file_paths, labels, transform=None, preload=True):
            self.file_paths = file_paths
            self.labels = labels
            self.transform = transform
            self.preload = preload

            if preload:
                self.images = [self._load_image(f) for f in self.file_paths]
            else:
                self.images = None

        def _load_image(self, path):
            img = nib.load(path).get_fdata(dtype=np.float32)
            img = np.nan_to_num(img)

            slice = np.argmax(np.std(img, axis=(0, 1)))
            img = img[:, :, slice]
            img = img[None, :, :]  # Add channel
            return img

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            img = self.images[idx] if self.preload else self._load_image(self.file_paths[idx])
            label = self.labels[idx]

            data = {"image": img, "label": label}
            if self.transform:
                data = self.transform(data)
            return data

    def __init__(self, data_root, test_root=None, batch_size=16, val_ratio=0.1, preload=True, seed=42):
        self.data_root = data_root
        self.test_root = test_root
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.preload = preload
        self.seed = seed
        self.label_map = {"t1": 0, "t2": 1, "t1ce": 2, "flair": 3}

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.setup()

    def _get_transforms(self, train=True):
        base_transforms = [
            CenterSpatialCropd(keys=["image"], roi_size=(128, 128)), 
            ResizeD(keys=["image"], spatial_size=(224, 224)),
        ]
        
        if train:
            aug_transforms = [ 
                RandGaussianNoised(keys=["image"], prob=0.45, std=0.09),
                RandAdjustContrastD(keys=["image"], gamma=(0.5, 1.5), prob=0.45),
                RandGaussianSmoothD(keys=["image"], sigma_x=(0.5, 1.5), prob=0.45),
                RandAffined(keys=["image"], prob=0.5, rotate_range=0.1, scale_range=0.1),
                RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.4),
                RandCoarseDropoutd(keys=["image"], holes=4, spatial_size=(48, 48), fill_value=0,  prob=0.5),
            ]
            transforms = base_transforms + aug_transforms + [NormalizeIntensityD(keys=["image"], nonzero=False, channel_wise=True),ToTensorD(keys=["image", "label"])]
        else:
            transforms = base_transforms + [NormalizeIntensityD(keys=["image"], nonzero=False, channel_wise=True), ToTensorD(keys=["image", "label"])]
        return Compose(transforms)

    def _gather_paths_and_labels(self, root_dir):
        file_paths, labels = [], []
        for patient_dir in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, patient_dir)
            nii_files = glob.glob(os.path.join(pdir, "*.nii*"))

            for f in nii_files:
                fname = os.path.basename(f).lower()
                for key in self.label_map:
                    if key in fname and not (key == "t1" and "t1ce" in fname):
                        file_paths.append(f)
                        labels.append(self.label_map[key])
                        break
        return file_paths, labels

    def setup(self):
        # Train/Valid
        file_paths, labels = self._gather_paths_and_labels(self.data_root)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            file_paths, labels, test_size=self.val_ratio, stratify=labels, random_state=self.seed
        )

        self.train_dataset = self.MRIDataset(train_paths, train_labels, transform=self._get_transforms(train=True), preload=self.preload)
        self.valid_dataset = self.MRIDataset(val_paths, val_labels, transform=self._get_transforms(train=False), preload=self.preload)

        # Test (optional)
        if self.test_root:
            test_paths, test_labels = self._gather_paths_and_labels(self.test_root)
            self.test_dataset = self.MRIDataset(test_paths, test_labels, transform=self._get_transforms(train=False), preload=self.preload)

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_valid_loader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def get_test_loader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            raise ValueError("No test_root specified. Cannot build test_loader.")


class MRIDataModuleForTest:
    class MRIDataset(Dataset):
        def __init__(self, file_paths, labels, transform=None, preload=True):
            self.file_paths = file_paths
            self.labels = labels
            self.transform = transform
            self.preload = preload

            if preload:
                self.images = [self._load_image(f) for f in self.file_paths]
            else:
                self.images = None

        def _load_image(self, path):
            img = nib.load(path).get_fdata(dtype=np.float32)
            img = np.nan_to_num(img)

            slice = np.argmax(np.std(img, axis=(0, 1)))
            img = img[:, :, slice]
            img = img[None, :, :]  # Add channel
            return img

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            img = self.images[idx] if self.preload else self._load_image(self.file_paths[idx])
            label = self.labels[idx]

            data = {"image": img, "label": label}
            if self.transform:
                data = self.transform(data)
            return data

    def __init__(self, data_root, gd_label='t1ce', batch_size=16, val_ratio=0.1, preload=True, seed=42):
        self.data_root = data_root
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.preload = preload
        self.seed = seed
        self.gd_label = gd_label
        self.label_map = {"t1": 0, "t2": 1, self.gd_label: 2, "flair": 3}

        self.test_dataset = None

        self.setup()

    def _get_transforms(self):
        base_transforms = [
            CenterSpatialCropd(keys=["image"], roi_size=(128, 128)), 
            ResizeD(keys=["image"], spatial_size=(224, 224)),
            NormalizeIntensityD(keys=["image"], nonzero=False, channel_wise=True),
        ]
        transforms = base_transforms + [ToTensorD(keys=["image", "label"])]
        return Compose(transforms)

    def _gather_paths_and_labels(self, root_dir):
        file_paths, labels = [], []
        for patient_dir in sorted(os.listdir(root_dir)):
            pdir = os.path.join(root_dir, patient_dir)
            nii_files = glob.glob(os.path.join(pdir, "*.nii*"))

            for f in nii_files:
                fname = os.path.basename(f).lower()
                for key in self.label_map:
                    if key in fname and not (key == "t1" and self.gd_label in fname):
                        file_paths.append(f)
                        labels.append(self.label_map[key])
                        break
        return file_paths, labels

    def setup(self):
        test_paths, test_labels = self._gather_paths_and_labels(self.data_root)
        self.test_dataset = self.MRIDataset(test_paths, test_labels, transform=self._get_transforms(), preload=self.preload)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)