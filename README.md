# MPS: MRI Pulse Sequence Classification Model

### model parameter

https://drive.google.com/drive/folders/176T0DLI5cb2faLDeFzmYNFvcFP0uT2IY?usp=drive_link

## Data

### Train
Brats 2018 (Train & Validation) + Brats 2020 (Train)
<img width="697" alt="image" src="https://github.com/user-attachments/assets/e733656e-bb7a-4ad5-b386-fea2eb5409f1" />


### Test
Brats 2020 (Validation) + Severence Brain MRI Data (Glioma + Metastasis)
<img width="695" alt="image" src="https://github.com/user-attachments/assets/74d2a217-7975-4f7b-8846-f8ffe2f66ff9" />


### Transform

**Base**

`CenterSpatialCropd(keys=["image"], roi_size=(128, 128))`

`ResizeD(keys=["image"], spatial_size=(224, 224))`

`NormalizeIntensityD(keys=["image"], nonzero=False, channel_wise=True)`



**Augment**

`RandGaussianNoised(keys=["image"], prob=0.45, std=0.09)`

`RandAdjustContrastD(keys=["image"], gamma=(0.5, 1.5), prob=0.45)`

`RandGaussianSmoothD(keys=["image"], sigma_x=(0.5, 1.5), prob=0.45)`

`RandAffined(keys=["image"], prob=0.5, rotate_range=0.1, scale_range=0.1)`

`RandHistogramShiftd(keys=["image"], num_control_points=10, prob=0.4)`



## Train

`optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )`
        
`criterion = nn.NLLLoss()`

`scheduler = CosineAnnealingLR(optimizer, len(train_loader) * self.num_epochs)`

epochs = 50 / Early Stopping at 23 Epochs

![image](https://github.com/user-attachments/assets/b7420dbb-7797-4bd5-8e4e-2aeab462940c)



## Performance
Loss : 0.0803 / Acc : 0.9779

![image](https://github.com/user-attachments/assets/6ab80fdf-0c78-4cb4-b1e7-3cdb276e357b)


### Accuracy per class
T1 : 100%

T2 : 99.55%

T1 Contrast-Enhenced : 93.39%

FLAIR : 98.57%
