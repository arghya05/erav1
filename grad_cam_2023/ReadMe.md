# Grad Cam Assingment 
## 

## Goals:
1. train resnet18 for 20 epochs on the CIFAR10 dataset
2. show loss curves for test and train datasets
3. show a gallery of 10 misclassified images
4. show gradcamLinks to an external site. output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬
5. Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
Train for 20 epochs
6. Get 10 misclassified images
7. Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
8. Apply these transforms while training:
9. RandomCrop(32, padding=4)
10. CutOut(16x16)


## Torch CV Utils

Additional support/wrapper repo was updated with augmentations required for this assignment and LR range test helper files, it can be found here: [Torch_CV_Utils](https://github.com/a-pujahari/Torch_CV_Utils)

## Notebook

The notebook for this assignment can be accessed here: [https://github.com/arghya05/erav1/blob/master/grad_cam_2023/Grad_cam_assingment_final.ipynb)

## Analysis

Epochs - 24   
Best Training Accuracy - 94.59% (24th Epoch)     
Best Testing Accuracy -  90.63% (24th Epoch)    

Optimizer - Adam
Scheduler - OneCycleLR with pct_start = 0.2 (~5/24) since max_lr is required at Epoch 5, out of 24 total epochs

## Learning Rate Range Test Curve
![LR_test](https://github.com/arghya05/erav1/blob/master/grad_cam_2023/LR_test.png)

## Loss Curves
![Loss and Accuracy](https://github.com/arghya05/erav1/blob/master/grad_cam_2023/Loss%20and%20accuracy.png)

## Sample Misclassified Images
![Misclassified](https://github.com/arghya05/erav1/blob/master/grad_cam_2023/misclassified.png)

## GradCam Output
![gradcam1](https://github.com/arghya05/erav1/blob/master/grad_cam_2023/gradcam1.png)
![gradcam2](https://github.com/arghya05/erav1/blob/master/grad_cam_2023/gradcam2.png)

## Training Logs 

EPOCH: 1
  0%|          | 0/98 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
Loss=1.8810629844665527 Batch_id=97 LR=0.00037 Accuracy=44.06: 100%|██████████| 98/98 [00:15<00:00,  6.24it/s]

Test set: Average loss: 0.0040, Accuracy: 4566/10000 (45.66%)

EPOCH: 2
Loss=1.7911227941513062 Batch_id=97 LR=0.00107 Accuracy=60.97: 100%|██████████| 98/98 [00:15<00:00,  6.25it/s]

Test set: Average loss: 0.0037, Accuracy: 6156/10000 (61.56%)

EPOCH: 3
Loss=1.7530632019042969 Batch_id=97 LR=0.00189 Accuracy=69.41: 100%|██████████| 98/98 [00:15<00:00,  6.23it/s]

Test set: Average loss: 0.0037, Accuracy: 6238/10000 (62.38%)

EPOCH: 4
Loss=1.6939486265182495 Batch_id=97 LR=0.00251 Accuracy=72.38: 100%|██████████| 98/98 [00:15<00:00,  6.22it/s]

Test set: Average loss: 0.0036, Accuracy: 6455/10000 (64.55%)

EPOCH: 5
Loss=1.6740559339523315 Batch_id=97 LR=0.00268 Accuracy=75.99: 100%|██████████| 98/98 [00:15<00:00,  6.27it/s]

Test set: Average loss: 0.0034, Accuracy: 7676/10000 (76.76%)

EPOCH: 6
Loss=1.6826244592666626 Batch_id=97 LR=0.00266 Accuracy=78.70: 100%|██████████| 98/98 [00:15<00:00,  6.22it/s]

Test set: Average loss: 0.0034, Accuracy: 7360/10000 (73.60%)

EPOCH: 7
Loss=1.6443225145339966 Batch_id=97 LR=0.00260 Accuracy=80.54: 100%|██████████| 98/98 [00:15<00:00,  6.23it/s]

Test set: Average loss: 0.0034, Accuracy: 7565/10000 (75.65%)

EPOCH: 8
Loss=1.660696268081665 Batch_id=97 LR=0.00250 Accuracy=81.77: 100%|██████████| 98/98 [00:15<00:00,  6.22it/s]

Test set: Average loss: 0.0033, Accuracy: 7983/10000 (79.83%)

EPOCH: 9
Loss=1.6140000820159912 Batch_id=97 LR=0.00238 Accuracy=82.87: 100%|██████████| 98/98 [00:15<00:00,  6.24it/s]

Test set: Average loss: 0.0033, Accuracy: 7945/10000 (79.45%)

EPOCH: 10
Loss=1.6176668405532837 Batch_id=97 LR=0.00222 Accuracy=84.34: 100%|██████████| 98/98 [00:15<00:00,  6.22it/s]

Test set: Average loss: 0.0033, Accuracy: 8120/10000 (81.20%)

EPOCH: 11
Loss=1.6293424367904663 Batch_id=97 LR=0.00205 Accuracy=85.53: 100%|██████████| 98/98 [00:15<00:00,  6.23it/s]

Test set: Average loss: 0.0033, Accuracy: 8304/10000 (83.04%)

EPOCH: 12
Loss=1.5810519456863403 Batch_id=97 LR=0.00185 Accuracy=86.44: 100%|██████████| 98/98 [00:15<00:00,  6.25it/s]

Test set: Average loss: 0.0032, Accuracy: 8361/10000 (83.61%)

EPOCH: 13
Loss=1.5949000120162964 Batch_id=97 LR=0.00164 Accuracy=87.61: 100%|██████████| 98/98 [00:15<00:00,  6.26it/s]

Test set: Average loss: 0.0032, Accuracy: 8526/10000 (85.26%)

EPOCH: 14
Loss=1.5578066110610962 Batch_id=97 LR=0.00143 Accuracy=89.05: 100%|██████████| 98/98 [00:15<00:00,  6.31it/s]

Test set: Average loss: 0.0032, Accuracy: 8653/10000 (86.53%)

EPOCH: 15
Loss=1.5586748123168945 Batch_id=97 LR=0.00121 Accuracy=89.49: 100%|██████████| 98/98 [00:15<00:00,  6.27it/s]

Test set: Average loss: 0.0032, Accuracy: 8693/10000 (86.93%)

EPOCH: 16
Loss=1.5678797960281372 Batch_id=97 LR=0.00099 Accuracy=90.53: 100%|██████████| 98/98 [00:15<00:00,  6.25it/s]

Test set: Average loss: 0.0032, Accuracy: 8815/10000 (88.15%)

EPOCH: 17
Loss=1.532879114151001 Batch_id=97 LR=0.00079 Accuracy=91.43: 100%|██████████| 98/98 [00:15<00:00,  6.23it/s]

Test set: Average loss: 0.0031, Accuracy: 8868/10000 (88.68%)

EPOCH: 18
Loss=1.5467233657836914 Batch_id=97 LR=0.00059 Accuracy=92.33: 100%|██████████| 98/98 [00:15<00:00,  6.25it/s]

Test set: Average loss: 0.0031, Accuracy: 8928/10000 (89.28%)

EPOCH: 19
Loss=1.540803074836731 Batch_id=97 LR=0.00042 Accuracy=92.92: 100%|██████████| 98/98 [00:15<00:00,  6.22it/s]

Test set: Average loss: 0.0031, Accuracy: 9000/10000 (90.00%)

EPOCH: 20
Loss=1.5131628513336182 Batch_id=97 LR=0.00028 Accuracy=93.71: 100%|██████████| 98/98 [00:15<00:00,  6.25it/s]

Test set: Average loss: 0.0031, Accuracy: 9016/10000 (90.16%)

EPOCH: 21
Loss=1.5177315473556519 Batch_id=97 LR=0.00016 Accuracy=94.13: 100%|██████████| 98/98 [00:15<00:00,  6.23it/s]

Test set: Average loss: 0.0031, Accuracy: 9035/10000 (90.35%)

EPOCH: 22
Loss=1.5127336978912354 Batch_id=97 LR=0.00007 Accuracy=94.41: 100%|██████████| 98/98 [00:15<00:00,  6.21it/s]

Test set: Average loss: 0.0031, Accuracy: 9055/10000 (90.55%)

EPOCH: 23
Loss=1.5245492458343506 Batch_id=97 LR=0.00002 Accuracy=94.51: 100%|██████████| 98/98 [00:15<00:00,  6.23it/s]

Test set: Average loss: 0.0031, Accuracy: 9056/10000 (90.56%)

EPOCH: 24
Loss=1.496837854385376 Batch_id=97 LR=0.00000 Accuracy=94.59: 100%|██████████| 98/98 [00:15<00:00,  6.30it/s]

Test set: Average loss: 0.0031, Accuracy: 9063/10000 (90.63%)
