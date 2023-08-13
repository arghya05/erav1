# Assignment 12 Submission: `way to lightning`

<br>

- [try on spaces](https://huggingface.co/spaces/anantgupta129/miniresnet_gradcam) 
- [main repo](https://github.com/anantgupta129/TorcHood) 
- [training notebook](https://colab.research.google.com/github/anantgupta129/ERA-V1/blob/main/session12/notebooks/s12_train.ipynb)

## Results

```bash
[x] Accuracy of ::
	[*] airplane : 94 %
	[*] automobile : 100 %
	[*]     bird : 81 %
	[*]      cat : 84 %
	[*]     deer : 95 %
	[*]      dog : 75 %
	[*]     frog : 84 %
	[*]    horse : 95 %
	[*]     ship : 100 %
	[*]    truck : 91 %
```

## Grad Cam Results

![](./images/cam.png)

## Misclassified Images

![](./images/misclf.png)

## Learning Curve

![](./images/learning_curve.png)

## Model Summary

```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CustomResNet                             [1, 10]                   --
├─Sequential: 1-1                        [1, 64, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 64, 32, 32]           1,728
│    └─BatchNorm2d: 2-2                  [1, 64, 32, 32]           128
│    └─ReLU: 2-3                         [1, 64, 32, 32]           --
│    └─Dropout2d: 2-4                    [1, 64, 32, 32]           --
├─Sequential: 1-2                        [1, 128, 16, 16]          --
│    └─Conv2d: 2-5                       [1, 128, 32, 32]          73,728
│    └─MaxPool2d: 2-6                    [1, 128, 16, 16]          --
│    └─BatchNorm2d: 2-7                  [1, 128, 16, 16]          256
│    └─ReLU: 2-8                         [1, 128, 16, 16]          --
│    └─Dropout2d: 2-9                    [1, 128, 16, 16]          --
│    └─ResBlock: 2-10                    [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-1                  [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-2             [1, 128, 16, 16]          256
│    │    └─Dropout2d: 3-3               [1, 128, 16, 16]          --
│    │    └─Conv2d: 3-4                  [1, 128, 16, 16]          147,456
│    │    └─BatchNorm2d: 3-5             [1, 128, 16, 16]          256
│    │    └─Dropout2d: 3-6               [1, 128, 16, 16]          --
├─Sequential: 1-3                        [1, 256, 8, 8]            --
│    └─Conv2d: 2-11                      [1, 256, 16, 16]          294,912
│    └─MaxPool2d: 2-12                   [1, 256, 8, 8]            --
│    └─BatchNorm2d: 2-13                 [1, 256, 8, 8]            512
│    └─ReLU: 2-14                        [1, 256, 8, 8]            --
│    └─Dropout2d: 2-15                   [1, 256, 8, 8]            --
├─Sequential: 1-4                        [1, 512, 4, 4]            --
│    └─Conv2d: 2-16                      [1, 512, 8, 8]            1,179,648
│    └─MaxPool2d: 2-17                   [1, 512, 4, 4]            --
│    └─BatchNorm2d: 2-18                 [1, 512, 4, 4]            1,024
│    └─ReLU: 2-19                        [1, 512, 4, 4]            --
│    └─Dropout2d: 2-20                   [1, 512, 4, 4]            --
│    └─ResBlock: 2-21                    [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-7                  [1, 512, 4, 4]            2,359,296
│    │    └─BatchNorm2d: 3-8             [1, 512, 4, 4]            1,024
│    │    └─Dropout2d: 3-9               [1, 512, 4, 4]            --
│    │    └─Conv2d: 3-10                 [1, 512, 4, 4]            2,359,296
│    │    └─BatchNorm2d: 3-11            [1, 512, 4, 4]            1,024
│    │    └─Dropout2d: 3-12              [1, 512, 4, 4]            --
├─MaxPool2d: 1-5                         [1, 512, 1, 1]            --
├─Conv2d: 1-6                            [1, 10, 1, 1]             5,120
==========================================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
Total mult-adds (M): 379.27
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 4.65
Params size (MB): 26.29
Estimated Total Size (MB): 30.96
==========================================================================================
```


## Training Logs

```bash
Epoch 0/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.43it/s v_num: 0 train/loss: 1.578          
                                                                               train/acc: 0.396                    
Epoch 1/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:21 • 0:00:00 4.57it/s v_num: 0 train/loss: 1.23 train/acc:
                                                                               0.542 val/loss: 1.371 val/acc: 0.51 
Epoch 2/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:23 • 0:00:00 4.26it/s v_num: 0 train/loss: 1.004          
                                                                               train/acc: 0.64 val/loss: 1.053     
                                                                               val/acc: 0.631                      
Epoch 3/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.32it/s v_num: 0 train/loss: 0.782          
                                                                               train/acc: 0.693 val/loss: 0.942    
                                                                               val/acc: 0.68                       
Epoch 4/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.44it/s v_num: 0 train/loss: 0.702          
                                                                               train/acc: 0.747 val/loss: 0.762    
                                                                               val/acc: 0.739                      
Epoch 5/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.43it/s v_num: 0 train/loss: 0.609          
                                                                               train/acc: 0.81 val/loss: 0.623     
                                                                               val/acc: 0.789                      
Epoch 6/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.39it/s v_num: 0 train/loss: 0.6 train/acc: 
                                                                               0.792 val/loss: 0.582 val/acc: 0.801
Epoch 7/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.45it/s v_num: 0 train/loss: 0.556          
                                                                               train/acc: 0.795 val/loss: 0.504    
                                                                               val/acc: 0.831                      
Epoch 8/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:23 • 0:00:00 4.31it/s v_num: 0 train/loss: 0.484          
                                                                               train/acc: 0.827 val/loss: 0.447    
                                                                               val/acc: 0.847                      
Epoch 9/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.38it/s v_num: 0 train/loss: 0.491          
                                                                               train/acc: 0.839 val/loss: 0.437    
                                                                               val/acc: 0.85                       
Epoch 10/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.43it/s v_num: 0 train/loss: 0.45          
                                                                                train/acc: 0.83 val/loss: 0.386    
                                                                                val/acc: 0.869                     
Epoch 11/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:23 • 0:00:00 4.27it/s v_num: 0 train/loss: 0.393         
                                                                                train/acc: 0.869 val/loss: 0.415   
                                                                                val/acc: 0.861                     
Epoch 12/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.46it/s v_num: 0 train/loss: 0.417         
                                                                                train/acc: 0.875 val/loss: 0.374   
                                                                                val/acc: 0.872                     
Epoch 13/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.49it/s v_num: 0 train/loss: 0.4 train/acc:
                                                                                0.869 val/loss: 0.353 val/acc:     
                                                                                0.878                              
Epoch 14/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:23 • 0:00:00 4.29it/s v_num: 0 train/loss: 0.413         
                                                                                train/acc: 0.863 val/loss: 0.326   
                                                                                val/acc: 0.891                     
Epoch 15/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.45it/s v_num: 0 train/loss: 0.342         
                                                                                train/acc: 0.893 val/loss: 0.315   
                                                                                val/acc: 0.896                     
Epoch 16/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.44it/s v_num: 0 train/loss: 0.337         
                                                                                train/acc: 0.887 val/loss: 0.318   
                                                                                val/acc: 0.894                     
Epoch 17/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.35it/s v_num: 0 train/loss: 0.335         
                                                                                train/acc: 0.893 val/loss: 0.296   
                                                                                val/acc: 0.9                       
Epoch 18/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.47it/s v_num: 0 train/loss: 0.298         
                                                                                train/acc: 0.896 val/loss: 0.277   
                                                                                val/acc: 0.908                     
Epoch 19/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.37it/s v_num: 0 train/loss: 0.231         
                                                                                train/acc: 0.911 val/loss: 0.275   
                                                                                val/acc: 0.906                     
Epoch 20/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:23 • 0:00:00 4.23it/s v_num: 0 train/loss: 0.313         
                                                                                train/acc: 0.869 val/loss: 0.275   
                                                                                val/acc: 0.908                     
Epoch 21/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:22 • 0:00:00 4.42it/s v_num: 0 train/loss: 0.218         
                                                                                train/acc: 0.935 val/loss: 0.251   
                                                                                val/acc: 0.916                     
Epoch 22/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:23 • 0:00:00 4.23it/s v_num: 0 train/loss: 0.239         
                                                                                train/acc: 0.938 val/loss: 0.246   
                                                                                val/acc: 0.919                     
Epoch 23/23 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 98/98 0:00:23 • 0:00:00 4.29it/s v_num: 0 train/loss: 0.201         
                                                                                train/acc: 0.92 val/loss: 0.233    
                                                                                val/acc: 0.921     
```
