# EVA 7 Session 6 Assignment - Experimentation with Normalizations


## Goals
1.Change the dataset to CIFAR10
2. Make this network:
C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
3 .Keep the parameter count less than 50000
4 .Try and add one layer to another
5.Max Epochs is 20
6.You are making 3 versions of the above code (in each case achieve above 70% accuracy):
7.Network with Group Normalization
8. Network with Layer Normalization
9.Network with Batch Normalization
10.Share these details
11.Training accuracy for 3 models
12.Test accuracy for 3 models
13.Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images. 
14.write an explanatory README file that explains:
15.what is your code all about,
16.your findings for normalization techniques,
17.add all your graphs
18.your collection-of-misclassified-images 
19.Upload your complete assignment on GitHub and share the link on LMS


## Normalization
Normalization refers to the process of standardizing inputs to a neural network. Different normalization techniques can standardize different segments of the input. 

Batch normalization - standardizes each mini-batch input to a layer.\
Layer normalization - normalizes the activations along the feature/channel direction instead of the batch direction. Removes depdency on batch size. \
Group normalization - similar to layer normalization, however, it divides the features/channels into groups and normalizes each group separately.

## Normalization Experiments Conducted

Batch size - 128 \
Dropout value - 0.05 \
Optimizer - SGD with learning rate 0.01 and momentum 0.9 \
Scheduler - None \
Number of groups used for Group Normalization - 2 for 1st Convolution block, 4 for the rest, No normalization applied to last layer

|Regularization|	Best Train Accuracy	| Best Test Accuracy |	Best Test Loss| L1 Factor |
|------------|-----------------|-------------|----------|---|
|Batch Normalization|85.25 (Epoch 2)|93.48 (Epoch 1)|0.2468|0.01
|Layer Normalization|98.07 (Epoch 20)|99.34 (Epoch 19)|0.0215|0
|Group Normalization|98.59|99.58|0.0151|0

## Observations

Session 5 assignment included training a network with Batch Normalization included. In this assignment (Session 6), L1 normalization is added to the network in additiona to BN. Results indicate instability in the network during the training process, with jumps observed in the progression of Test Accuracy through epochs. Training accuracy eventually settles to around 83%.

Both layer and group normalization perform in a more "expected" fashion with both training and test accuracies showing gradual improvement, and test accuracies for both models eventually exceeding 99%.

Overfitting is not observed for both models with layer and group normalizations, with test accuracies staying above training accuracies for all later epochs (>10).

[model.py](https://github.com/arghya05/erav1/blob/master/Session%206-%20Batch%20layer%20group%20normalization/model.py) contains the NN model which can accept multiple normalization selection inputs through an added input variable.

## Test & Training Accuracies & Loss
![Session 6- Batch layer group normalization/LossAccuracyGraphs.png)

Please note that the curves for Batch Normalization above include L1 regularization with a weight of 0.01.
Instabilities in the learning process are evident from the jumps in the test loss and test accuracy curves for Batch Normalization.

## Misclassified Images

### Batch Normalization with L1 Regularization
![BatchNorm](https://github.com/arghya05/erav1/blob/master/Session%206-%20Batch%20layer%20group%20normalization/BatchNorm_L1Reg_misclassified.png)

### Layer Normalization
![LayerNorm](https://github.com/arghya05/erav1/blob/master/Session%206-%20Batch%20layer%20group%20normalization/LayerNorm_misclassified.png)

### Group Normalization
![GroupNorm](https://github.com/arghya05/erav1/blob/master/Session%206-%20Batch%20layer%20group%20normalization/LayerNorm_misclassified.png)

