# Assignment 5
## Divyam Malay Shah (divyam096@gmail.com)
<br>
<br>



### Model 1:
#### Target
Default model from the previous session:

Rotation based image augmentation(+- 7 degrees)
#### Results:
Parameters: 13.8k

Best Train Accuracy: 99.03

Best Test Accuracy: 99.45 (15th Epoch)
#### Analysis:
This model serves as a starting point for our problem.

Existing model violates the 10k parameters limit
 
<br>
 
### Model 2:
#### Target
Decrease the number of parameters by reducing the number of kernels to 10 from 16 for the second convolution block.

Adding an LR scheduler. 
#### Results
Parameters: 9560

Best Train Accuracy: 98.61

Best Test Accuracy: 99.19 (15th Epoch)
#### Analysis
Model begins overfitting 8th epoch onwards.

Accuracy stagnates at 99.19.

<br>

### Model 3
#### Target
Add more augmentations to make learning more challenging.
#### Results
Parameters: 9560

Best Train Accuracy: 98.54

Best Test Accuracy: 99.20 (15th Epoch)
#### Analysis
Slightly better accuracy than model 3. 

Training saturates 8th epoch onwards(stagnation of train and test accuracy).

<br>

### Model 4:
#### Target
Apply image augmentations to test set (Test Time Augmentations)
#### Results
Parameters: 9560

Best Train Accuracy: 98.70

Best Test Accuracy: 99.36 (15th Epoch)
#### Analysis
Model is an underfit. Train and Test accuracies are consistently increasing.

The model can be improved further by trying out more augmentations and scheduling strategies like one cycle learning.
