Shashank Pathak
Target:
	1.	Get the basic code setup (train/test loops) and dataloader. 
	2.	Use basic transforms (ToTensor, Normalize)
	3.	15 epochs
	4.	Results:
	1.	Parameters:  194,884
	2.	Best Train Accuracy: 99.03
	3.	Best Test Accuracy: 98.78
	5.	Analysis:
	1.	The model is quite large for MNIST task 
	2.	 Overfitting


Target:
	1.	Decrease the number of parameters (Decrease Kernels, Add GAP Layer)
	2.	Add regularisation(Dropout Layers) and 
	3.	Results:
	1.	Parameters: 6,950
	2.	Best Train Accuracy: 98.86 %
	3.	Best Test Accuracy: 99.25%
	4.	Analysis:
	1.	The model performs decently. Potential to improve
	2.	Underfitting



Target:
	1.	As the previous model was undefitting increase the capacity 
	1.	Results:
	1.	Parameters: 9,060
	2.	Best Train Accuracy: 98.81
	3.	Best Test Accuracy: 99.39
	2.	Analysis:
	1. Capcity can be increased further
	2. Transforms can be applied to make train set more like test set




Target:
	1.	Get the basic code setup and finalise the base skeleton with Batch-norm and Gap layers
	2.	Results:
	1.	Parameters: 9,880
	2.	Best Train Accuracy: 98.33
	3.	Best Test Accuracy: 99.43
	3.	Analysis:
	1.	The can be trained for more epochs, has potential 
