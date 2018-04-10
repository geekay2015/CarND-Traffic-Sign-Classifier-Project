
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

# Gangadhar's Deep learning model that learns to recognize traffic signs

**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Highlights of this approach
* The traffic sign dataset used is German Traffic Signs dataset.
* The approach used is deep learning with Convolutional Neural Network (CNN)
* The architecture used will be an adaptation of the LeNet.
* Python is the language used to program this.
* The complete source code can be found [here](https://github.com/geekay2015/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)




### Dataset Exploration
----

### Understand the Dataset

For a classification problem like traffic signs, the first step is to understand that the dataset components
* Features — The actual data that will be fed into the network / model i.e images containing traffic signs
* Labels — The category under which the data that is fed, is likely to fall unde i.e the name of the traffic signs

The per label and across label distribution of data
* Image type — RGB/ Grayscale , 8 bit/16 bit/ 32 bit datatype
* The non uniformity in distribution of data( More images in one traffic sign and lesser in another)
* Even the quality of the images has to come up with a general image pre-processing and augmentation algorithm

The pickled data is a dictionary with 4 key/value pairs:
'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 2D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES

![png](https://github.com/geekay2015/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-Copy2_files/Dataset%20Exploration.png)

#### Dataset Summary
I used the numpy library to calculate summary statistics of the traffic signs data set:
* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization
Here is an exploratory visualization of the data set. 
It pulls in a random set of eight images and labels them with the correct names in reference with the csv file to their respective id's.

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_7_0.png)


## Data Distribution to check the potential pitfall
After this point I also detail the dataset structure by plotting the occurrence of each image class to get an idea of how the data is distributed. This can help understand where potential pitfalls could occur if the dataset isn't uniform in terms of a baseline occurrence.


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_0.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_1.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_2.png)

### Design and Test a Model Architecture
----

#### Preprocessing Techniques useed

My preprocessing pipeline consists of the following steps:
1. Converting to grayscale - This worked well for Sermanet and LeCun as described in their traffic sign classification article. It also helps to reduce training time and can be tried when a GPU is nor available.

2. Normalizing the data to the range (-1,1) - This was done using the line of code X_train_normalized = X_train/127.5-1. I chose to do this mostly because it was suggested in the lessons and it was fairly easy to do. How it helps is a bit nebulous to me, but this site has an explanation, the gist of which is that having a wider distribution in the data would make it more difficult to train using a singlar learning rate. Different features could encompass far different ranges and a single learning rate might make some weights diverge.

Here is an example of a traffic sign images that were randomly selected and gray scaled

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_14_0.png)

Here is a look at the normalized images. Which should look identical, but for some small random alterations such as opencv affine and rotation.

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_16_0.png)

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_1.png)
![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_2.png)
![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_3.png)

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_18_0.png)

#### Model Architecture


![Original LeNet architecture](https://github.com/geekay2015/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-Copy2_files/LeNet5%20architecture.jpg)

My architecture is a deep convolutional neural network inspired by LeNet[1].
I used LeNet as my base model and started playing with it. This is the point where I experimented a lot and tried to tune the parameter in the network.

LeNet5 model has below features:
* convolutional neural network use sequence of 3 layers: convolution, pooling, non-linearity –> This is be the key feature of Deep Learning for images
* uses convolution to extract spatial features
* subsample using spatial average of maps
* non-linearity in the form of tanh or sigmoids
* multi-layer neural network (MLP) as final classifier
* sparse connection matrix between layers to avoid large computational cost

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep        									|
| Fully connected		| input 84, output 43        									|


#### Model Training
* Optimization algorithm
To train the model, I used AdamOptimizer (adaptive moment estimation) which is A Method for Stochastic Optimization. 
I used the  Adam optimization algorithms as it is
    * Straightforward to implement.
    * Computationally efficient.
    * requires Little memory.
    * Well suited for problems that are large in terms of data and/or parameters.
    * Appropriate for non-stationary objectives.
    * Appropriate for problems with very noisy/or sparse gradients.
    * Hyper-parameters have intuitive interpretation and typically require little tuning.

* Batch Size - a batch size of 100, 
* EPOCH - One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE. I used at most 27 epochs
* a learn rate of 0.009

I saved the model with the best validation accuracy. My final model results were:
* training set accuracy of 100.0%
* validation set accuracy of 99.4%
* test set accuracy of 94.9%

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_26_0.png)
![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_26_1.png)


#### Solution Approach
The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.
My final model results were:
* training set accuracy of 100.0%
* validation set accuracy of 99.4%
* test set accuracy of 94.9%

### Testing my model on New images
----
#### Acquiring New Images
The submission includes below new German Traffic signs found on the web, and the images are visualized. 
![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_30_0.png)

Normalized the test images
![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_31_0.png)


#### Performance on New Images
My model predicted 6 out of 6 signs correctly, 
The performance on the new images is compared to the accuracy results of the test set is as below:
   
   Image 1
   Image Accuracy = 1.000

   Image 2
   Image Accuracy = 1.000

   Image 3
   Image Accuracy = 1.000

   Image 4
   Image Accuracy = 1.000

   Image 5
   Image Accuracy = 1.000

   Image 6
   Image Accuracy = 1.000
    
#### Model Certainty - Softmax Probabilities
The top five softmax probabilities of the predictions on the captured images are outputted. 

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_0.png)

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_1.png)

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_2.png)

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_3.png)

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_4.png)

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_5.png)


### References
----
[1] Lecun(1998): Gradient-Based Learning Applied to Document Recognition

[2] Sermanet(2011): Traffic Sign Recognition with Multi-Scale Convolutional Networks

[3] Ciresan (2012): Multi-Column Deep Neural Network for Traffic Sign Classification


### Conclusion
----
This project gave me steep learning curve and for the first time, I experimented different architecture of Convolutional Neural Network. This motivated me to experiment more and think of new architecture also.
