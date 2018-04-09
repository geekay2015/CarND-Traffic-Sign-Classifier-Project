
# Gangadhar's Traffic Sign Recognition using Neural Network

##Build a Traffic Sign Recognition Project

**The goals / steps of this project are the following:

Load the data set (see below for links to the project data set)
Explore, summarize and visualize the data set
Design, train and test a model architecture
Use the model to make predictions on new images
Analyze the softmax probabilities of the new images
Summarize the results with a written report


# Rubric Points

## Here I will consider the rubric points individually and describe how I addressed each point in my implementation. This is a summary. Continue further to read each in depth.

1. Files submitted, from everything I read a HTML file, notebook, and write up file is required. So this will meet requirements.
2. Dataset summary & visualization, the rubric is referring to explaining the size of the dataset and shape of the data therein. It also would like visual explorations.
3 .Design & test model: which includes preprocessing, model architecture, training, and solution. This was basically given to us for the most part in the with LeNet. The preprocessing from what I have read is many common tasks especially in the paper referenced by the instructors. It outlined the types of data preprocessing and I have tried to implement as much I could time permitting. This includes changing from 3 color channels to 1 so grayscale and then also normalizing the image data so it is smaller numbers. I also did some random augmentation mostly slight edging of the image in random directions, or tilting and then adding those new images to the data set and redistributing it to the train and validation sets while leaving the test set alone. For training again this was basically given using the AdamOptimizer. It worked really well so I didn't change it from the last quiz before this project. The more important parts of the training in the instance I think is the epoch which was 27 and batch size was 158. I also used a learning rate of 0.00097 because it gave good results. Lastly in regards to the solution, or model design I used the default given to me except I did add two drops outs and adjusted the size of the layers to better represent the actual data since it is 32x32 and not 28x28. I also added another convolution.
4. Test model on new images, I found new images on the internet and tried to find images that were already classified out of the 43 classes. It wasn't difficult, but at first I did try images that were very difficult to classify and it didn't do that well. After I found images of signs that were severely damaged it identified the images fairly well. I did scale my images perfectly to 32x32 as well which is probably some what limiting to real scenarios. Finally I also show the probabilities reduced for softmax probabilities and also individual performance along with total performance for all.

# Data Set Summary & Exploration

1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I used the numpy library to calculate summary statistics of the traffic signs data set:
The size of training set is 34799
The size of test set is 12630
The shape of a traffic sign image is (32, 32, 3)
The number of unique classes/labels in the data set is 43
2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set. It pulls in a random set of eight images and labels them with the correct names in reference with the csv file to their respective id's.


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_7_0.png)


## Data Distribution to check the potential pitfall
After this point I also detail the dataset structure by plotting the occurrence of each image class to get an idea of how the data is distributed. This can help understand where potential pitfalls could occur if the dataset isn't uniform in terms of a baseline occurrence.


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_0.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_1.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_2.png)

##Design and Test a Model Architecture

1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth, sixth, seventh, eighth, ninth, and tenth code cell of the IPython notebook.

At first I tried to convert it to YUV as that was what the technical paper described that was authored by Pierre Sermanet and Yann LeCun. I had difficulty getting this working at so I skipped over this in order to meet my time requirements.

The next step, I decided to convert the images to grayscale because in the technical paper it outlined several steps they used to achieve 99.7%. I assume this works better because the excess information only adds extra confusion into the learning process. After the grayscale I also normalized the image data because I've read it helps in speed of training and performance because of things like resources. Also added additional images to the datasets through randomized modifications.

Here is an example of a traffic sign images that were randomly selected.

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_14_0.png)

Here is a look at the normalized images. Which should look identical, but for some small random alterations such as opencv affine and rotation.

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_16_0.png)

2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

At first I wasn't going to do this part because I didn't have enough time, but I took an extra day and decided to turn this in on the 28th rather then the 27th. I did a few random alterations to the images and saved multiple copies of them depending on the total images in the dataset class type.

Here is an example of 1 image I changed at random. More can be seen further in the document, but the original is on the right and the randomized opencv affine change is on the left. Small rotations are also visible further along as stated.

![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_1.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_2.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_3.png)


I increased the train dataset size to 89860 and also merged and then remade another validation dataset. Now no image class in the train set has less then 1000 images. Test


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_18_0.png)



### Model Architecture

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.



    (?, 28, 28, 6)
    (?, 14, 14, 6)
    
    (?, 10, 10, 16)
    (?, 5, 5, 16)
    
    (?, 1, 1, 412)
    (?, 1, 1, 412)
    




    Training...
    
    EPOCH 1 ...
    Test Accuracy = 0.907
    Validation Accuracy = 0.790
    
    EPOCH 2 ...
    Test Accuracy = 0.968
    Validation Accuracy = 0.909
    
    EPOCH 3 ...
    Test Accuracy = 0.985
    Validation Accuracy = 0.949
    
    EPOCH 4 ...
    Test Accuracy = 0.987
    Validation Accuracy = 0.959
    
    EPOCH 5 ...
    Test Accuracy = 0.993
    Validation Accuracy = 0.968
    
    EPOCH 6 ...
    Test Accuracy = 0.993
    Validation Accuracy = 0.969
    
    EPOCH 7 ...
    Test Accuracy = 0.996
    Validation Accuracy = 0.975
    
    EPOCH 8 ...
    Test Accuracy = 0.997
    Validation Accuracy = 0.981
    
    EPOCH 9 ...
    Test Accuracy = 0.998
    Validation Accuracy = 0.980
    
    EPOCH 10 ...
    Test Accuracy = 0.998
    Validation Accuracy = 0.985
    
    EPOCH 11 ...
    Test Accuracy = 0.998
    Validation Accuracy = 0.986
    
    EPOCH 12 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.989
    
    EPOCH 13 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.987
    
    EPOCH 14 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.989
    
    EPOCH 15 ...
    Test Accuracy = 0.996
    Validation Accuracy = 0.988
    
    EPOCH 16 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.987
    
    EPOCH 17 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.990
    
    EPOCH 18 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.990
    
    EPOCH 19 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.990
    
    EPOCH 20 ...
    Test Accuracy = 1.000
    Validation Accuracy = 0.990
    
    EPOCH 21 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.990
    
    EPOCH 22 ...
    Test Accuracy = 0.998
    Validation Accuracy = 0.986
    
    EPOCH 23 ...
    Test Accuracy = 1.000
    Validation Accuracy = 0.992
    
    EPOCH 24 ...
    Test Accuracy = 1.000
    Validation Accuracy = 0.986
    
    EPOCH 25 ...
    Test Accuracy = 1.000
    Validation Accuracy = 0.993
    
    EPOCH 26 ...
    Test Accuracy = 0.999
    Validation Accuracy = 0.991
    
    EPOCH 27 ...
    Test Accuracy = 1.000
    Validation Accuracy = 0.994
    
    Model saved





![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_26_0.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_26_1.png)




    Train Accuracy = 1.000
    Valid Accuracy = 0.994
    Test Accuracy = 0.949


## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_30_0.png)





![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_31_0.png)


### Predict the Sign Type for Each Image



    My Data Set Accuracy = 1.000


### Analyze Performance



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
    


### Output Top 5 Softmax Probabilities For Each Image Found on the Web



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_0.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_1.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_2.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_3.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_4.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_5.png)

