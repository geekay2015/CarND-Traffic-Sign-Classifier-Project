
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



```python
unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)
plt.grid()
plt.title("Train Dataset Sign Counts")
plt.show()

unique_test, counts_test = np.unique(y_test, return_counts=True)
plt.bar(unique_test, counts_test)
plt.grid()
plt.title("Test Dataset Sign Counts")
plt.show()

unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
plt.bar(unique_valid, counts_valid)
plt.grid()
plt.title("Valid Dataset Sign Counts")
plt.show()

```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_0.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_1.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_8_2.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from math import ceil
from sklearn.utils import shuffle

# Convert to grayscale
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

X_valid_rgb = X_valid
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)

print('Train RGB shape:', X_train_rgb.shape)
print('Train Grayscale shape', X_train_gry.shape)

print('Test RGB shape:', X_test_rgb.shape)
print('Test Grayscale shape', X_test_gry.shape)

```

    Train RGB shape: (34799, 32, 32, 3)
    Train Grayscale shape (34799, 32, 32, 1)
    Test RGB shape: (12630, 32, 32, 3)
    Test Grayscale shape (12630, 32, 32, 1)



```python
X_train = X_train_gry
X_test = X_test_gry
X_valid = X_valid_gry
```


```python
image_depth_channels = X_train.shape[3]

# print(image_depth_channels)

number_to_stop = 8
figures = {}
random_signs = []
for i in range(number_to_stop):
    index = random.randint(0, n_train-1)
    labels[i] = name_values[y_train[index]][1].decode('ascii')
    figures[i] = X_train[index].squeeze()
    random_signs.append(index)
    
# print(random_signs)
plot_figures(figures, 4, 2, labels)
```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_14_0.png)



```python
import cv2

more_X_train = []
more_y_train = []

more2_X_train = []
more2_y_train = []

new_counts_train = counts_train
for i in range(n_train):
    if(new_counts_train[y_train[i]] < 3000):
        for j in range(3):
            dx, dy = np.random.randint(-1.7, 1.8, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            dst = cv2.warpAffine(X_train[i], M, (X_train[i].shape[0], X_train[i].shape[1]))
            dst = dst[:,:,None]
            more_X_train.append(dst)
            more_y_train.append(y_train[i])

            random_higher_bound = random.randint(27, 32)
            random_lower_bound = random.randint(0, 5)
            points_one = np.float32([[0,0],[32,0],[0,32],[32,32]])
            points_two = np.float32([[0, 0], [random_higher_bound, random_lower_bound], [random_lower_bound, 32],[32, random_higher_bound]])
            M = cv2.getPerspectiveTransform(points_one, points_two)
            dst = cv2.warpPerspective(X_train[i], M, (32,32))
            more2_X_train.append(dst)
            more2_y_train.append(y_train[i])
            
            tilt = random.randint(-12, 12)
            M = cv2.getRotationMatrix2D((X_train[i].shape[0]/2, X_train[i].shape[1]/2), tilt, 1)
            dst = cv2.warpAffine(X_train[i], M, (X_train[i].shape[0], X_train[i].shape[1]))
            more2_X_train.append(dst)
            more2_y_train.append(y_train[i])
            
            new_counts_train[y_train[i]] += 2
    
more_X_train = np.array(more_X_train)
more_y_train = np.array(more_y_train)
X_train = np.concatenate((X_train, more_X_train), axis=0)
y_train = np.concatenate((y_train, more_y_train), axis=0)

more2_X_train = np.array(more_X_train)
more2_y_train = np.array(more_y_train)
more2_X_train = np.reshape(more2_X_train, (np.shape(more2_X_train)[0], 32, 32, 1))
X_train = np.concatenate((X_train, more2_X_train), axis=0)
y_train = np.concatenate((y_train, more2_y_train), axis=0)

X_train = np.concatenate((X_train, X_valid), axis=0)
y_train = np.concatenate((y_train, y_valid), axis=0)
```


```python
figures1 = {}
labels = {}
figures1[0] = X_train[n_train+1].squeeze()
labels[0] = y_train[n_train+1]
figures1[1] = X_train[0].squeeze()
labels[1] = y_train[0]

plot_figures(figures1, 1, 2, labels)
```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_16_0.png)



```python
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print("New Dataset Size : {}".format(X_train.shape[0]))

unique, counts = np.unique(y_train, return_counts=True)
plt.bar(unique, counts)
plt.grid()
plt.title("Train Dataset Sign Counts")
plt.show()

unique, counts = np.unique(y_test, return_counts=True)
plt.bar(unique, counts)
plt.grid()
plt.title("Test Dataset Sign Counts")
plt.show()

unique, counts = np.unique(y_valid, return_counts=True)
plt.bar(unique, counts)
plt.grid()
plt.title("Valid Dataset Sign Counts")
plt.show()
```

    New Dataset Size : 89860



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_1.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_2.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_17_3.png)



```python
def normalize(im):
    return -np.log(1/((1 + im)/257) - 1)

# X_train_normalized = normalize(X_train)
# X_test_normalized = normalize(X_test)
X_train_normalized = X_train/127.5-1
X_test_normalized = X_test/127.5-1

number_to_stop = 8
figures = {}
count = 0
for i in random_signs:
    labels[count] = name_values[y_train[i]][1].decode('ascii')
    figures[count] = X_train_normalized[i].squeeze()
    count += 1;
    
plot_figures(figures, 4, 2, labels)
```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_18_0.png)



```python
X_train = X_train_normalized
X_test = X_test_normalized
```

### Model Architecture


```python
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    print(x.get_shape())
    return tf.nn.relu(x)

def LeNet(x):
    mu = 0
    sigma = 0.1
    
    W_one = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth_channels, 6), mean = mu, stddev = sigma))
    b_one = tf.Variable(tf.zeros(6))
    layer_one = conv2d(x, W_one, b_one, 1)
    
    layer_one = tf.nn.max_pool(layer_one, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(layer_one.get_shape())
    print()
    
    W_two = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    b_two = tf.Variable(tf.zeros(16))
    layer_two = conv2d(layer_one, W_two, b_two, 1)

    layer_two = tf.nn.max_pool(layer_two, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(layer_two.get_shape())
    print()
    
    W_two_a = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 412), mean = mu, stddev = sigma))
    b_two_a = tf.Variable(tf.zeros(412))
    layer_two_a = conv2d(layer_two, W_two_a, b_two_a, 1)
    print(layer_two_a.get_shape())
    print() 

    flat = flatten(layer_two_a)

    W_three = tf.Variable(tf.truncated_normal(shape=(412, 122), mean = mu, stddev = sigma))
    b_three = tf.Variable(tf.zeros(122))
    layer_three = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, W_three), b_three))
    layer_three = tf.nn.dropout(layer_three, keep_prob)
    
    W_four = tf.Variable(tf.truncated_normal(shape=(122, 84), mean = mu, stddev = sigma))
    b_four = tf.Variable(tf.zeros(84))
    layer_four = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer_three, W_four), b_four))
    layer_four = tf.nn.dropout(layer_four, keep_prob)
    
    W_five = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    b_five = tf.Variable(tf.zeros(43))
    layer_five = tf.nn.bias_add(tf.matmul(layer_four, W_five), b_five)
    
    return layer_five

x = tf.placeholder(tf.float32, (None, 32, 32, image_depth_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

keep_prob = tf.placeholder(tf.float32)

print("Done")
```

    Done


### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
EPOCHS = 27
BATCH_SIZE = 100

rate = 0.0009

logits = LeNet(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = rate) 
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

    (?, 28, 28, 6)
    (?, 14, 14, 6)
    
    (?, 10, 10, 16)
    (?, 5, 5, 16)
    
    (?, 1, 1, 412)
    (?, 1, 1, 412)
    



```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    validation_accuracy_figure = []
    test_accuracy_figure = []
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        validation_accuracy_figure.append(validation_accuracy)
        
        test_accuracy = evaluate(X_train, y_train)
        test_accuracy_figure.append(test_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

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



```python
plt.plot(validation_accuracy_figure)
plt.title("Test Accuracy")
plt.show()

plt.plot(validation_accuracy_figure)
plt.title("Validation Accuracy")
plt.show()
```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_26_0.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_26_1.png)



```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    train_accuracy = evaluate(X_train, y_train)
    print("Train Accuracy = {:.3f}".format(train_accuracy))
    
    valid_accuracy = evaluate(X_valid, y_valid)
    print("Valid Accuracy = {:.3f}".format(valid_accuracy))    
    
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Train Accuracy = 1.000
    Valid Accuracy = 0.994
    Test Accuracy = 0.949


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import glob
import cv2

my_images = sorted(glob.glob('mysigns/*.png'))
my_labels = np.array([1, 22, 35, 15, 37, 18])

figures = {}
labels = {}
my_signs = []
index = 0
for my_image in my_images:
    img = cv2.cvtColor(cv2.imread(my_image), cv2.COLOR_BGR2RGB)
    my_signs.append(img)
    figures[index] = img
    labels[index] = name_values[my_labels[index]][1].decode('ascii')
    index += 1

plot_figures(figures, 3, 2, labels)
```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_30_0.png)



```python
my_signs = np.array(my_signs)
my_signs_gray = np.sum(my_signs/3, axis=3, keepdims=True)
my_signs_normalized = my_signs_gray/127.5-1

number_to_stop = 6
figures = {}
labels = {}
for i in range(number_to_stop):
    labels[i] = name_values[my_labels[i]][1].decode('ascii')
    figures[i] = my_signs_gray[i].squeeze()
    
plot_figures(figures, 3, 2, labels)
```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_31_0.png)


### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./lenet")
    my_accuracy = evaluate(my_signs_normalized, my_labels)
    print("My Data Set Accuracy = {:.3f}".format(my_accuracy))
```

    My Data Set Accuracy = 1.000


### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

my_single_item_array = []
my_single_item_label_array = []

for i in range(6):
    my_single_item_array.append(my_signs_normalized[i])
    my_single_item_label_array.append(my_labels[i])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
#         saver = tf.train.import_meta_graph('./lenet.meta')
        saver.restore(sess, "./lenet")
        my_accuracy = evaluate(my_single_item_array, my_single_item_label_array)
        print('Image {}'.format(i+1))
        print("Image Accuracy = {:.3f}".format(my_accuracy))
        print()
```

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

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

k_size = 5
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=k_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_signs_normalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_signs_normalized, keep_prob: 1.0})

    for i in range(6):
        figures = {}
        labels = {}
        
        figures[0] = my_signs[i]
        labels[0] = "Original"
        
        for j in range(k_size):
            labels[j+1] = 'Guess {} : ({:.0f}%)'.format(j+1, 100*my_top_k[0][i][j])
            figures[j+1] = X_valid[np.argwhere(y_valid == my_top_k[1][i][j])[0]].squeeze()
            
        plot_figures(figures, 1, 6, labels)
```


![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_0.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_1.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_2.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_3.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_4.png)



![png](Traffic_Sign_Classifier-Copy2_files/Traffic_Sign_Classifier-Copy2_38_5.png)


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.


```python

```
