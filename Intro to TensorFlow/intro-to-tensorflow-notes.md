
# Intro to TensorFlow

## Hello, Tensor World!


```python
import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operatin in the session
    output = sess.run(hello_constant)
    print(output)
```

    b'Hello World!'


### Tensor

In TensorFlow, data isn’t stored as integers, floats, or strings. These values are encapsulated in an object called a tensor. In the case of `hello_constant = tf.constant('Hello World!')`, hello_constant is a 0-dimensional string tensor, but tensors come in a variety of sizes as shown below:


```python
# A is a 0-dimensional int32 tensor
A = tf.constant(1234)

# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789])

# C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```

The tensor returned by `tf.constant()` is called a constant tensor, because the value of the tensor never changes.

### Session

TensorFlow’s api is built around the idea of a computational graph, a way of visualizing a mathematical process.  Let’s take the TensorFlow code and turn that into a graph:

![Session](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580feadb_session/session.png)

A "TensorFlow Session", as shown above, is an environment for running a graph. The session is in charge of allocating the operations to GPU(s) and/or CPU(s), including remote machines. Let’s see how you use it:


```python
with tf.Session() as sess:
    output = sess.run(hello_constant)
```

The code has already created the tensor, `hello_constant`, from the previous lines. The next step is to evaluate the tensor in a session.

The code creates a session instance, `sess`, using `tf.Session`. The `sess.run()` function then evaluates the tensor and returns the results.

## TensorFlow Input 

In the last section, a tensor was passed into a session and it returned the result. What if we want to use a non-constant? This is where `tf.placeholder()` and `feed_dict` come into place. In this section, we'll go over the basics of feeding data into TensorFlow.

### tf.placeholder()

Sadly you can’t just set `x` to your dataset and put it in TensorFlow, because over time you'll want your TensorFlow model to take in different datasets with different parameters. You need `tf.placeholder()`!

`tf.placeholder()` returns a tensor that gets its value from data passed to the `tf.session.run()` function, allowing you to set the input right before the session runs.

### Session's feed_dict


```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
    print(output)
```

    Hello World


Use the feed_dict parameter in `tf.session.run()` to set the placeholder tensor. The above example shows the tensor `x` being set to the string `"Hello, world"`. It's also possible to set more than one tensor using `feed_dict` as shown below:


```python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output_x = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    output_y = sess.run(y, feed_dict={x: 'Test String', y: 123, z:45.67})
    print(output_x)
    print(output_y)
```

    Test String
    123


**Note**: If the data passed to the `feed_dict` doesn’t match the tensor type and can’t be cast into the tensor type, you’ll get the error `“ValueError: invalid literal for...”`.

### Quiz


```python
import tensorflow as tf


def run():
    output = None
    x = tf.placeholder(tf.int32)

    with tf.Session() as sess:
        # TODO: Feed the x tensor 123
        output = sess.run(x, feed_dict={x: 123})

    return output

run()
```




    array(123, dtype=int32)



## TensorFlow Math

Getting the input is great, but now you need to use it. We're going to use basic math functions that everyone knows and loves - add, subtract, multiply, and divide - with tensors. (There's many more math functions you can check out in the [documentation](https://www.tensorflow.org/api_docs/python/math_ops/).)

### Addition


```python
x = tf.add(5, 2)  # 7
```

Let's start with the add function. The `tf.add()` function does exactly what you expect it to do. It takes in two numbers, two tensors, or one of each, and returns their sum as a tensor.

### Subraction and Multiplication


```python
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10
```

The `x` tensor will evaluate to `6`, because `10 - 4 = 6`. The `y` tensor will evaluate to `10`, because `2 * 5 = 10`. That was easy!

### Converting Types

It may be necessary to convert between types to make certain operators work together. For example, if you tried the following, it would fail with an exception:


```python
tf.subtract(tf.constant(2.0),tf.constant(1))  
# Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
```

That's because the constant `1` is an integer but the constant `2.0` is a floating point value and subtract expects them to match.

In cases like these, you can either make sure your data is all of the same type, or you can cast a value to another type. In this case, converting the `2.0` to an integer before subtracting, like so, will give the correct result:


```python
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```




    <tf.Tensor 'Sub_3:0' shape=() dtype=int32>



### Quiz

Let's apply what you learned to convert an algorithm to TensorFlow. The code below is a simple algorithm using division and subtraction. Convert the following algorithm in regular Python to TensorFlow and print the results of the session. You can use `tf.constant()` for the values `10`, `2`, and `1`.


```python
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x, y), 1)

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
```

    4.0


## TensorFlow Linear Functions

The most common operation in neural networks is calculating the linear combination of inputs, weights, and biases. As a reminder, we can write the output of the linear operation as:

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a4d8b3_linear-equation/linear-equation.gif)

Here, **W** is a matrix of the weights connecting two layers. The output **y**, the input **x**, and the biases **b** are all vectors.

### Weights and Bias in TensorFlow

The goal of training a neural network is to modify weights and biases to best predict the labels. In order to use weights and bias, you'll need a Tensor that can be modified. This leaves out `tf.placeholder()` and `tf.constant()`, since those Tensors can't be modified. This is where `tf.Variable` class comes in.

### tf.Variable()


```python
x = tf.Variable(5)
```

The `tf.Variable` class creates a tensor with an initial value that can be modified, much like a normal Python variable. This tensor stores its state in the session, so you must initialize the state of the tensor manually. You'll use the `tf.global_variables_initializer()` function to initialize the state of all the Variable tensors:


```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

The `tf.global_variables_initializer()` call returns an operation that will initialize all TensorFlow variables from the graph. You call the operation using a session to initialize all the variables as shown above. Using the `tf.Variable` class allows us to change the weights and bias, but an initial value needs to be chosen.

Initializing the weights with random numbers from a normal distribution is good practice. Randomizing the weights helps the model from becoming stuck in the same place every time you train it. You'll learn more about this in the next lesson, gradient descent.

Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights. We'll use the `tf.truncated_normal()` function to generate random numbers from a normal distribution.

### tf.truncated_normal()


```python
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```

The `tf.truncated_normal()` function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean.

Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias. Let's use the simplest solution, setting the bias to 0.

### tf.zeros()


```python
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))
```

The `tf.zeros()` function returns a tensor with all zeros.

## TensorFlow Softmax

In the Intro to TFLearn lesson we used the softmax function to calculate class probabilities as output from the network. The softmax function squashes it's inputs, typically called **logits** or **logit scores**, to be between 0 and 1 and also normalizes the outputs such that they all sum to 1. This means the output of the softmax function is equivalent to a categorical probability distribution. It's the perfect function to use as the output activation for a network predicting multiple classes.

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58950908_softmax-input-output/softmax-input-output.png)

We're using TensorFlow to build neural networks and, appropriately, there's a function for calculating softmax.


```python
x = tf.nn.softmax([2.0, 1.0, 0.2])
```

Easy as that! `tf.nn.softmax()` implements the softmax function for you. It takes in logits and returns softmax activations.

### Quiz


```python
import tensorflow as tf


def run_2():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    
    # TODO: Calculate the softmax of the logits
    softmax = tf.nn.softmax(logit_data)   
    
    with tf.Session() as sess:
        # TODO: Feed in the logit data
        output = sess.run(softmax, feed_dict={logits: logit_data})

    return output

print(run_2())
```

    [ 0.65900117  0.24243298  0.09856589]


## One-Hot Encoding

Transforming your labels into one-hot encoded vectors is pretty simple with scikit-learn using `LabelBinarizer`. Check it out below!


```python
import numpy as np
from sklearn import preprocessing

# Example labels
labels = np.array([1,5,3,2,1,4,2,1,3])

# Create the encoder
lb = preprocessing.LabelBinarizer()

# Here the encoder finds the classes and assigns one-hot vectors 
lb.fit(labels)

# And finally, transform the labels into one-hot encoded vectors
lb.transform(labels)
```




    array([[1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0]])



## TensorFlow Cross Entropy

In the Intro to TFLearn lesson we discussed using cross entropy as the cost function for classification with one-hot encoded labels. Again, TensorFlow has a function to do the cross entropy calculations for us.

![](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589b18f5_cross-entropy-diagram/cross-entropy-diagram.png)

To create a cross entropy function in TensorFlow, you'll need to use two new functions:

* `tf.reduce_sum()`
* `tf.log()`

### Reduce Sum


```python
x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15
```

The `tf.reduce_sum()` function takes an array of numbers and sums them together.

### Natural Log


```python
l = tf.log(100)  # 4.60517
```

This function does exactly what you would expect it to do. `tf.log()` takes the natural log of a number.

### Quiz

Print the cross entropy using `softmax_data` and `one_hot_encod_label`.


```python
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# TODO: Print cross entropy from session
cross_entropy = -tf.reduce_sum(tf.multiply(one_hot_data, tf.log(softmax_data)))

with tf.Session() as session:
    output = session.run(cross_entropy, feed_dict={one_hot: one_hot_data, softmax: softmax_data})
    print(output)
```

    0.356675


## Mini-batching

In this section, you'll go over what mini-batching is and how to apply it in TensorFlow.

Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.

Mini-batching is computationally inefficient, since you can't calculate the loss simultaneously across all samples. However, this is a small price to pay in order to be able to run the model at all.

It's also quite useful combined with SGD. The idea is to randomly shuffle the data at the start of each epoch, then create the mini-batches. For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.

## Epochs

An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data. 
