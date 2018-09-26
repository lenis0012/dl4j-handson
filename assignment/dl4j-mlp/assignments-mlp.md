Assignments (MLP)
=================

## 1: ND4J (N-dimensional arrays)
In java, we do not have a poweful method to work with n-dimensional arrays.  
You may think of using `int[][][]...` to represent a n-dimensional array, but this comes with limitations.  
* No easy (and fast) way to perform large scale matrix manipulations.
* Limited to `Integer.MAX_VALUE` entries, which is easy to exceed with many dimensions.
* Allocated within java, which brings overheard during allocation and when sharing with native code.

To get around this problem, ND4J was created.  
ND4J allocates native memory outside of java's heap, this provides some benefits.
* Faster interaction with native code (used for matrix math with native acceleration)
* Not limited to `Integer.MAX_VALUE`
* Fully managed by ND4J, which exposes are robust API for working with N-dimensional arrays

**Asignment: Experiment with n-dimensional arrays using ND4J's API**  
[Online documentation is available.](https://deeplearning4j.org/docs/latest/nd4j-overview)  
[Examples can be found on github](https://github.com/deeplearning4j/dl4j-examples/blob/master/nd4j-examples/src/main/java/org/nd4j/examples/Nd4jEx3_GettingAndSettingSubsets.java)  
You may try out the following to get a feel for the API:
* Create a 3x5 matrix filled with zeros
* Create a 3x5 matrix filled with ones
* Create a matrix with random values
* Modify entries (scalars) from a matrix
* Reschape a 1-dimensional array into a 2-dimensional matrix
* Modify a row from a matrix, and watch the content of the underlying matrix change.

Example:
```java
INDArray array = Nd4j.zeros(rows, cols);
```

## 2: Neural network math
Let's use the power of ND4J to implement a very simple neural net, to solve the Perfect Roommate scenario.  
Your roommate is perfect, because he cooks for you every day.  
On sunny days, he cooks pie, because he is happy, and on rainy days, he cooks a burger. he can also cook chicken.

Build a neural network to predict what your roommate will cook based on the weather.  
The input should be a vector with 2 entries, the amount of sun (0-1) and the amount of rain (0-1).  
You may use the following matrix to accurately predict the outcome:  
```
[1, 0,
 0, 1,
 0, 0]
```

The outcome should be a vector with a length of 3, representing the likelyhood of the given meal.  
0th entry: Pie  
1st entry: Burger  
2nd entry: Chicken

Given all this information, when inserting `[1, 0]`, the output should be `[1, 0, 0]`

## 3: DL4J (Multi-layer Network)
Manually doing the network calculation can get very complicated if you are dealing with more complex networks and optimization algorithms.  
This is why developers commonly use a framework/library to manage this, and we will be using Deeplearning4java.  

[A quickstart guide is available on deeplearning4j's website](https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart)  
[And examples are available on github](https://github.com/deeplearning4j/dl4j-examples) (you should only pay attention to dl4j-examples/dataexamples and dl4j-examples/feedforward/mnist for now)

We are going to solve the MNIST dataset problem with a neural network.  
MNIST is a commonly used dataset of hand written digits, that has been labeled with 50,000 training entries and 10,000 evaluation entries.  
Build a network with 1 input layer, 1 hidden layer and 1 output layer.  
The input layer should be 28 x 28 long, which are the dimensions of the MNIST images.  
The hidden layer should be 1000 long, and the output layer should be 10 long, one neuron for each number (0-9).

Construct a new `MultiLayerNetwork` with a `NeuralNetConfiguration.Builder`.
```java
int rngSeed = 123; // this number allows randomly generated values to be consistant across runs.
NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .list()
                // Add a layer
                .layer(new DenseLayer.Builder()
                        .nIn(rows * cols)
                        .nOut(1000)
                        .build())
                // Add more layers..
                .build();
MultiLayerNetwork model = new MultiLayerNetwork(conf);
```
Use `DenseLayer` for the hidden layer, and `OutputLayer` for the output layer.  
The input layer will be created automatically, using the `nIn` value from the first hidden layer.

Initialize your new neural network by calling `model.init()`.  
Test your network by inserting an input vector (eg. `Nd4j.zeros(rows, cols)`) into the `model.output()` method, then print the result.

The current network is untrained, so don't be surprised if the output is garbage. In a perfect world, an empty input vector should return a 0% probability for all numbers. `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.

## 4: Evaluation
Checking the output of an empty input vector doesn't really give us much insight on the effectiveness of our network.  
Using the `MnistDataSetIterator` as a test dataset, and the `Evaluation` class to evaluate the network with the test data,
we can gain a lot more insight.

Initialize a new DataSetIterator with the Mnist data set.
```java
MnistDataSetIterator testDataSet = new MnistDataSetIterator(batchSize, false, rngSeed);
```
The batch size defines how many entries will be ran though the network in one batch, this is used during training and evaluation, but not when making a regular prediction.
128 is a number that generally works well with this dataset.

Create a new Evaluation instance, and use the test data set to obtain some useful metrics.
```java
Evaluation evaluation = new Evaluation(outputNum);
while(testDataSet.hasNext()) {
    DataSet dataSet = testDataSet.next(); // each dataset contains as many entries as you defined in your batchSize.
    INDArray output = model.output(dataSet.getFeatures()); // run the entire dataset through the network
    evaluation.eval(dataSet.getLabels(), output); // record the metrics of this specific batch
}
```
You can print `evaluation.stats()` to see the metrics you just collected.

## 5: "Learning" with back propagation
Now that we have some numbers, we definitively know that our network is pretty rubbish, let's try to fix that.  
Currently, our network does not learn. Luckily, Mnist contains 50,000 test entries that we can use to train the network.
Training networks is done by supplying a DataSet to the `model.fit()` method, which contains `batchSize` elements.

Training the network can take some time, so we usually limit the amount of batches we use to train, these are called `epochs`.  
For now, let's use `5` epochs.

Create a new dataset iterator for our training data.
```java
MnistDataSetIterator trainDataSet = new MnistDataSetIterator(batchSize, true, rngSeed);
```
Very similar to the test iterator, except we pass true to the `isTrainData` property.

You also need to enable backpropegation.Bbefore calling `.build()` on your network confgiruation, add `.pretrain(false).backprop(true)`

Now, to train the network, we have to create an iterator loop, looping once for each epoch. (i = 0; i < 5; i++).  
then, call `model.fit()` with your dataset iterator in each epoch, to perform the training.
Before doing this, you can also add a score listener which shows some metrics during the training.
```java
MultiLayerNetwork model = new MultiLayerNetwork(conf);
model.init();
model.setListeners(new ScoreIterationListener()); // NEW: Print the scores while trinaing
// now, perform the training.
```
Scores are printed to a slf4j logger.  
If you look at the evaluation now, the results should be slightly less terrible, you can see this in the confusion matrix.

## 6: Hyper parameters
Our network still doesn't perform very well.  
We have to define some hyperparameters, these are the tweaks and optimization algorithms that allow the network to perform well.

#### Actication function
First off, we need to pick an activation function. An activation function is responsible for taking the result of the input signals and the weights of a neuron,
and translating it to a normalized output the represents the signal of the neuron.  
There are many activation functions, but these are the most commonly used ones:
* Sigmoid: Reduces extreme numbers without removing them, was used a lot in the past but is not very effective when training from scratch.
* Tanh: Similar to sigmoid, but deals better with negative numbers
* ReLU (Rectified Linear Units): quickly eliminates negative signals while keeping the positive signals constant. Very effective for training hidden layers.
* Softmax: Highlight the largest signals and suppress values which are significantly below the maximum signal. Commonly used in output layers.

In your hidden layer, add `.activation(Activation.RELU)`, and in your output layer, add Softmax.

#### Weight Initialization
You can fill a network with random weights on each neuron connections, but it's not a great idea.
Depending on your activation function, some values may never really propagate through the network well, this is why you often see `Xavier` being used to initialize weights.
Based on a paper from Xavier Glorot & Yoshua Bengio. You can read more about it here if you like.

Apply Xavier on all your layers by calling `.weightInit(WeightInit.XAVIER)`

#### "Updaters"
By now you should have some pretty decent results, but there are many more hyper parameters that can be added.  
One that is worth mentioning for this specific network is "updaters".  
In DL4J, updaters modify the output of Gradient Descent to control the learning rate. if the rate is too low, training might take too long. if it's too high, you may shoot over the local minama and miss out the best configuration.  
You can add the following to your network configuration before calling `.list()` and adding your layers, to optimize your backpropagation.
```java
.updater(new Nesterovs(0.006, 0.9)) // Optimization algorithm for gradient descent
```
Nesterovs is a Stochastic gradient descent optimizer that utilizes learning rate and momentum to tweak the output of the gradient descent.  
You can read [more about SGD (Stochastic gradient descent) on wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) if you like.

By now, you should be able to reach roughly 95% accuracy.
Not bad for 1 hidden layer and 5 * 128 training entries.

#### Some additional information
If you wonder when to use which hyperparameters, and how to configure them, this is typically automated.  
DL4J contains methods to automatically experiment with hyperparameter configuration to reach an optimal configuration.  
If you find the time, read about some optimization algorithms and when they would be applicable, and experiment with the different results.  
Here are some ways to find otu hyperparameters:
* [Grid Search](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
* Random Search
* Bayesian Optimization

There are also optimization algorthms that optimize performance by dropping out unnecessary parts of the network,
that don't really affect the accuracy. this is one of the secrets to well optimized, fast and accurate neural networks.