Spring DL4J Assignments
=======================

#### 1: Take your neural network to spring.
In the MLP assignments, we have created a neural network in DL4J.  
It is time to add this to a spring application, so that you may deploy it.  

Start by adding the maven dependencies for deeplearning4j to your pom.xml.
And then, create a new class annotated with `@Configuration` where you will put your MNIST neural network configuration.  

Define a new `@Bean` for your MultiLayerNetwork.  
Keep the code that trains the network as part of the bean setup, this will train the network during your applicaion startup.  
Because DL4J uses SLF4J for logging, all logs will get printed in the spring format out of the box, nice.

#### 2: Add a prediction endpoint
Create a new controller for your mnist network `/predict/{digit}/{index}`, which reads a PNG image file and passes it to the network.
You can use a `ClassPathResource()` to /mnist/{digit}/{index}.png as resource. 3 samples for each digits are included in your project.

[Use DL4J DataVec to convert the image to a vector.](https://github.com/deeplearning4j/DataVec/blob/master/datavec-data/datavec-data-image/src/main/java/org/datavec/image/loader/ImageLoader.java)

the output will contain the probability for each digit.  
Find the highest probability, it's index will be the number of the prediction.  
Return this as response for your endpoint.

Bonus: You can also a POST request and upload files to the server to use for prediction instead of using a classpath resource.  
See [spring's tutorial on file uploading to get this done](https://spring.io/guides/gs/uploading-files/)

#### 3: Serialize your network
You don't want to train your network every time during boot, instead you probably want to load a pre-trained network.  
On [DL4J's cheatsheet page](https://deeplearning4j.org/docs/latest/deeplearning4j-cheat-sheet), they mention "Network Saving and Loading", with the `ModelSerializer` class.

You can add an endpoint (like, /train), to train your network, and save it to a file.  
When creating your bean, you can check for the existence of this file, and deserialize your network from this file.  
This will speed up your application startup time by a huge amount.

#### 4: Animal prediction
In our application, we have a H2 database with a table called animals.
This contains information about 30 different animals.  
The dataset contains humans, dogs and cats, with information about their age, what they eat, the sound they make and their weight.  

Make a neural network that can predict an animal based on these properties, and train it from your JDBC database.  
For reference, you can use the DL4J example project.  

Add the following maven dependencies ot your project to be able to load data from jdbc:
```xml
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-datavec-iterators</artifactId>
    <version>${dl4j.version}</version>
</dependency>
<dependency>
    <groupId>org.datavec</groupId>
    <artifactId>datavec-jdbc</artifactId>
    <version>${dl4j.version}</version>
</dependency>
```

Use the `JDBCRecordReader` and `RecordReaderDataSetIterator` to use JDBC as a datasource.

#### 5: (Bonus) Apply Regularization Techniques to optimize performance.

To obtain higher accuracy and performance, you can utilize Regularization Techniques.  
These techniques are used to prevent overfitting.

Use the DL4J cheatsheet, and [this article from analytics vidhya](https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/) to optimize your network.