package nl.quintor.handson.dl4jmlp;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * In this assignment we add Back propagation to the neural network.
 *
 * Back propagation utilizes gradient descent to find an optimal configuration for the network weights.
 * This is done by running gradient descent over the error of various examples from a training dataset.
 */
public class Assignment5 {

    public static void main(String[] args) throws Exception {
        final int rngSeed = 123; // Seed, so we always get the same output when using randomly generated values
        final int rows = 28;
        final int cols = 28;
        final int outputNum = 10; // numbers (0-9)
        final int batchSize = 128; // batch size for each epoch

        // Load the MNIST data set
        MnistDataSetIterator trainDataSet = new MnistDataSetIterator(batchSize, true, rngSeed); // NEW
        MnistDataSetIterator testDataSet = new MnistDataSetIterator(batchSize, false, rngSeed);

        System.out.println("Configuring network."); // Configure
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .list()
                // Hidden layer 1
                .layer(new DenseLayer.Builder()
                        .nIn(rows * cols)
                        .nOut(1000)
                        .build())
                // Output layer
                .layer(new OutputLayer.Builder()
                        .nIn(1000)
                        .nOut(outputNum)
                        .build())
                .pretrain(false).backprop(true) // NEW: Enable back propagation
                .build();

        System.out.println("Initializing network."); // Initialize
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener()); // NEW: Print the scores while trinaing

        // NEW: Train the model
        System.out.println("Training model.");
        for(int i = 0; i < 5; i++) { // 15 epochs (take 15 samples from the training data)
            model.fit(trainDataSet);
        }

        System.out.println("Evaluating model.");
        Evaluation evaluation = new Evaluation(outputNum);
        while(testDataSet.hasNext()) {
            DataSet dataSet = testDataSet.next();
            INDArray output = model.output(dataSet.getFeatures());
            evaluation.eval(dataSet.getLabels(), output);
        }

        // Print evaluation
        System.out.println(evaluation.stats());
    }
}
