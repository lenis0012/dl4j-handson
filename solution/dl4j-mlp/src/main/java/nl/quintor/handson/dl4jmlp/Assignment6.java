package nl.quintor.handson.dl4jmlp;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;

/**
 * In this assignment we enhance our network with various hyper parameters.
 *
 * By adding an activation function, weight initialization algorithm and optimization algorithm we can make our network much more effective.
 */
public class Assignment6 {

    public static void main(String[] args) throws Exception {
        final int rngSeed = 123; // Seed, so we always get the same output when using randomly generated values
        final int rows = 28;
        final int cols = 28;
        final int outputNum = 10; // numbers (0-9)
        final int batchSize = 128; // batch size for each epoch

        // Load the MNIST data set
        DataSetIterator trainDataSet = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator testDataSet = new MnistDataSetIterator(batchSize, false, rngSeed);

        System.out.println("Configuring network."); // Configure
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new Nesterovs(0.006, 0.9)) // Optimization algorithm for gradient descent
                .list()
                // Hidden layer 1
                .layer(0, new DenseLayer.Builder()
                        .nIn(rows * cols)
                        .nOut(1000)
                        .activation(Activation.RELU) // Rectified linear activation function - optimal for hidden layers
                        .weightInit(WeightInit.XAVIER)
                        .build())
                // Output layer
                .layer(1, new OutputLayer.Builder()
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX) // Softmax activation function - optimal for output layers
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true)
                .build();

        System.out.println("Initializing network."); // Initialize
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener());

        System.out.println("Training model.");
        for(int i = 0; i < 5; i++) { // 15 epochs (take 15 samples from the training data)
            model.fit(trainDataSet);
        }

        System.out.println("Evaluating model.");
        Evaluation evaluation = new Evaluation(outputNum);
        while(testDataSet.hasNext()) {
            DataSet next = testDataSet.next();
            INDArray output = model.output(next.getFeatures());
            evaluation.eval(next.getLabels(), output);
        }

        // Print evaluation
        System.out.println(evaluation.stats());
    }
}
