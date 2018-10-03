package nl.quintor.handson.dl4jmlp;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

/**
 * In this assignment, we apply Evaluation to get matrix on the accuracy of our model.
 *
 * Unfortunately, our model isn't very good yet.
 */
public class Assignment4 {

    public static void main(String[] args) throws Exception {
        final int rngSeed = 123; // Seed, so we always get the same output when using randomly generated values
        final int rows = 28;
        final int cols = 28;
        final int outputNum = 10; // numbers (0-9)
        final int batchSize = 128; // batch size for each epoch

        // Load the MNIST data set
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
                .build();

        System.out.println("Initializing network."); // Initialize
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

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
