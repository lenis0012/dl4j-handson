package nl.quintor.handson.dl4jmlp;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A neural network that is intended to solve the MNIST dataset problem.
 *
 * This network is untrained, and thus gives a somewhat random  and incorrect prediction.
 */
public class Assignment3 {

    public static void main(String[] args) {
        final int rngSeed = 123; // Seed, so we always get the same output when using randomly generated values
        final int rows = 28;
        final int cols = 28;
        final int outputNum = 10; // numbers (0-9)

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

        System.out.println("Making prediction."); // Predict
        INDArray input = Nd4j.zeros(rows * cols); // empty pane
        INDArray output = model.output(input);

        System.out.println("Output = " + output);
        System.out.println("Best guess = " + ((int) ((float) output.argMax().element())));
    }
}
