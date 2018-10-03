package nl.quintor.handson.dl4j.spring.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.io.IOException;

@Configuration
public class MnistConfiguration {
    private static final Logger LOG = LoggerFactory.getLogger(MnistConfiguration.class);
    private static final int RNG_SEED = 123;

    @Bean(name = "mnistTrainData")
    public DataSetIterator trainDataSet() throws IOException {
        return new MnistDataSetIterator(128, true, RNG_SEED);
    }

    @Bean(name = "mnistNetwork")
    public MultiLayerNetwork network(@Qualifier("mnistTrainData") DataSetIterator trainDataSet) throws IOException {
        final int rows = 28;
        final int cols = 28;
        final int outputNum = 10; // numbers (0-9)

        File networkFile = new File("network.bin");
        if(networkFile.exists()) {
            return ModelSerializer.restoreMultiLayerNetwork(networkFile);
        }

        LOG.info("Configuring network."); // Configure
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(RNG_SEED)
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

        LOG.info("Initializing network."); // Initialize
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener());

        return model;
    }
}
