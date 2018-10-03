package nl.quintor.handson.dl4j.spring.animals;

import org.datavec.api.records.reader.impl.jdbc.JDBCRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.flywaydb.core.Flyway;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;
import java.io.IOException;

@Configuration
public class AnimalsConfiguration {
    private static final Logger LOG = LoggerFactory.getLogger(AnimalsConfiguration.class);
    private static final int RNG_SEED = 123;

    @Bean(name = "animalsTrainData")
    public DataSetIterator dataSet(DataSource dataSource, Flyway flyway) throws IOException, InterruptedException {
        flyway.migrate();

        Thread.sleep(2000); // wait for flyway

        // create a jdbc record reader that reads the animals table
        JDBCRecordReader recordReader = new JDBCRecordReader("SELECT * FROM animals;", dataSource);
        recordReader.initialize(null);

        return new RecordReaderDataSetIterator(recordReader, 30,4, 3);
    }

    @Bean
    public DataNormalization normalization(@Qualifier("animalsTrainData") DataSetIterator trainingData) {
        DataNormalization normalization = new NormalizerStandardize(); // normalize all the data based on the values from the training set.
        normalization.fit(trainingData);
        return normalization;
    }

    @Bean(name = "animalsNetwork")
    public MultiLayerNetwork network(@Qualifier("animalsTrainData") DataSetIterator trainingData, DataNormalization normalization) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(RNG_SEED)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(4)
                        .nOut(3)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(3)
                        .nOut(3)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3)
                        .nOut(3)
                        .build())
                .backprop(true).pretrain(false)
                .build();

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Prepare the dataset
        trainingData.hasNext();
        DataSet dataSet = trainingData.next();
        normalization.transform(dataSet);

        for( int i=0; i< 1000; i++ ) {
            model.fit(dataSet);
        }

        // Test the network
        AnimalInputDTO animal = new AnimalInputDTO();
        animal.setYearsLived(19); // 19 years old
        animal.setEats(Food.HOTDOGS); // eats hotdogs
        animal.setSounds(Sound.TALKING); // talks
        animal.setWeight(65); // weighs 65
        INDArray input = Nd4j.create(new float[] { animal.getYearsLived(), animal.getEats().ordinal(), animal.getSounds().ordinal(), animal.getWeight() });
        normalization.transform(input);

        INDArray output = model.output(input);
        LOG.info("Output: " + output);
        LOG.info("Animal: " + Animal.values()[(int) ((Float) output.argMax(1).element()).floatValue()]);

        return model;
    }
}
