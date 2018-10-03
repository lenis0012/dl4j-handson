package nl.quintor.handson.dl4j.spring.mnist;

import org.datavec.image.loader.ImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;

@RestController
@RequestMapping("/mnist")
public class MnistController {
    private static final Logger LOG = LoggerFactory.getLogger(MnistController.class);

    private final MultiLayerNetwork network;
    private final DataSetIterator dataSetIterator;
    private final ImageLoader imageLoader;

    @Autowired
    public MnistController(@Qualifier("mnistNetwork") MultiLayerNetwork network, @Qualifier("mnistTrainData") DataSetIterator dataSetIterator) {
        this.network = network;
        this.dataSetIterator = dataSetIterator;

        this.imageLoader = new ImageLoader(28, 28);
    }

    // Train on train data
    @RequestMapping("/train/{epochs}")
    public void train(@PathVariable int epochs)  {
        LOG.info("Training model.");
        network.clear();
        for(int i = 0; i < epochs; i++) { // 15 epochs (take 15 samples from the training data)
            network.fit(dataSetIterator);
        }

        // Save the newly trained model
        try {
            ModelSerializer.writeModel(network, new File("network.bin"), true);
        } catch(IOException e) {
            LOG.error("Failed to save network", e);
        }
    }

    // Predict with classpath resource
    @GetMapping({"/predict/{number}/{index}", "/predict/{number}"})
    public Prediction predict(@PathVariable int number, @PathVariable(required = false) Optional<Integer> index) throws Exception {
        // Read image file from classpath, and convert it to a row vector with DL4J DataVec.
        ClassPathResource resource = new ClassPathResource("mnist/" + number + "/" + index.orElse(0) + ".png");
        return getPrediction(resource.getInputStream());
    }

    // Predict with uploaded PNG file
    @PostMapping("/predict")
    public Prediction predict(@RequestParam("file") MultipartFile file) throws IOException {
        return getPrediction(file.getInputStream());
    }

    @NotNull
    private Prediction getPrediction(InputStream resource) throws IOException {
        // convert it to a row vector with DL4J DataVec.
        INDArray input = imageLoader.asRowVector(resource);

        // Normalize pixel values between 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(input);

        INDArray output = network.output(input);
        return new Prediction(
                (int) ((Float) output.argMax(1).element()).floatValue(), // argMax returns the index of the largest number in the vector
                (double) output.maxNumber() // maxNumber returns the value of the largest number in the vector
        );
    }
}
