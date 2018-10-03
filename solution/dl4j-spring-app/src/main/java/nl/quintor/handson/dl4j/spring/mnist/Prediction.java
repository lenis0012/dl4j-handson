package nl.quintor.handson.dl4j.spring.mnist;

public class Prediction {
    private int value;
    private double confidence;

    public Prediction() {
    }

    public Prediction(int value, double confidence) {
        this.value = value;
        this.confidence = confidence;
    }

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }
}
