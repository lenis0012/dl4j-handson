package nl.quintor.handson.dl4jmlp;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Assignment 2.
 *
 * Using ND4J, we will implement a very simple neural network to solve the Perfect Roommate problem.
 */
public class Assignment2 {

    public static void main(String[] args) {
        // Create the input (sunny = 1, rainy = 0)
        INDArray input = Nd4j.create(new float[] { 1, 0 });
        System.out.println("Input: \n" + input);

        // Create the matrix representing the weights
        INDArray matrix = Nd4j.create(3, 2);
        matrix.putScalar(0, 0, 1);
        matrix.putScalar(1, 1, 1);
        System.out.println("Matrix: \n" + matrix);

        // Run the matrix multiplication
        INDArray output = matrix.mulRowVector(input); // Multiply the matrix with the input
        output = output.max(1);                       // Convert matrix to vector (converting to an array of maxes across it's 1st dimension)

        System.out.println("Output vector = \n" + output);
        System.out.println("Pie = " + output.getScalar(0));
        System.out.println("Burger = " + output.getScalar(1));
        System.out.println("Chicken = " + output.getScalar(2));
    }
}
