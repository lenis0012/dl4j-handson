package nl.quintor.handson.dl4jmlp;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Assignment1 {

    public static void main(String[] args) {
        int rows = 3;
        int cols = 5;

        System.out.println("Zeros:");
        INDArray array = Nd4j.zeros(rows, cols);
        System.out.println(array);

        System.out.println("Ones:");
        array = Nd4j.ones(rows, cols);
        System.out.println(array);

        System.out.println("Random array:");
        array = Nd4j.rand(rows, cols);
        System.out.println(array);

        System.out.println("Modifying array:");
        array.putScalar(0, 1, 2.0);             // Set value at row 0, column 1 to value 2.0
        array.putScalar(2, 3, 5.0);             // Set value at row 2, column 3 to value 5.0
        array.add(1);                                           // Add 1 to each value
        array.mul(2);                                           // Multiply eac
        System.out.println(array);

        System.out.println("Reshape:");
        array = Nd4j.linspace(1, 15, 15);
        System.out.println(array);                              // 1x15 array
        array = array.reshape('c', 3, 5);
        System.out.println(array);

        System.out.println("Row modification:");
        INDArray row = array.getRow(0);
        System.out.println(row);

        System.out.println("Add");
        row.add(1);                                             // Add one to the row. (returns a new array)
        System.out.println(row);
        System.out.println(array);

        System.out.println("Addi");
        row.addi(1);                                            // Add one to the row, and to the original array (returns itself)
        System.out.println(row);
        System.out.println(array);
    }
}
