package com.recommendationengine.index;

import org.springframework.stereotype.Component;

import java.util.Random;

/**
 * Generates a random hyperplane matrix used by the LSH index.
 *
 * <p>Each row is a random unit vector (hyperplane normal). An input vector
 * is hashed by computing the sign of its dot product with each hyperplane:
 * <pre>
 *   bit_i = '1'  if (v · hyperplane_i) >= 0
 *           '0'  otherwise
 * </pre>
 *
 * <p>Vectors that are close in cosine space tend to land on the same side
 * of each hyperplane, and therefore produce the same binary hash string.
 * This is the "random projection" variant of LSH for cosine similarity.
 */
@Component
public class RandomProjection {

    // [numHashBits][vectorDimensions]
    private double[][] hyperplanes;
    private int numHashBits;
    private int vectorDimensions;

    /**
     * Initialises the hyperplane matrix with a fixed random seed
     * so hashes are reproducible across restarts.
     *
     * @param numHashBits      number of hyperplanes (= bit length of each hash)
     * @param vectorDimensions dimensionality of the input vectors
     */
    public void initialize(int numHashBits, int vectorDimensions) {
        this.numHashBits = numHashBits;
        this.vectorDimensions = vectorDimensions;
        this.hyperplanes = new double[numHashBits][vectorDimensions];

        Random rng = new Random(42L); // fixed seed for reproducibility
        for (int i = 0; i < numHashBits; i++) {
            double norm = 0.0;
            for (int j = 0; j < vectorDimensions; j++) {
                hyperplanes[i][j] = rng.nextGaussian();
                norm += hyperplanes[i][j] * hyperplanes[i][j];
            }
            norm = Math.sqrt(norm);
            for (int j = 0; j < vectorDimensions; j++) {
                hyperplanes[i][j] /= norm;
            }
        }
    }

    /**
     * Computes the binary hash string for a given vector.
     *
     * @param vector input vector — must match the dimension used in {@link #initialize}
     * @return a string of {@code '0'} and {@code '1'} characters, one per hyperplane
     */
    public String hash(double[] vector) {
        if (!isInitialized()) {
            throw new IllegalStateException("RandomProjection must be initialized before hashing");
        }
        if (vector.length != vectorDimensions) {
            throw new IllegalArgumentException(
                    "Vector dimension mismatch: expected " + vectorDimensions + ", got " + vector.length);
        }

        StringBuilder sb = new StringBuilder(numHashBits);
        for (int i = 0; i < numHashBits; i++) {
            double dot = 0.0;
            for (int j = 0; j < vectorDimensions; j++) {
                dot += hyperplanes[i][j] * vector[j];
            }
            sb.append(dot >= 0.0 ? '1' : '0');
        }
        return sb.toString();
    }

    public boolean isInitialized() {
        return hyperplanes != null;
    }

    public int getNumHashBits() {
        return numHashBits;
    }

    public int getVectorDimensions() {
        return vectorDimensions;
    }
}
