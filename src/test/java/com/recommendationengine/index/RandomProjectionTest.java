package com.recommendationengine.index;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RandomProjectionTest {

    private RandomProjection projection;

    @BeforeEach
    void setUp() {
        projection = new RandomProjection();
    }

    @Test
    void isNotInitializedBeforeInit() {
        assertFalse(projection.isInitialized());
    }

    @Test
    void isInitializedAfterInit() {
        projection.initialize(8, 50);
        assertTrue(projection.isInitialized());
    }

    @Test
    void hashLengthEqualsNumHashBits() {
        projection.initialize(8, 4);
        String hash = projection.hash(new double[]{0.1, 0.2, 0.3, 0.4});
        assertEquals(8, hash.length());
    }

    @Test
    void hashContainsOnlyZerosAndOnes() {
        projection.initialize(12, 4);
        String hash = projection.hash(new double[]{0.5, 0.1, 0.9, 0.3});
        assertTrue(hash.chars().allMatch(c -> c == '0' || c == '1'));
    }

    @Test
    void sameVectorProducesSameHash() {
        projection.initialize(8, 3);
        double[] vector = {0.3, 0.7, 0.1};
        assertEquals(projection.hash(vector), projection.hash(vector));
    }

    @Test
    void hashIsReproducibleAcrossInstances() {
        // Fixed seed (42) means two fresh instances produce identical hashes
        RandomProjection p1 = new RandomProjection();
        RandomProjection p2 = new RandomProjection();
        p1.initialize(8, 4);
        p2.initialize(8, 4);

        double[] vector = {0.2, 0.5, 0.8, 0.1};
        assertEquals(p1.hash(vector), p2.hash(vector));
    }

    @Test
    void dimensionMismatchThrows() {
        projection.initialize(4, 3);
        assertThrows(IllegalArgumentException.class,
                () -> projection.hash(new double[]{0.1, 0.2})); // wrong dim
    }

    @Test
    void hashBeforeInitThrows() {
        assertThrows(IllegalStateException.class,
                () -> projection.hash(new double[]{0.1, 0.2}));
    }

    @Test
    void closeVectorsTendToShareHash() {
        // Two very similar vectors should collide more often than two random ones
        projection.initialize(4, 10);

        double[] base = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05};
        double[] similar = {0.91, 0.79, 0.71, 0.61, 0.51, 0.39, 0.31, 0.19, 0.11, 0.04};

        // With only 4 bits there's a high chance similar vectors hash identically
        String h1 = projection.hash(base);
        String h2 = projection.hash(similar);

        // Count matching bits
        int matching = 0;
        for (int i = 0; i < h1.length(); i++) {
            if (h1.charAt(i) == h2.charAt(i)) matching++;
        }
        // Similar vectors should agree on most bits
        assertTrue(matching >= 3, "Similar vectors should share at least 3 of 4 hash bits");
    }
}
