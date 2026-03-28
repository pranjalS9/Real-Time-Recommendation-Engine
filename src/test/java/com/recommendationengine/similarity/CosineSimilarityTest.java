package com.recommendationengine.similarity;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CosineSimilarityTest {

    private CosineSimilarity cosine;

    @BeforeEach
    void setUp() {
        cosine = new CosineSimilarity();
    }

    @Test
    void identicalVectorsReturnOne() {
        double[] a = {1.0, 2.0, 3.0};
        assertEquals(1.0, cosine.compute(a, a), 1e-10);
    }

    @Test
    void orthogonalVectorsReturnZero() {
        double[] a = {1.0, 0.0};
        double[] b = {0.0, 1.0};
        assertEquals(0.0, cosine.compute(a, b), 1e-10);
    }

    @Test
    void parallelVectorsOfDifferentMagnitudeReturnOne() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {2.0, 4.0, 6.0};
        assertEquals(1.0, cosine.compute(a, b), 1e-10);
    }

    @Test
    void zeroVectorAReturnsZero() {
        double[] a = {0.0, 0.0, 0.0};
        double[] b = {1.0, 2.0, 3.0};
        assertEquals(0.0, cosine.compute(a, b), 1e-10);
    }

    @Test
    void zeroVectorBReturnsZero() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {0.0, 0.0, 0.0};
        assertEquals(0.0, cosine.compute(a, b), 1e-10);
    }

    @Test
    void bothZeroVectorsReturnZero() {
        double[] a = {0.0, 0.0};
        double[] b = {0.0, 0.0};
        assertEquals(0.0, cosine.compute(a, b), 1e-10);
    }

    @Test
    void scoreIsBetweenZeroAndOneForNonNegativeVectors() {
        double[] a = {0.5, 0.1, 0.9, 0.0, 0.3};
        double[] b = {0.2, 0.8, 0.4, 0.6, 0.1};
        double score = cosine.compute(a, b);
        assertTrue(score >= 0.0 && score <= 1.0);
    }

    @Test
    void computeIsSymmetric() {
        double[] a = {0.3, 0.7, 0.1};
        double[] b = {0.9, 0.2, 0.5};
        assertEquals(cosine.compute(a, b), cosine.compute(b, a), 1e-10);
    }

    @Test
    void dimensionMismatchThrowsException() {
        double[] a = {1.0, 2.0};
        double[] b = {1.0, 2.0, 3.0};
        assertThrows(IllegalArgumentException.class, () -> cosine.compute(a, b));
    }

    @Test
    void knownValueIsComputedCorrectly() {
        // a = [1, 0], b = [1, 1] → dot=1, |a|=1, |b|=√2 → similarity = 1/√2 ≈ 0.7071
        double[] a = {1.0, 0.0};
        double[] b = {1.0, 1.0};
        assertEquals(1.0 / Math.sqrt(2.0), cosine.compute(a, b), 1e-10);
    }
}
