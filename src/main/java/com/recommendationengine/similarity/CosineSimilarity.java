package com.recommendationengine.similarity;

import org.springframework.stereotype.Component;

/**
 * Computes cosine similarity between two vectors.
 *
 * <pre>
 *   similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)
 * </pre>
 *
 * Returns a value in [0.0, 1.0] for non-negative TF-IDF vectors:
 * <ul>
 *   <li>1.0 — vectors point in the same direction (maximally similar)</li>
 *   <li>0.0 — vectors are orthogonal (nothing in common)</li>
 * </ul>
 *
 * Returns 0.0 if either vector is a zero vector (no terms matched vocabulary).
 */
@Component
public class CosineSimilarity {

    public double compute(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    "Vectors must have equal dimensions: " + a.length + " vs " + b.length);
        }

        double dotProduct = 0.0;
        double magnitudeA  = 0.0;
        double magnitudeB  = 0.0;

        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }

        magnitudeA = Math.sqrt(magnitudeA);
        magnitudeB = Math.sqrt(magnitudeB);

        if (magnitudeA == 0.0 || magnitudeB == 0.0) {
            return 0.0;
        }

        return dotProduct / (magnitudeA * magnitudeB);
    }
}
