package com.recommendationengine.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RecommendationResultTest {

    @Test
    void recordAccessorsReturnCorrectValues() {
        RecommendationResult result = new RecommendationResult("item_1", "Inception", "A thief steals secrets", 0.91);

        assertEquals("item_1", result.id());
        assertEquals("Inception", result.name());
        assertEquals("A thief steals secrets", result.description());
        assertEquals(0.91, result.score(), 1e-9);
    }

    @Test
    void recordEqualityBasedOnValues() {
        RecommendationResult a = new RecommendationResult("item_1", "Inception", "desc", 0.9);
        RecommendationResult b = new RecommendationResult("item_1", "Inception", "desc", 0.9);

        assertEquals(a, b);
    }

    @Test
    void scoreCanBeZero() {
        RecommendationResult result = new RecommendationResult("item_1", "Name", "desc", 0.0);
        assertEquals(0.0, result.score());
    }
}
