package com.recommendationengine.model;

/**
 * Immutable response DTO returned by the recommendation API.
 *
 * @param id          item identifier
 * @param name        display name
 * @param description short description of the item
 * @param score       cosine similarity score in [0.0, 1.0] — higher is more similar
 */
public record RecommendationResult(
        String id,
        String name,
        String description,
        double score
) {}
