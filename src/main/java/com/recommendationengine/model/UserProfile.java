package com.recommendationengine.model;

import java.util.List;

/**
 * Represents a user and the items they have already viewed.
 * The viewed item IDs are used to build a preference vector
 * for generating recommendations.
 */
public record UserProfile(
        String userId,
        List<String> viewedItemIds
) {}
