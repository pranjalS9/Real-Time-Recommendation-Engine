package com.recommendationengine.model;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class UserProfileTest {

    @Test
    void recordAccessorsReturnCorrectValues() {
        UserProfile profile = new UserProfile("user_1", List.of("item_1", "item_2"));

        assertEquals("user_1", profile.userId());
        assertEquals(List.of("item_1", "item_2"), profile.viewedItemIds());
    }

    @Test
    void recordEqualityBasedOnValues() {
        UserProfile a = new UserProfile("user_1", List.of("item_1"));
        UserProfile b = new UserProfile("user_1", List.of("item_1"));

        assertEquals(a, b);
    }

    @Test
    void emptyViewedItemsIsAllowed() {
        UserProfile profile = new UserProfile("user_new", List.of());
        assertTrue(profile.viewedItemIds().isEmpty());
    }
}
