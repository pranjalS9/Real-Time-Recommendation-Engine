package com.recommendationengine.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ItemTest {

    @Test
    void constructorSetsFieldsCorrectly() {
        Item item = new Item("id_1", "Inception", "A thief steals secrets through dreams");

        assertEquals("id_1", item.getId());
        assertEquals("Inception", item.getName());
        assertEquals("A thief steals secrets through dreams", item.getDescription());
        assertNull(item.getVector());
    }

    @Test
    void vectorIsNullBeforeVectorizationAndSettableAfter() {
        Item item = new Item("id_1", "Inception", "A thief steals secrets through dreams");
        assertNull(item.getVector());

        double[] vector = {0.1, 0.5, 0.3};
        item.setVector(vector);

        assertArrayEquals(vector, item.getVector());
    }

    @Test
    void defaultConstructorAllowsPropertySetting() {
        Item item = new Item();
        item.setId("id_2");
        item.setName("Interstellar");
        item.setDescription("Astronauts travel through a wormhole");

        assertEquals("id_2", item.getId());
        assertEquals("Interstellar", item.getName());
        assertEquals("Astronauts travel through a wormhole", item.getDescription());
    }

    @Test
    void toStringContainsIdAndName() {
        Item item = new Item("id_1", "Inception", "description");
        String str = item.toString();

        assertTrue(str.contains("id_1"));
        assertTrue(str.contains("Inception"));
    }
}
