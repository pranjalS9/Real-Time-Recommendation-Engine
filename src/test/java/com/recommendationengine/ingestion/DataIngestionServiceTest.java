package com.recommendationengine.ingestion;

import com.recommendationengine.model.Item;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DataIngestionServiceTest {

    private DataIngestionService service;

    @BeforeEach
    void setUp() {
        service = new DataIngestionService();
    }

    @Test
    void ingestsAllRowsFromTestCsv() {
        List<Item> items = service.ingest("/data/test-items.csv");
        assertEquals(3, items.size());
    }

    @Test
    void parsesFieldsCorrectly() {
        List<Item> items = service.ingest("/data/test-items.csv");
        Item first = items.get(0);

        assertEquals("test_1", first.getId());
        assertEquals("Dark Knight", first.getName());
        assertFalse(first.getDescription().isBlank());
    }

    @Test
    void vectorIsNullAfterIngestion() {
        List<Item> items = service.ingest("/data/test-items.csv");
        // Vector is not set by ingestion — it is set later by TfIdfVectorizer
        items.forEach(item -> assertNull(item.getVector()));
    }

    @Test
    void findByIdReturnsCorrectItem() {
        service.ingest("/data/test-items.csv");
        Item item = service.findById("test_2");

        assertNotNull(item);
        assertEquals("Inception", item.getName());
    }

    @Test
    void findByIdReturnsNullForUnknownId() {
        service.ingest("/data/test-items.csv");
        assertNull(service.findById("does_not_exist"));
    }

    @Test
    void sizeReflectsIngestedCount() {
        service.ingest("/data/test-items.csv");
        assertEquals(3, service.size());
    }

    @Test
    void reingestClearsPreviousState() {
        service.ingest("/data/test-items.csv");
        assertEquals(3, service.size());

        // Second ingest on same file should not accumulate
        service.ingest("/data/test-items.csv");
        assertEquals(3, service.size());
    }

    @Test
    void returnsEmptyListForMissingFile() {
        List<Item> items = service.ingest("/data/nonexistent.csv");
        assertTrue(items.isEmpty());
        assertEquals(0, service.size());
    }

    @Test
    void getAllItemsReturnsAllIngestedItems() {
        service.ingest("/data/test-items.csv");
        assertEquals(3, service.getAllItems().size());
    }
}
