package com.recommendationengine.index;

import com.recommendationengine.model.Item;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class LSHIndexTest {

    private LSHIndex index;
    private RandomProjection projection;

    @BeforeEach
    void setUp() {
        projection = new RandomProjection();
        index = new LSHIndex(projection);
    }

    // --- helpers -------------------------------------------------------------

    private Item itemWithVector(String id, double[] vector) {
        Item item = new Item(id, "Name " + id, "description");
        item.setVector(vector);
        return item;
    }

    private List<Item> buildThreeItems() {
        return List.of(
                itemWithVector("i1", new double[]{1.0, 0.0, 0.0, 0.0}),
                itemWithVector("i2", new double[]{0.0, 1.0, 0.0, 0.0}),
                itemWithVector("i3", new double[]{0.0, 0.0, 1.0, 0.0})
        );
    }

    // --- build() -------------------------------------------------------------

    @Test
    void buildPopulatesBuckets() {
        index.build(buildThreeItems());
        assertTrue(index.getBucketCount() > 0);
    }

    @Test
    void buildSetsCorrectTotalItemCount() {
        index.build(buildThreeItems());
        assertEquals(3, index.getTotalItemCount());
    }

    @Test
    void buildWithEmptyListProducesEmptyIndex() {
        index.build(List.of());
        assertEquals(0, index.getBucketCount());
        assertEquals(0, index.getTotalItemCount());
    }

    @Test
    void rebuildClearsPreviousState() {
        index.build(buildThreeItems());
        int firstCount = index.getTotalItemCount();

        List<Item> two = List.of(
                itemWithVector("x1", new double[]{1.0, 0.0, 0.0, 0.0}),
                itemWithVector("x2", new double[]{0.0, 1.0, 0.0, 0.0})
        );
        index.build(two);

        assertEquals(2, index.getTotalItemCount());
        assertNotEquals(firstCount, index.getTotalItemCount());
    }

    // --- getCandidates() -----------------------------------------------------

    @Test
    void getCandidatesReturnsNonEmptyList() {
        List<Item> items = buildThreeItems();
        index.build(items);

        List<Item> candidates = index.getCandidates(new double[]{1.0, 0.0, 0.0, 0.0});
        assertFalse(candidates.isEmpty());
    }

    @Test
    void getCandidatesForEmptyIndexReturnsAllItems() {
        // build was never called — should fall back safely
        List<Item> candidates = index.getCandidates(new double[]{1.0, 0.0, 0.0, 0.0});
        assertTrue(candidates.isEmpty()); // allItems is empty too
    }

    @Test
    void identicalVectorsAlwaysReturnSameBucket() {
        List<Item> items = buildThreeItems();
        index.build(items);

        double[] query = items.get(0).getVector();
        List<Item> first  = index.getCandidates(query);
        List<Item> second = index.getCandidates(query);

        assertEquals(first.size(), second.size());
        assertTrue(first.containsAll(second));
    }

    @Test
    void similarVectorsLandInSameOrAdjacentBucket() {
        // Build index with items spanning the space
        List<Item> items = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            double[] v = new double[8];
            v[i % 8] = 1.0;
            items.add(itemWithVector("item_" + i, v));
        }
        index.build(items);

        // A query very close to item_0's vector should return candidates
        double[] query = new double[]{0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        List<Item> candidates = index.getCandidates(query);
        assertFalse(candidates.isEmpty());
    }

    @Test
    void getCandidatesReturnedListIsUnmodifiable() {
        index.build(buildThreeItems());
        List<Item> candidates = index.getCandidates(new double[]{1.0, 0.0, 0.0, 0.0});
        assertThrows(UnsupportedOperationException.class, () -> candidates.add(null));
    }
}
