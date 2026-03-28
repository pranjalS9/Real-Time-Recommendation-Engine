package com.recommendationengine.index;

import com.recommendationengine.model.Item;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * Locality-Sensitive Hashing index for approximate nearest-neighbour search.
 *
 * <p>Items whose TF-IDF vectors are close in cosine space are grouped into the
 * same hash bucket. At query time only the matching bucket is searched,
 * reducing the candidate set from O(N) to roughly O(N / 2^k).
 *
 * <p>If the exact bucket is empty, the index expands the search to all
 * adjacent buckets (Hamming distance 1) before falling back to a full scan.
 */
@Component
public class LSHIndex {

    private static final Logger log = LoggerFactory.getLogger(LSHIndex.class);

    private final RandomProjection projection;

    @Value("${lsh.num-hash-bits:12}")
    private int numHashBits = 12;

    // hash string → items in that bucket
    private final Map<String, List<Item>> buckets = new HashMap<>();

    private List<Item> allItems = new ArrayList<>();

    public LSHIndex(RandomProjection projection) {
        this.projection = projection;
    }

    /**
     * Hashes all items into buckets.
     * Must be called after TF-IDF vectors have been set on every item.
     */
    public void build(List<Item> items) {
        if (items == null || items.isEmpty()) {
            log.warn("LSHIndex.build() called with empty item list — index will be empty");
            return;
        }

        buckets.clear();
        allItems = new ArrayList<>(items);

        int vectorDim = items.get(0).getVector().length;
        if (!projection.isInitialized()) {
            projection.initialize(numHashBits, vectorDim);
        }

        for (Item item : items) {
            String hash = projection.hash(item.getVector());
            buckets.computeIfAbsent(hash, k -> new ArrayList<>()).add(item);
        }

        log.info("LSH index built: {} items in {} buckets (avg size: {:.1f})",
                items.size(),
                buckets.size(),
                (double) items.size() / buckets.size());
    }

    /**
     * Returns the candidate items for a given query vector.
     *
     * <ol>
     *   <li>Look up the exact hash bucket.</li>
     *   <li>If empty, expand to adjacent buckets (Hamming distance 1).</li>
     *   <li>If still empty, fall back to all items.</li>
     * </ol>
     */
    public List<Item> getCandidates(double[] queryVector) {
        if (!projection.isInitialized() || buckets.isEmpty()) {
            log.warn("LSH index is empty — returning all items as candidates");
            return Collections.unmodifiableList(allItems);
        }

        String hash = projection.hash(queryVector);
        List<Item> bucket = buckets.get(hash);

        if (bucket != null && !bucket.isEmpty()) {
            return Collections.unmodifiableList(bucket);
        }

        List<Item> expanded = expandedSearch(hash);
        if (!expanded.isEmpty()) {
            log.debug("Exact bucket empty for hash {} — using {} expanded candidates", hash, expanded.size());
            return Collections.unmodifiableList(expanded);
        }

        log.debug("No candidates in adjacent buckets for hash {} — falling back to full scan", hash);
        return Collections.unmodifiableList(allItems);
    }

    /**
     * Collects items from all buckets at Hamming distance 1 from the given hash.
     */
    private List<Item> expandedSearch(String hash) {
        List<Item> candidates = new ArrayList<>();
        char[] bits = hash.toCharArray();

        for (int i = 0; i < bits.length; i++) {
            char original = bits[i];
            bits[i] = (original == '0') ? '1' : '0';
            List<Item> neighbour = buckets.get(new String(bits));
            if (neighbour != null) {
                candidates.addAll(neighbour);
            }
            bits[i] = original;
        }

        return candidates;
    }

    public int getBucketCount() {
        return buckets.size();
    }

    public int getTotalItemCount() {
        return allItems.size();
    }
}
