package com.recommendationengine.engine;

import com.recommendationengine.index.LSHIndex;
import com.recommendationengine.index.RandomProjection;
import com.recommendationengine.ingestion.DataIngestionService;
import com.recommendationengine.model.Item;
import com.recommendationengine.model.RecommendationResult;
import com.recommendationengine.model.UserProfile;
import com.recommendationengine.nlp.TextPreprocessor;
import com.recommendationengine.nlp.TfIdfVectorizer;
import com.recommendationengine.similarity.CosineSimilarity;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RecommendationEngineTest {

    private RecommendationEngine engine;
    private DataIngestionService ingestionService;

    @BeforeEach
    void setUp() {
        // Wire all real collaborators — no mocks, this is a unit integration test
        ingestionService = new DataIngestionService();
        TfIdfVectorizer vectorizer = new TfIdfVectorizer(new TextPreprocessor());
        LSHIndex lshIndex = new LSHIndex(new RandomProjection());
        CosineSimilarity cosine = new CosineSimilarity();

        // 2 hash bits → 4 buckets for 50 items → ~12 items per bucket on average,
        // guaranteeing candidates always include unviewed items in tests
        lshIndex.setNumHashBits(2);

        engine = new RecommendationEngine(ingestionService, lshIndex, cosine);

        List<Item> items = ingestionService.ingest("/data/items.csv");
        vectorizer.fitAndTransform(items);
        lshIndex.build(items);
    }

    @Test
    void returnsResultsForValidUser() {
        UserProfile profile = new UserProfile("u1", List.of("item_1"));
        List<RecommendationResult> results = engine.recommend(profile);
        assertFalse(results.isEmpty());
    }

    @Test
    void doesNotIncludeAlreadyViewedItems() {
        UserProfile profile = new UserProfile("u1", List.of("item_1"));
        List<RecommendationResult> results = engine.recommend(profile);

        boolean containsViewed = results.stream().anyMatch(r -> r.id().equals("item_1"));
        assertFalse(containsViewed, "Already viewed items must not appear in recommendations");
    }

    @Test
    void resultsAreSortedByScoreDescending() {
        UserProfile profile = new UserProfile("u1", List.of("item_1"));
        List<RecommendationResult> results = engine.recommend(profile);

        for (int i = 0; i < results.size() - 1; i++) {
            assertTrue(results.get(i).score() >= results.get(i + 1).score(),
                    "Results must be sorted by score descending");
        }
    }

    @Test
    void respectsTopNLimit() {
        UserProfile profile = new UserProfile("u1", List.of("item_1"));
        List<RecommendationResult> results = engine.recommend(profile, 1);
        assertTrue(results.size() <= 1);
    }

    @Test
    void scoresAreBetweenZeroAndOne() {
        UserProfile profile = new UserProfile("u1", List.of("item_1"));
        List<RecommendationResult> results = engine.recommend(profile);
        results.forEach(r ->
                assertTrue(r.score() >= 0.0 && r.score() <= 1.0,
                        "Score out of range: " + r.score())
        );
    }

    @Test
    void returnsEmptyForUserWithNoHistory() {
        UserProfile profile = new UserProfile("new_user", List.of());
        List<RecommendationResult> results = engine.recommend(profile);
        assertTrue(results.isEmpty());
    }

    @Test
    void returnsEmptyForUnknownItemIds() {
        UserProfile profile = new UserProfile("u1", List.of("does_not_exist"));
        List<RecommendationResult> results = engine.recommend(profile);
        assertTrue(results.isEmpty());
    }

    @Test
    void multipleViewedItemsProduceValidRecommendations() {
        // User has viewed two items — preference vector is their average
        UserProfile profile = new UserProfile("u1", List.of("item_1", "item_2"));
        List<RecommendationResult> results = engine.recommend(profile);

        assertFalse(results.isEmpty());
        results.forEach(r -> {
            assertNotEquals("item_1", r.id());
            assertNotEquals("item_2", r.id());
        });
    }

    @Test
    void returnedListIsUnmodifiable() {
        UserProfile profile = new UserProfile("u1", List.of("test_1"));
        List<RecommendationResult> results = engine.recommend(profile);
        assertThrows(UnsupportedOperationException.class, () -> results.add(null));
    }
}
