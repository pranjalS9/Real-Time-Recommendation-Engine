package com.recommendationengine.nlp;

import com.recommendationengine.model.Item;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class TfIdfVectorizerTest {

    private TfIdfVectorizer vectorizer;

    private static final Item ITEM_1 = new Item("1", "Dark Knight", "batman battles joker gotham city crime");
    private static final Item ITEM_2 = new Item("2", "Inception",   "thief steals secrets dreams corporation");
    private static final Item ITEM_3 = new Item("3", "Interstellar","astronauts travel wormhole space galaxy");

    @BeforeEach
    void setUp() {
        vectorizer = new TfIdfVectorizer(new TextPreprocessor());
    }

    // --- fit() ---------------------------------------------------------------

    @Test
    void fitBuildsNonEmptyVocabulary() {
        vectorizer.fit(List.of(ITEM_1, ITEM_2, ITEM_3));
        assertTrue(vectorizer.getVocabularySize() > 0);
    }

    @Test
    void fitMarksVectorizerAsFitted() {
        assertFalse(vectorizer.isFitted());
        vectorizer.fit(List.of(ITEM_1, ITEM_2));
        assertTrue(vectorizer.isFitted());
    }

    @Test
    void fitThrowsOnEmptyList() {
        assertThrows(IllegalArgumentException.class, () -> vectorizer.fit(List.of()));
    }

    @Test
    void fitThrowsOnNullList() {
        assertThrows(IllegalArgumentException.class, () -> vectorizer.fit(null));
    }

    @Test
    void vocabularyContainsTermsFromDescriptions() {
        vectorizer.fit(List.of(ITEM_1, ITEM_2, ITEM_3));
        // These terms appear in descriptions and should be in vocabulary
        assertTrue(vectorizer.getVocabulary().containsKey("batman")
                || vectorizer.getVocabulary().containsKey("joker")
                || vectorizer.getVocabulary().containsKey("gotham"));
    }

    // --- transform() ---------------------------------------------------------

    @Test
    void transformThrowsIfNotFitted() {
        assertThrows(IllegalStateException.class, () -> vectorizer.transform(ITEM_1));
    }

    @Test
    void transformReturnsVectorOfCorrectDimension() {
        vectorizer.fit(List.of(ITEM_1, ITEM_2, ITEM_3));
        double[] vector = vectorizer.transform(ITEM_1);
        assertEquals(vectorizer.getVocabularySize(), vector.length);
    }

    @Test
    void transformReturnsNonZeroVectorForKnownTerms() {
        vectorizer.fit(List.of(ITEM_1, ITEM_2, ITEM_3));
        double[] vector = vectorizer.transform(ITEM_1);
        double sum = 0;
        for (double v : vector) sum += v;
        assertTrue(sum > 0, "Vector should have at least some non-zero entries");
    }

    @Test
    void transformReturnsZeroVectorForEmptyDescription() {
        vectorizer.fit(List.of(ITEM_1, ITEM_2, ITEM_3));
        Item empty = new Item("e", "Empty", "");
        double[] vector = vectorizer.transform(empty);
        for (double v : vector) assertEquals(0.0, v, 1e-10);
    }

    @Test
    void rareTermsGetHigherIdfThanCommonTerms() {
        // "space" appears in only one doc; "travel" appears in only one doc
        // Common words are filtered by stop-word list, so rare corpus terms get higher IDF
        Item i1 = new Item("1", "A", "dragon dragon dragon wizard");
        Item i2 = new Item("2", "B", "dragon knight castle");
        Item i3 = new Item("3", "C", "wizard knight forest");

        vectorizer.fit(List.of(i1, i2, i3));

        // "dragon" appears in 2 docs, "wizard" appears in 2 docs, "forest" in 1 doc
        // "forest" should have higher IDF than "dragon"
        Map<String, Integer> vocab = vectorizer.getVocabulary();
        if (vocab.containsKey("forest") && vocab.containsKey("dragon")) {
            double[] forestItem = vectorizer.transform(i3);
            double[] dragonItem = vectorizer.transform(i1);
            int forestIdx = vocab.get("forest");
            int dragonIdx = vocab.get("dragon");
            assertTrue(forestItem[forestIdx] / (1.0 / 1) >= dragonItem[dragonIdx] / (3.0 / 1),
                    "Rare term 'forest' should contribute more per occurrence than common term 'dragon'");
        }
    }

    // --- fitAndTransform() ---------------------------------------------------

    @Test
    void fitAndTransformWritesVectorsOntoItems() {
        List<Item> items = List.of(
                new Item("1", "Dark Knight", "batman battles joker gotham city crime"),
                new Item("2", "Inception",   "thief steals secrets dreams corporation"),
                new Item("3", "Interstellar","astronauts travel wormhole space galaxy")
        );

        vectorizer.fitAndTransform(items);

        for (Item item : items) {
            assertNotNull(item.getVector(), "Vector must be set on item: " + item.getId());
            assertEquals(vectorizer.getVocabularySize(), item.getVector().length);
        }
    }

    @Test
    void differentItemsProduceDifferentVectors() {
        List<Item> items = List.of(
                new Item("1", "A", "dragon wizard castle magic spell"),
                new Item("2", "B", "space rocket orbit planet galaxy")
        );
        vectorizer.fitAndTransform(items);

        double[] v1 = items.get(0).getVector();
        double[] v2 = items.get(1).getVector();

        boolean different = false;
        for (int i = 0; i < v1.length; i++) {
            if (Math.abs(v1[i] - v2[i]) > 1e-10) {
                different = true;
                break;
            }
        }
        assertTrue(different, "Vectors for semantically different items must differ");
    }
}
