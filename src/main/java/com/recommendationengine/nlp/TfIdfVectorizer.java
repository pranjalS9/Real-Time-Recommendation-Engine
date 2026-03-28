package com.recommendationengine.nlp;

import com.recommendationengine.model.Item;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * Converts item descriptions into TF-IDF vectors.
 *
 * <p>How it works:
 * <ul>
 *   <li><b>TF</b> (Term Frequency) — how often a word appears in this item's
 *       description, normalized by the total number of tokens in that description.</li>
 *   <li><b>IDF</b> (Inverse Document Frequency) — {@code log(N / df)} where N is the
 *       total number of items and df is how many items contain that word.
 *       Penalizes common words, rewards rare discriminating words.</li>
 *   <li><b>TF-IDF</b> — {@code TF × IDF}, the final weight for each term.</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>
 *   vectorizer.fitAndTransform(items);  // builds vocab + writes vectors onto items
 *   double[] v = vectorizer.transform(newItem);  // after fit, transform any item
 * </pre>
 */
@Component
public class TfIdfVectorizer {

    private static final Logger log = LoggerFactory.getLogger(TfIdfVectorizer.class);

    private final TextPreprocessor preprocessor;

    @Value("${lsh.vector-dimensions:500}")
    private int maxVocabSize = 500;

    // term → its index in the output vector
    private final Map<String, Integer> vocabulary = new LinkedHashMap<>();

    // term → precomputed IDF score
    private final Map<String, Double> idfScores = new HashMap<>();

    private boolean fitted = false;

    public TfIdfVectorizer(TextPreprocessor preprocessor) {
        this.preprocessor = preprocessor;
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Builds vocabulary and IDF scores from the corpus.
     * Keeps only the top {@code maxVocabSize} most discriminating terms.
     */
    public void fit(List<Item> items) {
        if (items == null || items.isEmpty()) {
            throw new IllegalArgumentException("Cannot fit on an empty item list");
        }

        int totalDocs = items.size();

        // Count how many documents contain each term (document frequency)
        Map<String, Integer> docFrequency = new HashMap<>();
        for (Item item : items) {
            Set<String> uniqueTerms = new HashSet<>(preprocessor.tokenize(item.getDescription()));
            for (String term : uniqueTerms) {
                docFrequency.merge(term, 1, Integer::sum);
            }
        }

        // Compute IDF for each term
        Map<String, Double> rawIdf = new HashMap<>();
        for (Map.Entry<String, Integer> entry : docFrequency.entrySet()) {
            // +1 smoothing prevents division by zero for unseen terms
            double idf = Math.log((double) (totalDocs + 1) / (entry.getValue() + 1));
            rawIdf.put(entry.getKey(), idf);
        }

        // Keep the top-N terms ranked by IDF (most discriminating first)
        List<Map.Entry<String, Double>> sorted = new ArrayList<>(rawIdf.entrySet());
        sorted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        vocabulary.clear();
        idfScores.clear();

        int vocabSize = Math.min(maxVocabSize, sorted.size());
        for (int i = 0; i < vocabSize; i++) {
            String term = sorted.get(i).getKey();
            vocabulary.put(term, i);
            idfScores.put(term, sorted.get(i).getValue());
        }

        fitted = true;
        log.info("TF-IDF fitted on {} documents — vocabulary size: {}", totalDocs, vocabulary.size());
    }

    /**
     * Converts a single item's description into a TF-IDF vector.
     * Must call {@link #fit(List)} first.
     *
     * @return a {@code double[]} of length equal to the vocabulary size
     */
    public double[] transform(Item item) {
        if (!fitted) {
            throw new IllegalStateException("Vectorizer must be fitted before calling transform()");
        }

        List<String> tokens = preprocessor.tokenize(item.getDescription());
        if (tokens.isEmpty()) {
            return new double[vocabulary.size()];
        }

        int totalTerms = tokens.size();

        // Term frequency count for this document
        Map<String, Integer> termCount = new HashMap<>();
        for (String token : tokens) {
            termCount.merge(token, 1, Integer::sum);
        }

        double[] vector = new double[vocabulary.size()];
        for (Map.Entry<String, Integer> entry : termCount.entrySet()) {
            String term = entry.getKey();
            Integer index = vocabulary.get(term);
            if (index == null) continue; // term outside vocabulary

            double tf = (double) entry.getValue() / totalTerms;
            double idf = idfScores.get(term);
            vector[index] = tf * idf;
        }

        return vector;
    }

    /**
     * Fits the vectorizer on the corpus then transforms every item,
     * writing the resulting vector directly onto each {@link Item}.
     */
    public void fitAndTransform(List<Item> items) {
        fit(items);
        for (Item item : items) {
            item.setVector(transform(item));
        }
        log.info("TF-IDF vectors computed for {} items (dimensions: {})", items.size(), vocabulary.size());
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public int getVocabularySize() {
        return vocabulary.size();
    }

    public boolean isFitted() {
        return fitted;
    }

    /** Exposed for testing — returns an unmodifiable view of the vocabulary. */
    public Map<String, Integer> getVocabulary() {
        return Collections.unmodifiableMap(vocabulary);
    }
}
