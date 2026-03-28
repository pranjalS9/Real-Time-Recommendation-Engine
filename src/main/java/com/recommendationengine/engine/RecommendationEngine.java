package com.recommendationengine.engine;

import com.recommendationengine.index.LSHIndex;
import com.recommendationengine.ingestion.DataIngestionService;
import com.recommendationengine.model.Item;
import com.recommendationengine.model.RecommendationResult;
import com.recommendationengine.model.UserProfile;
import com.recommendationengine.similarity.CosineSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Core recommendation logic.
 *
 * <p>Given a {@link UserProfile} (a list of viewed item IDs):
 * <ol>
 *   <li>Fetch the TF-IDF vectors for all viewed items.</li>
 *   <li>Average them into a single <em>user preference vector</em>.</li>
 *   <li>Ask {@link LSHIndex} for the candidate set (the matching hash bucket).</li>
 *   <li>Score each candidate with {@link CosineSimilarity}.</li>
 *   <li>Filter out already-viewed items and return the top-N results.</li>
 * </ol>
 */
@Service
public class RecommendationEngine {

    private static final Logger log = LoggerFactory.getLogger(RecommendationEngine.class);

    private final DataIngestionService ingestionService;
    private final LSHIndex lshIndex;
    private final CosineSimilarity cosineSimilarity;

    @Value("${recommendation.top-n:10}")
    private int defaultTopN = 10;

    public RecommendationEngine(DataIngestionService ingestionService,
                                LSHIndex lshIndex,
                                CosineSimilarity cosineSimilarity) {
        this.ingestionService = ingestionService;
        this.lshIndex = lshIndex;
        this.cosineSimilarity = cosineSimilarity;
    }

    public List<RecommendationResult> recommend(UserProfile profile) {
        return recommend(profile, defaultTopN);
    }

    /**
     * Returns up to {@code topN} recommendations for the given user.
     *
     * @param profile user containing their viewed item IDs
     * @param topN    maximum number of results to return
     * @return ranked list of recommendations, highest score first
     */
    public List<RecommendationResult> recommend(UserProfile profile, int topN) {
        List<String> viewedIds = profile.viewedItemIds();

        if (viewedIds == null || viewedIds.isEmpty()) {
            log.warn("User {} has no viewing history — returning empty recommendations", profile.userId());
            return Collections.emptyList();
        }

        // Step 1 — collect vectors for viewed items
        List<double[]> viewedVectors = new ArrayList<>();
        for (String id : viewedIds) {
            Item item = ingestionService.findById(id);
            if (item != null && item.getVector() != null) {
                viewedVectors.add(item.getVector());
            } else {
                log.debug("Item not found or not vectorized: {}", id);
            }
        }

        if (viewedVectors.isEmpty()) {
            log.warn("No vectorized items found for user {} — returning empty recommendations", profile.userId());
            return Collections.emptyList();
        }

        // Step 2 — build user preference vector by averaging viewed vectors
        double[] preferenceVector = average(viewedVectors);

        // Step 3 — retrieve candidates from LSH bucket
        List<Item> candidates = lshIndex.getCandidates(preferenceVector);
        log.debug("User {}: {} candidates retrieved from LSH", profile.userId(), candidates.size());

        // Step 4 — score and filter
        Set<String> viewed = new HashSet<>(viewedIds);
        List<RecommendationResult> results = new ArrayList<>();

        for (Item candidate : candidates) {
            if (viewed.contains(candidate.getId())) continue;
            if (candidate.getVector() == null) continue;

            double score = cosineSimilarity.compute(preferenceVector, candidate.getVector());
            results.add(new RecommendationResult(
                    candidate.getId(),
                    candidate.getName(),
                    candidate.getDescription(),
                    score
            ));
        }

        // Step 5 — sort descending by score, return top-N
        results.sort(Comparator.comparingDouble(RecommendationResult::score).reversed());
        return Collections.unmodifiableList(results.subList(0, Math.min(topN, results.size())));
    }

    // -------------------------------------------------------------------------

    private double[] average(List<double[]> vectors) {
        int dim = vectors.get(0).length;
        double[] avg = new double[dim];
        for (double[] v : vectors) {
            for (int i = 0; i < dim; i++) {
                avg[i] += v[i];
            }
        }
        for (int i = 0; i < dim; i++) {
            avg[i] /= vectors.size();
        }
        return avg;
    }
}
