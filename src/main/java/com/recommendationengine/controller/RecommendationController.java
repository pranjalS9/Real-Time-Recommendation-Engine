package com.recommendationengine.controller;

import com.recommendationengine.cache.RecommendationCacheService;
import com.recommendationengine.engine.RecommendationEngine;
import com.recommendationengine.model.RecommendationResult;
import com.recommendationengine.model.UserProfile;
import com.recommendationengine.user.UserProfileRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * REST API for the recommendation engine.
 *
 * <pre>
 *   GET  /recommend/{userId}          → top-N recommendations (cache-aside)
 *   POST /users/{userId}/items/{itemId} → record a view event
 *   DELETE /cache/{userId}            → evict cached recommendations
 * </pre>
 */
@RestController
public class RecommendationController {

    private static final Logger log = LoggerFactory.getLogger(RecommendationController.class);

    private final RecommendationEngine engine;
    private final RecommendationCacheService cache;
    private final UserProfileRepository userProfileRepository;

    public RecommendationController(RecommendationEngine engine,
                                    RecommendationCacheService cache,
                                    UserProfileRepository userProfileRepository) {
        this.engine = engine;
        this.cache = cache;
        this.userProfileRepository = userProfileRepository;
    }

    /**
     * Returns top-N recommendations for the user.
     *
     * <p>Cache-aside strategy:
     * <ol>
     *   <li>Check Redis. Return immediately on HIT.</li>
     *   <li>On MISS: compute via engine, store in Redis, return results.</li>
     * </ol>
     */
    @GetMapping("/recommend/{userId}")
    public ResponseEntity<List<RecommendationResult>> recommend(@PathVariable String userId) {
        // 1. Cache lookup
        List<RecommendationResult> cached = cache.get(userId);
        if (cached != null) {
            return ResponseEntity.ok(cached);
        }

        // 2. Profile lookup
        UserProfile profile = userProfileRepository.findById(userId).orElse(null);
        if (profile == null) {
            log.warn("User not found: {}", userId);
            return ResponseEntity.notFound().build();
        }

        // 3. Compute and cache
        List<RecommendationResult> results = engine.recommend(profile);
        cache.put(userId, results);

        return ResponseEntity.ok(results);
    }

    /**
     * Records that a user viewed an item and evicts their stale cached recommendations.
     */
    @PostMapping("/users/{userId}/items/{itemId}")
    public ResponseEntity<Void> recordView(@PathVariable String userId,
                                           @PathVariable String itemId) {
        userProfileRepository.addViewedItem(userId, itemId);
        cache.evict(userId); // stale after new view event
        log.info("Recorded view: user={} item={}", userId, itemId);
        return ResponseEntity.ok().build();
    }

    /**
     * Evicts cached recommendations for a user — next request recomputes.
     */
    @DeleteMapping("/cache/{userId}")
    public ResponseEntity<Void> evictCache(@PathVariable String userId) {
        cache.evict(userId);
        return ResponseEntity.noContent().build();
    }
}
