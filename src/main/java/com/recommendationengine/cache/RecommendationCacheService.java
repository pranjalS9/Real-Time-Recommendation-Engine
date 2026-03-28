package com.recommendationengine.cache;

import com.recommendationengine.model.RecommendationResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisOperations;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.List;

/**
 * Cache-aside layer for recommendation results.
 *
 * <p>Key format: {@code "rec:{userId}"}
 * <p>TTL is configurable via {@code recommendation.cache.ttl-minutes} (default 10 min).
 */
@Service
public class RecommendationCacheService {

    private static final Logger log = LoggerFactory.getLogger(RecommendationCacheService.class);
    private static final String KEY_PREFIX = "rec:";

    private final RedisOperations<String, Object> redisTemplate;

    @Value("${recommendation.cache.ttl-minutes:10}")
    private long ttlMinutes;

    public RecommendationCacheService(RedisOperations<String, Object> redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    /**
     * Returns cached recommendations for the user, or {@code null} on a cache miss.
     */
    @SuppressWarnings("unchecked")
    public List<RecommendationResult> get(String userId) {
        String key = key(userId);
        try {
            Object cached = redisTemplate.opsForValue().get(key);
            if (cached != null) {
                log.debug("Cache HIT for user: {}", userId);
                return (List<RecommendationResult>) cached;
            }
        } catch (Exception e) {
            log.warn("Redis GET failed for key {} — treating as cache miss", key, e);
        }
        log.debug("Cache MISS for user: {}", userId);
        return null;
    }

    /**
     * Stores recommendations for the user with the configured TTL.
     */
    public void put(String userId, List<RecommendationResult> results) {
        String key = key(userId);
        try {
            redisTemplate.opsForValue().set(key, results, Duration.ofMinutes(ttlMinutes));
            log.debug("Cached {} recommendations for user: {} (TTL: {}m)", results.size(), userId, ttlMinutes);
        } catch (Exception e) {
            log.warn("Redis SET failed for key {} — continuing without caching", key, e);
        }
    }

    /**
     * Evicts the cached recommendations for the user.
     */
    public void evict(String userId) {
        String key = key(userId);
        try {
            Boolean deleted = redisTemplate.delete(key);
            log.info("Cache evicted for user: {} (key existed: {})", userId, deleted);
        } catch (Exception e) {
            log.warn("Redis DELETE failed for key {}", key, e);
        }
    }

    private String key(String userId) {
        return KEY_PREFIX + userId;
    }
}
