package com.recommendationengine.cache;

import com.recommendationengine.model.RecommendationResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.data.redis.core.RedisOperations;
import org.springframework.data.redis.core.ValueOperations;

import java.time.Duration;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class RecommendationCacheServiceTest {

    @Mock
    private RedisOperations<String, Object> redisTemplate;

    @Mock
    private ValueOperations<String, Object> valueOps;

    private RecommendationCacheService cacheService;

    private static final List<RecommendationResult> RESULTS = List.of(
            new RecommendationResult("item_1", "Inception", "A thief steals secrets", 0.91),
            new RecommendationResult("item_2", "Interstellar", "Astronauts travel through space", 0.85)
    );

    @BeforeEach
    void setUp() {
        when(redisTemplate.opsForValue()).thenReturn(valueOps);
        cacheService = new RecommendationCacheService(redisTemplate);
    }

    // --- get() ---------------------------------------------------------------

    @Test
    void getReturnsCachedResultOnHit() {
        when(valueOps.get("rec:user_1")).thenReturn(RESULTS);

        List<RecommendationResult> result = cacheService.get("user_1");

        assertEquals(RESULTS, result);
    }

    @Test
    void getReturnsNullOnCacheMiss() {
        when(valueOps.get("rec:user_1")).thenReturn(null);

        assertNull(cacheService.get("user_1"));
    }

    @Test
    void getReturnsNullWhenRedisThrows() {
        // Make opsForValue() itself throw — avoids default-method stubbing pitfalls
        when(redisTemplate.opsForValue()).thenThrow(new RuntimeException("Redis down"));

        assertNull(cacheService.get("user_1"));
    }

    @Test
    void getUsesCorrectKeyPrefix() {
        when(valueOps.get("rec:user_42")).thenReturn(RESULTS);

        cacheService.get("user_42");

        verify(valueOps).get("rec:user_42");
    }

    // --- put() ---------------------------------------------------------------

    @Test
    void putStoresResultsWithTtl() {
        cacheService.put("user_1", RESULTS);

        verify(valueOps).set(eq("rec:user_1"), eq(RESULTS), any(Duration.class));
    }

    @Test
    void putUsesCorrectKeyPrefix() {
        cacheService.put("user_42", RESULTS);

        verify(valueOps).set(eq("rec:user_42"), any(), any(Duration.class));
    }

    @Test
    void putDoesNotThrowWhenRedisThrows() {
        when(redisTemplate.opsForValue()).thenThrow(new RuntimeException("Redis down"));

        assertDoesNotThrow(() -> cacheService.put("user_1", RESULTS));
    }

    // --- evict() -------------------------------------------------------------

    @Test
    void evictDeletesCorrectKey() {
        cacheService.evict("user_1");

        verify(redisTemplate).delete("rec:user_1");
    }

    @Test
    void evictDoesNotThrowWhenRedisThrows() {
        when(redisTemplate.delete(anyString())).thenThrow(new RuntimeException("Redis down"));

        assertDoesNotThrow(() -> cacheService.evict("user_1"));
    }
}
