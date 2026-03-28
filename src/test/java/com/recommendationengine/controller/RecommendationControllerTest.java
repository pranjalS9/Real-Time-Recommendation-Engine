package com.recommendationengine.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.recommendationengine.cache.RecommendationCacheService;
import com.recommendationengine.engine.RecommendationEngine;
import com.recommendationengine.model.RecommendationResult;
import com.recommendationengine.model.UserProfile;
import com.recommendationengine.user.UserProfileRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(RecommendationController.class)
class RecommendationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private RecommendationEngine engine;

    @MockBean
    private RecommendationCacheService cache;

    @MockBean
    private UserProfileRepository userProfileRepository;

    private static final List<RecommendationResult> RESULTS = List.of(
            new RecommendationResult("item_2", "Inception", "A thief steals secrets", 0.91),
            new RecommendationResult("item_3", "Interstellar", "Astronauts travel through space", 0.85)
    );

    private static final UserProfile PROFILE =
            new UserProfile("user_1", List.of("item_1"));

    // --- GET /recommend/{userId} ---------------------------------------------

    @Test
    void returnsCachedResultsOnCacheHit() throws Exception {
        when(cache.get("user_1")).thenReturn(RESULTS);

        mockMvc.perform(get("/recommend/user_1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2))
                .andExpect(jsonPath("$[0].id").value("item_2"))
                .andExpect(jsonPath("$[0].score").value(0.91));

        verify(engine, never()).recommend(any());
    }

    @Test
    void computesAndCachesOnCacheMiss() throws Exception {
        when(cache.get("user_1")).thenReturn(null);
        when(userProfileRepository.findById("user_1")).thenReturn(Optional.of(PROFILE));
        when(engine.recommend(PROFILE)).thenReturn(RESULTS);

        mockMvc.perform(get("/recommend/user_1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(2));

        verify(cache).put(eq("user_1"), eq(RESULTS));
    }

    @Test
    void returns404ForUnknownUser() throws Exception {
        when(cache.get("unknown")).thenReturn(null);
        when(userProfileRepository.findById("unknown")).thenReturn(Optional.empty());

        mockMvc.perform(get("/recommend/unknown"))
                .andExpect(status().isNotFound());
    }

    @Test
    void returnsEmptyListWhenEngineFindsNoRecommendations() throws Exception {
        when(cache.get("user_1")).thenReturn(null);
        when(userProfileRepository.findById("user_1")).thenReturn(Optional.of(PROFILE));
        when(engine.recommend(PROFILE)).thenReturn(List.of());

        mockMvc.perform(get("/recommend/user_1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.length()").value(0));
    }

    // --- POST /users/{userId}/items/{itemId} ---------------------------------

    @Test
    void recordViewReturns200() throws Exception {
        mockMvc.perform(post("/users/user_1/items/item_5"))
                .andExpect(status().isOk());

        verify(userProfileRepository).addViewedItem("user_1", "item_5");
        verify(cache).evict("user_1");
    }

    @Test
    void recordViewEvictsCacheForThatUser() throws Exception {
        mockMvc.perform(post("/users/user_1/items/item_5"))
                .andExpect(status().isOk());

        verify(cache).evict("user_1");
    }

    // --- DELETE /cache/{userId} ----------------------------------------------

    @Test
    void evictCacheReturns204() throws Exception {
        mockMvc.perform(delete("/cache/user_1"))
                .andExpect(status().isNoContent());

        verify(cache).evict("user_1");
    }
}
