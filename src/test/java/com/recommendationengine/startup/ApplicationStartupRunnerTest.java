package com.recommendationengine.startup;

import com.recommendationengine.index.LSHIndex;
import com.recommendationengine.ingestion.DataIngestionService;
import com.recommendationengine.nlp.TfIdfVectorizer;
import com.recommendationengine.user.UserProfileRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.test.context.ActiveProfiles;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies the startup pipeline ran end-to-end.
 *
 * Redis is mocked so no running Redis instance is needed in CI.
 */
@SpringBootTest
@ActiveProfiles("test")
class ApplicationStartupRunnerTest {

    // Mock the connection factory so the context starts without a live Redis instance.
    // Our RedisConfig uses this to build RedisTemplate, which stays unconnected in tests.
    @MockBean
    private RedisConnectionFactory redisConnectionFactory;

    @Autowired
    private DataIngestionService ingestionService;

    @Autowired
    private TfIdfVectorizer vectorizer;

    @Autowired
    private LSHIndex lshIndex;

    @Autowired
    private UserProfileRepository userProfileRepository;

    @Test
    void itemsAreIngestedOnStartup() {
        assertTrue(ingestionService.size() > 0, "Items must be ingested during startup");
    }

    @Test
    void vectorizerIsFittedOnStartup() {
        assertTrue(vectorizer.isFitted(), "TF-IDF vectorizer must be fitted during startup");
        assertTrue(vectorizer.getVocabularySize() > 0, "Vocabulary must be non-empty after startup");
    }

    @Test
    void allItemsHaveVectorsAfterStartup() {
        ingestionService.getAllItems().forEach(item ->
                assertNotNull(item.getVector(),
                        "Every item must have a vector after startup: " + item.getId())
        );
    }

    @Test
    void lshIndexIsBuiltOnStartup() {
        assertTrue(lshIndex.getBucketCount() > 0, "LSH index must have buckets after startup");
        assertEquals(ingestionService.size(), lshIndex.getTotalItemCount(),
                "LSH index item count must match ingested item count");
    }

    @Test
    void demoUserProfilesAreSeededOnStartup() {
        assertTrue(userProfileRepository.size() > 0, "Demo user profiles must be seeded during startup");
        assertTrue(userProfileRepository.findById("user_sci_fi").isPresent());
        assertTrue(userProfileRepository.findById("user_action").isPresent());
        assertTrue(userProfileRepository.findById("user_horror").isPresent());
        assertTrue(userProfileRepository.findById("user_animation").isPresent());
        assertTrue(userProfileRepository.findById("user_drama").isPresent());
    }

    @Test
    void vectorDimensionsMatchVocabularySize() {
        int vocabSize = vectorizer.getVocabularySize();
        ingestionService.getAllItems().forEach(item ->
                assertEquals(vocabSize, item.getVector().length,
                        "Vector dimension must match vocabulary size for item: " + item.getId())
        );
    }
}
