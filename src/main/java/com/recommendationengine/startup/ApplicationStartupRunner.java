package com.recommendationengine.startup;

import com.recommendationengine.index.LSHIndex;
import com.recommendationengine.ingestion.DataIngestionService;
import com.recommendationengine.model.Item;
import com.recommendationengine.model.UserProfile;
import com.recommendationengine.nlp.TfIdfVectorizer;
import com.recommendationengine.user.UserProfileRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Runs once on application startup and initialises the recommendation pipeline:
 *
 * <ol>
 *   <li>Ingest items from CSV into the in-memory item store.</li>
 *   <li>Fit the TF-IDF vectorizer on the corpus and compute all item vectors.</li>
 *   <li>Build the LSH index over the computed vectors.</li>
 *   <li>Seed demo user profiles for manual testing.</li>
 * </ol>
 *
 * If any step produces zero items the application logs an error but continues
 * rather than crashing — endpoints will return empty results until data is
 * available via the re-ingest endpoint.
 */
@Component
public class ApplicationStartupRunner implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(ApplicationStartupRunner.class);

    private final DataIngestionService ingestionService;
    private final TfIdfVectorizer vectorizer;
    private final LSHIndex lshIndex;
    private final UserProfileRepository userProfileRepository;

    public ApplicationStartupRunner(DataIngestionService ingestionService,
                                    TfIdfVectorizer vectorizer,
                                    LSHIndex lshIndex,
                                    UserProfileRepository userProfileRepository) {
        this.ingestionService = ingestionService;
        this.vectorizer = vectorizer;
        this.lshIndex = lshIndex;
        this.userProfileRepository = userProfileRepository;
    }

    @Override
    public void run(String... args) {
        log.info("=== Recommendation Engine startup pipeline beginning ===");
        long start = System.currentTimeMillis();

        // Step 1 — ingest
        List<Item> items = ingestionService.ingest();
        if (items.isEmpty()) {
            log.error("No items ingested — recommendation engine will return empty results");
            return;
        }
        log.info("Step 1/3 complete — {} items ingested", items.size());

        // Step 2 — vectorize
        vectorizer.fitAndTransform(items);
        log.info("Step 2/3 complete — TF-IDF vectors computed (dim={})", vectorizer.getVocabularySize());

        // Step 3 — index
        lshIndex.build(items);
        log.info("Step 3/3 complete — LSH index built ({} buckets)", lshIndex.getBucketCount());

        // Seed demo profiles
        seedDemoProfiles();

        long elapsed = System.currentTimeMillis() - start;
        log.info("=== Startup pipeline complete in {}ms — service is ready ===", elapsed);
    }

    private void seedDemoProfiles() {
        List<UserProfile> demoProfiles = List.of(
                new UserProfile("user_sci_fi",   List.of("item_2", "item_3", "item_4")),   // Inception, Interstellar, Matrix
                new UserProfile("user_action",   List.of("item_1", "item_5", "item_9")),   // Dark Knight, Endgame, Mad Max
                new UserProfile("user_horror",   List.of("item_11", "item_12", "item_29")),// Get Out, Hereditary, Us
                new UserProfile("user_animation",List.of("item_40", "item_41", "item_42")),// Coco, Soul, Ratatouille
                new UserProfile("user_drama",    List.of("item_38", "item_39", "item_6"))  // Portrait, Manchester, Parasite
        );

        demoProfiles.forEach(userProfileRepository::save);
        log.info("Seeded {} demo user profiles", demoProfiles.size());
    }
}
