package com.recommendationengine.user;

import com.recommendationengine.model.UserProfile;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory store for user profiles.
 *
 * <p>Seeded with demo profiles at startup via {@code ApplicationStartupRunner}.
 * In a production system this would be backed by a database or an event-driven
 * view-history service.
 */
@Repository
public class UserProfileRepository {

    private final Map<String, UserProfile> store = new ConcurrentHashMap<>();

    public void save(UserProfile profile) {
        store.put(profile.userId(), profile);
    }

    public Optional<UserProfile> findById(String userId) {
        return Optional.ofNullable(store.get(userId));
    }

    public void addViewedItem(String userId, String itemId) {
        store.compute(userId, (id, existing) -> {
            if (existing == null) {
                return new UserProfile(userId, List.of(itemId));
            }
            List<String> updated = new java.util.ArrayList<>(existing.viewedItemIds());
            if (!updated.contains(itemId)) {
                updated.add(itemId);
            }
            return new UserProfile(userId, updated);
        });
    }

    public int size() {
        return store.size();
    }
}
