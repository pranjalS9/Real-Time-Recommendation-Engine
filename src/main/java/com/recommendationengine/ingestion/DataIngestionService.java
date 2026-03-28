package com.recommendationengine.ingestion;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import com.recommendationengine.model.Item;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Loads item data from a CSV file on the classpath and maintains
 * an in-memory store for fast lookup by ID.
 *
 * <p>Expected CSV format (header required):
 * <pre>
 *   id,name,description
 *   item_1,The Dark Knight,A vigilante battles a criminal mastermind...
 * </pre>
 *
 * <p>Call {@link #ingest()} on startup to populate the store.
 * Call it again to re-load after the file changes (e.g. via the admin endpoint).
 */
@Service
public class DataIngestionService {

    private static final Logger log = LoggerFactory.getLogger(DataIngestionService.class);
    private static final String DEFAULT_CSV_PATH = "/data/items.csv";

    // id → Item; ConcurrentHashMap so reads during a re-ingest are safe
    private final Map<String, Item> itemStore = new ConcurrentHashMap<>();

    public List<Item> ingest() {
        return ingest(DEFAULT_CSV_PATH);
    }

    /**
     * Loads items from the given classpath resource path.
     * Clears the existing store before loading.
     *
     * @param classpathResource e.g. {@code "/data/items.csv"}
     * @return the list of successfully parsed items
     */
    public List<Item> ingest(String classpathResource) {
        itemStore.clear();

        InputStream stream = getClass().getResourceAsStream(classpathResource);
        if (stream == null) {
            log.error("CSV file not found on classpath: {}", classpathResource);
            return Collections.emptyList();
        }

        List<Item> items = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new InputStreamReader(stream))) {
            String[] headers = reader.readNext();
            if (headers == null) {
                log.warn("CSV file is empty: {}", classpathResource);
                return Collections.emptyList();
            }

            String[] row;
            int lineNumber = 1;
            while ((row = reader.readNext()) != null) {
                lineNumber++;
                if (row.length < 3) {
                    log.warn("Skipping malformed row at line {}: {}", lineNumber, Arrays.toString(row));
                    continue;
                }
                String id          = row[0].trim();
                String name        = row[1].trim();
                String description = row[2].trim();

                if (id.isEmpty()) {
                    log.warn("Skipping row at line {} — empty id", lineNumber);
                    continue;
                }

                Item item = new Item(id, name, description);
                items.add(item);
                itemStore.put(id, item);
            }
        } catch (IOException | CsvValidationException e) {
            log.error("Failed to read CSV file: {}", classpathResource, e);
            return Collections.emptyList();
        }

        log.info("Ingested {} items from {}", items.size(), classpathResource);
        return items;
    }

    public Item findById(String id) {
        return itemStore.get(id);
    }

    public Collection<Item> getAllItems() {
        return Collections.unmodifiableCollection(itemStore.values());
    }

    public int size() {
        return itemStore.size();
    }
}
