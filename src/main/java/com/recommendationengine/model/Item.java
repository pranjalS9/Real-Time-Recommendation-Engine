package com.recommendationengine.model;

/**
 * Represents a catalogued item (e.g. a movie, product, article).
 *
 * The {@code vector} field is null at ingestion time and gets populated
 * by {@code TfIdfVectorizer.fitAndTransform()} during startup.
 */
public class Item {

    private String id;
    private String name;
    private String description;
    private double[] vector;

    public Item() {}

    public Item(String id, String name, String description) {
        this.id = id;
        this.name = name;
        this.description = description;
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public double[] getVector() { return vector; }
    public void setVector(double[] vector) { this.vector = vector; }

    @Override
    public String toString() {
        return "Item{id='" + id + "', name='" + name + "'}";
    }
}
