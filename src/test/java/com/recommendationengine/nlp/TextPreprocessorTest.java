package com.recommendationengine.nlp;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TextPreprocessorTest {

    private TextPreprocessor preprocessor;

    @BeforeEach
    void setUp() {
        preprocessor = new TextPreprocessor();
    }

    @Test
    void lowercasesInput() {
        List<String> tokens = preprocessor.tokenize("Action THRILLER Drama");
        assertTrue(tokens.contains("action"));
        assertTrue(tokens.contains("thriller"));
        assertTrue(tokens.contains("drama"));
    }

    @Test
    void stripsPunctuation() {
        List<String> tokens = preprocessor.tokenize("sci-fi, thriller: action!");
        // hyphens become spaces so "sci-fi" → ["sci", "fi"]
        assertFalse(tokens.stream().anyMatch(t -> t.contains("-")));
        assertFalse(tokens.stream().anyMatch(t -> t.contains(",")));
        assertFalse(tokens.stream().anyMatch(t -> t.contains("!")));
    }

    @Test
    void removesStopWords() {
        List<String> tokens = preprocessor.tokenize("a man and the woman");
        assertFalse(tokens.contains("a"));
        assertFalse(tokens.contains("and"));
        assertFalse(tokens.contains("the"));
    }

    @Test
    void removesSingleCharacterTokens() {
        List<String> tokens = preprocessor.tokenize("x action y thriller z");
        assertFalse(tokens.contains("x"));
        assertFalse(tokens.contains("y"));
        assertFalse(tokens.contains("z"));
        assertTrue(tokens.contains("action"));
        assertTrue(tokens.contains("thriller"));
    }

    @Test
    void returnsEmptyListForNullInput() {
        List<String> tokens = preprocessor.tokenize(null);
        assertTrue(tokens.isEmpty());
    }

    @Test
    void returnsEmptyListForBlankInput() {
        List<String> tokens = preprocessor.tokenize("   ");
        assertTrue(tokens.isEmpty());
    }

    @Test
    void handlesNormalDescription() {
        List<String> tokens = preprocessor.tokenize("A vigilante battles a criminal mastermind in Gotham City");
        assertTrue(tokens.contains("vigilante"));
        assertTrue(tokens.contains("battles"));
        assertTrue(tokens.contains("criminal"));
        assertTrue(tokens.contains("mastermind"));
        assertTrue(tokens.contains("gotham"));
        assertTrue(tokens.contains("city"));
        assertFalse(tokens.contains("a"));
        assertFalse(tokens.contains("in"));
    }
}
