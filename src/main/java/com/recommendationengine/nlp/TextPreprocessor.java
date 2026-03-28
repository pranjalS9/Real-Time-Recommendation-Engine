package com.recommendationengine.nlp;

import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Converts raw text into a list of clean, meaningful tokens.
 *
 * Pipeline:
 *   1. Lowercase
 *   2. Strip non-alphabetic characters
 *   3. Split on whitespace
 *   4. Remove single-character tokens
 *   5. Remove stop words
 */
@Component
public class TextPreprocessor {

    private static final Set<String> STOP_WORDS = Set.of(
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
            "so", "yet", "both", "either", "neither", "one", "two", "three", "as",
            "it", "its", "he", "she", "they", "we", "his", "her", "their", "our",
            "who", "which", "that", "this", "these", "those", "what", "when",
            "where", "while", "after", "before", "during", "into", "through",
            "about", "against", "between", "up", "down", "out", "off", "over",
            "under", "again", "then", "once", "only", "also", "just", "if", "how"
    );

    public List<String> tokenize(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        return Arrays.stream(
                        text.toLowerCase()
                            .replaceAll("[^a-z\\s]", " ")
                            .split("\\s+")
                )
                .filter(token -> !token.isBlank())
                .filter(token -> token.length() > 1)
                .filter(token -> !STOP_WORDS.contains(token))
                .toList();
    }
}
