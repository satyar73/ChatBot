from difflib import SequenceMatcher
import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download necessary NLTK resources (uncomment these when first running)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

class SimilarityEngines:
    """Class containing comprehensive similarity testing methods for text comparison"""

    @staticmethod
    def preprocess_text(text: str, remove_stopwords: bool = True,
                        lemmatize: bool = True) -> List[str]:
        """Preprocess text for better comparison"""
        # Convert to lowercase
        text = text.lower()

        # Simple tokenization fallback when NLTK is not available
        try:
            # Try to use NLTK tokenization
            tokens = word_tokenize(text)
        except (ImportError, LookupError):
            # Fallback to basic tokenization
            import re
            tokens = re.findall(r'\b\w+\b', text)

        # Remove stopwords if requested
        if remove_stopwords:
            try:
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
            except (ImportError, LookupError):
                # Fallback - just use a basic list of common English stopwords
                basic_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                                  'is', 'in', 'into', 'of', 'for', 'with', 'by', 'to', 'from', 'at', 'on'}
                tokens = [token for token in tokens if token not in basic_stopwords]

        # Lemmatize if requested
        if lemmatize:
            try:
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            except (ImportError, LookupError):
                # Skip lemmatization if NLTK is not available
                pass

        return tokens

    @staticmethod
    def basic_similarity(text1: str, text2: str) -> float:
        """Calculate basic character-level similarity using SequenceMatcher"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    @staticmethod
    def jaccard_similarity(text1: str, text2: str, preprocess: bool = True) -> float:
        """Calculate Jaccard similarity (intersection over union) between two texts"""
        if preprocess:
            tokens1 = set(SimilarityEngines.preprocess_text(text1))
            tokens2 = set(SimilarityEngines.preprocess_text(text2))
        else:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def dice_coefficient(text1: str, text2: str, preprocess: bool = True) -> float:
        """Calculate Dice coefficient (2*intersection over sum of sizes) between two texts"""
        if preprocess:
            tokens1 = set(SimilarityEngines.preprocess_text(text1))
            tokens2 = set(SimilarityEngines.preprocess_text(text2))
        else:
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())

        intersection = len(tokens1.intersection(tokens2))
        total = len(tokens1) + len(tokens2)

        if total == 0:
            return 0.0
        return 2 * intersection / total

    @staticmethod
    def cosine_similarity_tfidf(text1: str, text2: str) -> float:
        """Calculate cosine similarity using TF-IDF vectors"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]

    @staticmethod
    def cosine_similarity_count(text1: str, text2: str) -> float:
        """Calculate cosine similarity using count vectors"""
        vectorizer = CountVectorizer()
        count_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(count_matrix[0:1], count_matrix[1:2])
        return cosine_sim[0][0]

    @staticmethod
    def extract_key_concepts(text: str, include_common_words: bool = True) -> List[str]:
        """Extract key concepts from text (noun phrases, entities, etc.)"""
        # Extract capitalized phrases (potential proper nouns/concepts)
        concepts = []

        capitalized_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        concepts.extend(re.findall(capitalized_pattern, text))

        # Add numerical values which might be important data points
        numbers_pattern = r'\b\d+(?:\.\d+)?%?\b'
        concepts.extend(re.findall(numbers_pattern, text))

        # Extract quoted phrases which often represent key concepts
        quoted_pattern = r'"([^"]*)"'
        concepts.extend(re.findall(quoted_pattern, text))

        # Include common but important words that might be missed by capitalization
        if include_common_words:
            common_important_words = ["this", "with", "these", "those", "where", "when", "how",
                                      "why", "what", "who", "which"]
            for word in common_important_words:
                word_pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(word_pattern, text, re.IGNORECASE):
                    concepts.append(word)

        # Extract key noun phrases using POS tagging (if nltk is available)
        try:
            tokens = nltk.word_tokenize(text)
            # Use pos_tag directly without the '_eng' suffix which causes the error
            pos_tags = nltk.pos_tag(tokens)
            noun_phrases = []
            current_np = []

            for word, tag in pos_tags:
                if tag.startswith('NN'):
                    current_np.append(word)
                elif current_np:
                    if len(current_np) > 0:
                        noun_phrases.append(' '.join(current_np))
                    current_np = []

            if current_np:
                noun_phrases.append(' '.join(current_np))

            concepts.extend(noun_phrases)
        except Exception as e:
            print(f"Error extracting key concepts: {e}")
            # If NLTK processing fails, continue without it
            pass

        return concepts

    @staticmethod
    def calculate_concept_coverage(text1: str, text2: str) -> Dict:
        """Calculate coverage of key concepts between two texts"""
        concepts1 = SimilarityEngines.extract_key_concepts(text1)
        concepts2 = SimilarityEngines.extract_key_concepts(text2)

        # Normalize concepts for comparison
        normalized_concepts1 = set(c.lower() for c in concepts1)
        normalized_concepts2 = set(c.lower() for c in concepts2)

        # Extract key phrases (2-3 word) for semantic coverage
        def extract_key_phrases(text):
            words = re.findall(r'\b\w+\b', text.lower())
            phrases = []
            # Get bigrams
            for i in range(len(words) - 1):
                phrases.append(f"{words[i]} {words[i + 1]}")
            # Get trigrams
            for i in range(len(words) - 2):
                phrases.append(f"{words[i]} {words[i + 1]} {words[i + 2]}")
            return set(phrases)

        phrases1 = extract_key_phrases(text1)
        phrases2 = extract_key_phrases(text2)
        phrase_intersection = phrases2.intersection(phrases1)

        # Calculate semantic phrase coverage
        if not phrases2:
            phrase_coverage = 1.0
        else:
            phrase_coverage = len(phrase_intersection) / len(phrases2)

        # Calculate concept coverage with more weight to important concepts
        # Create important domain-specific concepts list
        important_concepts = ['incrementality', 'testing', 'marketing', 'customer',
                              'acquisition', 'data', 'measurement', 'impact']

        # Check if important concepts are covered
        important_in_text1 = [c for c in important_concepts if any(c in nc.lower() for nc in normalized_concepts1)]
        important_in_text2 = [c for c in important_concepts if any(c in nc.lower() for nc in normalized_concepts2)]

        # Calculate important concept coverage
        if not important_in_text2:
            important_coverage = 1.0
        else:
            important_intersection = set(important_in_text1).intersection(set(important_in_text2))
            important_coverage = len(important_intersection) / len(important_in_text2)

        # Calculate standard concept coverage
        if not normalized_concepts2:
            concept_coverage = 1.0  # If no concepts in expected, then full coverage
        else:
            concept_intersection = normalized_concepts2.intersection(normalized_concepts1)
            concept_coverage = len(concept_intersection) / len(normalized_concepts2)

        # Concepts missing from the actual response
        concepts_missing = list(normalized_concepts2 - normalized_concepts1)

        # Calculate Jaccard similarity for concepts
        if not normalized_concepts1 and not normalized_concepts2:
            concept_jaccard = 1.0
        elif not normalized_concepts1 or not normalized_concepts2:
            concept_jaccard = 0.0
        else:
            concept_intersection = normalized_concepts2.intersection(normalized_concepts1)
            concept_union = normalized_concepts2.union(normalized_concepts1)
            concept_jaccard = len(concept_intersection) / len(concept_union)

        # Create a weighted concept coverage that gives more weight to important concepts and phrases
        weighted_concept_coverage = (concept_coverage * 0.4) + (important_coverage * 0.4) + (phrase_coverage * 0.2)

        return {
            "concept_coverage": weighted_concept_coverage,  # Use the weighted version
            "raw_concept_coverage": concept_coverage,  # Keep the original as reference
            "concept_jaccard": concept_jaccard,
            "phrase_coverage": phrase_coverage,
            "important_concept_coverage": important_coverage,
            "concepts_missing": concepts_missing,
            "concepts_extra": list(normalized_concepts1 - normalized_concepts2)
        }

    @staticmethod
    def levenshtein_distance(text1: str, text2: str) -> int:
        """Calculate Levenshtein (edit) distance between two strings"""
        if len(text1) < len(text2):
            return SimilarityEngines.levenshtein_distance(text2, text1)

        if len(text2) == 0:
            return len(text1)

        previous_row = range(len(text2) + 1)
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def levenshtein_similarity(text1: str, text2: str) -> float:
        """Convert Levenshtein distance to a similarity score between 0 and 1"""
        distance = SimilarityEngines.levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        return 1 - (distance / max_len)

    @staticmethod
    def ngram_similarity(text1: str, text2: str, n: int = 2) -> float:
        """Calculate n-gram overlap similarity"""

        def get_ngrams(text):
            tokens = text.lower().split()
            ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
            return set(ngrams)

        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)

        if not ngrams1 and not ngrams2:
            return 1.0
        elif not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)

        return len(intersection) / len(union)

    @staticmethod
    def sentiment_similarity(text1: str, text2: str) -> float:
        """Compare sentiment polarity between two texts"""
        try:
            from textblob import TextBlob

            sentiment1 = TextBlob(text1).sentiment.polarity
            sentiment2 = TextBlob(text2).sentiment.polarity

            # Calculate similarity as 1 minus the normalized absolute difference
            max_diff = 2.0  # Maximum possible difference in polarity (-1 to 1)
            similarity = 1 - (abs(sentiment1 - sentiment2) / max_diff)

            return similarity
        except ImportError:
            # If TextBlob is not available, return None
            return -1.0

    @staticmethod
    def comprehensive_test(actual: str, expected: str) -> Dict:
        """Perform a comprehensive similarity test with multiple metrics"""
        results = {
            "character_similarity": SimilarityEngines.basic_similarity(actual, expected),
            "jaccard_similarity": SimilarityEngines.jaccard_similarity(actual, expected),
            "dice_coefficient": SimilarityEngines.dice_coefficient(actual, expected),
            "levenshtein_similarity": SimilarityEngines.levenshtein_similarity(actual, expected)
        }

        # Add cosine similarities
        try:
            results["cosine_tfidf"] = SimilarityEngines.cosine_similarity_tfidf(actual, expected)
            results["cosine_count"] = SimilarityEngines.cosine_similarity_count(actual, expected)
        except Exception as e:
            print(f"sklearn error: {e}")
            # If sklearn is not available
            pass

        # Add n-gram similarities
        results["bigram_similarity"] = SimilarityEngines.ngram_similarity(actual, expected, 2)
        results["trigram_similarity"] = SimilarityEngines.ngram_similarity(actual, expected, 3)

        # Add sentiment similarity if available
        sentiment_sim = SimilarityEngines.sentiment_similarity(actual, expected)
        if sentiment_sim is not None:
            results["sentiment_similarity"] = sentiment_sim

        # Add concept coverage data
        concept_data = SimilarityEngines.calculate_concept_coverage(actual, expected)
        results.update(concept_data)

        # Calculate a weighted composite score
        weights = {
            "character_similarity": 0.05,
            "jaccard_similarity": 0.15,
            "cosine_tfidf": 0.25,
            "concept_coverage": 0.15,
            "bigram_similarity": 0.2,
            "trigram_similarity": 0.2
        }

        # Only use available metrics
        available_weights = {k: v for k, v in weights.items() if k in results}
        if available_weights:
            # Normalize weights
            total_weight = sum(available_weights.values())
            normalized_weights = {k: v / total_weight for k, v in available_weights.items()}

            weighted_score = sum(results[k] * normalized_weights[k] for k in normalized_weights)
            results["weighted_composite_score"] = weighted_score

        return results

    @staticmethod
    def quick_test(actual: str, expected: str) -> Dict:
        """Perform a quick algorithmic test with essential metrics"""
        # Basic similarity
        basic_similarity = SimilarityEngines.basic_similarity(actual, expected)

        # Jaccard similarity
        jaccard_similarity = SimilarityEngines.jaccard_similarity(actual, expected)

        # N-gram similarity for context and structure
        bigram_similarity = SimilarityEngines.ngram_similarity(actual, expected, 2)
        trigram_similarity = SimilarityEngines.ngram_similarity(actual, expected, 3)

        # Concept coverage
        concept_data = SimilarityEngines.calculate_concept_coverage(actual, expected)

        # Create weighted score with more focus on contextual similarity than exact matches
        # (15% basic, 20% Jaccard, 25% bigrams, 20% trigrams, 20% concept coverage)
        weighted_score = (basic_similarity * 0.15) + (jaccard_similarity * 0.20) + \
                         (bigram_similarity * 0.25) + (trigram_similarity * 0.20) + \
                         (concept_data["concept_coverage"] * 0.20)

        # Add a similarity threshold adjustment
        # For LLM outputs that are semantically similar but use different words
        if jaccard_similarity >= 0.35 and concept_data["concept_coverage"] >= 0.7:
            # Boost the score if key concepts are covered but wording differs
            weighted_score = min(0.85, weighted_score * 1.2)

        return {
            "basic_similarity": basic_similarity,
            "jaccard_similarity": jaccard_similarity,
            "bigram_similarity": bigram_similarity,
            "trigram_similarity": trigram_similarity,
            "concept_coverage": concept_data["concept_coverage"],
            "concepts_missing": concept_data["concepts_missing"],
            "weighted_similarity": weighted_score
        }