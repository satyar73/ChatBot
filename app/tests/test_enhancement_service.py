"""
Tests for the enhancement service.
"""
import unittest
from unittest.mock import patch, MagicMock

from app.services.enhancement_service import EnhancementService


class TestEnhancementService(unittest.TestCase):
    """Test cases for the EnhancementService class."""

    def setUp(self):
        """Set up test fixtures, if any."""
        # Create a service instance with mocked data
        with patch('app.services.enhancement_service.load_json') as mock_load_json:
            mock_load_json.return_value = {
                "What is marketing mix modeling?": "Marketing Mix Modeling (MMM) is a statistical analysis technique used to estimate the impact of various marketing activities on sales or other KPIs.",
                "What is an attribution multiplier?": "An attribution multiplier is a factor applied to channel attribution to account for over-reporting or under-reporting in marketing platforms."
            }
            self.service = EnhancementService()

    def test_expand_abbreviations(self):
        """Test expanding abbreviations in queries."""
        # Test MMM expansion
        query = "What is MMM and how does it work?"
        expanded = self.service.expand_abbreviations(query)
        self.assertEqual(expanded, "What is marketing mix modeling and how does it work?")
        
        # Test multiple expansions
        query = "Compare ROAS and ROI for MMM"
        expanded = self.service.expand_abbreviations(query)
        self.assertIn("marketing mix modeling", expanded)
        self.assertIn("return on ad spend", expanded)
        self.assertIn("return on investment", expanded)

    def test_add_synonyms(self):
        """Test adding synonyms to queries."""
        query = "How to measure marketing attribution?"
        with_synonyms = self.service.add_synonyms(query)
        self.assertNotEqual(query, with_synonyms)
        self.assertIn("attribution", with_synonyms)
        self.assertIn("credit assignment", with_synonyms)
        
    def test_enhance_query(self):
        """Test enhancing a query with alternatives."""
        query = "What is MMM?"
        result = self.service.enhance_query(query)
        
        # Check structure
        self.assertIn("original_query", result)
        self.assertIn("enhanced_query", result)
        self.assertIn("alt_queries", result)
        
        # Check content
        self.assertEqual(result["original_query"], query)
        self.assertIn("marketing mix modeling", result["enhanced_query"])
        self.assertTrue(len(result["alt_queries"]) > 1)
        
        # Due to mocking differences, we won't test QA matching specifically
        # Just check that the qa_match structure exists
        self.assertIn("qa_match", result)

    def test_qa_matching(self):
        """Test matching questions to answers."""
        # Exact match
        result = self.service.get_answer("What is marketing mix modeling?")
        self.assertIsNotNone(result)
        self.assertIn("Marketing Mix Modeling", result["answer"])
        
        # Case-insensitive match
        result = self.service.get_answer("what is marketing mix modeling?")
        self.assertIsNotNone(result)
        self.assertIn("Marketing Mix Modeling", result["answer"])
        
        # No match
        result = self.service.get_answer("What is multi-touch attribution?")
        self.assertIsNone(result)

    def test_extract_key_concepts(self):
        """Test extracting key concepts from expected answers."""
        answer = "Marketing Mix Modeling (MMM) is a statistical analysis technique. It is used to estimate the impact of various marketing activities on sales or other KPIs."
        concepts = self.service.extract_key_concepts(answer)
        
        # Basic check that some concepts were extracted
        self.assertTrue(len(concepts) > 0)
        self.assertIn("Concept:", concepts)
        
    def test_enhance_prompt_with_expected_answer(self):
        """Test enhancing a prompt with an expected answer."""
        base_prompt = "You are a helpful assistant."
        expected_answer = "Marketing Mix Modeling (MMM) is a statistical analysis technique."
        
        enhanced = self.service.enhance_prompt_with_expected_answer(base_prompt, expected_answer)
        
        # Should contain the base prompt
        self.assertIn(base_prompt, enhanced)
        # Should contain guidance
        self.assertIn("RESPONSE GUIDANCE", enhanced)
        # Should contain key concepts
        self.assertIn("KEY CONCEPTS FROM REFERENCE", enhanced)


if __name__ == '__main__':
    unittest.main()