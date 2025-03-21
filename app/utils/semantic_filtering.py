"""
Semantic filtering functionality for query rewriting and retrieval
"""
from typing import List, Dict
from app.utils.similarity_engines import SimilarityEngines

class SemanticFilter:
    """Class for filtering and comparing text based on semantic similarity"""
    
    @staticmethod
    def filter_similar_queries(queries: List[str], similarity_threshold: float = 0.7) -> List[str]:
        """
        Filter out semantically similar queries from a list of candidate queries
        
        Args:
            queries: List of query strings
            similarity_threshold: Minimum semantic difference required to keep a query
                                 (higher = more strict filtering)
            
        Returns:
            List of semantically diverse queries
        """
        if not queries:
            return []
        
        # Always keep the original query (first in the list)
        filtered_queries = [queries[0]]
        
        # Compare each subsequent query with those already in the filtered list
        for query in queries[1:]:
            is_unique = True
            
            for existing_query in filtered_queries:
                # Get similarity score between current query and an existing one
                similarity = SimilarityEngines.quick_test(query, existing_query)
                
                # If similarity is above threshold, consider it too similar to keep
                if similarity["weighted_similarity"] >= similarity_threshold:
                    is_unique = False
                    break
            
            # If this query is semantically distinct, add it to filtered list
            if is_unique:
                filtered_queries.append(query)
        
        return filtered_queries
    
    @staticmethod
    def rank_queries_by_diversity(queries: List[str], reference_query: str) -> List[str]:
        """
        Rank queries by their semantic diversity from a reference query
        
        Args:
            queries: List of query strings to rank
            reference_query: The query to compare against
            
        Returns:
            List of queries ranked by semantic diversity (most diverse first)
        """
        if not queries:
            return []
        
        # Calculate similarity scores for each query compared to reference
        scored_queries = []
        for query in queries:
            if query == reference_query:
                # Always put the reference query first with score 1.0
                scored_queries.append((query, 1.0))
            else:
                # Get similarity score
                similarity = SimilarityEngines.quick_test(query, reference_query)
                # Lower similarity means more diverse, so we use 1 - similarity
                diversity_score = 1.0 - similarity["weighted_similarity"]
                scored_queries.append((query, diversity_score))
        
        # Sort by diversity score (descending order - most diverse first)
        ranked_queries = [q[0] for q in sorted(scored_queries, key=lambda x: x[1], reverse=True)]
        return ranked_queries