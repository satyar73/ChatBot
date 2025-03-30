"""
Text splitting utilities for the application.
Provides various text splitting implementations including token-based splitting.
"""
from typing import List
from langchain.docstore.document import Document
from tiktoken import encoding_for_model


class TokenTextSplitter:
    """
    A text splitter that splits text based on token count rather than character count.
    This is particularly useful for LLM-based applications where token limits are important.
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int, model_name: str = "gpt-4"):
        """
        Initialize the token-based text splitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: Name of the model to use for tokenization
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            
            # If adding this paragraph would exceed chunk size
            if current_tokens + paragraph_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep last paragraph for overlap
                    overlap_text = current_chunk[-1]
                    overlap_tokens = self.count_tokens(overlap_text)
                    current_chunk = [overlap_text]
                    current_tokens = overlap_tokens
            
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks based on token count.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata=doc.metadata.copy()
                )
                new_doc.metadata['chunk'] = i
                split_docs.append(new_doc)
        return split_docs 