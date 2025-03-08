#!/usr/bin/env python
"""
NLTK Resource Installer

This script ensures that all necessary NLTK resources are properly installed.
Run this script before using the chat tester to avoid 'Resource not found' errors.
"""
import os
import nltk
import sys


def ensure_nltk_resources():
    """Download and verify all necessary NLTK resources."""
    print("NLTK Resource Installer")
    print("======================")

    # Create a custom download directory if needed
    home_dir = os.path.expanduser("~")
    nltk_data_dir = os.path.join(home_dir, "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)

    print(f"Using NLTK data directory: {nltk_data_dir}")

    # Resources needed for the chat tester
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]

    success = True

    for resource, path in resources:
        print(f"\nChecking for {resource}...")
        try:
            # Try to find the resource
            found = nltk.data.find(path)
            print(f"✅ Found: {found}")
        except LookupError:
            # Download if not found
            print(f"❌ Not found. Downloading {resource}...")
            try:
                nltk.download(resource, download_dir=nltk_data_dir)
                print(f"✅ Successfully downloaded {resource}")
            except Exception as e:
                print(f"❌ Error downloading {resource}: {str(e)}")
                success = False

    # Verify that we can initialize the tokenizer
    print("\nVerifying tokenizer functionality...")
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize("Testing the NLTK tokenizer.")
        print(f"✅ Tokenizer works! Sample: {tokens}")
    except Exception as e:
        print(f"❌ Tokenizer verification failed: {str(e)}")
        success = False

    # Final status
    print("\n======================")
    if success:
        print("✅ All NLTK resources are properly installed!")
    else:
        print("❌ Some NLTK resources could not be installed.")
        print("   You may need to manually install them or check your internet connection.")

    return success


if __name__ == "__main__":
    if ensure_nltk_resources():
        sys.exit(0)
    else:
        sys.exit(1)