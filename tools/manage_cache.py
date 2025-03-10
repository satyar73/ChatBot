#!/usr/bin/env python3
"""
CLI tool for managing the chat cache.

Usage:
    python manage_cache.py stats         - Show cache statistics
    python manage_cache.py clear         - Clear all cache entries
    python manage_cache.py clear 7       - Clear entries older than 7 days
"""

import argparse
import json
import sys
import os
from pathlib import Path
import requests

def get_api_url():
    """Get the API URL from environment or use default."""
    return os.environ.get("API_URL", "http://localhost:8005")

def get_stats():
    """Get and display cache statistics."""
    api_url = get_api_url()
    try:
        response = requests.get(f"{api_url}/chat/cache/stats")
        response.raise_for_status()
        stats = response.json()
        
        # Print the statistics in a formatted way
        print("\n=== Cache Statistics ===")
        print(f"Total entries: {stats.get('total_entries', 0)}")
        print(f"Total hits: {stats.get('total_hits', 0)}")
        print(f"Hit rate: {stats.get('hit_rate_percent', 0)}%")
        print(f"Average hit time: {stats.get('avg_hit_time_ms', 0)} ms")
        print(f"Average miss time: {stats.get('avg_miss_time_ms', 0)} ms")
        print(f"Estimated time saved: {stats.get('time_saved_ms', 0)/1000:.2f} seconds")
        
        # Print frequent queries if available
        frequent_queries = stats.get('frequent_queries', [])
        if frequent_queries:
            print("\nMost frequent queries:")
            for query, count in frequent_queries:
                # Truncate long queries
                if len(query) > 60:
                    query = query[:57] + "..."
                print(f"  - '{query}' ({count} times)")
        
        return 0
    except Exception as e:
        print(f"Error getting cache statistics: {e}", file=sys.stderr)
        return 1

def clear_cache(days=None):
    """Clear cache entries, optionally only those older than specified days."""
    api_url = get_api_url()
    try:
        params = {}
        if days is not None:
            params = {"older_than_days": days}
            print(f"Clearing cache entries older than {days} days...")
        else:
            print("Clearing all cache entries...")
        
        response = requests.delete(f"{api_url}/chat/cache", params=params)
        response.raise_for_status()
        result = response.json()
        
        print(f"Success: {result.get('message', 'Cache cleared')}")
        return 0
    except Exception as e:
        print(f"Error clearing cache: {e}", file=sys.stderr)
        return 1

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage the chat cache")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear cache entries")
    clear_parser.add_argument("days", nargs="?", type=int, help="Clear entries older than this many days")
    
    args = parser.parse_args()
    
    if args.command == "stats":
        return get_stats()
    elif args.command == "clear":
        return clear_cache(args.days)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())