#!/usr/bin/env python3
"""
CLI interface for neuromorphic memory.
Used by Node.js wrapper for Clawdbot integration.

Usage:
  python3 cli.py store "content" [--id N]
  python3 cli.py recall "query"
  python3 cli.py search "query" [--limit N]
  python3 cli.py stats
"""

import sys
import json
import argparse

# Add skill directory to path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_memory import get_memory


def main():
    parser = argparse.ArgumentParser(description='Neuromorphic Memory CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Store command
    store_parser = subparsers.add_parser('store', help='Store a memory')
    store_parser.add_argument('content', help='Content to store')
    store_parser.add_argument('--id', type=int, help='Memory ID')
    
    # Recall command
    recall_parser = subparsers.add_parser('recall', help='Recall a memory')
    recall_parser.add_argument('query', help='Query text')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search memories')
    search_parser.add_argument('query', help='Query text')
    search_parser.add_argument('--limit', type=int, default=5, help='Max results')
    
    # Stats command
    subparsers.add_parser('stats', help='Get statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    memory = get_memory()
    
    if args.command == 'store':
        result = memory.store(args.content, memory_id=args.id)
        output = {'success': True, 'memory_id': result}
        
    elif args.command == 'recall':
        result = memory.recall(args.query)
        output = {
            'success': True,
            'content': result.get('content'),
            'memory_id': result.get('memory_id'),
            'confidence': result.get('confidence'),
            'latency_ms': result.get('latency_ms')
        }
        
    elif args.command == 'search':
        results = memory.search(args.query, limit=args.limit)
        output = {'success': True, 'results': results}
        
    elif args.command == 'stats':
        stats = memory.get_stats()
        output = {'success': True, 'stats': stats}
    
    print(json.dumps(output))


if __name__ == '__main__':
    main()
