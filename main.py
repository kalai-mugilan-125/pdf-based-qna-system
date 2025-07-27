import argparse
import os
import sys
from dotenv import load_dotenv
from pipeline.qna_reader import run_qna_pipeline
from pipeline.query_mode import run_query_mode

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Hybrid QG-QA Document Processor")
    parser.add_argument('--mode', choices=['pipeline', 'query'], required=True, 
                       help='Mode: pipeline (generate Q&A) or query (answer specific question)')
    parser.add_argument('--file', type=str, required=True, 
                       help='Path to input file (PDF, DOCX, EML, or TXT)')
    parser.add_argument('--output_dir', type=str, default='data/outputs/', 
                       help='Output directory for pipeline mode')
    parser.add_argument('--query', type=str, 
                       help='User query for query mode')
    parser.add_argument('--max_tokens', type=int, default=300, 
                       help='Maximum tokens per chunk')
    
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        sys.exit(1)

    if args.mode == 'pipeline':
        print(f"Running pipeline mode on {args.file}")
        results = run_qna_pipeline(args.file, args.output_dir, args.max_tokens)
        print(f"Generated {len(results)} Q&A pairs")

    elif args.mode == 'query':
        if not args.query:
            print("Error: Query mode requires --query argument")
            sys.exit(1)
        
        print(f"Running query mode on {args.file}")
        result = run_query_mode(args.file, args.query)
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    main()