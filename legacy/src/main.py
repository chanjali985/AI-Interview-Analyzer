"""CLI script for running the AI Interview Analyzer."""
import argparse
import json
import os
import sys
from analyzer import InterviewAnalyzer


def read_resume_file(file_path: str) -> str:
    """Read resume from text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading resume file: {str(e)}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Interview Analyzer - Analyze interview responses"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The interview question"
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file (MP3, WAV, etc.)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        required=True,
        help="Path to resume text file or resume text directly"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (optional, prints to stdout if not provided)"
    )
    
    args = parser.parse_args()
    
    # Check if resume is a file path or direct text
    if os.path.exists(args.resume):
        resume_text = read_resume_file(args.resume)
    else:
        resume_text = args.resume
    
    # Initialize analyzer
    analyzer = InterviewAnalyzer()
    
    # Run analysis
    try:
        result = analyzer.analyze(args.question, args.audio, resume_text)
        
        # Output results
        output_json = json.dumps(result, indent=2, ensure_ascii=False)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"Results saved to {args.output}")
        else:
            print("\n" + "="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            print(output_json)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

