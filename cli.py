"""
cli.py — Quick command-line/notebook-friendly inference, no server needed.
Good for testing directly inside Colab.

Usage:
    python cli.py --text "John Doe, john@mail.com, Software Engineer at Acme (2019-2023)"
    python cli.py --file path/to/resume.txt
"""
import argparse
import json
import sys

from resume_parser import get_parser, ModelLoadError, ParseError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Resume text passed directly as a string")
    parser.add_argument("--file", help="Path to a .txt file containing resume text")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Error: provide --text or --file", file=sys.stderr)
        sys.exit(1)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            resume_text = f.read()
    else:
        resume_text = args.text

    try:
        engine = get_parser()
    except ModelLoadError as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = engine.parse_resume(resume_text)
    except ParseError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
