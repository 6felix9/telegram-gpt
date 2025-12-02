#!/usr/bin/env python3
"""CLI benchmark harness for testing LLM provider performance."""
import asyncio
import argparse
import time
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def run_benchmark(provider, prompt: str):
    """
    Run single benchmark test.

    Args:
        provider: LLM provider instance
        prompt: Test prompt to send
    """
    messages = [{"role": "user", "content": prompt}]

    print(f"\n{'='*60}")
    print(f"Provider: {provider.get_model_name()}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"{'='*60}\n")

    # Measure latency
    start = time.perf_counter()
    response = await provider.get_completion(messages)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    # Count tokens
    try:
        prompt_tokens = provider.count_tokens(messages)
        response_tokens = provider.count_tokens([{"role": "assistant", "content": response}])
        total_tokens = prompt_tokens + response_tokens
        tokens_per_sec = response_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
    except Exception as e:
        print(f"Warning: Token counting failed: {e}")
        prompt_tokens = response_tokens = total_tokens = tokens_per_sec = 0

    # Print results
    print(f"{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"Latency:           {latency_ms:.2f} ms")
    print(f"Response length:   {len(response)} chars")
    print(f"Prompt tokens:     {prompt_tokens}")
    print(f"Response tokens:   {response_tokens}")
    print(f"Total tokens:      {total_tokens}")
    print(f"Throughput:        {tokens_per_sec:.2f} tokens/sec")
    print(f"\nResponse preview:")
    print(f"{'-'*60}")
    print(response[:500] + ('...' if len(response) > 500 else ''))
    print(f"{'-'*60}\n")


def main():
    """Main entry point for benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark LLM provider performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark OpenAI
  python benchmark.py --provider openai --model gpt-4o-mini --prompt "Hello, world!"

  # Benchmark Gemini
  python benchmark.py --provider gemini --model gemini-2.5-flash-preview-09-2025 --prompt "Explain quantum computing"

  # Benchmark Groq
  python benchmark.py --provider groq --model openai/gpt-oss-120b --prompt "Write a Python function"
        """
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "gemini", "groq"],
        help="LLM provider to benchmark"
    )
    parser.add_argument(
        "--model",
        help="Model name (uses provider default if not specified)"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Test prompt to send to the model"
    )

    args = parser.parse_args()

    # Import provider classes
    try:
        if args.provider == "openai":
            from providers.openai_provider import OpenAIProvider
            api_key = os.getenv("OPENAI_API_KEY")
            default_model = "gpt-4o-mini"
            if not api_key:
                print("Error: OPENAI_API_KEY environment variable not set")
                sys.exit(1)
            provider = OpenAIProvider(
                api_key=api_key,
                model=args.model or default_model,
                timeout=60
            )

        elif args.provider == "gemini":
            from providers.gemini_provider import GeminiProvider
            api_key = os.getenv("GEMINI_API_KEY")
            default_model = "gemini-2.5-flash-preview-09-2025"
            if not api_key:
                print("Error: GEMINI_API_KEY environment variable not set")
                sys.exit(1)
            provider = GeminiProvider(
                api_key=api_key,
                model=args.model or default_model,
                timeout=60
            )

        elif args.provider == "groq":
            from providers.groq_provider import GroqProvider
            api_key = os.getenv("GROQ_API_KEY")
            default_model = "openai/gpt-oss-120b"
            if not api_key:
                print("Error: GROQ_API_KEY environment variable not set")
                sys.exit(1)
            provider = GroqProvider(
                api_key=api_key,
                model=args.model or default_model,
                timeout=60
            )

        else:
            print(f"Error: Unknown provider {args.provider}")
            sys.exit(1)

    except Exception as e:
        print(f"Error initializing provider: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run benchmark
    try:
        asyncio.run(run_benchmark(provider, args.prompt))
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
