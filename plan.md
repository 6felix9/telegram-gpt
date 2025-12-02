Implementation Plan: Pluggable LLM Providers + Local Speed Tests (Revised)

 Overview

 Refactor the Telegram bot to support multiple LLM providers (OpenAI, Gemini, Groq) through a provider interface, and add a CLI benchmark harness for local
 latency/throughput testing.

 Architecture Design

 1. Provider Interface (llm_provider.py)

 Create an abstract base class that all providers implement:

 class LLMProvider(ABC):
     @abstractmethod
     async def get_completion(self, messages: list[dict], **kwargs) -> str:
         """Get completion from LLM. Returns response text."""
         pass

     @abstractmethod
     async def test_connection(self) -> bool:
         """Test API connectivity."""
         pass

     @abstractmethod
     def get_model_name(self) -> str:
         """Return current model identifier."""
         pass

     @abstractmethod
     def get_max_context_tokens(self) -> int:
         """Return max context window for current model."""
         pass

     @abstractmethod
     def count_tokens(self, messages: list[dict]) -> int:
         """Count tokens in message list."""
         pass

     @abstractmethod
     def format_messages(self, messages: list[dict], is_group: bool) -> list[dict]:
         """Format messages for provider-specific API."""
         pass

 Rationale:
 - Encapsulates all provider-specific logic
 - Token counting moves into provider (providers know their own tokenization)
 - Message formatting is provider-specific (system prompts, role names)
 - Each provider handles its own error mapping

 2. Provider Implementations

 2.1 OpenAI Provider (providers/openai_provider.py)

 Refactor existing openai_client.py to implement the interface:

 - Move current OpenAIClient code into OpenAIProvider
 - Keep existing tiktoken-based token counting
 - Preserve current system prompt logic (group vs private)
 - Maintain group message formatting with [SenderName]
 - Keep existing error handling (AuthenticationError, RateLimitError, etc.)

 Context Limits (from existing config):
 - gpt-4o-mini: 128k tokens
 - gpt-4o: 128k tokens
 - gpt-3.5-turbo: 16k tokens
 - gpt-4-turbo: 128k tokens
 - gpt-4: 8k tokens

2.2 Gemini Provider (providers/gemini_provider.py)

 Use the updated Gemini SDK (`from google import genai`):

 from google import genai

 class GeminiProvider(LLMProvider):
     def __init__(self, api_key: str, model: str, timeout: int):
         self.client = genai.Client(api_key=api_key)
         self.model_name = model
         self.timeout = timeout
         self.system_prompt = ...
         self.chat = None  # lazily created

 Key Implementation Details:
 - Create chat session per bot instance: self.chat = self.client.chats.create(model=model, system_instruction=system_prompt)
 - Send turns with chat.send_message(prompt) (or chat.send_message_stream for streaming benchmarks)
 - Message Format: Gemini expects role:user/model; prepend [SenderName]: for group user messages to keep context
 - Token Counting: use client.models.count_tokens(model=model, contents=formatted_messages) where formatted_messages mirrors what you send (system + user/assistant turns); fall back to heuristic if unavailable
 - Error Mapping: catch genai.AuthError, RateLimitError, ServiceUnavailable, etc., and return user-friendly messages

 Model Default:
 - gemini-2.5-flash-preview-09-2025 (per request). Check current context window in docs and use that for limits.

2.3 Groq Provider (providers/groq_provider.py)

 Use Groq's SDK (OpenAI-style chat completions):

 from groq import Groq

 class GroqProvider(LLMProvider):
     def __init__(self, api_key: str, model: str, timeout: int):
         self.client = Groq(api_key=api_key, timeout=timeout)
         self.model = model

 Key Implementation Details:
 - API Compatibility: client.chat.completions.create(messages=[...], model="llama-3.3-70b-versatile")
 - Token Counting: Use tiktoken with model-specific encoding; fallback to cl100k_base if unknown
 - Message Format: Same as OpenAI (role: system/user/assistant), keep [SenderName]: prefix for group user messages
 - Streaming: stream=True supported for benchmarks
 - Error Mapping: Similar to OpenAI (AuthenticationError, RateLimitError, Timeout, etc.)

 Supported Models (sample):
 - openai/gpt-oss-120b (default)

 3. Configuration Changes (config.py)

 Add new environment variables:

 # Provider selection
 PROVIDER: str = "openai"  # openai|gemini|groq

 # API Keys (only required key is loaded based on PROVIDER)
 OPENAI_API_KEY: Optional[str] = None
 GEMINI_API_KEY: Optional[str] = None
 GROQ_API_KEY: Optional[str] = None

 # Model (provider-specific)
 MODEL: str = "gpt-4o-mini"  # default for OpenAI

 # Shared settings
 TIMEOUT: int = 60
 # Working context cap (trimmed history will use min(cap, model limit))
 MAX_CONTEXT_TOKENS: int = 16000

Provider-Specific Defaults:
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash-preview-09-2025",
    "groq": "llama-3.3-70b-versatile"
}

 MODEL_CONTEXT_LIMITS = {
     # OpenAI
     "gpt-4o-mini": 128000,
     "gpt-4o": 128000,
     "gpt-3.5-turbo": 16385,
     # Gemini (from current docs; output limit ~65,536)
     "gemini-2.5-flash-preview-09-2025": 1048576,  # input tokens
     # Groq
     "llama-3.3-70b-versatile": 128000,
     "llama-3.1-8b-instant": 128000,
     "mixtral-8x7b-32768": 32768,
 }

 Validation Logic:
 - Check PROVIDER is valid (openai|gemini|groq)
 - Ensure corresponding API key exists for selected provider
 - Validate MODEL exists in known models (warn if unknown, don't fail)
 - Use MAX_CONTEXT_TOKENS or model-specific limit (whichever is lower)

 Backward Compatibility:
 - Preferred: MODEL/TIMEOUT with PROVIDER.
 - Temporary fallback: if MODEL is unset, read OPENAI_MODEL; if TIMEOUT is unset, read OPENAI_TIMEOUT; log a warning to migrate. Remove fallbacks later.

 4. Factory Implementation (llm_factory.py)

 Create factory to instantiate providers:

 def create_provider(config: Config) -> LLMProvider:
     """Factory to create LLM provider based on config."""
     provider_type = config.PROVIDER.lower()

     if provider_type == "openai":
         return OpenAIProvider(
             api_key=config.OPENAI_API_KEY,
             model=config.MODEL,
             timeout=config.TIMEOUT
         )
     elif provider_type == "gemini":
         return GeminiProvider(
             api_key=config.GEMINI_API_KEY,
             model=config.MODEL,
             timeout=config.TIMEOUT
         )
     elif provider_type == "groq":
         return GroqProvider(
             api_key=config.GROQ_API_KEY,
             model=config.MODEL,
             timeout=config.TIMEOUT
         )
     else:
         raise ValueError(f"Unknown provider: {provider_type}")

 Integration Point: Replace OpenAIClient initialization in bot.py:

 # Old:
 openai_client = OpenAIClient(config.OPENAI_API_KEY, config.OPENAI_MODEL, config.OPENAI_TIMEOUT)

 # New:
 llm_provider = llm_factory.create_provider(config)

 5. Token Management Refactoring

 Option A: Remove TokenManager (Recommended)
 - Each provider implements count_tokens() method
 - Providers know their own tokenization logic
 - Simplifies architecture (one less layer)

 Option B: Keep TokenManager as Wrapper
 - TokenManager delegates to provider.count_tokens()
 - Adds context trimming logic (provider-agnostic)
 - Maintains existing interface for handlers

 Recommendation: Option A - Remove TokenManager
 - Token counting logic moves to providers
 - Context trimming moves to handlers or a utility function
 - Reduces indirection and complexity

 Context Trimming Utility (utils/context_trimmer.py):
 def trim_messages_to_fit(
     messages: list[dict],
     max_tokens: int,
     provider: LLMProvider,
     reserve_tokens: int = 1000
 ) -> list[dict]:
     """Trim messages to fit within token budget."""
     # Implementation similar to current TokenManager.trim_to_fit()
     # Uses provider.count_tokens() for counting

 6. Handler Changes (handlers.py)

 Minimal changes required:

 1. Replace openai_client with llm_provider in init_handlers()
 2. Replace token_manager.count_tokens() with provider.count_tokens()
 3. Replace token_manager.trim_to_fit() with trim_messages_to_fit() utility
 4. Update process_request() to use provider methods

 Example Change:
 # Old:
 token_count = token_manager.count_tokens([{"role": "user", "content": content}])
 messages = token_manager.trim_to_fit(messages, max_tokens)
 response = await openai_client.get_completion(messages, is_group)

 # New:
 token_count = llm_provider.count_tokens([{"role": "user", "content": content}])
 messages = trim_messages_to_fit(messages, max_tokens, llm_provider)
 response = await llm_provider.get_completion(messages, is_group=is_group)

 7. CLI Benchmark Harness (benchmark.py)

 Create minimal standalone CLI tool for speed testing:

python benchmark.py --provider openai --model gpt-4o-mini --prompt "Hello, world!"
python benchmark.py --provider gemini --model gemini-2.5-flash-preview-09-2025 --prompt "Explain quantum computing"

 Simple Implementation:
 - Arguments: --provider, --model, --prompt
 - Metrics: End-to-end latency, response length, tokens
 - Output: Simple printed results (latency in ms, throughput)

 async def run_benchmark(provider: LLMProvider, prompt: str):
     """Run single benchmark test."""
     messages = [{"role": "user", "content": prompt}]

     start = time.perf_counter()
     response = await provider.get_completion(messages)
     end = time.perf_counter()

     latency_ms = (end - start) * 1000
     tokens = provider.count_tokens([{"role": "assistant", "content": response}])
     tokens_per_sec = tokens / (latency_ms / 1000) if latency_ms > 0 else 0

     print(f"\n=== Benchmark Results ===")
     print(f"Provider: {provider.get_model_name()}")
     print(f"Latency: {latency_ms:.2f} ms")
     print(f"Response: {len(response)} chars, {tokens} tokens")
     print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
     print(f"\nResponse preview:\n{response[:200]}...")

 Keep it simple: No warmup, iterations, or fancy output formats initially

 File Structure

 telegram-gpt/
 ├── bot.py                          # Modified: Use factory instead of OpenAIClient
 ├── config.py                       # Modified: Add provider-specific env vars
 ├── handlers.py                     # Modified: Use provider interface
 ├── database.py                     # No changes
 ├── token_manager.py                # DEPRECATED: Token counting moves to providers
 ├── openai_client.py                # DEPRECATED: Becomes providers/openai_provider.py
 │
 ├── llm_provider.py                 # NEW: Abstract provider interface
 ├── llm_factory.py                  # NEW: Provider factory
 │
 ├── providers/                      # NEW: Provider implementations
 │   ├── __init__.py
 │   ├── openai_provider.py          # Refactored from openai_client.py
 │   ├── gemini_provider.py          # NEW: Google Gemini implementation
 │   └── groq_provider.py            # NEW: Groq implementation
 │
 ├── utils/                          # NEW: Shared utilities
 │   ├── __init__.py
 │   └── context_trimmer.py          # Token-budget trimming logic
 │
 ├── benchmark.py                    # NEW: CLI benchmark harness
 ├── requirements.txt                # Modified: Add google-genai, groq
 ├── .env.example                    # Modified: Add new env vars
 └── README.md                       # Modified: Document new features

 Implementation Order

 Phase 1: Provider Abstraction (Foundation)

 1. Create llm_provider.py with abstract interface
 2. Create providers/ directory structure
 3. Refactor openai_client.py → providers/openai_provider.py to implement interface
 4. Create llm_factory.py with OpenAI-only support
 5. Update bot.py to use factory (still OpenAI only)
 6. Test: Verify bot works identically with refactored code

Phase 2: Gemini Provider

 1. Add google-genai to requirements.txt
 2. Implement providers/gemini_provider.py
 3. Update config.py with GEMINI_API_KEY, validate Gemini models
 4. Update llm_factory.py to support "gemini" provider
 5. Test: Set PROVIDER=gemini, verify bot works with Gemini

 Phase 3: Groq Provider

 1. Add groq SDK to requirements.txt
 2. Implement providers/groq_provider.py
 3. Update config.py with GROQ_API_KEY, validate Groq models
 4. Update llm_factory.py to support "groq" provider
 5. Test: Set PROVIDER=groq, verify bot works with Groq

 Phase 4: Token Management Refactoring

 1. Create utils/context_trimmer.py with trim_messages_to_fit()
 2. Update handlers.py to use provider.count_tokens() directly
 3. Replace token_manager.trim_to_fit() with context_trimmer utility
 4. Update bot.py to remove TokenManager initialization
 5. Mark token_manager.py as deprecated (or delete)
 6. Test: Verify token counting and trimming work for all providers

 Phase 5: CLI Benchmark Harness

 1. Create benchmark.py with simple argument parsing
 2. Implement single run_benchmark() function
 3. Support --provider, --model, --prompt flags
 4. Print latency, throughput, and response preview
 5. Test: Quick check against all three providers

 Phase 6: Documentation & Configuration

 1. Update .env.example with new variables and examples
 2. Document provider-specific models in README
 3. Add simple benchmark usage example
 4. Update requirements.txt with all dependencies

 Configuration Examples

 OpenAI

 PROVIDER=openai
 OPENAI_API_KEY=sk-...
 MODEL=gpt-4o-mini
 TIMEOUT=60
 MAX_CONTEXT_TOKENS=16000

Gemini

PROVIDER=gemini
GEMINI_API_KEY=AIza...
MODEL=gemini-2.5-flash-preview-09-2025
TIMEOUT=60
MAX_CONTEXT_TOKENS=128000  # cap below the 1,048,576 model limit unless you need more

 Groq

 PROVIDER=groq
 GROQ_API_KEY=gsk_...
 MODEL=llama-3.3-70b-versatile
 TIMEOUT=30
 MAX_CONTEXT_TOKENS=32000

 Testing Strategy

 Quick Manual Test Checklist

 Provider Switching:
 - Bot works with PROVIDER=openai
 - Bot works with PROVIDER=gemini
 - Bot works with PROVIDER=groq
 - Error message if PROVIDER invalid or API key missing

 Basic Functionality:
 - Private chat responses work
 - Group chat with keyword trigger works
 - /clear, /stats, /grant, /revoke commands work
 - Token counting and context trimming work

 Benchmark Harness:
 - Runs for each provider
 - Prints latency and throughput
 - Shows response preview

 Dependencies

 Add to requirements.txt:
 # Existing
 python-telegram-bot==21.7
 openai==1.56.2
 tiktoken==0.8.0
 python-dotenv==1.0.1

 # New
 google-genai  # Gemini API (latest client, replace older google-generativeai)
 groq>=0.11.0  # Groq API

 Critical Files to Modify

 1. bot.py (lines 51-131): Replace OpenAIClient with factory
 2. config.py (entire file): Add provider-specific configuration
 3. handlers.py (lines 10-23, 40-150): Use provider interface, remove token_manager
 4. openai_client.py: Refactor to providers/openai_provider.py
 5. requirements.txt: Add new SDKs

 Migration Path for Existing Deployments

 Breaking changes require .env updates:

 1. Add PROVIDER variable: Set to "openai", "gemini", or "groq"
 2. Rename variables:
   - OPENAI_MODEL → MODEL
   - OPENAI_TIMEOUT → TIMEOUT (optional, has default)
 3. Keep API key: OPENAI_API_KEY stays the same (or add GEMINI_API_KEY/GROQ_API_KEY)
 4. Update .env.example: Reference new variable names

 Example migration:
 # Old .env
 OPENAI_API_KEY=sk-...
 OPENAI_MODEL=gpt-4o-mini
 OPENAI_TIMEOUT=60

 # New .env
 PROVIDER=openai
 OPENAI_API_KEY=sk-...
 MODEL=gpt-4o-mini
 TIMEOUT=60

 Trade-offs & Considerations

Token Counting Accuracy

 - OpenAI/Groq: Tiktoken provides exact token counts
 - Gemini: Use client.models.count_tokens(...); fallback to heuristic if unavailable
 - Impact: Context trimming reliable for all providers

 System Prompts

 - Different providers may interpret system prompts differently
 - Groq (Llama models) may need prompt engineering adjustments
 - Consider per-provider system prompt customization if needed

 Rate Limits

 - OpenAI: 10k RPM (gpt-4o-mini)
 - Gemini: 15 RPM (free tier), 1000 RPM (paid)
 - Groq: 30 RPM (free tier)
 - Bot should handle rate limit errors gracefully (already does)

 Streaming (Deferred)

 - Not included in initial implementation
 - Can be added later if latency measurements show benefit
 - Telegram bot sends complete messages anyway

 Error Handling

 - Each provider has different exception types
 - Map to common user-friendly messages
 - Log provider-specific errors for debugging

 Cost Considerations

 - OpenAI gpt-4o-mini: $0.15/$0.60 per 1M tokens
 - Gemini 1.5 Flash: Free tier available, then $0.075/$0.30
 - Groq: Free tier available, very fast inference
 - Document cost/performance trade-offs in README

 Success Criteria

 - Bot works with all three providers (OpenAI, Gemini, Groq)
 - Switching providers requires only PROVIDER + API_KEY + MODEL change
 - Token counting accurate for each provider
 - Benchmark harness prints latency and throughput
 - README documents all providers, models, and usage
 - All existing bot functionality preserved (/clear, /stats, /grant, /revoke)

 Future Enhancements (Not in Initial Scope)

 1. Streaming responses: Measure first-token latency
 2. Per-chat provider selection: Different chats use different providers
 3. Automatic fallback: If one provider fails, try another
 4. Cost tracking: Log spending per provider/model
 5. Advanced benchmarking: Multiple iterations, warmup, statistical analysis
