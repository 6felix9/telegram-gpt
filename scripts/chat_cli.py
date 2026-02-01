#!/usr/bin/env python3
"""
Interactive CLI for simulating conversations with the AI bot.

This script allows you to:
- Test conversations using a dedicated "test" chat_id (default)
- Simulate conversations in real group chats (read-only mode)
- Clear conversation history (only for chat_id="test")
- Change bot personality for group chats

Usage:
    # Test mode (default, writes to database)
    python scripts/chat_cli.py --chat-id test

    # Simulate real group chat (read-only, doesn't write to database)
    python scripts/chat_cli.py --chat-id 15223921 --group

    # Clear test conversation history
    /clear

Commands:
    /clear - Clear conversation history (only works when chat_id="test")
    /stats - Show chat statistics
    /personality [name] - View or change active personality (group chats only)
    /exit or /quit - Exit the CLI
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database import Database
from token_manager import TokenManager
from openai_client import OpenAIClient

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.LOG_LEVEL)
)
logger = logging.getLogger(__name__)


class ChatCLI:
    """Interactive CLI for chat simulation."""

    def __init__(self, chat_id: str, is_group: bool = False):
        """
        Initialize CLI with database and OpenAI client.

        Args:
            chat_id: Chat ID to use (default "test" for writable mode)
            is_group: Whether to treat as group chat (affects formatting)
        """
        self.chat_id = str(chat_id)
        self.is_group = is_group
        self.is_test_mode = (self.chat_id == "test")

        # Initialize components (mirroring bot.py setup)
        logger.info("Initializing components...")
        config.validate()

        # Database
        self.db = Database(config.DATABASE_URL)

        # Token manager
        model_limit = config.get_model_context_limit(config.OPENAI_MODEL)
        max_tokens = min(config.MAX_CONTEXT_TOKENS, model_limit - 2000)
        self.token_manager = TokenManager(config.OPENAI_MODEL, max_tokens)

        # OpenAI client
        self.openai_client = OpenAIClient(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            timeout=config.OPENAI_TIMEOUT,
        )

        logger.info(f"CLI initialized for chat_id={self.chat_id}, group={is_group}, test_mode={self.is_test_mode}")

    async def process_message(self, user_input: str) -> str:
        """
        Process a user message and return assistant response.

        Args:
            user_input: User's message text

        Returns:
            Assistant's response text
        """
        try:
            # Count tokens in user's message
            user_tokens = self.token_manager.count_message_tokens("user", user_input)

            # For test mode, store user message
            if self.is_test_mode:
                self.db.add_message(
                    chat_id=self.chat_id,
                    role="user",
                    content=user_input,
                    user_id=int(config.AUTHORIZED_USER_ID),
                    message_id=None,
                    token_count=user_tokens,
                    sender_name="CLI User",
                    sender_username=None,
                    is_group_chat=self.is_group,
                )

            # Get conversation history within token budget
            max_tokens = config.MAX_CONTEXT_TOKENS
            messages = self.db.get_messages_by_tokens(self.chat_id, max_tokens)

            # For non-test mode, append current user message in-memory (not persisted)
            if not self.is_test_mode:
                messages.append({
                    "role": "user",
                    "content": user_input,
                    "sender_name": "CLI User",
                    "sender_username": None,
                    "is_group_chat": self.is_group,
                })

            # Final trim to ensure we fit (accounting for response)
            messages = self.token_manager.trim_to_fit(messages, reserve_tokens=1000)

            logger.info(
                f"Processing request for chat {self.chat_id}: "
                f"{len(messages)} messages, {user_tokens} tokens"
            )

            # Get completion from OpenAI
            # For group chats, fetch active personality and use custom prompt if available
            custom_prompt = None
            if self.is_group:
                try:
                    active_personality = self.db.get_active_personality()
                    # If personality is "normal", use default SYSTEM_PROMPT_GROUP
                    # Otherwise fetch custom prompt from database
                    if active_personality != "normal":
                        custom_prompt = self.db.get_personality_prompt(active_personality)
                        # If custom prompt not found, fall back to default
                        if not custom_prompt:
                            logger.warning(f"Personality '{active_personality}' not found in database, using default")
                except Exception as e:
                    logger.error(f"Error fetching personality: {e}", exc_info=True)
                    # Continue with default prompt on error

            response = await self.openai_client.get_completion(messages, self.is_group, custom_system_prompt=custom_prompt)

            # For test mode, store assistant's response
            if self.is_test_mode:
                assistant_tokens = self.token_manager.count_message_tokens("assistant", response)
                self.db.add_message(
                    chat_id=self.chat_id,
                    role="assistant",
                    content=response,
                    token_count=assistant_tokens,
                    is_group_chat=self.is_group,
                )

            logger.info(f"Response generated for chat {self.chat_id}")
            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"

    def clear_history(self) -> bool:
        """
        Clear conversation history for current chat.

        Returns:
            True if cleared, False if not allowed
        """
        if not self.is_test_mode:
            return False

        try:
            self.db.clear_history(self.chat_id)
            logger.info(f"History cleared for chat {self.chat_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}", exc_info=True)
            return False

    def get_stats(self) -> dict:
        """Get statistics for current chat."""
        try:
            return self.db.get_stats(self.chat_id)
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "first_message": "N/A",
                "last_message": "N/A",
            }

    def handle_personality_command(self, args: list[str]) -> None:
        """
        Handle /personality command.

        Args:
            args: Command arguments (empty list or [personality_name])
        """
        try:
            # If no args, show current active personality
            if not args or len(args) == 0:
                active_personality = self.db.get_active_personality()
                print(f"\nCurrent personality: {active_personality}")
                print("Usage: /personality <name>")
                print("Example: /personality villain\n")
                return

            # Parse personality name from argument
            personality_name = args[0].strip()

            # Check if personality exists
            if not self.db.personality_exists(personality_name):
                print(f"\n❌ No personality '{personality_name}' found.\n")
                return

            # Set active personality
            self.db.set_active_personality(personality_name)
            print(f"\n✅ Personality changed to '{personality_name}'\n")
            logger.info(f"Personality changed to {personality_name}")

        except Exception as e:
            logger.error(f"Error handling personality command: {e}", exc_info=True)
            print(f"\n❌ Failed to change personality: {e}\n")

    def handle_list_personality_command(self) -> None:
        """Handle /list_personality command."""
        try:
            personalities = self.db.list_personalities()
            active = self.db.get_active_personality()
            
            if not personalities:
                print("\nNo custom personalities available.")
                print(f"Currently using: {active} (default)\n")
                return
            
            print(f"\n**Available Personalities:**")
            print(f"Currently active: {active}\n")
            
            for name, prompt_preview in personalities:
                marker = "✓" if name == active else "-"
                print(f"{marker} {name}")
                print(f"  {prompt_preview}\n")
        except Exception as e:
            logger.error(f"Error listing personalities: {e}", exc_info=True)
            print(f"\n❌ Failed to list personalities: {e}\n")

    async def run(self):
        """Run the interactive CLI loop."""
        mode_str = "TEST MODE" if self.is_test_mode else "READ-ONLY MODE"
        group_str = " (GROUP)" if self.is_group else ""
        print(f"\n{'='*60}")
        print(f"Chat CLI - {mode_str}{group_str}")
        print(f"Chat ID: {self.chat_id}")
        print(f"Model: {config.OPENAI_MODEL}")
        print(f"{'='*60}\n")

        if not self.is_test_mode:
            stats = self.get_stats()
            print(f"📊 Existing history: {stats['total_messages']} messages, "
                  f"{stats['total_tokens']:,} tokens")
            print("⚠️  READ-ONLY MODE: Your prompts/responses will NOT be saved to database\n")

        print("Type your message (or /clear, /stats, /personality [name], /list_personality, /exit to quit):\n")

        while True:
            try:
                # Read user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["/exit", "/quit"]:
                    print("\nExiting...")
                    break

                if user_input.lower() == "/clear":
                    if self.is_test_mode:
                        if self.clear_history():
                            print("✅ Conversation history cleared.\n")
                        else:
                            print("❌ Failed to clear history.\n")
                    else:
                        print("❌ /clear is only available in TEST MODE (chat_id='test').\n")
                    continue

                if user_input.lower() == "/stats":
                    stats = self.get_stats()
                    first_msg = stats["first_message"]
                    if first_msg != "N/A":
                        first_msg = first_msg.split("T")[0]
                    print(f"\n📊 Chat Statistics:")
                    print(f"  Messages: {stats['total_messages']}")
                    print(f"  Total tokens: {stats['total_tokens']:,}")
                    print(f"  Since: {first_msg}\n")
                    continue

                if user_input.lower().startswith("/personality"):
                    # Parse command: /personality or /personality <name>
                    parts = user_input.split(None, 1)
                    args = parts[1:] if len(parts) > 1 else []
                    self.handle_personality_command(args)
                    continue

                if user_input.lower().startswith("/list_personality"):
                    self.handle_list_personality_command()
                    continue

                # Process message
                print("\n🤔 Thinking...")
                response = await self.process_message(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except EOFError:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                print(f"\n❌ Unexpected error: {e}\n")

        # Cleanup
        self.db.close()
        print("Goodbye!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for simulating conversations with the AI bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (default, writes to database)
  python scripts/chat_cli.py --chat-id test

  # Simulate real group chat (read-only, doesn't write to database)
  python scripts/chat_cli.py --chat-id 15223921 --group

  # Test mode with group formatting
  python scripts/chat_cli.py --chat-id test --group
        """
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        default="test",
        help="Chat ID to use (default: 'test' for writable mode)"
    )
    parser.add_argument(
        "--group",
        action="store_true",
        help="Treat as group chat (affects message formatting)"
    )

    args = parser.parse_args()

    # Create and run CLI
    cli = ChatCLI(chat_id=args.chat_id, is_group=args.group)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()

