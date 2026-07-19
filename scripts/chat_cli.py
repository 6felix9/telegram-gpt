#!/usr/bin/env python3
"""
Interactive CLI for simulating conversations with the AI bot.

This script allows you to:
- Test conversations using a dedicated "test" chat_id (default)
- Simulate conversations against a real chat_id's live agent memory
- Clear conversation history (only for chat_id="test")
- Change bot personality for group chats

Note: every run persists turns to the LangGraph checkpoint thread for the
given chat_id. Use "test" (or another throwaway id) unless you intend to
write into a real chat's conversation history.

Usage:
    # Test mode (default, writes to the "test" checkpoint thread)
    python scripts/chat_cli.py --chat-id test

    # Simulate a real group chat (writes to that group's real checkpoint thread!)
    python scripts/chat_cli.py --chat-id 15223921 --group

    # Clear test conversation history
    /clear

Commands:
    /clear - Clear conversation history (only works when chat_id="test")
    /stats - Show chat statistics
    /model [name] - View or change the active AI model
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
from prompt_builder import PromptBuilder
from agent import Agent, CompletionError
from model_registry import MODEL_PROVIDERS
import agent as agent_module
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

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

        # Load persisted active model (seeds from DEFAULT_MODEL on first run)
        self.db.init_active_model(config.DEFAULT_MODEL)
        effective_model = self.db.get_active_model()

        # Checkpointer pool (tables created out-of-band; do NOT call .setup()).
        self.checkpointer_pool = ConnectionPool(
            conninfo=config.DATABASE_URL,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        checkpointer = PostgresSaver(self.checkpointer_pool)

        # Prompt builder + agent (mirrors bot.py wiring).
        self.prompt_builder = PromptBuilder(
            default_private_prompt=agent_module.SYSTEM_PROMPT,
            default_group_prompt=agent_module.SYSTEM_PROMPT_GROUP,
            get_active_personality=self.db.get_active_personality,
            get_personality_prompt=self.db.get_personality_prompt,
        )
        self.agent = Agent(
            config=config,
            prompt_builder=self.prompt_builder,
            checkpointer=checkpointer,
            model_name=effective_model,
            db=self.db,
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
            # Context now lives in the checkpoint thread; run through the agent.
            human = self.prompt_builder.to_lc_human_message(
                text=user_input, is_group=self.is_group, sender_name="CLI User")
            response = await self.agent.run(self.chat_id, human, self.is_group)

            logger.info(f"Response generated for chat {self.chat_id}")
            return response

        except CompletionError as e:
            return e.user_message
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
            self.agent.clear_thread(self.chat_id)
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

    def handle_model_command(self, args: list[str]) -> None:
        """Handle /model command."""
        available = ", ".join(MODEL_PROVIDERS.keys())
        if not args:
            current = self.db.get_active_model()
            print(f"\nCurrent model: {current}")
            print(f"Available: {available}")
            print("Usage: /model <name>\n")
            return

        new_model = args[0].strip()
        if new_model not in MODEL_PROVIDERS:
            print(f"\n❌ Unknown model '{new_model}'.")
            print(f"Available: {available}\n")
            return

        try:
            self.db.set_active_model(new_model)
            self.agent.set_model(new_model)
            print(f"\n✅ Model switched to '{new_model}'\n")
        except Exception as e:
            logger.error(f"Error switching model: {e}", exc_info=True)
            print(f"\n❌ Failed to switch model: {e}\n")

    async def run(self):
        """Run the interactive CLI loop."""
        mode_str = "TEST MODE" if self.is_test_mode else "LIVE MODE"
        group_str = " (GROUP)" if self.is_group else ""
        print(f"\n{'='*60}")
        print(f"Chat CLI - {mode_str}{group_str}")
        print(f"Chat ID: {self.chat_id}")
        print(f"Model: {self.agent.model_name}")
        print(f"{'='*60}\n")

        if not self.is_test_mode:
            stats = self.get_stats()
            print(f"📊 Existing history: {stats['total_messages']} messages, "
                  f"{stats['total_tokens']:,} tokens")
            print("⚠️  LIVE MODE: prompts/responses ARE persisted to this chat_id's "
                  "checkpoint thread — use a throwaway chat_id to avoid polluting "
                  "a real conversation's memory\n")

        print("Type your message (or /clear, /stats, /model [name], /personality [name], /list_personality, /exit to quit):\n")

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

                if user_input.lower().startswith("/model"):
                    parts = user_input.split(None, 1)
                    args = parts[1:] if len(parts) > 1 else []
                    self.handle_model_command(args)
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
        self.checkpointer_pool.close()
        print("Goodbye!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for simulating conversations with the AI bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (default, writes to the "test" checkpoint thread)
  python scripts/chat_cli.py --chat-id test

  # Simulate a real group chat (writes to that group's real checkpoint thread!)
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

