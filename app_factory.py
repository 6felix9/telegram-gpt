"""Single composition point for the db/prompt-builder/agent stack, shared by
bot.py and scripts/chat_cli.py so their bootstrap can't drift apart."""
from dataclasses import dataclass

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

import agent as agent_module
from agent import Agent
from database import Database
from prompt_builder import PromptBuilder


@dataclass
class AppStack:
    db: Database
    prompt_builder: PromptBuilder
    agent: Agent
    checkpointer_pool: ConnectionPool


def build_app_stack(config) -> AppStack:
    """Build the db/prompt-builder/agent stack shared by bot.py and chat_cli.py.

    Tables are created out-of-band by scripts/setup_checkpointer.py (deploy
    preDeployCommand); this does NOT call PostgresSaver.setup().
    """
    db = Database(config.DATABASE_URL)
    db.init_active_model(config.DEFAULT_MODEL)
    effective_model = db.get_active_model()

    checkpointer_pool = ConnectionPool(
        conninfo=config.DATABASE_URL,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=ConnectionPool.check_connection,
    )
    checkpointer = PostgresSaver(checkpointer_pool)

    prompt_builder = PromptBuilder(
        default_private_prompt=agent_module.SYSTEM_PROMPT,
        default_group_prompt=agent_module.SYSTEM_PROMPT_GROUP,
        get_active_personality=db.get_active_personality,
        get_personality_prompt=db.get_personality_prompt,
    )

    bot_agent = Agent(
        config=config,
        prompt_builder=prompt_builder,
        checkpointer=checkpointer,
        model_name=effective_model,
        db=db,
    )

    return AppStack(
        db=db,
        prompt_builder=prompt_builder,
        agent=bot_agent,
        checkpointer_pool=checkpointer_pool,
    )
