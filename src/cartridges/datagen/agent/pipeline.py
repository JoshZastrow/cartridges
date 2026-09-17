"""Shared implementation for the prepare-data workflow."""

import logging
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from cartridges.datagen.agent.tools import prepare_data_server
from cartridges.datagen.cache import init_cache

logger = logging.getLogger(__name__)

ALLOWED_TOOLS = [
    "Skill",
    "mcp__prepare_data__list_pdf_pages",
    "mcp__prepare_data__pdf_page_to_image",
    "mcp__prepare_data__ocr_pdf_page",
    "mcp__prepare_data__append_page",
]


async def prepare_data(pdf_path: Path, output_path: Path, model: str = "") -> None:
    init_cache()
    options = ClaudeAgentOptions(
        tools=["Skill"],
        mcp_servers={"prepare_data": prepare_data_server},
        allowed_tools=ALLOWED_TOOLS,
        skills=["prepare-data"],
        cwd=str(Path.cwd()),
        permission_mode="bypassPermissions",
        # long documents need many sequential tool-call turns (2+ per page)
        max_turns=2000,
    )
    model_clause = f" using model '{model}' for every ocr_pdf_page call" if model else ""
    prompt = (
        f"Use the prepare-data skill to convert {pdf_path.resolve()} into "
        f"markdown and write it to {output_path.resolve()}{model_clause}."
    )
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    logger.info(block.text)
