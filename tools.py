"""Agent tools: web search (Tavily or DuckDuckGo) and page fetch."""
from __future__ import annotations

import logging

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def web_search_backend(config) -> str:
    """Which search backend will be used given the current config."""
    return "tavily" if getattr(config, "TAVILY_API_KEY", "").strip() else "duckduckgo"


@tool
def fetch_url(url: str) -> str:
    """Fetch the text content of a web page.

    Use this to read a specific URL returned by web search.

    Args:
        url: The full http(s) URL to fetch.
    """
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (telegram-gpt bot)"})
        resp.raise_for_status()
        text = resp.text
        return text[:8000]
    except Exception as e:  # tool errors are surfaced to the model, not the user
        return f"Failed to fetch {url}: {e}"


def _format_results(results: list[dict]) -> str:
    """Render search hits as the title/url/snippet blocks both backends return."""
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('url') or r.get('href', '')}\n{r.get('content') or r.get('body', '')}"
        for r in results
    )


def _tavily_search_tool(config):
    """Wrap TavilySearch so it shares the fallback's name and single-arg schema.

    TavilySearch is named "tavily_search" and exposes include_domains,
    search_depth, topic, time_range and more as tool arguments. Wrapping it
    keeps one stable tool name across both backends and one parameter for the
    model to get right.
    """
    from langchain_tavily import TavilySearch

    backend = TavilySearch(max_results=5, tavily_api_key=config.TAVILY_API_KEY)

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information.

        Use when you need recent facts, news, or data you don't already know.

        Args:
            query: The search query (2-10 words works best).
        """
        try:
            raw = backend.invoke({"query": query})
        except Exception as e:  # includes ToolException on zero results
            return f"Search failed: {e}"
        if isinstance(raw, dict):
            if raw.get("error"):
                return f"Search failed: {raw['error']}"
            return _format_results(raw.get("results", []))
        return str(raw)

    return web_search


def _duckduckgo_search_tool():
    from ddgs import DDGS

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information.

        Use when you need recent facts, news, or data you don't already know.

        Args:
            query: The search query (2-10 words works best).
        """
        try:
            with DDGS() as ddgs:
                return _format_results(list(ddgs.text(query, max_results=5)))
        except Exception as e:
            return f"Search failed: {e}"

    return web_search


def build_tools(config, db=None) -> list:
    """Assemble the agent's tool set based on configuration.

    When db is provided, includes the get_image retrieval tool."""
    if web_search_backend(config) == "tavily":
        search = _tavily_search_tool(config)
    else:
        logger.info("TAVILY_API_KEY not set — using DuckDuckGo web search")
        search = _duckduckgo_search_tool()
    built = [search, fetch_url]
    if db is not None:
        from image_store import build_image_tool
        built.append(build_image_tool(db))
    return built
