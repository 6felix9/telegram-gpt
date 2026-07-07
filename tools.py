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
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            return "\n\n".join(
                f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
                for r in results
            )
        except Exception as e:
            return f"Search failed: {e}"

    return web_search


def build_tools(config) -> list:
    """Assemble the agent's tool set based on configuration."""
    if web_search_backend(config) == "tavily":
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=5, tavily_api_key=config.TAVILY_API_KEY)
    else:
        logger.info("TAVILY_API_KEY not set — using DuckDuckGo web search")
        search = _duckduckgo_search_tool()
    return [search, fetch_url]
