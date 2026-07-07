"""Tool selection: Tavily when key present, DuckDuckGo fallback otherwise."""
import tools


class _CfgTavily:
    TAVILY_API_KEY = "tvly-x"


class _CfgNoKey:
    TAVILY_API_KEY = ""


def test_backend_selection():
    assert tools.web_search_backend(_CfgTavily) == "tavily"
    assert tools.web_search_backend(_CfgNoKey) == "duckduckgo"


def test_build_tools_returns_search_and_fetch():
    built = tools.build_tools(_CfgNoKey)
    names = {t.name for t in built}
    assert "fetch_url" in names
    assert any("search" in n for n in names)
