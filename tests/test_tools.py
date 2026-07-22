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
    built = tools.build_tools(_CfgNoKey, db=None)
    names = {t.name for t in built}
    assert "fetch_url" in names
    assert any("search" in n for n in names)
    assert "get_image" not in names


def test_build_tools_includes_get_image_when_db_present():
    from types import SimpleNamespace
    built = tools.build_tools(_CfgNoKey, db=SimpleNamespace(get_image=lambda *a: None))
    names = {t.name for t in built}
    assert "get_image" in names
