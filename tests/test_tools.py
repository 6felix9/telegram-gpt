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
    assert "web_search" in names
    assert "get_image" not in names


def test_search_tool_named_web_search_on_both_backends():
    """The prompt's tool section must not change name with TAVILY_API_KEY."""
    for cfg in (_CfgNoKey, _CfgTavily):
        names = {t.name for t in tools.build_tools(cfg, db=None)}
        assert "web_search" in names, cfg
        assert "tavily_search" not in names, cfg


def test_tavily_search_tool_takes_only_a_query_arg():
    schema = tools._tavily_search_tool(_CfgTavily).args_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}


def test_format_results_handles_both_backend_shapes():
    tavily = tools._format_results([{"title": "T", "url": "u", "content": "c"}])
    ddg = tools._format_results([{"title": "T", "href": "u", "body": "c"}])
    assert tavily == ddg == "T\nu\nc"


def test_format_results_empty():
    assert tools._format_results([]) == "No results found."


def test_build_tools_includes_get_image_when_db_present():
    from types import SimpleNamespace
    built = tools.build_tools(_CfgNoKey, db=SimpleNamespace(get_image=lambda *a: None))
    names = {t.name for t in built}
    assert "get_image" in names
