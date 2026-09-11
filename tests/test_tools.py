import pytest

from foresight_agent.tools import CalculatorTool, LocalTextReadTool, LocalTextSearchTool


def test_calculator_allows_arithmetic_and_blocks_code():
    assert CalculatorTool().run({"expression": "(2 + 3) ** 2"})["observation"] == "25"
    with pytest.raises(ValueError, match="Unsupported"):
        CalculatorTool().run({"expression": "__import__('os').getcwd()"})


def test_local_text_tools_are_root_scoped(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("Alpha\nForesight agent\nOmega", encoding="utf-8")
    assert "Foresight" in LocalTextReadTool(tmp_path).run({"path": "note.txt"})["observation"]
    result = LocalTextSearchTool(tmp_path).run({"path": "note.txt", "query": "agent"})
    assert result["match_count"] == 1
    with pytest.raises(ValueError, match="inside"):
        LocalTextReadTool(tmp_path).run({"path": "../secret.txt"})
