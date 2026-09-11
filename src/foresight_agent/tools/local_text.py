from pathlib import Path
from typing import Any


class _LocalTextTool:
    allowed_suffixes = {".txt", ".md", ".csv", ".json", ".py"}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _safe_path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("Path must remain inside the configured root.")
        if candidate.suffix.lower() not in self.allowed_suffixes:
            raise ValueError("Unsupported text file type.")
        return candidate


class LocalTextReadTool(_LocalTextTool):
    name, description = "read_text", "Read a UTF-8 text file. Argument: path."

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = self._safe_path(str(arguments.get("path", ""))).read_text(encoding="utf-8")
        return {"observation": text[:20_000], "success": True, "confidence": 1.0}


class LocalTextSearchTool(_LocalTextTool):
    name, description = "search_text", "Search lines in a UTF-8 text file. Arguments: path, query."

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path, query = self._safe_path(str(arguments.get("path", ""))), str(arguments.get("query", "")).casefold()
        if not query:
            raise ValueError("Search query cannot be empty.")
        matches = [f"{n}: {line}" for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                   if query in line.casefold()]
        return {"observation": "\n".join(matches[:100]), "success": True,
                "confidence": 1.0, "match_count": len(matches)}
