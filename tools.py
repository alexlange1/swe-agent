"""
Tool implementations for the tau mining agent.

Provides: bash, read_file, write_file, edit_file, multi_edit

Key improvements over standard implementations:
- edit_file gives richer diagnostics on failure (shows nearby content)
- multi_edit allows batching multiple edits in one tool call
- bash has a tighter timeout (30s) to prevent hanging
- read_file shows line numbers for easier edit_file targeting
"""

import os
import subprocess
from difflib import SequenceMatcher


class ToolExecutor:
    MAX_OUTPUT_CHARS = 30_000
    MAX_FILE_READ_CHARS = 150_000
    BASH_TIMEOUT = 30

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.repo_path, path)

    def definitions(self) -> list[dict]:
        return [
            {
                "name": "bash",
                "description": (
                    "Run a bash command in the repo directory. Use for: "
                    "grep, find, ls, git commands. Timeout: 30 seconds."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to run"}
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "read_file",
                "description": (
                    "Read a file's contents with line numbers. "
                    "MUST be called before editing any file. "
                    "Path can be relative to the repo root or absolute."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path (relative or absolute)"},
                        "start_line": {"type": "integer", "description": "Optional: first line to read (1-based)"},
                        "end_line": {"type": "integer", "description": "Optional: last line to read (1-based)"},
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": (
                    "Write complete content to a file (create or overwrite). "
                    "Use for new files or complete rewrites. Prefer edit_file for targeted changes."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "content": {"type": "string", "description": "Complete file content"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": (
                    "Replace a specific string in a file. The old_string must be unique in the file "
                    "and match the file content EXACTLY (including whitespace and indentation). "
                    "On failure, shows nearby content to help you correct old_string. "
                    "Preferred over write_file for targeted edits."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to replace (must be unique in file)",
                        },
                        "new_string": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
            {
                "name": "multi_edit",
                "description": (
                    "Apply multiple edits to a single file in one call. "
                    "Each edit is an {old_string, new_string} pair applied sequentially. "
                    "Use when you need to make several changes to the same file."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old_string": {"type": "string"},
                                    "new_string": {"type": "string"},
                                },
                                "required": ["old_string", "new_string"],
                            },
                            "description": "List of {old_string, new_string} pairs to apply sequentially",
                        },
                    },
                    "required": ["path", "edits"],
                },
            },
        ]

    def execute(self, name: str, inp: dict) -> str:
        try:
            handler = getattr(self, f"_run_{name}", None)
            if handler is None:
                return f"[tool-error] Unknown tool: {name}"
            return handler(inp)
        except Exception as e:
            return f"[tool-error] {name} raised: {type(e).__name__}: {e}"

    def _run_bash(self, inp: dict) -> str:
        cmd = inp["command"]
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.BASH_TIMEOUT,
                cwd=self.repo_path,
            )
        except subprocess.TimeoutExpired:
            return f"[tool-error] Command timed out after {self.BASH_TIMEOUT}s"

        out = ""
        if result.stdout:
            out += result.stdout
        if result.stderr:
            out += f"\n[stderr]\n{result.stderr}"
        if not out.strip():
            out = f"(no output; exit code {result.returncode})"
        return out[: self.MAX_OUTPUT_CHARS]

    def _run_read_file(self, inp: dict) -> str:
        path = self.resolve_path(inp["path"])
        if not os.path.exists(path):
            return f"[tool-error] File not found: {path}"
        if os.path.isdir(path):
            return f"[tool-error] {path} is a directory — use bash('ls {path}')"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            return f"[tool-error] Cannot read {path}: {e}"

        start_line = inp.get("start_line")
        end_line = inp.get("end_line")

        if start_line or end_line:
            lines = content.split("\n")
            s = max(0, (start_line or 1) - 1)
            e = min(len(lines), end_line or len(lines))
            numbered = [f"{i+1}|{line}" for i, line in enumerate(lines[s:e], start=s)]
            result = "\n".join(numbered)
            if len(result) > self.MAX_FILE_READ_CHARS:
                result = result[: self.MAX_FILE_READ_CHARS] + "\n[... truncated ...]"
            return result

        if len(content) > self.MAX_FILE_READ_CHARS:
            return content[: self.MAX_FILE_READ_CHARS] + "\n\n[... truncated — file too large ...]"
        return content

    def _run_write_file(self, inp: dict) -> str:
        path = self.resolve_path(inp["path"])
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(inp["content"])
        lines = inp["content"].count("\n")
        return f"Wrote {len(inp['content'])} chars ({lines} lines) to {path}"

    def _run_edit_file(self, inp: dict) -> str:
        path = self.resolve_path(inp["path"])
        if not os.path.exists(path):
            return f"[tool-error] File not found: {path}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        old = inp["old_string"]
        new = inp["new_string"]

        occurrences = content.count(old)
        if occurrences == 0:
            return self._edit_not_found_diagnostic(path, content, old)
        if occurrences > 1:
            return (
                f"[tool-error] old_string appears {occurrences} times in {path}. "
                f"Include more surrounding lines to make it unique."
            )

        new_content = content.replace(old, new, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        old_lines = old.count("\n")
        new_lines = new.count("\n")
        delta = new_lines - old_lines
        sign = "+" if delta >= 0 else ""
        return f"Edited {path} ({sign}{delta} lines)"

    def _run_multi_edit(self, inp: dict) -> str:
        path = self.resolve_path(inp["path"])
        if not os.path.exists(path):
            return f"[tool-error] File not found: {path}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        applied = 0
        errors = []
        for i, edit in enumerate(inp.get("edits", [])):
            old = edit.get("old_string", "")
            new = edit.get("new_string", "")
            if not old:
                errors.append(f"Edit {i+1}: empty old_string")
                continue
            occ = content.count(old)
            if occ == 0:
                errors.append(f"Edit {i+1}: old_string not found")
                continue
            if occ > 1:
                errors.append(f"Edit {i+1}: old_string appears {occ} times (not unique)")
                continue
            content = content.replace(old, new, 1)
            applied += 1

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        parts = [f"Applied {applied}/{len(inp.get('edits', []))} edits to {path}"]
        if errors:
            parts.append("Errors: " + "; ".join(errors))
        return ". ".join(parts)

    def _edit_not_found_diagnostic(self, path: str, content: str, old_string: str) -> str:
        """Rich diagnostic when edit_file can't find old_string."""
        lines = content.split("\n")
        first_line = old_string.split("\n")[0].strip() if old_string else ""

        best_ratio = 0.0
        best_line_idx = -1
        if first_line and len(first_line) > 3:
            for i, line in enumerate(lines):
                ratio = SequenceMatcher(None, first_line, line.strip()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_line_idx = i

        parts = [f"[tool-error] old_string not found in {path}."]

        if best_ratio > 0.5 and best_line_idx >= 0:
            start = max(0, best_line_idx - 2)
            end = min(len(lines), best_line_idx + 5)
            context_lines = lines[start:end]
            parts.append(f"Closest match near line {best_line_idx + 1} (similarity {best_ratio:.0%}):")
            for j, line in enumerate(context_lines, start=start + 1):
                parts.append(f"  {j}|{line}")
            parts.append("Re-read the file and use the exact content from the file.")
        else:
            snippet = repr(old_string[:80]) + ("..." if len(old_string) > 80 else "")
            parts.append(f"Searched for: {snippet}")
            parts.append("Tip: use read_file to see exact content, then retry.")

        return "\n".join(parts)
