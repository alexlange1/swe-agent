"""
Pre-LLM reconnaissance engine.

Runs BEFORE the first API call in under 5 seconds. Builds a rich context
package that eliminates the exploration phase entirely:

  1. File tree snapshot — Claude sees the full project layout instantly
  2. Explicit file path extraction — tasks often name files directly
  3. Identifier grep — maps code symbols to files
  4. Ranked target list — files sorted by relevance signals

This is the primary competitive edge: while other miners waste 15-30s
on tool calls to explore the repo, we hand Claude a complete map.
"""

import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_STOP_WORDS = frozenset({
    "should_be", "instead_of", "make_sure", "such_as", "as_well",
    "each_time", "at_least", "no_longer", "does_not", "do_not",
    "is_not", "can_be", "will_be", "has_been", "have_been",
    "for_each", "set_up", "up_to", "based_on", "end_of",
    "None", "True", "False", "self", "this", "that", "with",
    "from", "import", "return", "class", "function", "method",
    "file", "code", "task", "change", "update", "fix", "bug",
    "error", "issue", "should", "must", "need", "when", "then",
    "make", "ensure", "currently", "also", "note", "like",
    "want", "would", "could", "does", "have", "been",
})

_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".swift", ".kt", ".scala", ".sh",
    ".lua", ".ex", ".exs", ".erl",
    ".hs", ".ml", ".clj", ".vue", ".svelte",
    ".dart", ".zig", ".sql", ".m", ".mm",
})

_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "vendor", "dist", "build", ".next", ".nuxt", "target",
    ".tox", ".eggs", "egg-info", ".mypy_cache", ".pytest_cache",
})


def get_file_tree(repo_path: str, max_files: int = 300) -> list[str]:
    """Fast file tree using find. Returns relative paths sorted alphabetically."""
    try:
        prune_clauses = " -o ".join(
            f'-name "{d}" -prune' for d in sorted(_SKIP_DIRS)
        )
        cmd = (
            f'find . \\( {prune_clauses} \\) -o -type f -print '
            f'| head -{max_files}'
        )
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=3, cwd=repo_path,
        )
        files = sorted(
            f.lstrip("./") for f in proc.stdout.strip().split("\n")
            if f.strip() and f != "."
        )
        return files
    except Exception:
        return []


def extract_file_paths(text: str) -> list[str]:
    """Extract explicit file paths mentioned in the task description."""
    paths = []
    ext_pattern = "|".join(
        re.escape(e) for e in sorted(_CODE_EXTENSIONS)
    )
    for m in re.finditer(
        rf'\b([\w/.-]+(?:{ext_pattern}))\b', text
    ):
        path = m.group(1)
        if "/" in path or path.count(".") == 1:
            paths.append(path)

    for m in re.finditer(r'`([^`]{3,80})`', text):
        token = m.group(1).strip()
        if "/" in token and any(token.endswith(e) for e in _CODE_EXTENSIONS):
            if token not in paths:
                paths.append(token)

    seen = set()
    result = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def extract_identifiers(text: str) -> list[str]:
    """Extract code identifiers from a task description."""
    found: list[str] = []

    for m in re.finditer(r"`([^`]{2,60})`", text):
        token = m.group(1).strip()
        if token and not token.startswith("/") and not any(
            token.endswith(e) for e in _CODE_EXTENSIONS
        ):
            found.append(token)

    for m in re.finditer(r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b", text):
        found.append(m.group(0))

    for m in re.finditer(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", text):
        found.append(m.group(0))

    for m in re.finditer(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b", text):
        found.append(m.group(0))

    for m in re.finditer(r"\b[a-z][a-z0-9]*\.[a-z_][a-z0-9_.]*\b", text):
        found.append(m.group(0))

    seen = set()
    result = []
    for ident in found:
        key = ident.lower()
        if key in seen or key in _STOP_WORDS or len(ident) < 4:
            continue
        seen.add(key)
        result.append(ident)

    return result


def detect_repo_info(repo_path: str) -> dict:
    """Quick repo language/framework detection."""
    info: dict = {"language": "unknown", "framework": "", "file_count": 0}
    ext_count: Counter = Counter()
    total = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            total += 1
            ext = Path(f).suffix.lower()
            if ext:
                ext_count[ext] += 1
        if total > 2000:
            break

    info["file_count"] = total
    lang_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".tsx": "TypeScript", ".go": "Go", ".rs": "Rust",
        ".java": "Java", ".rb": "Ruby", ".php": "PHP",
        ".c": "C", ".cpp": "C++", ".cs": "C#",
        ".swift": "Swift", ".kt": "Kotlin",
    }
    for ext, count in ext_count.most_common(3):
        if ext in lang_map:
            info["language"] = lang_map[ext]
            break

    markers = {
        "package.json": "Node.js", "requirements.txt": "Python",
        "Cargo.toml": "Rust", "go.mod": "Go", "pom.xml": "Java/Maven",
        "build.gradle": "Java/Gradle", "Gemfile": "Ruby",
        "composer.json": "PHP/Composer", "setup.py": "Python",
        "pyproject.toml": "Python",
    }
    for marker, fw in markers.items():
        if os.path.exists(os.path.join(repo_path, marker)):
            info["framework"] = fw
            break

    return info


def grep_identifiers(identifiers: list[str], repo_path: str) -> dict[str, list[str]]:
    """Grep for identifiers in parallel. Returns {identifier: [file_paths]}."""
    results: dict[str, list[str]] = {}
    idents_to_search = identifiers[:10]
    if not idents_to_search:
        return results

    def _grep_one(ident: str) -> tuple[str, list[str]]:
        safe = re.escape(ident)
        cmd = f"grep -rn '{safe}' . --include='*.py' --include='*.js' --include='*.ts' --include='*.tsx' --include='*.jsx' --include='*.go' --include='*.rs' --include='*.java' --include='*.rb' --include='*.c' --include='*.cpp' --include='*.h' --include='*.cs' --include='*.php' -l 2>/dev/null | head -10"
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=3, cwd=repo_path,
            )
            files = [
                f.lstrip("./")
                for f in proc.stdout.strip().split("\n")
                if f.strip()
            ]
            return ident, files
        except subprocess.TimeoutExpired:
            return ident, []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_grep_one, ident): ident for ident in idents_to_search}
        for future in as_completed(futures, timeout=8):
            try:
                ident, files = future.result()
                if files:
                    results[ident] = files
            except Exception:
                continue

    return results


def resolve_file_paths(mentioned_paths: list[str], file_tree: list[str]) -> list[str]:
    """Match mentioned file paths against the actual file tree."""
    resolved = []
    for mentioned in mentioned_paths:
        for actual in file_tree:
            if actual.endswith(mentioned) or mentioned in actual:
                if actual not in resolved:
                    resolved.append(actual)
                break
    return resolved


def build_recon_context(task_prompt: str, repo_path: str) -> str:
    """
    Full pre-LLM recon. Returns context string for prompt injection.
    Runs everything in parallel where possible. Target: < 5 seconds total.
    """
    parts: list[str] = []

    info = detect_repo_info(repo_path)
    parts.append(
        f"Repository: {info['language']} project, "
        f"~{info['file_count']} files"
        + (f", {info['framework']}" if info['framework'] else "")
    )

    file_tree = get_file_tree(repo_path, max_files=300)
    mentioned_paths = extract_file_paths(task_prompt)
    identifiers = extract_identifiers(task_prompt)

    file_map: dict[str, list[str]] = {}
    if identifiers:
        file_map = grep_identifiers(identifiers, repo_path)

    resolved_paths = resolve_file_paths(mentioned_paths, file_tree) if mentioned_paths else []

    if resolved_paths:
        parts.append("\nFiles explicitly mentioned in task (verified in repo):")
        for p in resolved_paths[:8]:
            parts.append(f"  {p}")

    if file_map:
        parts.append("\nIdentifier locations (pre-grepped):")
        for ident, files in file_map.items():
            files_str = ", ".join(files[:5])
            parts.append(f"  '{ident}' -> {files_str}")

    file_freq: Counter = Counter()
    for p in resolved_paths:
        file_freq[p] += 3
    for files in file_map.values():
        for f in files:
            file_freq[f] += 1

    if file_freq:
        top_files = file_freq.most_common(8)
        parts.append("\nRANKED TARGET FILES (start here):")
        for i, (f, score) in enumerate(top_files, 1):
            parts.append(f"  {i}. {f} (relevance: {score})")
    elif not resolved_paths and not file_map:
        if identifiers:
            parts.append(f"\nIdentifiers extracted: {', '.join(identifiers[:5])}")
        parts.append("(no matches found — use grep to locate target files)")

    code_files = [f for f in file_tree if any(f.endswith(e) for e in _CODE_EXTENSIONS)]
    tree_sample = code_files[:150] if code_files else file_tree[:150]
    if tree_sample:
        parts.append(f"\nFile tree ({len(tree_sample)} of {len(file_tree)} files):")
        for f in tree_sample:
            parts.append(f"  {f}")
        if len(file_tree) > len(tree_sample):
            parts.append(f"  ... and {len(file_tree) - len(tree_sample)} more files")

    return "\n".join(parts)
