"""
Core solver for Bittensor subnet 66 mining.

Competitive edges:

  1. PRE-COMPUTED CONTEXT WITH FILE TREE:
     Before the first API call we build a complete file tree, extract and
     grep all identifiers, resolve mentioned file paths, and rank targets
     by relevance. Claude gets a full map on turn 1 — no exploration waste.

  2. DYNAMIC TIME MANAGEMENT:
     The validator's actual timeout is min(baseline_elapsed * 2 + 1, 300).
     We detect the budget from the environment or default conservatively,
     then set phase gates as fractions of the budget.

  3. GIT DIFF SELF-CHECK:
     After the main edit phase, we run 'git diff' and inject it as context.
     Claude verifies coverage of acceptance criteria and catches obvious
     misses in a single focused turn — much cheaper than a full review.

  4. ABSOLUTE SCORING AWARENESS:
     Win condition is matched_changed_lines > king_lines (absolute count).
     The prompt is tuned to maximize matched lines, not minimize diff size.

  5. EDIT RESILIENCE:
     Per-file error tracking, automatic re-read steering on failure, and
     smart blocking after repeated failures on the same file.
"""

import os
import subprocess
import time
from collections import defaultdict
from typing import Optional

import anthropic
import httpx

from prompts import (
    SYSTEM_PROMPT,
    build_task_prompt,
    TIME_WARNING_PROMPT,
    FORCE_EDIT_PROMPT,
    ANTI_ZERO_PROMPT,
    DIFF_CHECK_PROMPT,
)
from recon import build_recon_context
from tools import ToolExecutor


def _detect_time_budget() -> float:
    """Detect available time budget from environment or default conservatively.

    The validator sets agent_timeout = min(baseline_elapsed * 2 + 1, 300).
    We look for TAU_AGENT_TIMEOUT / PI_AGENT_TIMEOUT env vars, otherwise
    fall back to a safe 170s that works under any configuration.
    """
    for var in ("TAU_AGENT_TIMEOUT", "PI_AGENT_TIMEOUT"):
        val = os.environ.get(var)
        if val:
            try:
                return max(30.0, float(val) - 10.0)
            except ValueError:
                pass
    return 170.0


class TauSolver:
    MAX_READS_BEFORE_EDIT = 5
    EDIT_ERROR_THRESHOLD = 2
    MAX_ITERATIONS = 60
    MAX_TOKENS = 16384
    PROVIDER_RETRY_MAX = 2

    def __init__(self, repo_path: str, model: str = "docker-proxy-model") -> None:
        self.repo_path = repo_path
        self.model = model
        self.tools = ToolExecutor(repo_path)
        self.client = self._build_client()

        self.time_budget = _detect_time_budget()
        self.hard_exit_s = self.time_budget
        self.slow_start_s = self.time_budget * 0.45
        self.anti_zero_s = self.time_budget * 0.85
        self.diff_check_cutoff_s = 25.0  # seconds remaining required to attempt diff check

    def _build_client(self) -> anthropic.Anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "proxy-key")
        socket_path = (
            os.environ.get("TAU_PROXY_SOCKET_PATH")
            or os.environ.get("PI_PROXY_SOCKET_PATH")
        )
        proxy_port = (
            os.environ.get("TAU_PROXY_LISTEN_PORT")
            or os.environ.get("PI_PROXY_LISTEN_PORT")
            or "4318"
        )

        if socket_path and os.path.exists(socket_path):
            transport = httpx.HTTPTransport(uds=socket_path)
            http_client = httpx.Client(
                transport=transport,
                base_url="http://openrouter-proxy",
            )
            return anthropic.Anthropic(
                api_key=api_key,
                base_url="http://openrouter-proxy",
                http_client=http_client,
            )

        return anthropic.Anthropic(
            api_key=api_key,
            base_url=f"http://127.0.0.1:{proxy_port}",
        )

    def solve(self, task_prompt: str) -> tuple[str, bool]:
        start = time.time()

        # Phase 0: Pre-LLM reconnaissance (2-5s, no API calls)
        recon_context = ""
        try:
            recon_context = build_recon_context(task_prompt, self.repo_path)
        except Exception:
            pass

        # Phase 1: Main solving loop
        messages: list[dict] = [
            {"role": "user", "content": build_task_prompt(
                task_prompt, self.repo_path, recon_context
            )}
        ]

        output_lines: list[str] = []
        has_edited = False
        reads_without_edit = 0
        slow_warned = False
        anti_zero_fired = False
        edit_errors: dict[str, int] = defaultdict(int)

        for iteration in range(self.MAX_ITERATIONS):
            elapsed = time.time() - start

            if elapsed >= self.hard_exit_s:
                break

            if not has_edited and not anti_zero_fired and elapsed >= self.anti_zero_s:
                anti_zero_fired = True
                messages.append({"role": "user", "content": ANTI_ZERO_PROMPT})

            if not has_edited and not slow_warned and elapsed >= self.slow_start_s:
                slow_warned = True
                remaining = int(self.hard_exit_s - elapsed)
                messages.append({"role": "user", "content": (
                    f"URGENT: {int(elapsed)}s elapsed, only {remaining}s remain, "
                    f"and you have no edits yet. " + TIME_WARNING_PROMPT
                )})

            if reads_without_edit >= self.MAX_READS_BEFORE_EDIT:
                messages.append({"role": "user", "content": FORCE_EDIT_PROMPT})
                reads_without_edit = 0

            response = self._call_api_with_retry(messages, start)
            if response is None:
                break

            text = self._extract_text(response)
            if text:
                output_lines.append(text)

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_results, edit_made, read_count, file_edit_errors = (
                    self._run_tools(response, edit_errors)
                )

                if edit_made:
                    has_edited = True
                    reads_without_edit = 0
                else:
                    reads_without_edit += read_count

                for path, count in file_edit_errors.items():
                    edit_errors[path] += count
                    if edit_errors[path] >= self.EDIT_ERROR_THRESHOLD:
                        tool_results.append({
                            "type": "text",
                            "text": (
                                f"STOP editing '{path}'. {edit_errors[path]} failed "
                                f"attempts. Re-read it with read_file first, or try "
                                f"write_file to replace the entire file."
                            ),
                        })

                messages.append({"role": "user", "content": tool_results})
            else:
                messages.append({
                    "role": "user",
                    "content": "Continue. Call a tool — do not write narrative text.",
                })

        # Phase 2: Git diff self-check (if time remains and edits were made)
        if has_edited and (self.hard_exit_s - (time.time() - start)) > self.diff_check_cutoff_s:
            check_text = self._diff_check(messages, start)
            if check_text:
                output_lines.append(check_text)

        # Result
        diff_stat = self._get_diff_stat()
        if diff_stat:
            has_edited = True

        return "\n\n".join(output_lines), has_edited

    def _call_api_with_retry(
        self, messages: list[dict], start: float = 0
    ) -> Optional[anthropic.types.Message]:
        remaining = self.hard_exit_s - (time.time() - start) if start else 60
        if remaining < 5:
            return None

        for attempt in range(self.PROVIDER_RETRY_MAX + 1):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=self.tools.definitions(),
                    messages=messages,
                )
            except anthropic.APIStatusError:
                if attempt == self.PROVIDER_RETRY_MAX:
                    return None
                time.sleep(1)
            except Exception:
                return None
        return None

    def _run_tools(
        self, response: anthropic.types.Message, edit_errors: dict[str, int]
    ) -> tuple[list[dict], bool, int, dict[str, int]]:
        """Returns (results, any_edit_made, read_count, new_file_errors)."""
        results = []
        edit_made = False
        read_count = 0
        new_errors: dict[str, int] = defaultdict(int)

        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name in ("edit_file", "write_file"):
                path = block.input.get("path", "")
                if edit_errors.get(path, 0) >= self.EDIT_ERROR_THRESHOLD:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            f"[blocked] Too many failed edits on '{path}'. "
                            f"Re-read it first or try write_file to replace entirely."
                        ),
                    })
                    continue

            output = self.tools.execute(block.name, block.input)
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })

            if block.name in ("write_file", "edit_file"):
                if output.startswith("[tool-error]"):
                    path = block.input.get("path", "")
                    new_errors[path] += 1
                else:
                    edit_made = True
            elif block.name in ("read_file", "grep", "find_files"):
                read_count += 1

        return results, edit_made, read_count, dict(new_errors)

    def _diff_check(self, prior_messages: list[dict], start: float) -> str:
        """Run git diff and let Claude verify coverage in one focused turn."""
        if (time.time() - start) >= self.hard_exit_s - 15:
            return ""

        diff_output = self._get_diff()
        if not diff_output or len(diff_output) < 10:
            return ""

        if len(diff_output) > 8000:
            diff_output = diff_output[:8000] + "\n... (diff truncated)"

        messages = list(prior_messages) + [
            {"role": "user", "content": (
                f"Here is your current git diff:\n```\n{diff_output}\n```\n\n"
                + DIFF_CHECK_PROMPT
            )}
        ]

        try:
            response = self._call_api_with_retry(messages, start)
            if response is None:
                return ""

            text = self._extract_text(response)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                results, _, _, _ = self._run_tools(response, defaultdict(int))
                # One fix attempt: send results back so model confirms or signals done
                messages.append({"role": "user", "content": results})
                confirm = self._call_api_with_retry(messages, start)
                if confirm:
                    extra = self._extract_text(confirm)
                    if extra:
                        return (text or "") + "\n" + extra
            return text or ""
        except Exception:
            return ""

    def _extract_text(self, response: anthropic.types.Message) -> str:
        return "\n".join(
            block.text for block in response.content
            if hasattr(block, "text") and block.text
        )

    def _get_diff_stat(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True, text=True,
                cwd=self.repo_path, timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def _get_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True, text=True,
                cwd=self.repo_path, timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""
