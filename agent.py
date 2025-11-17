#!/usr/bin/env python3
# PercepNet Agent - policy + markdown instructions aware (single-file)
# Usage: see CLI subcommands at bottom. Drops in as agent.py at repo root.

from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import json
import re
import shutil

REPO_ROOT = Path(__file__).resolve().parent

# -------------------------
# Policy loader (yaml or json or simple fallback)
# -------------------------
POLICY_PATHS = [REPO_ROOT / "policy.yaml", REPO_ROOT / "policy.yml", REPO_ROOT / "policy.json"]

def load_policy() -> dict:
    defaults = {
        "require_approval_for": ["shell", "patch"],
        "summarize_from_comments_only": True,
        "search_mode": "literal",
        "max_search_hits": 50,
        "show_policy_on_startup": False
    }
    for p in POLICY_PATHS:
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            if p.suffix == ".json":
                try:
                    parsed = json.loads(text)
                    return {**defaults, **(parsed or {})}
                except Exception:
                    break
            else:
                # Prefer PyYAML if available
                try:
                    import yaml  # type: ignore
                    parsed = yaml.safe_load(text)
                    return {**defaults, **(parsed or {})}
                except Exception:
                    # simple fallback parser for key: value (and simple list)
                    policy = {}
                    for line in text.splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if ":" not in line:
                            continue
                        k, v = line.split(":", 1)
                        k = k.strip()
                        v = v.strip()
                        if v.startswith("[") and v.endswith("]"):
                            items = [it.strip().strip("'\"") for it in v[1:-1].split(",") if it.strip()]
                            policy[k] = items
                        elif v.lower() in ("true", "false"):
                            policy[k] = v.lower() == "true"
                        else:
                            try:
                                policy[k] = int(v)
                            except Exception:
                                policy[k] = v
                    return {**defaults, **policy}
    return defaults

POLICY = load_policy()

# -------------------------
# Instructions loader & parser (Markdown)
# -------------------------
INSTR_PATHS = [
    REPO_ROOT / ".github" / "copilot-instructions.md",
    REPO_ROOT / ".github" / "copilot-instructions.md",  # duplicate kept intentionally if you have multiple variants
    REPO_ROOT / "README.md",
    REPO_ROOT / "agent-instructions.md"
]

def load_instructions() -> Tuple[Optional[Path], str]:
    for p in INSTR_PATHS:
        if p.exists():
            return p, p.read_text(encoding="utf-8", errors="ignore")
    return None, ""

def parse_instructions_rules(md_text: str) -> dict:
    """
    Conservative extraction: look for explicit YAML-like lines in the Markdown.
    Only extracts if Markdown contains explicit key: value or key: [a,b] style lines.
    """
    rules = {}
    patterns = {
        "require_approval_for": re.compile(r"require_approval_for\s*[:\-]\s*\[?([^\]\n]+)\]?", re.IGNORECASE),
        "summarize_from_comments_only": re.compile(r"summarize_from_comments_only\s*[:\-]\s*(true|false)", re.IGNORECASE),
        "search_mode": re.compile(r"search_mode\s*[:\-]\s*([a-z_]+)", re.IGNORECASE),
        "max_search_hits": re.compile(r"max_search_hits\s*[:\-]\s*(\d+)", re.IGNORECASE)
    }
    for k, pat in patterns.items():
        m = pat.search(md_text)
        if m:
            v = m.group(1).strip()
            if k == "require_approval_for":
                items = [it.strip().strip("',\"") for it in re.split(r"[,\s]+", v) if it.strip()]
                rules[k] = items
            elif k == "summarize_from_comments_only":
                rules[k] = (v.lower() == "true")
            elif k == "max_search_hits":
                try:
                    rules[k] = int(v)
                except:
                    pass
            else:
                rules[k] = v
    return rules

def compare_policy_vs_instructions(policy: dict, instr_rules: dict) -> List[str]:
    mismatches = []
    for k, instr_v in instr_rules.items():
        pol_v = policy.get(k)
        if pol_v != instr_v:
            mismatches.append(f"Mismatch for '{k}': policy={pol_v!r} vs instructions={instr_v!r}")
    return mismatches

# -------------------------
# Utility functions (robust command launch)
# -------------------------
def run_cmd(cmd: str, cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr). Handles launch errors."""
    try:
        p = subprocess.Popen(
            shlex.split(cmd), cwd=str(cwd or REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd}"
    except PermissionError as e:
        return 126, "", f"Permission denied when executing: {cmd}: {e}"
    except Exception as e:
        return 1, "", f"Error launching command: {e}"
    out, err = p.communicate()
    return p.returncode, out, err

def ripgrep_available() -> bool:
    try:
        rg_path = shutil.which("rg")
        return bool(rg_path and os.access(rg_path, os.X_OK))
    except Exception:
        return False

def safe_read_file(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return path.read_text(encoding="utf-8", errors="ignore")

# -------------------------
# Search & explain helpers
# -------------------------
def search_codebase(query: str, repo_root: Path = REPO_ROOT, max_hits: Optional[int] = None) -> List[Tuple[Path, int, str]]:
    max_hits = max_hits or POLICY.get("max_search_hits", 50)
    hits = []
    if ripgrep_available():
        cmd = f"rg -n --hidden --no-ignore-vcs -S --max-filesize 2M -- {shlex.quote(query)} {shlex.quote(str(repo_root))}"
        rc, out, err = run_cmd(cmd)
        if rc in (0, 1):
            for line in out.splitlines():
                try:
                    p, ln, content = line.split(":", 2)
                    hits.append((Path(p), int(ln), content.strip()))
                    if len(hits) >= max_hits:
                        break
                except ValueError:
                    continue
        return hits
    # fallback python grep
    for p in repo_root.rglob("*"):
        if p.is_file() and p.suffix in (".py", ".cpp", ".c", ".h", ".hpp", ".txt", ".md"):
            try:
                for i, l in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                    if query in l:
                        hits.append((p, i, l.strip()))
                        if len(hits) >= max_hits:
                            return hits
            except Exception:
                continue
    return hits

def extract_context(path: Path, line_no: int, context: int = 6) -> Tuple[int, int, List[str]]:
    text = safe_read_file(path)
    lines = text.splitlines()
    start = max(1, line_no - context)
    end = min(len(lines), line_no + context)
    snippet = lines[start-1:end]
    return start, end, snippet

# -------------------------
# Agent class
# -------------------------
class SimpleAgent:
    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self.policy = POLICY
        self.instr_path, self.instr_text = load_instructions()
        self.instr_rules = parse_instructions_rules(self.instr_text) if self.instr_text else {}

    def show_policy(self) -> str:
        return json.dumps(self.policy, indent=2)

    def show_instructions(self, length: int = 2000) -> str:
        if not self.instr_path:
            return "No instructions Markdown file found in known locations."
        text = self.instr_text
        out = [f"Loaded instructions from: {self.instr_path}", "-"*72]
        out.append(text[:length] + ("" if len(text)<=length else "\n\n... (truncated)"))
        return "\n".join(out)

    def check_consistency(self) -> str:
        if not self.instr_rules:
            return "No explicit policy lines found in instructions. Nothing to compare."
        mismatches = compare_policy_vs_instructions(self.policy, self.instr_rules)
        if not mismatches:
            return "Policy and instructions explicit rules match."
        return "Inconsistencies found:\n" + "\n".join(mismatches)

    def search(self, query: str, max_hits: Optional[int] = None) -> List[Tuple[Path, int, str]]:
        max_hits = max_hits or self.policy.get("max_search_hits", 50)
        return search_codebase(query, self.repo_root, max_hits=max_hits)

    def explain_symbol(self, symbol: str) -> str:
        hits = self.search(symbol, max_hits=self.policy.get("max_search_hits", 50))
        if not hits:
            return f"No occurrences of '{symbol}' found in repository under {self.repo_root}."

        # Prefer definitions heuristically
        preferred = None
        for p, ln, ltext in hits:
            lower = ltext.lower()
            if (p.suffix == ".py" and (lower.startswith("def ") or f"def {symbol}" in lower or f"class {symbol}" in lower)) \
               or (p.suffix in (".cpp", ".c", ".h", ".hpp") and ("(" in ltext and symbol in ltext and (";" not in ltext or "{" in ltext))):
                preferred = (p, ln)
                break
        if preferred is None:
            preferred = (hits[0][0], hits[0][1])

        p, ln = preferred
        start, end, snippet = extract_context(p, ln, context=8)

        observations = []
        for i, line in enumerate(snippet, start=start):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("#"):
                observations.append(f"{p}:{i}: {stripped}")
            else:
                if any(tok in line for tok in ("480", "FRAME", "ERB", "34", "FEATURE", "68", "NB_BANDS")):
                    observations.append(f"{p}:{i}: {stripped}")

        out_lines = []
        out_lines.append(f"Found occurrence for '{symbol}' in file: {p} (context lines {start}-{end}).")
        out_lines.append("Code snippet (lines included exactly):")
        out_lines.append("-" * 72)
        for i, line in enumerate(snippet, start=start):
            out_lines.append(f"{i:4d}: {line.rstrip()}")
        out_lines.append("-" * 72)
        if observations:
            out_lines.append("Observations (literal, derived from the snippet or adjacent comments):")
            for o in observations:
                out_lines.append(f" - {o}")
        else:
            out_lines.append("No code comments or obvious constants found in the snippet. Please inspect the snippet above.")
        out_lines.append("If you want a summarized explanation, run this agent with '--summarize' (the summary will be explicitly labelled and will only paraphrase text/comments present in the snippet according to policy).")
        return "\n".join(out_lines)

    def summarize_symbol(self, symbol: str) -> str:
        hits = self.search(symbol, max_hits=self.policy.get("max_search_hits", 50))
        if not hits:
            return f"No occurrences of '{symbol}' found."

        p, ln, _ = hits[0]
        start, end, snippet = extract_context(p, ln, context=8)
        comments = []
        for i, line in enumerate(snippet, start=start):
            s = line.strip()
            if s.startswith("#") or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
                comments.append(s.lstrip("#/ *"))

        if self.policy.get("summarize_from_comments_only", True):
            if not comments:
                return ("No inline comments found near the symbol. Policy requires summaries be derived from comments only. "
                        "I will not invent functionality. Please inspect the snippet with 'explain'.")
            paraphrase = " ".join([c.strip() for c in comments])
            return f"Paraphrase of nearby comments: {paraphrase}"
        else:
            parts = []
            if comments:
                parts.append(" ".join([c.strip() for c in comments]))
            code_lines = [line.strip() for line in snippet if line.strip() and not (line.strip().startswith("#") or line.strip().startswith("//") or line.strip().startswith("/*"))]
            if code_lines:
                parts.append("Code context: " + " ".join(code_lines[:6]))
            if not parts:
                return "No nearby comments or code context to summarize."
            return " | ".join(parts)

    def run_shell(self, cmd: str, approve: bool = False) -> str:
        require = self.policy.get("require_approval_for", [])
        if "shell" in require and not approve:
            return f"Refusing to run '{cmd}' without explicit approval (policy requires approval for: {require}). Re-run with --approve to execute."
        rc, out, err = run_cmd(cmd, cwd=self.repo_root)
        return f"Return code: {rc}\n--- STDOUT ---\n{out}\n--- STDERR ---\n{err}"

    def propose_patch(self, file_path: Path, new_contents: str, approve: bool = False) -> str:
        file_path = self.repo_root.joinpath(file_path)
        if not file_path.exists():
            old = ""
        else:
            old = file_path.read_text(encoding="utf-8", errors="ignore")

        import difflib
        old_lines = old.splitlines(keepends=True)
        new_lines = new_contents.splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=str(file_path), tofile=str(file_path) + " (new)"))

        require = self.policy.get("require_approval_for", [])
        if "patch" in require and not approve:
            return f"Patch (not applied). Policy requires approval for: {require}\n\n{diff or 'No changes.'}"
        # apply
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_contents, encoding="utf-8")
        rc, _, _ = run_cmd(f"git add {shlex.quote(str(file_path))}", cwd=self.repo_root)
        if rc != 0:
            return f"Applied file write but failed to 'git add'. See repo. Diff:\n\n{diff}"
        return f"Patch applied and staged. Diff:\n\n{diff or 'No changes.'}"

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(prog="percepnet-agent", description="Policy+instructions-aware agent for PercepNet codebase.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s_explain = sub.add_parser("explain", help="Find and show code snippet for SYMBOL")
    s_explain.add_argument("symbol", help="Symbol or term to search")
    s_explain.add_argument("--summarize", action="store_true", help="Produce short paraphrase from nearby comments")

    s_search = sub.add_parser("search", help="Search literal text in repo")
    s_search.add_argument("query", help="Query string")

    s_shell = sub.add_parser("shell", help="Run a shell command (requires approval by policy)")
    s_shell.add_argument("cmd", help="Command to run")
    s_shell.add_argument("--approve", action="store_true", help="Approve execution")

    s_patch = sub.add_parser("patch", help="Propose a patch (requires approval to apply)")
    s_patch.add_argument("file", help="File path relative to repo root")
    s_patch.add_argument("patchfile", help="Local file containing new contents")
    s_patch.add_argument("--approve", action="store_true", help="Approve applying the patch")

    s_policy = sub.add_parser("policy", help="Show the effective loaded policy (merged with defaults)")
    s_instructions = sub.add_parser("instructions", help="Show which Markdown instructions were loaded and a preview")
    s_check = sub.add_parser("check", help="Compare explicit rules in instructions vs policy and show mismatches")

    args = parser.parse_args()
    agent = SimpleAgent()

    # Optionally show policy at startup if configured
    if agent.policy.get("show_policy_on_startup", False):
        print("Effective policy (merged with defaults):")
        print(agent.show_policy())
        print("-" * 72)

    if args.cmd == "explain":
        if args.summarize:
            print(agent.summarize_symbol(args.symbol))
        else:
            print(agent.explain_symbol(args.symbol))
    elif args.cmd == "search":
        hits = agent.search(args.query)
        if not hits:
            print("No hits.")
            return
        for p, ln, text in hits[: agent.policy.get("max_search_hits", 50)]:
            print(f"{p}:{ln}: {text}")
    elif args.cmd == "shell":
        print(agent.run_shell(args.cmd, approve=args.approve))
    elif args.cmd == "patch":
        new_contents = safe_read_file(Path(args.patchfile))
        print(agent.propose_patch(Path(args.file), new_contents, approve=args.approve))
    elif args.cmd == "policy":
        print(agent.show_policy())
    elif args.cmd == "instructions":
        print(agent.show_instructions())
    elif args.cmd == "check":
        print(agent.check_consistency())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
