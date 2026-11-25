"""Runtime Agent Skills loader and tool.

Contract:
  - `SkillLibrary.catalog()` exposes only `name` and `description` for the base prompt.
  - Full `SKILL.md` instructions (and optional `references/` / `scripts/` / `assets/` files)
    are never permanently embedded in the base prompt; they are loaded at runtime via `load_skill`.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import config
from agent.types import ToolResult
from tools.base import Tool


SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
OPTIONAL_SUBDIRS = ("references", "scripts", "assets")


def parse_skill_markdown(text: str) -> Tuple[Dict[str, Any], str]:
    """Split SKILL.md into (frontmatter_dict, markdown_body)."""
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", text, flags=re.S)
    if not m:
        return {}, text.strip()
    raw_fm = m.group(1)
    body = text[m.end():].strip()
    try:
        import yaml

        loaded = yaml.safe_load(raw_fm)
        meta = loaded if isinstance(loaded, dict) else {}
    except Exception:
        meta = {}
        for line in raw_fm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip("'\"")
    return meta, body


class SkillLibrary:
    """Scans config.SKILLS_DIR for Agent Skills directories."""

    def __init__(self, skills_dir: str = config.SKILLS_DIR):
        self.dir = skills_dir
        self.skills: Dict[str, Dict[str, Any]] = {}
        if not os.path.isdir(skills_dir):
            return
        for entry in sorted(os.listdir(skills_dir)):
            sdir = os.path.join(skills_dir, entry)
            skill_md = os.path.join(sdir, "SKILL.md")
            if not (os.path.isdir(sdir) and os.path.isfile(skill_md)):
                continue
            with open(skill_md, "r", encoding="utf-8") as f:
                raw = f.read()
            meta, body = parse_skill_markdown(raw)
            name = str(meta.get("name", "")).strip()
            desc = str(meta.get("description", "")).strip()
            if not (name and desc and body):
                continue
            resources = self._list_resources(sdir)
            self.skills[name] = {
                "name": name,
                "description": desc,
                "dir": sdir,
                "path": skill_md,
                "resources": resources,
            }

    @staticmethod
    def _list_resources(skill_dir: str) -> List[str]:
        found: List[str] = []
        for sub in OPTIONAL_SUBDIRS:
            sub_path = os.path.join(skill_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            for root, _dirs, files in os.walk(sub_path):
                for fn in sorted(files):
                    rel = os.path.relpath(os.path.join(root, fn), skill_dir).replace("\\", "/")
                    found.append(rel)
        return found

    def names(self) -> List[str]:
        return list(self.skills.keys())

    def catalog(self) -> str:
        """Short catalog (name + description ONLY) for base prompt."""
        if not self.skills:
            return "(no skills available)"
        return "\n".join(f"- {name}: {info['description']}" for name, info in self.skills.items())

    def read(self, name: str, resource: str = "") -> Optional[str]:
        """Read full SKILL.md body (or an optional bundled resource file) at runtime."""
        info = self.skills.get(name)
        if info is None:
            return None
        if not resource:
            with open(info["path"], "r", encoding="utf-8") as f:
                raw = f.read()
            _meta, body = parse_skill_markdown(raw)
            if info["resources"]:
                res_list = ", ".join(info["resources"])
                body += f"\n\n[Optional bundled resources in this skill: {res_list}. Call load_skill(name={name!r}, resource='<path>') if needed.]"
            return body

        # Safe path resolution inside skill directory
        norm_rel = resource.replace("\\", "/").lstrip("/")
        base_dir = os.path.realpath(info["dir"])
        target = os.path.realpath(os.path.join(base_dir, norm_rel))
        if not target.startswith(base_dir + os.sep) or not os.path.isfile(target):
            return None
        with open(target, "r", encoding="utf-8") as f:
            return f.read()


class LoadSkillTool(Tool):
    name = "load_skill"
    description = (
        "Load the full runtime instructions of a reusable skill by name (and optional bundled resource path) "
        "when its description matches the current question or search state. Optional — skip if no skill is needed."
    )
    parameters = {
        "name": "string - skill name from the skill catalog",
        "resource": "string, optional - relative path inside the skill directory (e.g. 'references/clue-priority.md')",
    }
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact skill name from the skill catalog"},
            "resource": {
                "type": "string",
                "description": "Optional relative path to a bundled resource file inside the skill directory",
            },
        },
        "required": ["name"],
    }

    def __init__(self, library: SkillLibrary):
        self.library = library

    def run(self, name: str = "", resource: str = "", **_: Any) -> ToolResult:
        sname = str(name).strip()
        res = str(resource or "").strip()
        if sname not in self.library.skills:
            return ToolResult(
                f"Error: unknown skill {sname!r}. Available: {self.library.names()}",
                error=True,
            )
        body = self.library.read(sname, resource=res)
        if body is None:
            avail = self.library.skills[sname]["resources"]
            return ToolResult(
                f"Error: resource {res!r} not found in skill {sname!r}. Available resources: {avail}",
                error=True,
            )
        return ToolResult(body, data={"skill": sname, "resource": res or "SKILL.md"})
