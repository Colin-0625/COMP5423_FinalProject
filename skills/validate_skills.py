"""Validate all skills in code/skills/ against the Agent Skills specification
and runtime-only loading contract.

Usage:
  python skills/validate_skills.py
"""
from __future__ import annotations

import os
import sys
from typing import List

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import config
from agent.llm_client import LLMClient
from agent.loop import SearchAgent
from tools.base import ToolRegistry
from tools.skill_tool import LoadSkillTool, SKILL_NAME_RE, SkillLibrary, parse_skill_markdown


def validate_skills(skills_dir: str = config.SKILLS_DIR) -> List[str]:
    errors: List[str] = []
    if not os.path.isdir(skills_dir):
        return [f"Skills directory not found: {skills_dir}"]

    skill_dirs = [
        d for d in sorted(os.listdir(skills_dir))
        if os.path.isdir(os.path.join(skills_dir, d)) and not d.startswith((".", "__"))
    ]
    if not skill_dirs:
        return [f"No skill directories found in {skills_dir}"]

    valid_skills: List[str] = []
    bodies_by_name = {}

    for d in skill_dirs:
        sdir = os.path.join(skills_dir, d)
        skill_md = os.path.join(sdir, "SKILL.md")
        if not os.path.isfile(skill_md):
            errors.append(f"[{d}] Missing required SKILL.md")
            continue

        with open(skill_md, "r", encoding="utf-8") as f:
            raw = f.read()
        meta, body = parse_skill_markdown(raw)

        if not meta:
            errors.append(f"[{d}] Missing or invalid YAML frontmatter delimited by ---")
            continue

        name = meta.get("name")
        desc = meta.get("description")

        if not isinstance(name, str) or not name.strip():
            errors.append(f"[{d}] Frontmatter missing non-empty string 'name'")
        else:
            name = name.strip()
            if not (1 <= len(name) <= 64):
                errors.append(f"[{d}] 'name' length must be 1..64 (got {len(name)})")
            if not SKILL_NAME_RE.match(name):
                errors.append(f"[{d}] 'name' {name!r} must be lowercase alphanumeric with hyphens")
            if name != d:
                errors.append(f"[{d}] 'name' {name!r} must match directory name {d!r}")

        if not isinstance(desc, str) or not desc.strip():
            errors.append(f"[{d}] Frontmatter missing non-empty string 'description'")
        elif not (1 <= len(desc.strip()) <= 1024):
            errors.append(f"[{d}] 'description' length must be 1..1024 (got {len(desc.strip())})")

        if not body or len(body) < 40:
            errors.append(f"[{d}] Markdown instructions body is empty or too short")

        if not errors or all(not e.startswith(f"[{d}]") for e in errors):
            valid_skills.append(d)
            bodies_by_name[d] = body

    if len(valid_skills) < 2:
        errors.append(
            f"Automatic skill selection requires >= 2 valid skills (found {len(valid_skills)}: {valid_skills})"
        )

    # Verify runtime-loading contract: base prompt contains catalog only, never full SKILL.md bodies
    lib = SkillLibrary(skills_dir)
    if set(lib.names()) != set(valid_skills):
        errors.append(f"SkillLibrary loaded {lib.names()}, expected {valid_skills}")

    registry = ToolRegistry()
    registry.register(LoadSkillTool(lib))
    agent = SearchAgent(llm=LLMClient(api_key=""), registry=registry, skill_library=lib, use_skills=True)
    base_prompt = agent.build_system_prompt()
    for sname, body in bodies_by_name.items():
        if sname not in base_prompt:
            errors.append(f"[{sname}] Skill name missing from base prompt catalog")
        body_lines = [ln.strip() for ln in body.splitlines() if len(ln.strip()) >= 35 and not ln.startswith("#")]
        if body_lines and body_lines[0] in base_prompt:
            errors.append(f"[{sname}] Full skill instructions leaked into base prompt!")

    return errors


def main() -> None:
    errors = validate_skills(config.SKILLS_DIR)
    lib = SkillLibrary(config.SKILLS_DIR)
    if errors:
        print(f"SKILL VALIDATION FAILED ({len(errors)} error(s)):")
        for err in errors:
            print("  -", err)
        sys.exit(1)
    print(f"SKILL VALIDATION PASSED ({len(lib.names())} original skills):")
    for name, info in lib.skills.items():
        print(f"  - {name}: {info['description'][:90]}... (resources={info['resources']})")


if __name__ == "__main__":
    main()
