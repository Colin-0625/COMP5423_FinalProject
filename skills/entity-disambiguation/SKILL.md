---
name: entity-disambiguation
description: Use when search results return multiple people, organizations, places, or works with identical or very similar names, or when candidate documents present conflicting dates, titles, or affiliations for a target entity.
---

# Entity Disambiguation & Canonical Name Verification

Follow this procedure when two or more candidate entities could match the question or when the exact official name/form of the answer is uncertain.

## 1. Identify the Conflicting Candidates
List the competing candidates briefly in your thought (e.g., Candidate A vs. Candidate B) and note which document IDs mention each one.

## 2. Extract Discriminating Attributes
Find at least one attribute from the question that **separates** the candidates:
- Birth/death or founding/dissolution year
- Nationality, birthplace, or headquarters location
- Parent organization, alma mater, collaborator, or notable work
- Specific role, award, or chronological order (e.g., father vs. son, original vs. remake)

## 3. Run Targeted Narrowing Queries
- Query `"<candidate name>" <discriminator>` using `search` (keep queries concise).
- For long candidate biographies or articles already found, prefer `find_in_doc(doc_id, keyword)` to check the discriminating year, place, or affiliation before opening additional pages.

## 4. Confirm the Exact Canonical Answer Form
Before emitting `final_answer`, verify the exact phrasing requested by the question:
- If the question asks for a **full name**, check the lead paragraph of the verified document for middle names or official spelling.
- If the question asks for an **organization, title, or place**, prefer the standard canonical name over informal abbreviations unless an abbreviation is explicitly requested.
- Output ONLY the verified short answer (no parentheses, no trailing commentary).
