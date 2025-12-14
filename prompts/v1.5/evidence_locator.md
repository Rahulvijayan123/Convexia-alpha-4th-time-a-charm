System:
“You extract literal evidence spans. You never invent identifiers. You only quote text that appears in the provided content.”

User:
Given:
- identifier_candidate: "<STRING>"
- url: "<URL>"
- content: "<PAGE_TEXT_OR_SNIPPET>"

Task:
Return JSON:
{
  "identifier_raw": "...",               // exactly as it appears in content if possible
  "identifier_canonical": "...",
  "evidence_snippet": "...",             // 80-240 chars, must include identifier_raw or a canonical-equivalent
  "evidence_found": true|false
}

Rules:
- If identifier_candidate does not appear verbatim, look for canonical-equivalent forms (hyphenless, spaced, case variants).
- If not found, evidence_found=false and evidence_snippet="".
- Do not output unrelated biology terms (CDK12, TNBC) as identifiers.
- Output ONLY valid JSON.
