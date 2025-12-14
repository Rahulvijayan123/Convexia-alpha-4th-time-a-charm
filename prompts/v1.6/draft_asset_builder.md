System:
“You produce draft asset records with evidence anchoring. Unknown fields are allowed. Do not hallucinate.”

User:
Given:
- identifier_candidate
- evidence_url
- evidence_snippet

Task:
Return JSON:
{
  "identifier_raw": "...",
  "identifier_canonical": "...",
  "evidence_url": "...",
  "evidence_snippet": "...",
  "evidence_source_type": "patent|trial|registry|pipeline_pdf|paper|vendor|press_release|other",
  "confidence_discovery": 0.0,
  "sponsor": null,
  "target": null,
  "modality": null,
  "indication": null,
  "stage": null,
  "geography": null,
  "citations": [{"url":"...","snippet":"..."}]
}

Rules:
- sponsor/target/modality/etc may be null. Do not guess.
- evidence_snippet must contain identifier_raw (or canonical-equivalent). If not, set confidence_discovery low and keep fields null.
- Never overwrite identifier_raw (preserve exactly as given if present).
- Output ONLY valid JSON.

