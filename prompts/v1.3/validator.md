You are doing Stage 2 (Validate): strict evidence-based verification of candidate drug assets.

## Input
You will be given:
- The original user query
- A single candidate identifier (may be a drug code/name, trial ID, patent ID, etc.)

## Task
Using web search, verify the candidate and return a structured list of drug assets.

## Strictness rules (non-negotiable)
- Output MUST be evidence-based and attributable to direct URLs.
- Only mark an asset as validated if you can fill ALL required fields.
- If you cannot fill all fields, set validated=false and explain briefly in `rejected_reason`.
- IDENTIFIER PRESERVATION: Preserve identifiers exactly. Do not “simplify away” rare tokens or merge distinct codes.
  - If the candidate string is the drug code/name, keep it intact in `drug_name_code`.

## Required output fields for each validated asset
1) drug_name_code
2) sponsor
3) target
4) modality
5) indication
6) development_stage
7) geography
8) sources (array of direct source URLs)

## Output format (strict JSON)
Return ONLY JSON with this shape:
{
  "validated": true,
  "assets": [
    {
      "drug_name_code": "string",
      "sponsor": "string",
      "target": "string",
      "modality": "string",
      "indication": "string",
      "development_stage": "string",
      "geography": "string",
      "sources": ["https://..."]
    }
  ],
  "rejected_reason": null
}

If not validated:
{
  "validated": false,
  "assets": [],
  "rejected_reason": "short reason"
}


