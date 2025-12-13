You are doing Stage 1 (Recall): reckless mention harvesting from web snippets/text.

## Goal
Extract ANY plausible identifiers and drug asset mentions. DO NOT filter for relevance.

## What to extract
- Drug names and code names (e.g., ABC123, MK-3475, PF-06826647)
- Sponsor/company names
- Targets/pathways (as mentioned)
- Modalities (as mentioned)
- Indications (as mentioned)
- Trial IDs (NCT########, EudraCT ####-#######-##, CTRI/ACTRN/etc.)
- Patent/publication IDs (WO##########, US##########, etc.)

## Rules
- No relevance filtering. If it's plausible, include it.
- Preserve identifiers exactly as written (case, hyphens, rare tokens).
- If uncertain, still include, but mark lower confidence.

## Output format (strict JSON)
Return ONLY JSON:
{
  "mentions": [
    {
      "type": "drug_name|drug_code|sponsor|target|modality|indication|trial_id|patent_id|other",
      "raw": "exact string",
      "context": "short quote or fragment from the input that contains the raw mention",
      "confidence": 0.0
    }
  ]
}


