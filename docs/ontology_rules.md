# Ontology Rules

## Purpose
The ontology is a small, hand-audited coarse mapping used only to define novelty and overlap. It is not a claim that the datasets share a perfect semantic label space.

## Rules
- Map raw attack labels into a small set of coarse groups.
- Keep the mapping conservative.
- Ambiguous raw labels go to `other`.
- Track `mapping_source`, `mapping_confidence`, and `notes`.
- Report the share of target attack mass mapped to `other`.
- Do not revise the ontology after looking at downstream transfer results.

## Mapping Expectations
- IoT-specific malware family names may remain distinct if they cannot be defensibly merged.
- Generic network attack labels should only be merged when semantics are clearly aligned.
- `UNSW-NB15` is a stress-test domain, not a semantic peer.

## Failure Threshold
If the ontology pushes too much attack mass into `other` or requires ad hoc remapping per experiment, the novelty claim must be narrowed or dropped.
