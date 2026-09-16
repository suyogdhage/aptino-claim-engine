# Architecture Note

The claim engine makes policy decisions from the supplied Universal Sompo policy
PDF and the submitted synthetic claim facts. It treats the policy as the only
authority for coverage conclusions and returns `NEEDS_REVIEW` whenever a
material condition cannot be established.

## State flow and agent boundaries

`ClaimCase -> Case Analysis -> Policy Evidence -> Coverage and Exclusion -> Decision -> Validation`

The Case Analysis Agent converts the request into a structured investigation
plan (for example, waiting periods, exclusions, hospital definition, and
limits). The Policy Evidence Agent retrieves policy chunks separately for each
dimension. It combines dense Chroma results and sparse BM25 results with
reciprocal-rank fusion, then reranks the fused candidates with a cross-encoder.
Each evidence item retains its page, section, clause, and chunk identifier.

The Coverage and Exclusion Agent applies the retrieved clauses to the facts and
returns structured findings, blockers, and limits. The Decision Agent combines
only that structured state into a decision. The Validation Agent checks every
material finding and limit against the cited retrieved chunk. Validation failure
retries the decision once; a persistent failure is converted to `NEEDS_REVIEW`,
not presented as an unsupported adjudication. The visible trace contains agent
actions, counts, timings, and validation status, never hidden reasoning.

## Reliability and trade-offs

The index is built from semantic policy sections rather than arbitrary fixed
windows. A manifest records the policy SHA-256 and chunk count; startup checks
that Chroma and BM25 both match it. Reindexing is staged before replacement so
a failed build does not silently leave a partial index serving requests.

Groq is used when configured, while a deterministic mock path enables local
testing and CI without provider cost. The mock mode is a release-control tool,
not a substitute for review of live-model behavior. The evaluation suite covers
the 12 supplied cases plus five candidate cases and requires complete citation
coverage. The main trade-off is first-start latency: embedding/reranker model
downloads and initial index construction require network access and persistent
storage. The readiness endpoint communicates that state explicitly.
