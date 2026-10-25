# Final acceptance

Checked 25 October 2026 after the explanation-truncation fix. Runtime CSV files were not rewritten.

## Defects fixed

- `clean_plain()` kept only the text before the first `". "`, so the job-connection sentence never reached the screen. It now keeps up to two sentences and still strips rating codes and jargon.
- A Groq reply that omitted citation ids used to fail the whole batch and force fallback for every domain. Invented citation ids are still rejected. A missing id no longer discards a usable explanation. Each domain is labelled `groq` or `fallback`.
- Fallback text no longer stops at "domain name means definition." It names the stored occupational requirement and says when the link is only provisional.

## Score integrity

SHA-256 of the four runtime CSVs was identical before and after this change.

Data Scientists (`15-2051.00`), Artistic Vocational Interest (P047): the mapping file stores preliminary score **78.87**, rank 3 in its family, from Artistic interest OI=2.61 (percentile 78.9), link type direct. That is what the stored evidence supports. The score was not changed.

Estimated domain relevance remains the existing heuristic. It is not a calibrated probability of job fit or a validated hiring result. The crosswalk is still pending qualified review.

## Tests

`python3 -m pytest tests/ -q`

**42 passed, 0 failed.** Automated tests use mocked Groq responses.

One real Groq request succeeded for Quantitative Reasoning on Data Scientists. The model returned two sentences connecting statistical data analysis to mathematical reasoning and said the link is direct. The API key was not printed.

## Release

`/home/omen/Desktop/psychometric-job-mapper.zip`

The package excludes `.env` and `.git`.
