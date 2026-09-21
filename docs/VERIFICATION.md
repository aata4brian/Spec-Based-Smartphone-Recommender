# Verification — 21 September 2026

Base revision: `a56e9906f635d271f9bcc2ed73579b8013889c69`. Portfolio changes were developed on a separate branch.

| Check | Result |
| --- | --- |
| Python 3.12 dependency installation | Passed; top-level pins and constraints captured |
| Behavioral tests | 14 passed, including real bundled-data requests |
| Warnings treated as errors | 14 passed with `python -m pytest -q -W error` |
| Ruff correctness checks | Passed |
| Python compilation | Passed |
| Real Uvicorn process | Started and served HTTP |
| `/health`, `/ui`, `/openapi.json` | HTTP 200 |
| Real HTTP recommendation | OK; 830 cleaned rows, 216 candidates, 20 returned |
| Hosted deployment | Not verified; existing Render configuration only |
| Frontend browser interaction/mobile rendering | Pending; serving HTML is not a rendered browser test |
| Ranking relevance/accuracy | Not measured |

Request: medium budget, Camera + Battery priorities, minimum storage 256 GB. Full recorded response: [example-response.json](example-response.json). The top result in this dataset snapshot has a 76.79 heuristic score; that is not an accuracy metric.

Two initial dependency deprecations were resolved by using the current Starlette-supported HTTPX2 test client dependency and a compatible AnyIO constraint. The final warning-as-error run passed.

CI is added to run correctness lint, tests and Python compilation on push/PR without secrets. Its actual remote result is recorded in the portfolio audit after publication; a workflow file alone is not a green CI claim.

No existing `.txt` snapshots, tracked historical bytecode, deployment config or dataset files were deleted. No production deployment was requested.
