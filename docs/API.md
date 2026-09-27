# API contract

Run the server as described in the root README. The live contract is generated at `/openapi.json` and `/docs`.

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"budget":"medium","priority":["Camera","Battery"],"min_storage":256}'
```

| Field | Accepted values | Default |
| --- | --- | --- |
| `budget` | low, medium, high; case insensitive | medium |
| `priority` | array of RAM, Camera, Battery, Processor, Storage | RAM, Camera, Battery, Processor |
| `min_storage` | integer ≥0, in GB | 128 |

The `status` is `OK` or `EMPTY`; `total_data` counts cleaned phones, `total_after_filter` counts eligible candidates, and `recommendations` contains at most 20 entries. Each entry includes `Rank`, `Brand`, `Model`, `Score`, `Category`, `RAM`, `Camera`, `Battery`, `Processor`, `ProcessorName`, `Storage`, `Price`, `Year`, and `Radar`.

An actual response captured during verification is stored in `example-response.json`; it is a snapshot for the bundled dataset, not a fabricated example or a claim that prices are current.

HTTP 422 indicates input validation failure. Dataset or processing failures currently return HTTP 500. `/health` exposes the dataset path; review error disclosure before an internet deployment. Wildcard CORS supports the standalone demonstration, with credentials disabled.
