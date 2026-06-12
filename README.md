# Smartphone Recommender Web Final

Web app rekomendasi smartphone berbasis Fuzzy Inference System metode Mamdani.

## Run backend

```bash
pip install -r requirements.txt
uvicorn backend.fuzzy_smartphone_mamdani:app --reload
```

## Open frontend

Buka `frontend/index.html` di browser.

## API

`POST /recommend`

```json
{
  "budget": "medium",
  "priority": ["RAM", "Camera", "Battery", "Processor"],
  "min_storage": 128
}
```
