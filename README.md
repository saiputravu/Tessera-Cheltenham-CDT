# Cheltenham Road-Corridor Search (TESSERA + AURN)

## Quick start

```bash
uv sync
uv run --with jupyterlab jupyter lab cheltenham_similarity.ipynb
```

## Files

- `cheltenham_similarity.ipynb`: main walkthrough notebook.
- `aurn_no2_embeddings.py`: minimal AURN and TESSERA patch helpers.
- `data/aurn_no2_2023.csv`: cached 2023 UK AURN NO2 sites (`148` rows).
- `data/cheltenham_grid_2023.npz`: cached 10 m Cheltenham similarity grid (`353,064` pixels).
- `data/patches_640m_2023.npz`: cached 640 m site patches (gitignored due to 185 MB size; downloaded on cache miss).

## API keys

No API keys are required. AURN data, TESSERA Zarr stores, and Esri basemap tiles are public.
