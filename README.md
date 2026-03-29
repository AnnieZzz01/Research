# TikTok emotional cues and user engagement

Research project: how **emotional cues at different temporal positions** in short-form TikTok videos relate to **sharing and commenting** (and related engagement metrics). Analysis uses visual (DeepFace) and audio (librosa) features, machine-learning models, and supplementary mediation / ablation checks.

## Repository layout

| Path | Description |
|------|-------------|
| `analysis.ipynb` | Main notebook: data loading, EDA, models, figures, discussion |
| `extract_emotions.py` | Per-frame facial emotion extraction (DeepFace, 3 fps) → `full_visual_results.csv` |
| `extract_audio_features.py` | Per-window acoustic features → `full_audio_features.csv` |
| `data/` | Supporting CSVs (e.g. comment / emoji features) |
| `videos.csv` | TikTok metadata export (views, likes, shares, comments, …) |
| `run/videos/` | Local `.mp4` corpus (not tracked; see `.gitignore`) |

Paths in the scripts assume you run them from this directory (the repo root).

## Setup

Python 3.10+ recommended.

```bash
cd Research
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Running the analysis

1. Ensure feature CSVs and `videos.csv` are present (or run the extract scripts after placing videos under `run/videos/`).
2. Open `analysis.ipynb` in Jupyter or VS Code and run cells from the top (or use your saved outputs).

## License

This project is licensed under the [MIT License](LICENSE) (Copyright © 2026 Annie Zhang).
