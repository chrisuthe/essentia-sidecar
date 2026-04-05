# Essentia Analysis Sidecar

Audio analysis service for [Music Assistant](https://music-assistant.io/) using [Essentia](https://essentia.upf.edu/).

Licensed under AGPL-3.0 (due to Essentia's license). Distributed separately from Music Assistant (Apache 2.0).

## Two variants

| Image | Tag | Size | Features |
|---|---|---|---|
| **Lite** | `ghcr.io/chrisuthe/essentia-sidecar:lite` | ~500MB | BPM, key, loudness (EBU R128), spectral features |
| **Full** | `ghcr.io/chrisuthe/essentia-sidecar:gpu` | ~2.5GB | Everything in Lite + ML models + optional GPU acceleration |

`latest` is an alias for `lite`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ESSENTIA_USE_GPU` | `true` | Set to `false` to force CPU-only mode (even if GPU is available). Useful for running the Full image on CPU-only machines to get ML features without GPU. |
| `ESSENTIA_WORKERS` | auto | Number of parallel analysis workers. Auto-detected: 2 with GPU (GPU is the bottleneck), CPU cores / 2 without GPU (max 8). Override for your hardware. |
| `ESSENTIA_MODELS_PATH` | `/app/models` | Path to ML model .pb files. |
| `ESSENTIA_LOG_LEVEL` | `INFO` | Log verbosity: DEBUG, INFO, WARNING. |

## Quick start

**Lite (basic features, CPU only):**
```yaml
services:
  essentia-sidecar:
    image: ghcr.io/chrisuthe/essentia-sidecar:lite
    ports:
      - "5030:5030"
    restart: unless-stopped
```

**Full with GPU:**
```yaml
services:
  essentia-sidecar:
    image: ghcr.io/chrisuthe/essentia-sidecar:gpu
    ports:
      - "5030:5030"
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**Full without GPU (ML features on CPU):**
```yaml
services:
  essentia-sidecar:
    image: ghcr.io/chrisuthe/essentia-sidecar:gpu
    ports:
      - "5030:5030"
    restart: unless-stopped
    environment:
      ESSENTIA_USE_GPU: "false"
      ESSENTIA_WORKERS: "4"
```

**High-performance CPU server (many cores, no GPU):**
```yaml
services:
  essentia-sidecar:
    image: ghcr.io/chrisuthe/essentia-sidecar:gpu
    ports:
      - "5030:5030"
    restart: unless-stopped
    environment:
      ESSENTIA_USE_GPU: "false"
      ESSENTIA_WORKERS: "8"
    deploy:
      resources:
        limits:
          cpus: "16"
          memory: 16G
```

## How parallelism works

- **CPU-only mode**: Multiple workers run independent analyses in parallel. Each worker handles one track at a time. More workers = more tracks analyzed simultaneously. Rule of thumb: cores / 2 (each analysis is CPU-intensive).
- **GPU mode**: Basic analysis (BPM, key, spectral) runs on CPU per-worker. ML inference (embeddings + classification) is serialized through a GPU lock so multiple workers can do CPU work in parallel while taking turns on the GPU.
- The `--worker-class gthread --threads 2` config gives each worker 2 threads for I/O handling.

## ML models included (Full variant)

| Model | Field | What it detects |
|---|---|---|
| voice_instrumental | `instrumentalness` | Vocal vs instrumental (0-1) |
| DEAM valence/arousal | `valence`, `arousal` | Happy/sad dimension + calm/energetic dimension (0-1) |
| acoustic_electronic | `acousticness` | Acoustic vs electronic (0-1) |
| danceability | `danceability` | ML-based override of DFA score (0-1) |
| mood_happy/sad/aggressive/relaxed | `extra_data.moods` | Per-mood probability scores |
| genre_discogs400 | `extra_data.genre_predictions` | Top 5 genre predictions from 400-class taxonomy |

## API

- `GET /health` — returns status, ML availability, GPU state, worker count
- `POST /analyze?sample_rate=44100&bit_depth=16&channels=1` — raw PCM body, returns AudioAnalysisData JSON

## Configure in Music Assistant

1. Open the Sonic Analysis debug page (`http://<ma-ip>:8095/sonic_analysis/debug`)
2. Under "Audio Analysis Providers", set the sidecar URL to `http://<host-ip>:5030`
3. Click Save
4. Trigger a backfill to analyze your library
