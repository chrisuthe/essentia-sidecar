# Essentia Analysis Sidecar

Audio analysis service for [Music Assistant](https://music-assistant.io/) using [Essentia](https://essentia.upf.edu/).

Licensed under AGPL-3.0 (due to Essentia's license). Distributed separately from Music Assistant (Apache 2.0).

## Two variants

| Image | Tag | Size | Features |
|---|---|---|---|
| **Lite** | `ghcr.io/chrisuthe/essentia-sidecar:lite` | ~500MB | BPM, key, loudness (EBU R128), spectral features |
| **Full** | `ghcr.io/chrisuthe/essentia-sidecar:gpu` | ~2.5GB | Everything in Lite + ML models for instrumentalness, valence, acousticness, danceability, mood tags, genre predictions |

`latest` is an alias for `lite`.

The Lite image runs on CPU only. The Full image includes CUDA 11.8 + cuDNN 8 runtime and will use an NVIDIA GPU if available (pass `--gpus all` or use the deploy config below). Falls back to CPU gracefully if no GPU is present.

## Quick start

**Lite (basic features):**
```yaml
services:
  essentia-sidecar:
    image: ghcr.io/chrisuthe/essentia-sidecar:lite
    ports:
      - "5030:5030"
    restart: unless-stopped
```

**Full (with ML models + GPU acceleration):**
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

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. Without it, the image still works but uses CPU.

## ML models included (Full variant)

| Model | Field | What it detects |
|---|---|---|
| voice_instrumental | `instrumentalness` | Vocal vs instrumental (0-1) |
| DEAM valence/arousal | `valence` | Happy/sad mood dimension (0-1) |
| acoustic_electronic | `acousticness` | Acoustic vs electronic (0-1) |
| danceability | `danceability` | ML-based override of DFA score (0-1) |
| mood_happy/sad/aggressive/relaxed | `extra_data.moods` | Per-mood probability scores |
| genre_discogs400 | `extra_data.genre_predictions` | Top 5 genre predictions from 400-class taxonomy |

## API

- `GET /health` — returns `{"status": "ok", "ml_available": true/false, "ml_models": [...]}`
- `POST /analyze?sample_rate=44100&bit_depth=16&channels=1` — raw PCM body, returns AudioAnalysisData JSON

## Configure in Music Assistant

1. Open the Sonic Analysis debug page (`http://<ma-ip>:8095/sonic_analysis/debug`)
2. Under "Audio Analysis Providers", set the sidecar URL to `http://<host-ip>:5030`
3. Click Save
4. Trigger a backfill to analyze your library
