# Essentia Analysis Sidecar

Audio analysis service for [Music Assistant](https://music-assistant.io/) using [Essentia](https://essentia.upf.edu/).

Licensed under AGPL-3.0 (due to Essentia's license). Distributed separately from Music Assistant (Apache 2.0).

## Two variants

| Image | Tag | Size | Features |
|---|---|---|---|
| **Lite** | `ghcr.io/chrisuthe/essentia-sidecar:lite` | ~500MB | BPM, key, loudness (EBU R128), spectral features |
| **GPU** | `ghcr.io/chrisuthe/essentia-sidecar:gpu` | ~2.5GB | Everything in Lite + instrumentalness, valence, acousticness, danceability (ML), mood tags, genre predictions |

`latest` is an alias for `lite`.

## Quick start

**Lite (CPU):**
```yaml
services:
  essentia-sidecar:
    image: ghcr.io/chrisuthe/essentia-sidecar:lite
    ports:
      - "5030:5030"
    restart: unless-stopped
```

**GPU (with NVIDIA):**
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

## API

- `GET /health` — returns `{"status": "ok", "ml_available": true/false, "ml_models": [...]}`
- `POST /analyze?sample_rate=44100&bit_depth=16&channels=1` — raw PCM body, returns AudioAnalysisData JSON

## Configure in Music Assistant

1. Open the Sonic Analysis debug page (`http://<ma-ip>:8095/sonic_analysis/debug`)
2. Under "Audio Analysis Providers", set the sidecar URL to `http://<host-ip>:5030`
3. Click Save
