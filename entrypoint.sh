#!/bin/bash
# Auto-detect optimal worker count based on environment

# Default: auto-detect from CPU cores
# With GPU: fewer workers (GPU is the bottleneck, not CPU)
# Without GPU: more workers (CPU-bound, parallelize freely)
# Disable GPU before Python/TF starts if requested
USE_GPU="${ESSENTIA_USE_GPU:-true}"
if [ "$USE_GPU" = "false" ] || [ "$USE_GPU" = "0" ] || [ "$USE_GPU" = "no" ]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "GPU disabled via ESSENTIA_USE_GPU — set CUDA_VISIBLE_DEVICES=''"
fi

if [ -z "$ESSENTIA_WORKERS" ]; then
    CORES=$(nproc 2>/dev/null || echo 2)
    USE_GPU="${ESSENTIA_USE_GPU:-true}"

    if [ "$USE_GPU" = "true" ] || [ "$USE_GPU" = "1" ] || [ "$USE_GPU" = "yes" ]; then
        # GPU mode: 1 worker — gunicorn uses separate processes so threading.Lock
        # can't protect GPU memory across workers. Single worker avoids OOM.
        ESSENTIA_WORKERS=1
    else
        # CPU-only: use (cores / 2), min 2, max 8
        # Each worker is CPU-heavy so don't saturate all cores
        ESSENTIA_WORKERS=$(( CORES / 2 ))
        [ "$ESSENTIA_WORKERS" -lt 2 ] && ESSENTIA_WORKERS=2
        [ "$ESSENTIA_WORKERS" -gt 8 ] && ESSENTIA_WORKERS=8
    fi
fi

export ESSENTIA_WORKERS

echo "Starting essentia-sidecar with $ESSENTIA_WORKERS workers (GPU=${ESSENTIA_USE_GPU:-true})"

# Use python3 -m to avoid shebang path issues between build and runtime stages
exec python3 -m gunicorn \
    -b 0.0.0.0:5030 \
    -w "$ESSENTIA_WORKERS" \
    --timeout 300 \
    --worker-class gthread \
    --threads 2 \
    analyzer:app
