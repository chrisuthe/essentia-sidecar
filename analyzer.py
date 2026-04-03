"""Essentia audio analysis sidecar service.

Licensed under AGPL-3.0 — see LICENSE file.

Receives raw PCM audio over HTTP, runs Essentia analysis, returns JSON
compatible with Music Assistant's AudioAnalysisData model.

When ML models are available (via ESSENTIA_MODELS_PATH env var), also
extracts instrumentalness, valence, acousticness, danceability (ML),
mood tags, and genre predictions.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import essentia.standard as es
import numpy as np
from flask import Flask, Response, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("essentia-sidecar")

app = Flask(__name__)

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_TARGET_SR = 44100
_ML_SR = 16000
_FRAME_SIZE = 2048
_HOP_SIZE = 1024
_WAVEFORM_BINS = 800

_FLAT_TO_SHARP = {
    "Ab": "G#",
    "Bb": "A#",
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
}

# ML models directory — set via ESSENTIA_MODELS_PATH env var
MODELS_PATH = Path(os.environ.get("ESSENTIA_MODELS_PATH", "/app/models"))
_ml_available = False
_ml_models: dict[str, Path] = {}
_genre_labels: list[str] = []


def _init_ml_models() -> None:
    """Discover available ML model files at startup."""
    global _ml_available  # noqa: PLW0603
    if not MODELS_PATH.is_dir():
        logger.info("No models directory at %s — ML features disabled", MODELS_PATH)
        return

    # Each key can have multiple filename variants to try
    model_files: dict[str, list[str]] = {
        # Embedding extractors
        "vggish_embeddings": ["audioset-vggish-3.pb"],
        "musicnn_embeddings": ["msd-musicnn-1.pb"],
        "effnet_embeddings": ["discogs_artist_embeddings-effnet-bs64-1.pb"],
        # Classification heads — try multiple filename patterns
        "voice_instrumental": [
            "voice_instrumental-audioset-vggish-1.pb",
            "voice_instrumental-vggish-audioset-1.pb",
        ],
        "danceability": [
            "danceability-audioset-vggish-1.pb",
            "danceability-vggish-audioset-1.pb",
        ],
        "valence_arousal": ["deam-msd-musicnn-2.pb"],
        "acoustic_electronic": ["nsynth_acoustic_electronic-discogs-effnet-1.pb"],
        "mood_happy": ["mood_happy-audioset-vggish-1.pb"],
        "mood_sad": ["mood_sad-audioset-vggish-1.pb"],
        "mood_aggressive": ["mood_aggressive-audioset-vggish-1.pb"],
        "mood_relaxed": ["mood_relaxed-audioset-vggish-1.pb"],
        "genre_discogs400": ["genre_discogs400-discogs-effnet-1.pb"],
    }

    found = []
    for key, filenames in model_files.items():
        for filename in filenames:
            path = MODELS_PATH / filename
            if path.exists():
                _ml_models[key] = path
                found.append(f"{key} ({filename})")
                break

    # Load genre label mapping
    global _genre_labels  # noqa: PLW0603
    labels_file = MODELS_PATH / "discogs-effnet-bs64-1.json"
    if labels_file.exists():
        import json

        with open(labels_file) as f:
            data = json.load(f)
        _genre_labels = data.get("classes", [])
        logger.info("Genre labels loaded: %d classes", len(_genre_labels))

    if found:
        _ml_available = True
        logger.info("ML models loaded: %s", ", ".join(found))
        # Dump node names from each model for debugging
        for key, path in _ml_models.items():
            if key in ("vggish_embeddings", "musicnn_embeddings", "effnet_embeddings"):
                continue  # skip embedding extractors
            try:
                import tensorflow as tf
                with open(str(path), "rb") as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                node_names = [n.name for n in graph_def.node]
                logger.info("Model %s nodes: %s", key, node_names[:10])
            except Exception as exc:
                logger.warning("Could not inspect %s: %s", key, exc)
    else:
        logger.info("No ML model files found in %s", MODELS_PATH)


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _pcm_to_float32(pcm: bytes, bit_depth: int, channels: int) -> np.ndarray:
    """Convert raw PCM bytes to mono float32 array."""
    if bit_depth == 16:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif bit_depth == 24:
        n = len(pcm) // 3
        raw = np.frombuffer(pcm[: n * 3], dtype=np.uint8).reshape(-1, 3)
        i32 = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        i32[i32 >= 0x800000] -= 0x1000000
        samples = i32.astype(np.float32) / 8388608.0
    elif bit_depth == 32:
        samples = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def _extract_ml_features(audio_44k: np.ndarray) -> dict[str, object]:
    """Extract ML-based features via two-stage pipeline.

    Stage 1: Extract embeddings (VGGish, MusiCNN, EffNet)
    Stage 2: Feed embeddings to classification heads

    :param audio_44k: Mono float32 audio at 44100 Hz.
    :returns: Dict of field_name -> value.
    """
    results: dict[str, object] = {}
    if not _ml_available:
        return results

    audio_16k = es.Resample(
        inputSampleRate=float(_TARGET_SR), outputSampleRate=float(_ML_SR)
    )(audio_44k)

    # --- Stage 1: Extract embeddings ---

    vggish_emb = None
    if "vggish_embeddings" in _ml_models:
        try:
            vggish_emb = es.TensorflowPredictVGGish(
                graphFilename=str(_ml_models["vggish_embeddings"]),
                output="model/vggish/embeddings",
            )(audio_16k)
            logger.info("VGGish embeddings: shape %s", vggish_emb.shape)
        except Exception as exc:
            logger.error("VGGish embeddings failed: %s", exc, exc_info=True)

    musicnn_emb = None
    if "musicnn_embeddings" in _ml_models:
        try:
            musicnn_emb = es.TensorflowPredictMusiCNN(
                graphFilename=str(_ml_models["musicnn_embeddings"]),
                output="model/dense/BiasAdd",
            )(audio_16k)
            logger.info("MusiCNN embeddings: shape %s", musicnn_emb.shape)
        except Exception as exc:
            logger.error("MusiCNN embeddings failed: %s", exc, exc_info=True)

    effnet_emb = None
    if "effnet_embeddings" in _ml_models:
        try:
            effnet_emb = es.TensorflowPredictEffnetDiscogs(
                graphFilename=str(_ml_models["effnet_embeddings"]),
                output="PartitionedCall:1",
            )(audio_16k)
            logger.info("EffNet embeddings: shape %s", effnet_emb.shape)
        except Exception as exc:
            logger.error("EffNet embeddings failed: %s", exc, exc_info=True)

    # --- Stage 2: Classification heads ---
    # Note: TensorflowPredict2D default input="model/Placeholder" output="model/Sigmoid"
    # Many classification heads use different node names — we need to specify them.

    # Node name mappings per model type
    _HEAD_NODES: dict[str, dict[str, str]] = {
        "voice_instrumental": {"input": "model/Placeholder", "output": "model/Sigmoid"},
        "danceability": {"input": "model/Placeholder", "output": "model/Sigmoid"},
        "mood_happy": {"input": "model/Placeholder", "output": "model/Softmax"},
        "mood_sad": {"input": "model/Placeholder", "output": "model/Softmax"},
        "mood_aggressive": {"input": "model/Placeholder", "output": "model/Softmax"},
        "mood_relaxed": {"input": "model/Placeholder", "output": "model/Softmax"},
        "valence_arousal": {"input": "model/Placeholder", "output": "model/Identity"},
        "acoustic_electronic": {"input": "model/Placeholder", "output": "model/Softmax"},
        "genre_discogs400": {"input": "serving_default_model_Placeholder", "output": "PartitionedCall:0"},
    }

    def _predict_head(
        model_key: str, embeddings: np.ndarray, label: str
    ) -> np.ndarray | None:
        """Run a classification head on pre-computed embeddings."""
        if model_key not in _ml_models:
            return None
        model_path = str(_ml_models[model_key])
        nodes = _HEAD_NODES.get(model_key, {})
        try:
            preds = es.TensorflowPredict2D(
                graphFilename=model_path, **nodes
            )(embeddings)
            logger.info("%s OK (nodes: %s, shape: %s)", label, nodes.get("output", "default"), preds.shape)
            return preds
        except Exception as exc:
            logger.error("%s failed: %s", label, exc)
            return None

    # VGGish-based heads
    if vggish_emb is not None:
        preds = _predict_head("voice_instrumental", vggish_emb, "Voice/instrumental")
        if preds is not None:
            results["instrumentalness"] = _clamp(float(preds.mean(axis=0)[1]))
            logger.info("Instrumentalness: %.3f", results["instrumentalness"])

        preds = _predict_head("danceability", vggish_emb, "Danceability ML")
        if preds is not None:
            results["danceability"] = _clamp(float(preds.mean(axis=0)[1]))
            logger.info("Danceability ML: %.3f", results["danceability"])

        moods: dict[str, float] = {}
        for mood_name in ("mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed"):
            preds = _predict_head(mood_name, vggish_emb, mood_name)
            if preds is not None:
                moods[mood_name] = round(float(preds.mean(axis=0)[1]), 4)
    else:
        moods = {}

    # MusiCNN-based heads
    if musicnn_emb is not None:
        preds = _predict_head("valence_arousal", musicnn_emb, "Valence/arousal")
        if preds is not None:
            valence_raw = float(preds.mean(axis=0)[1])
            results["valence"] = _clamp((valence_raw - 1.0) / 8.0)
            logger.info("Valence: %.3f (raw %.2f)", results["valence"], valence_raw)

    # EffNet-based heads
    if effnet_emb is not None:
        preds = _predict_head("acoustic_electronic", effnet_emb, "Acoustic/electronic")
        if preds is not None:
            results["acousticness"] = _clamp(float(preds.mean(axis=0)[0]))
            logger.info("Acousticness: %.3f", results["acousticness"])

        genres: dict[str, float] = {}
        if "genre_discogs400" in _ml_models:
            try:
                preds = es.TensorflowPredict2D(
                    graphFilename=str(_ml_models["genre_discogs400"]),
                    input="serving_default_model_Placeholder",
                    output="PartitionedCall:0",
                )(effnet_emb)
                mean_preds = preds.mean(axis=0)
                top_indices = np.argsort(mean_preds)[::-1][:5]
                for idx in top_indices:
                    label = _genre_labels[idx] if idx < len(_genre_labels) else f"genre_{idx}"
                    genres[label] = round(float(mean_preds[idx]), 4)
            except Exception as exc:
                logger.error("Genre failed: %s", exc, exc_info=True)
    else:
        genres = {}

    # Pack moods + genres into extra_data
    extra: dict[str, object] = {}
    if moods:
        extra["moods"] = moods
    if genres:
        extra["genre_predictions"] = genres
    if extra:
        results["extra_data"] = extra

    return results


def analyze(audio: np.ndarray, sample_rate: int) -> dict:
    """Run full Essentia analysis and return AudioAnalysisData-compatible dict."""
    audio = audio.astype(np.float32)

    if sample_rate != _TARGET_SR:
        audio = es.Resample(
            inputSampleRate=float(sample_rate), outputSampleRate=float(_TARGET_SR)
        )(audio)
        sample_rate = _TARGET_SR

    # --- Rhythm ---
    rhythm = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, _conf, _, beats_intervals = rhythm(audio)

    rhythmic_regularity = 0.0
    if len(beats_intervals) >= 2:
        mean_i = float(np.mean(beats_intervals))
        std_i = float(np.std(beats_intervals))
        rhythmic_regularity = _clamp(1.0 - std_i / (mean_i + 1e-8))

    # --- Key ---
    key, scale, _strength = es.KeyExtractor()(audio)
    key = _FLAT_TO_SHARP.get(key, key)

    # --- Loudness (EBU R128) ---
    stereo = np.column_stack([audio, audio])
    _momentary, _short_term, integrated, loudness_range = es.LoudnessEBUR128(
        sampleRate=float(sample_rate)
    )(stereo)

    # --- True peak ---
    peak = float(np.max(np.abs(audio)))
    true_peak = float(20.0 * math.log10(peak)) if peak > 0 else -96.0

    # --- Energy ---
    mean_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    energy = _clamp(mean_rms / 0.25)

    # --- Danceability (DFA-based, may be overridden by ML) ---
    dance_score, _ = es.Danceability(sampleRate=float(sample_rate))(audio)
    danceability = _clamp(float(dance_score) / 3.0)

    # --- Frame features ---
    nyquist = sample_rate / 2.0
    spec_algo = es.Spectrum()
    w = es.Windowing(type="hann")
    dissonance_algo = es.Dissonance()
    peaks_algo = es.SpectralPeaks(
        sampleRate=float(sample_rate), maxPeaks=50, minFrequency=20.0, maxFrequency=nyquist
    )
    complexity_algo = es.SpectralComplexity(sampleRate=float(sample_rate))
    rms_algo = es.RMS()

    centroid_hz_vals = []
    roughness_vals = []
    complexity_vals = []
    rms_frames = []

    for frame in es.FrameGenerator(audio, frameSize=_FRAME_SIZE, hopSize=_HOP_SIZE):
        spectrum = spec_algo(w(frame))
        rms_frames.append(float(rms_algo(frame)))

        n_bins = len(spectrum)
        if n_bins > 0:
            bin_freqs = np.linspace(0, nyquist, n_bins)
            spec_sum = float(np.sum(spectrum))
            if spec_sum > 0:
                centroid_hz = float(np.sum(bin_freqs * spectrum) / spec_sum)
            else:
                centroid_hz = 0.0
            centroid_hz_vals.append(centroid_hz)

        freqs, mags = peaks_algo(spectrum)
        if len(freqs) > 1:
            roughness_vals.append(float(dissonance_algo(freqs, mags)))
        complexity_vals.append(float(complexity_algo(spectrum)))

    brightness = _clamp(float(np.mean(centroid_hz_vals)) / nyquist) if centroid_hz_vals else 0.0
    roughness = _clamp(float(np.mean(roughness_vals))) if roughness_vals else 0.0
    harmonic_complexity = _clamp(float(np.mean(complexity_vals)) / 50.0) if complexity_vals else 0.0

    # --- Per-second arrays ---
    fps = sample_rate / _HOP_SIZE
    n_sec = max(1, int(len(rms_frames) / fps))

    rms_per_sec = []
    centroid_per_sec = []
    for s in range(n_sec):
        start = int(s * fps)
        end = int((s + 1) * fps)
        rms_per_sec.append(float(np.mean(rms_frames[start:end])) if start < len(rms_frames) else 0.0)
        centroid_per_sec.append(
            float(np.mean(centroid_hz_vals[start:end])) if start < len(centroid_hz_vals) else 0.0
        )

    # --- Waveform ---
    abs_audio = np.abs(audio)
    n = len(abs_audio)
    if n >= _WAVEFORM_BINS:
        edges = np.linspace(0, n, _WAVEFORM_BINS + 1, dtype=int)
        waveform = [float(abs_audio[edges[i] : edges[i + 1]].max()) for i in range(_WAVEFORM_BINS)]
    else:
        indices = np.linspace(0, n - 1, _WAVEFORM_BINS, dtype=int)
        waveform = [float(abs_audio[i]) for i in indices]
    wf_max = max(waveform) if waveform else 0.0
    if wf_max > 0:
        waveform = [v / wf_max for v in waveform]

    result = {
        "bpm": float(bpm),
        "beats": [float(b) for b in beats],
        "key": str(key),
        "mode": str(scale),
        "loudness_integrated": float(integrated),
        "loudness_range": float(loudness_range),
        "true_peak": true_peak,
        "energy": energy,
        "danceability": danceability,
        "brightness": brightness,
        "roughness": roughness,
        "harmonic_complexity": harmonic_complexity,
        "rhythmic_regularity": rhythmic_regularity,
        "rms_energy_per_second": rms_per_sec,
        "spectral_centroid_per_second": centroid_per_sec,
        "wave_form": waveform,
        "duration": float(len(audio) / sample_rate),
    }

    # --- ML features (if models available) ---
    ml_features = _extract_ml_features(audio)
    if ml_features:
        ml_count = len([k for k in ml_features if k != "extra_data"])
        logger.info("ML features extracted: %d fields", ml_count)
        result.update(ml_features)

    return result


@app.route("/health")
def health() -> Response:
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "ml_available": _ml_available,
        "ml_models": list(_ml_models.keys()),
    })


@app.route("/analyze", methods=["POST"])
def analyze_endpoint() -> Response:
    """Analyze raw PCM audio and return AudioAnalysisData-compatible JSON."""
    t_start = time.monotonic()
    sample_rate = int(request.args.get("sample_rate", 44100))
    bit_depth = int(request.args.get("bit_depth", 16))
    channels = int(request.args.get("channels", 2))

    pcm_data = request.get_data()
    if not pcm_data:
        return jsonify({"error": "No audio data"}), 400

    pcm_mb = len(pcm_data) / (1024 * 1024)
    audio = _pcm_to_float32(pcm_data, bit_depth, channels)
    duration_sec = len(audio) / sample_rate

    if len(audio) < sample_rate * 2:
        logger.warning("Rejected: audio too short (%.1fs)", duration_sec)
        return jsonify({"error": "Audio too short (< 2 seconds)"}), 400

    logger.info(
        "Analyzing: %.1fs audio (%.1f MB, %dHz %dbit %dch) [ML=%s]",
        duration_sec, pcm_mb, sample_rate, bit_depth, channels,
        "yes" if _ml_available else "no",
    )

    result = analyze(audio, sample_rate)
    elapsed = time.monotonic() - t_start

    ml_fields = []
    for f in ("instrumentalness", "valence", "acousticness"):
        if f in result:
            ml_fields.append(f"{f}={result[f]:.2f}")

    logger.info(
        "Done: %.1fs audio -> BPM=%.1f key=%s %s (%.1fs, %.1fx RT)%s",
        duration_sec,
        result.get("bpm", 0),
        result.get("key", "?"),
        result.get("mode", "?"),
        elapsed,
        duration_sec / elapsed if elapsed > 0 else 0,
        " | " + " ".join(ml_fields) if ml_fields else "",
    )

    return jsonify(result)


# Initialize ML models at module load time
_init_ml_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5030)
