"""Essentia audio analysis sidecar service.

Licensed under AGPL-3.0 — see LICENSE file.

Receives raw PCM audio over HTTP, runs Essentia analysis, returns JSON
compatible with Music Assistant's AudioAnalysisData model.
"""

from __future__ import annotations

import math

import essentia.standard as es
import numpy as np
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_TARGET_SR = 44100
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

    # --- Danceability ---
    dance_score, _ = es.Danceability(sampleRate=float(sample_rate))(audio)
    danceability = _clamp(float(dance_score) / 3.0)

    # --- Frame features ---
    nyquist = sample_rate / 2.0
    spec_algo = es.Spectrum()
    w = es.Windowing(type="hann")
    centroid_algo = es.Centroid(range=nyquist)
    dissonance_algo = es.Dissonance()
    peaks_algo = es.SpectralPeaks(
        sampleRate=float(sample_rate), maxPeaks=50, minFrequency=20.0, maxFrequency=nyquist
    )
    complexity_algo = es.SpectralComplexity(sampleRate=float(sample_rate))
    rms_algo = es.RMS()

    centroids = []
    roughness_vals = []
    complexity_vals = []
    rms_frames = []
    centroid_frames = []

    for frame in es.FrameGenerator(audio, frameSize=_FRAME_SIZE, hopSize=_HOP_SIZE):
        spectrum = spec_algo(w(frame))
        c = float(centroid_algo(spectrum))
        centroids.append(c)
        centroid_frames.append(c * nyquist)
        rms_frames.append(float(rms_algo(frame)))

        freqs, mags = peaks_algo(spectrum)
        if len(freqs) > 1:
            roughness_vals.append(float(dissonance_algo(freqs, mags)))
        complexity_vals.append(float(complexity_algo(spectrum)))

    brightness = _clamp(float(np.mean(centroids))) if centroids else 0.0
    roughness = _clamp(float(np.mean(roughness_vals))) if roughness_vals else 0.0
    harmonic_complexity = _clamp(float(np.mean(complexity_vals)) / 20.0) if complexity_vals else 0.0

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
            float(np.mean(centroid_frames[start:end])) if start < len(centroid_frames) else 0.0
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

    return {
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


@app.route("/health")
def health() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze_endpoint() -> Response:
    """Analyze raw PCM audio and return AudioAnalysisData-compatible JSON."""
    sample_rate = int(request.args.get("sample_rate", 44100))
    bit_depth = int(request.args.get("bit_depth", 16))
    channels = int(request.args.get("channels", 2))

    pcm_data = request.get_data()
    if not pcm_data:
        return jsonify({"error": "No audio data"}), 400

    audio = _pcm_to_float32(pcm_data, bit_depth, channels)

    if len(audio) < sample_rate * 2:
        return jsonify({"error": "Audio too short (< 2 seconds)"}), 400

    result = analyze(audio, sample_rate)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5030)
