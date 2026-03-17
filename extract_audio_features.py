"""
Audio Emotion Feature Extraction Pipeline
==========================================
Extracts per-window acoustic features from all TikTok .mp4 videos.

Strategy
--------
1. ffmpeg decodes each .mp4 audio track to a 22 050 Hz mono WAV in /tmp
2. librosa reads the WAV (fast, no codec fallback needed)
3. Features are computed for whole / early / late windows
4. Results are appended to full_audio_features.csv every SAVE_EVERY videos
5. Already-processed videos are skipped (resume support)

Features per temporal window
-----------------------------
  tempo         : BPM — rhythmic arousal (whole video only)
  rms_energy    : loudness / emotional intensity
  spec_centroid : spectral brightness — valence proxy (bright = positive)
  zcr           : zero-crossing rate — roughness / speech presence
  chroma_std    : harmonic complexity (whole video only)

Derived features
----------------
  audio_energy_delta    = late_rms − early_rms   (builds up vs fades)
  audio_brightness_delta= late_centroid − early_centroid

Usage
-----
    nohup python extract_audio_features.py > logs/audio_extraction.log 2>&1 &
    echo $!
"""

import csv
import os
import subprocess
import tempfile
import time
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
VIDEO_DIR     = "run/videos"
OUTPUT_CSV    = "full_audio_features.csv"
FFMPEG        = "/opt/homebrew/bin/ffmpeg"
SAVE_EVERY    = 100
SESSION_LIMIT = 0          # 0 = process all remaining

SR         = 22050
HOP_LENGTH = 512
EARLY_FRAC = 0.25
LATE_FRAC  = 0.25

FIELDNAMES = [
    "video_id",
    # whole-video
    "audio_tempo",
    "audio_rms_energy",
    "audio_spec_centroid",
    "audio_zcr",
    "audio_chroma_std",
    # early window
    "early_audio_rms_energy",
    "early_audio_spec_centroid",
    "early_audio_zcr",
    # late window
    "late_audio_rms_energy",
    "late_audio_spec_centroid",
    "late_audio_zcr",
    # deltas (late − early)
    "audio_energy_delta",
    "audio_brightness_delta",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(x, default=0.0):
    try:
        v = float(np.atleast_1d(x).flat[0])
        return v if np.isfinite(v) else default
    except Exception:
        return default


def safe_mean(arr, default=0.0):
    try:
        v = float(np.nanmean(arr))
        return v if np.isfinite(v) else default
    except Exception:
        return default


def mp4_to_wav(mp4_path, wav_path):
    """Use ffmpeg to extract mono 22 050 Hz WAV from an mp4.  Returns True on success."""
    cmd = [
        FFMPEG, "-y", "-i", mp4_path,
        "-ac", "1",              # mono
        "-ar", str(SR),          # target sample rate
        "-vn",                   # drop video
        "-loglevel", "error",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0


def window_features(y_win, sr):
    """Compute RMS energy, spectral centroid, ZCR for a numpy audio window."""
    import librosa
    if len(y_win) < HOP_LENGTH * 2:
        return {"rms_energy": 0.0, "spec_centroid": 0.0, "zcr": 0.0}
    rms  = safe_mean(librosa.feature.rms(y=y_win, hop_length=HOP_LENGTH)[0])
    cent = safe_mean(librosa.feature.spectral_centroid(
                         y=y_win, sr=sr, hop_length=HOP_LENGTH)[0])
    zcr  = safe_mean(librosa.feature.zero_crossing_rate(
                         y_win, hop_length=HOP_LENGTH)[0])
    return {
        "rms_energy":    round(rms,  6),
        "spec_centroid": round(cent, 2),
        "zcr":           round(zcr,  6),
    }


def analyze_audio(wav_path):
    """Load WAV and compute all features.  Returns feature dict or None."""
    import librosa, soundfile as sf

    try:
        y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        # soundfile returns (samples, channels) for stereo; force mono
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        try:
            y, sr = librosa.load(wav_path, sr=SR, mono=True)
        except Exception:
            return None

    if len(y) < SR * 0.3:          # < 300 ms — skip
        return None

    n          = len(y)
    early_end  = int(n * EARLY_FRAC)
    late_start = int(n * (1 - LATE_FRAC))

    # Whole-video features
    w = window_features(y, sr)

    # Tempo
    try:
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
        tempo = safe_float(tempo_arr)
    except Exception:
        tempo = 0.0

    # Chroma std (harmonic complexity)
    try:
        chroma    = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
        chroma_std = round(safe_float(np.nanstd(chroma)), 6)
    except Exception:
        chroma_std = 0.0

    early = window_features(y[:early_end], sr)
    late  = window_features(y[late_start:], sr)

    return {
        "audio_tempo":           round(tempo, 2),
        "audio_rms_energy":      w["rms_energy"],
        "audio_spec_centroid":   w["spec_centroid"],
        "audio_zcr":             w["zcr"],
        "audio_chroma_std":      chroma_std,
        "early_audio_rms_energy":    early["rms_energy"],
        "early_audio_spec_centroid": early["spec_centroid"],
        "early_audio_zcr":           early["zcr"],
        "late_audio_rms_energy":     late["rms_energy"],
        "late_audio_spec_centroid":  late["spec_centroid"],
        "late_audio_zcr":            late["zcr"],
        "audio_energy_delta":
            round(late["rms_energy"] - early["rms_energy"], 6),
        "audio_brightness_delta":
            round(late["spec_centroid"] - early["spec_centroid"], 2),
    }


def process_video(video_path):
    """Full pipeline: mp4 → tmp wav → features → clean up."""
    video_id = os.path.basename(video_path).replace(".mp4", "")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    try:
        if not mp4_to_wav(video_path, wav_path):
            return None
        feats = analyze_audio(wav_path)
        if feats is None:
            return None
        feats["video_id"] = video_id
        return feats
    except Exception:
        return None
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def load_done_ids():
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add(row["video_id"])
    return done


def flush(batch, write_header):
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(batch)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    # Verify dependencies
    try:
        import librosa, soundfile
        print(f"librosa {librosa.__version__}  |  soundfile {soundfile.__version__}")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        raise SystemExit(1)

    if not os.path.exists(FFMPEG):
        print(f"ffmpeg not found at {FFMPEG}")
        raise SystemExit(1)

    video_files = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    done_ids    = load_done_ids()
    todo        = [f for f in video_files
                   if f.replace(".mp4", "") not in done_ids]

    if SESSION_LIMIT > 0:
        todo = todo[:SESSION_LIMIT]

    print(f"Total on disk   : {len(video_files)}")
    print(f"Already done    : {len(done_ids)}")
    print(f"To process      : {len(todo)}")
    print(f"Output          : {OUTPUT_CSV}")
    print()

    if not todo:
        print("Nothing left to process.")
        raise SystemExit(0)

    write_header = not os.path.exists(OUTPUT_CSV)
    batch, times = [], []
    failed = 0
    t0 = time.time()

    for i, fname in enumerate(tqdm(todo, desc="Audio features"), 1):
        t_vid = time.time()
        r = process_video(os.path.join(VIDEO_DIR, fname))
        elapsed = time.time() - t_vid
        times.append(elapsed)

        if r:
            batch.append(r)
        else:
            failed += 1

        if i % SAVE_EVERY == 0 or i == len(todo):
            flush(batch, write_header)
            write_header = False
            batch = []
            avg  = sum(times) / len(times)
            left = avg * (len(todo) - i)
            tqdm.write(
                f"  [{i}/{len(todo)}]  avg={avg:.1f}s/vid  "
                f"failed={failed}  est_remaining={left/3600:.1f}h"
            )

    total = time.time() - t0
    succeeded = len(todo) - failed
    print(f"\nFinished.  {succeeded}/{len(todo)} videos in {total/3600:.2f}h.")
    print(f"Failed / silent: {failed}")
    print(f"Output: {OUTPUT_CSV}")
