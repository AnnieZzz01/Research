"""
Full emotion extraction pipeline for all downloaded TikTok videos.

Optimisations vs. the notebook test version:
  - detector_backend='opencv'  (3-5× faster than default 'retinaface')
  - Frames resized to 224×224 before DeepFace (reduces preprocessing time)
  - Resume / checkpoint support: already-processed video_ids are skipped
  - Progress is flushed to CSV after every batch (SAVE_EVERY videos)

Usage:
    python extract_emotions.py

Or to run in the background (recommended for full dataset):
    nohup python extract_emotions.py \
        > logs/extraction.log 2>&1 &
    echo $!  # note the PID so you can kill it if needed
"""

import csv
import os
import time
from collections import Counter

import cv2
from deepface import DeepFace
from tqdm import tqdm

# ===== Configuration =====
FPS                  = 3         # Target sampling rate (frames per second)
EARLY_FRAC           = 0.25      # Early window: first 25 % of frames
LATE_FRAC            = 0.25      # Late window:  last  25 % of frames
CONFIDENCE_THRESHOLD = 10.0      # Minimum DeepFace confidence (%) to accept prediction
DETECTOR_BACKEND     = "opencv"  # Fastest backend; alternatives: "retinaface", "mtcnn"
FRAME_SIZE           = 224       # Resize shorter side to this before analysis

VIDEO_DIR   = "run/videos"
OUTPUT_CSV  = "full_visual_results.csv"
SAVE_EVERY  = 50                  # Flush results to CSV every N videos

# ── Batch / session control ──────────────────────────────────────────────────
# SESSION_LIMIT : max videos to process in this run (0 = no limit = full run)
#   Set to ~900 for a 12-hour session (44s/video × 900 ≈ 11h, leaving buffer)
#   Set to 0 to run until all videos are done
#   Already-processed videos are always skipped regardless of this limit.
SESSION_LIMIT = 0       # ← change to 900 for a 12-hour run

# TEST_MODE: quick functional test on a small sample
TEST_MODE  = False
TEST_LIMIT = 500

AROUSAL_MAP = {
    "surprise": 0.9, "fear": 0.9, "angry": 0.9,
    "happy":    0.7,
    "neutral":  0.3, "sad":  0.3, "disgust": 0.3,
}
EMOTION_TO_COL = {
    "angry":    "angry_intensity",
    "disgust":  "disgust_intensity",
    "fear":     "fear_intensity",
    "happy":    "happiness_intensity",
    "sad":      "sadness_intensity",
    "surprise": "surprise_intensity",
    "neutral":  "neutral_intensity",
}

FIELDNAMES = (
    ["video_id", "duration_s", "frames_analyzed", "dominant_emotion", "emotion_variety"]
    + [f"{pfx}_arousal" for pfx in ("early", "whole", "late")]
    + [f"{pfx}_{col}" for pfx in ("early", "whole", "late")
       for col in EMOTION_TO_COL.values()]
)


def resize_frame(frame, size=FRAME_SIZE):
    h, w = frame.shape[:2]
    scale = size / min(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def compute_window_features(frame_emotions, prefix):
    n = len(frame_emotions)
    if n == 0:
        feats = {f"{prefix}_arousal": 0.3}
        feats.update({f"{prefix}_{col}": 0.0 for col in EMOTION_TO_COL.values()})
        return feats
    arousals = [AROUSAL_MAP.get(e, 0.3) for e in frame_emotions]
    counts   = Counter(frame_emotions)
    feats    = {f"{prefix}_arousal": round(sum(arousals) / n, 4)}
    feats.update({f"{prefix}_{col}": round(counts.get(emo, 0) / n, 4)
                  for emo, col in EMOTION_TO_COL.items()})
    return feats


def analyze_video(video_path):
    cap          = cv2.VideoCapture(video_path)
    native_fps   = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if native_fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    duration_s     = total_frames / native_fps
    frame_interval = max(1, int(native_fps / FPS))
    early_end      = int(total_frames * EARLY_FRAC)
    late_start     = int(total_frames * (1 - LATE_FRAC))

    all_emo, early_emo, late_emo = [], [], []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            try:
                small = resize_frame(frame)
                res   = DeepFace.analyze(small, actions=["emotion"],
                                         enforce_detection=False,
                                         detector_backend=DETECTOR_BACKEND,
                                         silent=True)
                dom  = res[0]["dominant_emotion"]
                conf = res[0]["emotion"].get(dom, 0)
                if conf >= CONFIDENCE_THRESHOLD:
                    all_emo.append(dom)
                    if frame_idx <= early_end:
                        early_emo.append(dom)
                    if frame_idx >= late_start:
                        late_emo.append(dom)
            except Exception:
                pass
        frame_idx += 1
    cap.release()

    if not all_emo:
        return None

    counts = Counter(all_emo)
    record = {
        "video_id":         os.path.basename(video_path).replace(".mp4", ""),
        "duration_s":       round(duration_s, 2),
        "frames_analyzed":  len(all_emo),
        "dominant_emotion": counts.most_common(1)[0][0],
        "emotion_variety":  len(set(all_emo)),
    }
    record.update(compute_window_features(early_emo, "early"))
    record.update(compute_window_features(all_emo,   "whole"))
    record.update(compute_window_features(late_emo,  "late"))
    return record


def load_done_ids():
    """Load already-processed video_ids from OUTPUT_CSV for resume support."""
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["video_id"])
    return done


def flush(results, write_header):
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)

    video_files = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4"))
    done_ids    = load_done_ids()
    todo        = [f for f in video_files if f.replace(".mp4", "") not in done_ids]

    if TEST_MODE:
        todo       = todo[:TEST_LIMIT]
        OUTPUT_CSV = OUTPUT_CSV.replace(".csv", f"_test{TEST_LIMIT}.csv")
        print(f"*** TEST MODE: processing {TEST_LIMIT} videos → {OUTPUT_CSV} ***")
    elif SESSION_LIMIT > 0:
        todo = todo[:SESSION_LIMIT]
        print(f"*** SESSION LIMIT: processing {SESSION_LIMIT} videos this run ***")

    print(f"Total videos on disk : {len(video_files)}")
    print(f"Already processed    : {len(done_ids)}")
    print(f"Remaining            : {len(todo)}")
    print(f"Config               : {FPS} fps | backend={DETECTOR_BACKEND} | "
          f"frame_size={FRAME_SIZE} | early={EARLY_FRAC*100:.0f}% late={LATE_FRAC*100:.0f}%")
    print()

    if not todo:
        print("Nothing left to process. full_visual_results.csv is complete.")
        raise SystemExit(0)

    write_header = not os.path.exists(OUTPUT_CSV)
    batch        = []
    t0           = time.time()
    times        = []

    for i, fname in enumerate(tqdm(todo, desc="Extracting emotions"), 1):
        t_vid = time.time()
        r = analyze_video(os.path.join(VIDEO_DIR, fname))
        elapsed = time.time() - t_vid
        times.append(elapsed)

        if r:
            batch.append(r)

        # Periodic save
        if i % SAVE_EVERY == 0 or i == len(todo):
            flush(batch, write_header)
            write_header = False
            batch = []
            avg   = sum(times) / len(times)
            left  = avg * (len(todo) - i)
            tqdm.write(f"  Saved {i}/{len(todo)} | avg {avg:.1f}s/vid | "
                       f"est. {left/3600:.1f}h remaining")

    total = time.time() - t0
    print(f"\nDone. Processed {len(todo)} videos in {total/3600:.2f}h.")
    print(f"Output: {OUTPUT_CSV}")
