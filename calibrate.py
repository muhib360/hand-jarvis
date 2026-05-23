"""
Interactive calibration — run once to tune mouse range and gestures for your hand.

  python calibrate.py
"""

from __future__ import annotations

import os
import sys
import time
from statistics import median

# Quiet MediaPipe / TF console noise before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from config_store import AppConfig, save_config
from gestures import Gesture, _dist, detect_raw_gesture, hand_scale, is_index_extended
from tracking import HandTracker, ensure_model, flip_frame

STEPS = [
    (
        "mouse_left",
        "Extend INDEX toward the LEFT.\nOther fingers may be curled or hidden.\nPress SPACE when ready.",
    ),
    (
        "mouse_right",
        "Extend INDEX toward the RIGHT.\nOther fingers can be out of view.\nPress SPACE.",
    ),
    (
        "mouse_top",
        "Extend INDEX toward the TOP.\nPress SPACE.",
    ),
    (
        "mouse_bottom",
        "Extend INDEX toward the BOTTOM.\nOK if fist is hidden — only index needs to show.\nPress SPACE.",
    ),
    ("open_palm", "Show a clear OPEN PALM (all fingers spread).\nPress SPACE to capture."),
    ("pinch", "Make a PINCH (thumb + index touching).\nPress SPACE to capture."),
    ("fist", "Make a FIST.\nPress SPACE to capture."),
]


def draw_ui(frame, title: str, hint: str, extra: str = "") -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    y = 28
    for line in title.split("\n")[:2]:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 255), 2)
        y += 28
    y = h - 50
    for line in hint.split("\n")[:3]:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y += 22
    if extra:
        cv2.putText(frame, extra, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 255, 160), 1)


def collect_landmarks(tracker: HandTracker, cap, seconds: float = 1.0) -> list:
    samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = flip_frame(frame)
        lms = tracker.detect(frame)
        if lms is not None:
            samples.append(lms)
        cv2.imshow("Hand Jarvis — Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
    return samples


def main() -> None:
    ensure_model()
    config = AppConfig()
    pinch_dists: list[float] = []
    fist_scores: list[float] = []
    palm_scales: list[float] = []
    index_x: list[float] = []
    index_y: list[float] = []
    scales: list[float] = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    tracker = HandTracker()
    print("Hand Jarvis calibration. Press Q in the window to cancel.\n")

    try:
        for step_id, instruction in STEPS:
            captured = False
            while not captured:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = flip_frame(frame)
                lms = tracker.detect(frame)
                if lms is not None:
                    tracker.draw_skeleton(frame, lms)
                    tip = lms[8]
                    cv2.circle(
                        frame,
                        (int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])),
                        10,
                        (0, 255, 255),
                        2,
                    )

                draw_ui(
                    frame,
                    f"Step: {step_id}",
                    instruction,
                    "SPACE = capture | Q = quit",
                )
                cv2.imshow("Hand Jarvis — Calibration", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
                if key != ord(" "):
                    continue

                samples = collect_landmarks(tracker, cap, 0.8)
                if len(samples) < 5:
                    print("Not enough hand data — keep your hand in frame and try again.")
                    continue

                if step_id.startswith("mouse_"):
                    index_ok = sum(1 for s in samples if is_index_extended(s, config.gestures))
                    if index_ok < len(samples) * 0.35:
                        print(
                            "Keep your INDEX extended toward the target. "
                            "Other fingers can be hidden off-camera."
                        )
                        continue
                    xs = [s[8].x for s in samples]
                    ys = [s[8].y for s in samples]
                    scales.extend(hand_scale(s) for s in samples)
                    if step_id == "mouse_left":
                        index_x.append(min(xs))
                    elif step_id == "mouse_right":
                        index_x.append(max(xs))
                    elif step_id == "mouse_top":
                        index_y.append(min(ys))
                    elif step_id == "mouse_bottom":
                        index_y.append(max(ys))
                    print(f"  Saved {step_id}")
                    captured = True
                elif step_id == "open_palm":
                    for s in samples:
                        if detect_raw_gesture(s, config) == Gesture.OPEN_PALM:
                            palm_scales.append(hand_scale(s))
                    print(f"  Open palm samples: {len(palm_scales)}")
                    captured = True
                elif step_id == "pinch":
                    for s in samples:
                        pinch_dists.append(_dist(s[4], s[8]))
                    print(f"  Pinch distances captured: {len(pinch_dists)}")
                    captured = True
                elif step_id == "fist":
                    for s in samples:
                        wrist = s[0]
                        tips = [s[i] for i in (8, 12, 16, 20)]
                        fist_scores.append(float(np.mean([_dist(t, wrist) for t in tips])))
                    print(f"  Fist samples: {len(fist_scores)}")
                    captured = True

        if len(index_x) >= 2 and len(index_y) >= 2:
            pad = 0.03
            config.mouse.index_min_x = min(index_x) - pad
            config.mouse.index_max_x = max(index_x) + pad
            config.mouse.index_min_y = min(index_y) - pad
            config.mouse.index_max_y = max(index_y) + pad
            if scales:
                config.mouse.reference_hand_scale = float(median(scales))
        if pinch_dists:
            config.gestures.pinch_threshold = float(median(pinch_dists)) * 1.15
        if fist_scores:
            config.gestures.fist_tip_wrist_max = float(median(fist_scores)) * 1.08
        config.calibrated = True
        save_config(config)
        print("\nCalibration saved to config.json")
        print(f"  Mouse X: [{config.mouse.index_min_x:.2f}, {config.mouse.index_max_x:.2f}]")
        print(f"  Mouse Y: [{config.mouse.index_min_y:.2f}, {config.mouse.index_max_y:.2f}]")
        print(f"  Hand scale ref: {config.mouse.reference_hand_scale:.3f}")
        print(f"  Pinch threshold: {config.gestures.pinch_threshold:.3f}")
        print("\nRun: python main.py")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()
