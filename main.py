"""
Windows hand gesture controller — webcam + MediaPipe Hand Landmarker.

  python main.py           # run (calibrate first if prompted)
  python calibrate.py      # one-time setup for your hand & screen mapping
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import pyautogui

from config_store import AppConfig, load_config
from gesture_state import GestureStateMachine
from gestures import Gesture, detect_raw_gesture, is_teleport_pose
from pointer import PointerController
from tracking import HandTracker, flip_frame

CAMERA_INDEX = 0
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540

STABLE_FRAMES_PINCH = 4
STABLE_FRAMES_FIST = 8
STABLE_FRAMES_SWIPE = 5
STABLE_FRAMES_TELEPORT = 10

COOLDOWN_CLICK = 0.45
COOLDOWN_MUTE = 1.0
COOLDOWN_SWIPE = 0.9
COOLDOWN_PALM_TOGGLE = 2.0
COOLDOWN_TELEPORT = 1.5

SWIPE_DELTA_MIN = 0.14
SWIPE_WINDOW = 10

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


@dataclass
class AppState:
    config: AppConfig
    control_enabled: bool = False
    last_action_time: dict[str, float] = field(default_factory=dict)
    stable_count: dict[Gesture, int] = field(default_factory=dict)
    wrist_x_history: deque[float] = field(default_factory=lambda: deque(maxlen=SWIPE_WINDOW))
    palm_hold_start: float | None = None
    status_message: str = "Hold open palm 5s to enable control"
    current_gesture: Gesture = Gesture.NONE
    pointer_hint: str = ""
    teleport_streak: int = 0


def suppress_console_noise() -> None:
    logging.getLogger("absl").setLevel(logging.ERROR)
    logging.getLogger("mediapipe").setLevel(logging.ERROR)


def detect_swipe(state: AppState, landmarks) -> Gesture | None:
    state.wrist_x_history.append(landmarks[0].x)
    if len(state.wrist_x_history) < SWIPE_WINDOW:
        return None
    xs = list(state.wrist_x_history)
    delta = xs[-1] - xs[0]
    if abs(delta) < SWIPE_DELTA_MIN:
        return None
    if delta > 0:
        return Gesture.SWIPE_LEFT
    return Gesture.SWIPE_RIGHT


def on_cooldown(state: AppState, key: str, cooldown: float) -> bool:
    last = state.last_action_time.get(key, 0.0)
    return (time.monotonic() - last) < cooldown


def mark_action(state: AppState, key: str) -> None:
    state.last_action_time[key] = time.monotonic()


def toggle_volume_mute() -> None:
    if sys.platform != "win32":
        pyautogui.press("volumemute")
        return
    VK_VOLUME_MUTE = 0xAD
    KEYEVENTF_KEYUP = 0x0002
    user32 = ctypes.windll.user32
    user32.keybd_event(VK_VOLUME_MUTE, 0, 0, 0)
    user32.keybd_event(VK_VOLUME_MUTE, 0, KEYEVENTF_KEYUP, 0)


def stable_gesture(state: AppState, gesture: Gesture, required: int) -> bool:
    if gesture == Gesture.NONE:
        return False
    for g in list(state.stable_count):
        if g != gesture:
            del state.stable_count[g]
    state.stable_count[gesture] = state.stable_count.get(gesture, 0) + 1
    return state.stable_count.get(gesture, 0) >= required


def reset_stable(state: AppState, gesture: Gesture | None = None) -> None:
    if gesture is None:
        state.stable_count.clear()
    elif gesture in state.stable_count:
        del state.stable_count[gesture]


def palm_hold_progress(state: AppState, is_palm: bool) -> float:
    hold_s = state.config.palm_hold_seconds
    if not is_palm:
        state.palm_hold_start = None
        return 0.0
    now = time.monotonic()
    if state.palm_hold_start is None:
        state.palm_hold_start = now
    return min(1.0, (now - state.palm_hold_start) / hold_s)


def handle_gestures(
    state: AppState,
    landmarks,
    gesture_fsm: GestureStateMachine,
    pointer: PointerController,
    *,
    dt: float,
) -> None:
    raw = detect_raw_gesture(landmarks, state.config)
    effective = gesture_fsm.update(raw, True, landmarks)
    state.current_gesture = effective
    state.pointer_hint = ""

    swipe = None
    if state.control_enabled and effective in (Gesture.NONE, Gesture.POINT):
        swipe = detect_swipe(state, landmarks)
    elif effective not in (Gesture.NONE, Gesture.POINT):
        state.wrist_x_history.clear()

    if raw == Gesture.OPEN_PALM:
        progress = palm_hold_progress(state, True)
        hold_s = state.config.palm_hold_seconds
        if progress >= 1.0:
            if not on_cooldown(state, "palm_toggle", COOLDOWN_PALM_TOGGLE):
                state.control_enabled = not state.control_enabled
                mark_action(state, "palm_toggle")
                state.palm_hold_start = None
                pointer.reset()
                gesture_fsm.reset()
                if state.control_enabled:
                    state.status_message = "CONTROL ON — point to move | thumb tuck = clutch"
                else:
                    state.status_message = f"CONTROL OFF — hold open palm {hold_s:.0f}s"
            reset_stable(state, Gesture.OPEN_PALM)
        else:
            state.status_message = f"Hold open palm… {progress * hold_s:.1f}/{hold_s:.0f}s"
        return

    palm_hold_progress(state, False)

    if not state.control_enabled:
        return

    if effective == Gesture.PINCH:
        if stable_gesture(state, Gesture.PINCH, STABLE_FRAMES_PINCH):
            if not on_cooldown(state, "click", COOLDOWN_CLICK):
                pyautogui.click(_pause=False)
                mark_action(state, "click")
                state.status_message = "Left click (pinch)"
            reset_stable(state, Gesture.PINCH)
        return

    if effective == Gesture.FIST:
        if stable_gesture(state, Gesture.FIST, STABLE_FRAMES_FIST):
            if not on_cooldown(state, "mute", COOLDOWN_MUTE):
                toggle_volume_mute()
                mark_action(state, "mute")
                state.status_message = "Volume mute toggled"
            reset_stable(state, Gesture.FIST)
        return

    if swipe in (Gesture.SWIPE_LEFT, Gesture.SWIPE_RIGHT):
        key = "swipe_left" if swipe == Gesture.SWIPE_LEFT else "swipe_right"
        if stable_gesture(state, swipe, STABLE_FRAMES_SWIPE):
            if not on_cooldown(state, key, COOLDOWN_SWIPE):
                if swipe == Gesture.SWIPE_LEFT:
                    pyautogui.hotkey("alt", "shift", "tab", _pause=False)
                    state.status_message = "Alt+Shift+Tab"
                else:
                    pyautogui.hotkey("alt", "tab", _pause=False)
                    state.status_message = "Alt+Tab"
                mark_action(state, key)
                state.wrist_x_history.clear()
            reset_stable(state, swipe)
        return

    if is_teleport_pose(landmarks, state.config.gestures):
        state.teleport_streak += 1
        if state.teleport_streak >= STABLE_FRAMES_TELEPORT:
            if not on_cooldown(state, "teleport", COOLDOWN_TELEPORT):
                pointer.jump_to_aim(landmarks)
                mark_action(state, "teleport")
                state.status_message = "Cursor jump (index+middle)"
            state.teleport_streak = 0
        return
    state.teleport_streak = 0

    if effective == Gesture.POINT:
        reset_stable(state)
        hint = pointer.update(landmarks, pointing=True, dt=dt)
        if hint:
            state.pointer_hint = hint
        if not state.pointer_hint:
            p = state.config.pointer
            extras = []
            if p.dwell_click_enabled:
                extras.append("dwell=click")
            extras.append("thumb tuck=clutch")
            extras.append("pinky out=precision")
            state.status_message = "Pointing — " + ", ".join(extras)
        return

    reset_stable(state)


def draw_overlay(
    frame,
    state: AppState,
    hand_landmarks,
    tracker: HandTracker,
    pointer: PointerController,
) -> None:
    h, w = frame.shape[:2]
    if hand_landmarks is not None:
        tracker.draw_skeleton(frame, hand_landmarks)
        m = state.config.mouse
        x1 = int(m.index_min_x * w)
        x2 = int(m.index_max_x * w)
        y1 = int(m.index_min_y * h)
        y2 = int(m.index_max_y * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 1)
        if state.control_enabled:
            cx = int(pointer.state.norm_x * w)
            cy = int(pointer.state.norm_y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    control_color = (0, 220, 0) if state.control_enabled else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (w, 100), (30, 30, 30), -1)
    cv2.putText(
        frame,
        f"Control: {'ON' if state.control_enabled else 'OFF'}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        control_color,
        2,
    )
    gesture_name = state.current_gesture.name.replace("_", " ")
    cv2.putText(
        frame,
        f"Gesture: {gesture_name}",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )
    if state.pointer_hint:
        cv2.putText(
            frame,
            state.pointer_hint[:40],
            (12, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 255, 200),
            1,
        )

    if state.current_gesture == Gesture.OPEN_PALM and state.palm_hold_start:
        progress = palm_hold_progress(state, True)
        bar_w = int((w - 24) * progress)
        cv2.rectangle(frame, (12, 84), (w - 12, 94), (60, 60, 60), -1)
        cv2.rectangle(frame, (12, 84), (12 + bar_w, 94), (0, 200, 120), -1)

    cv2.putText(
        frame,
        state.status_message[:72],
        (12, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )
    mode = state.config.pointer.mode
    cal = "calibrated" if state.config.calibrated else "calibrate.py"
    cv2.putText(
        frame,
        f"Q=quit C=cal | {mode} | {cal}",
        (w - 280, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
    )


def run_app(config: AppConfig) -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open webcam. Try changing CAMERA_INDEX in main.py.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = HandTracker(smooth_window=config.pointer.landmark_smooth_window)
    tracker.set_fps(fps)
    pointer = PointerController(config=config, settings=config.pointer)
    pointer.set_fps(fps)
    gesture_fsm = GestureStateMachine(config.pointer)

    hold_s = config.palm_hold_seconds
    state = AppState(
        config=config,
        status_message=f"Hold open palm {hold_s:.0f}s | Point to move | Thumb tuck = clutch",
    )

    if not config.calibrated:
        print("Tip: Run 'python calibrate.py' once for better mouse + gesture accuracy.\n")

    print("Hand Jarvis running. Q=quit, C=calibrate.\n")
    print("  Thumb tucked + index out = clutch (freeze cursor)")
    print("  Pinky extended while pointing = precision (slow)")
    print("  Index + middle extended = cursor jump")
    if config.pointer.dwell_click_enabled:
        print("  Hold cursor still ~0.45s = dwell click\n")

    last_t = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            dt = max(now - last_t, 1 / 120)
            last_t = now

            frame = flip_frame(frame)
            raw_lms = tracker.detect(frame)
            present = raw_lms is not None
            lms = gesture_fsm.landmarks_for_control(present, raw_lms)

            if lms is not None:
                handle_gestures(state, lms, gesture_fsm, pointer, dt=dt)
            else:
                gesture_fsm.update(Gesture.NONE, False)
                state.current_gesture = gesture_fsm.effective
                palm_hold_progress(state, False)
                state.wrist_x_history.clear()
                if not present:
                    tracker.clear_smooth()

            draw_overlay(frame, state, raw_lms or lms, tracker, pointer)
            cv2.imshow("Hand Jarvis", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                cap.release()
                cv2.destroyAllWindows()
                tracker.close()
                subprocess.run([sys.executable, str(Path(__file__).parent / "calibrate.py")])
                config = load_config()
                state.config = config
                pointer = PointerController(config=config, settings=config.pointer)
                pointer.set_fps(fps)
                gesture_fsm = GestureStateMachine(config.pointer)
                tracker = HandTracker(smooth_window=config.pointer.landmark_smooth_window)
                tracker.set_fps(fps)
                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
                continue
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


def main() -> None:
    suppress_console_noise()

    parser = argparse.ArgumentParser(description="Hand Jarvis gesture control")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration wizard first",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Use absolute cursor mapping instead of relative (default)",
    )
    args = parser.parse_args()

    if args.calibrate:
        subprocess.run([sys.executable, str(Path(__file__).parent / "calibrate.py")])
        return

    config = load_config()
    if args.absolute:
        config.pointer.mode = "absolute"
    run_app(config)


if __name__ == "__main__":
    main()
