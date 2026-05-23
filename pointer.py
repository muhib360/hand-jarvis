"""Mouse pointer control: filtering, relative movement, clutch, dwell click."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pyautogui

from config_store import AppConfig, PointerSettings
from gestures import hand_scale, is_clutch_pose, is_precision_pose


class OneEuroFilter:
    """Adaptive low-pass filter — stable when still, responsive when moving."""

    def __init__(self, freq: float, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        edx = self.dx_prev + self._alpha(self.d_cutoff) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        filt = self.x_prev + self._alpha(cutoff) * (x - self.x_prev)
        self.x_prev = filt
        self.dx_prev = edx
        return filt

    def reset(self, x: float | None = None) -> None:
        self.x_prev = x
        self.dx_prev = 0.0


@dataclass
class PointerState:
    screen_x: float = 0.0
    screen_y: float = 0.0
    norm_x: float = 0.5
    norm_y: float = 0.5
    prev_norm_x: float | None = None
    prev_norm_y: float | None = None
    clutch_active: bool = False
    dwell_start: float | None = None
    dwell_anchor: tuple[float, float] | None = None
    initialized: bool = False


@dataclass
class PointerController:
    config: AppConfig
    settings: PointerSettings = field(default_factory=PointerSettings)
    state: PointerState = field(default_factory=PointerState)
    filter_x: OneEuroFilter | None = None
    filter_y: OneEuroFilter | None = None
    _last_frame_time: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._reset_filters(30.0)
        sw, sh = pyautogui.size()
        self.state.screen_x = sw / 2
        self.state.screen_y = sh / 2

    def _reset_filters(self, fps: float) -> None:
        s = self.settings
        self.filter_x = OneEuroFilter(fps, s.min_cutoff, s.beta, s.d_cutoff)
        self.filter_y = OneEuroFilter(fps, s.min_cutoff, s.beta, s.d_cutoff)

    def set_fps(self, fps: float) -> None:
        fps = max(fps, 10.0)
        self._reset_filters(fps)
        if self.filter_x:
            self.filter_x.freq = fps
        if self.filter_y:
            self.filter_y.freq = fps

    def aim_norm_from_landmarks(self, landmarks) -> tuple[float, float]:
        """Blend wrist + index MCP + tip for a stable aim point."""
        w = landmarks[0]
        mcp = landmarks[5]
        tip = landmarks[8]
        x = 0.5 * w.x + 0.3 * mcp.x + 0.2 * tip.x
        y = 0.5 * w.y + 0.3 * mcp.y + 0.2 * tip.y
        return x, y

    def _effective_bounds(self, landmarks) -> tuple[float, float, float, float]:
        m = self.config.mouse
        x_span = m.index_max_x - m.index_min_x
        y_span = m.index_max_y - m.index_min_y
        scale = hand_scale(landmarks)
        ref = max(m.reference_hand_scale, 0.06)
        distance_ratio = float(np.clip(ref / max(scale, 0.04), 0.75, 1.35))
        cx = (m.index_min_x + m.index_max_x) / 2
        cy = (m.index_min_y + m.index_max_y) / 2
        half_x = (x_span / 2) * distance_ratio
        half_y = (y_span / 2) * distance_ratio
        return cx - half_x, cx + half_x, cy - half_y, cy + half_y

    def norm_to_screen(self, nx: float, ny: float, landmarks) -> tuple[float, float]:
        eff_min_x, eff_max_x, eff_min_y, eff_max_y = self._effective_bounds(landmarks)
        nx = float(np.clip((nx - eff_min_x) / max(eff_max_x - eff_min_x, 0.05), 0, 1))
        ny = float(np.clip((ny - eff_min_y) / max(eff_max_y - eff_min_y, 0.05), 0, 1))
        sw, sh = pyautogui.size()
        return nx * (sw - 1), ny * (sh - 1)

    def update(
        self,
        landmarks,
        *,
        pointing: bool,
        dt: float,
    ) -> str | None:
        """
        Update pointer from landmarks. Returns optional status hint for HUD.
        """
        s = self.settings
        st = self.state

        if not pointing:
            st.prev_norm_x = None
            st.prev_norm_y = None
            st.dwell_start = None
            st.dwell_anchor = None
            return None

        clutch = is_clutch_pose(landmarks, self.config.gestures)
        if clutch and not st.clutch_active:
            st.clutch_active = True
            st.prev_norm_x = None
            st.prev_norm_y = None
        if not clutch:
            st.clutch_active = False

        raw_x, raw_y = self.aim_norm_from_landmarks(landmarks)
        if self.filter_x and self.filter_y:
            raw_x = self.filter_x(raw_x)
            raw_y = self.filter_y(raw_y)

        st.norm_x, st.norm_y = raw_x, raw_y
        target_x, target_y = self.norm_to_screen(raw_x, raw_y, landmarks)

        if not st.initialized:
            st.screen_x, st.screen_y = target_x, target_y
            st.initialized = True
            pyautogui.moveTo(int(st.screen_x), int(st.screen_y), _pause=False)
            st.prev_norm_x, st.prev_norm_y = raw_x, raw_y
            return "clutch: reposition hand" if st.clutch_active else None

        if st.clutch_active:
            st.prev_norm_x, st.prev_norm_y = raw_x, raw_y
            return "CLUTCH — move hand, cursor frozen"

        gain = s.relative_gain
        if is_precision_pose(landmarks, self.config.gestures):
            gain *= s.precision_gain

        hint = None
        if s.mode == "absolute":
            dx = target_x - st.screen_x
            dy = target_y - st.screen_y
            st.screen_x, st.screen_y = self._apply_motion(
                st.screen_x, st.screen_y, dx, dy, s
            )
        else:
            if st.prev_norm_x is None:
                st.prev_norm_x, st.prev_norm_y = raw_x, raw_y
            dnx = raw_x - st.prev_norm_x
            dny = raw_y - st.prev_norm_y
            st.prev_norm_x, st.prev_norm_y = raw_x, raw_y
            sw, sh = pyautogui.size()
            dx = dnx * sw * gain
            dy = dny * sh * gain
            st.screen_x, st.screen_y = self._apply_motion(
                st.screen_x, st.screen_y, dx, dy, s
            )
            if is_precision_pose(landmarks, self.config.gestures):
                hint = "precision mode"

        sw, sh = pyautogui.size()
        st.screen_x = float(np.clip(st.screen_x, 0, sw - 1))
        st.screen_y = float(np.clip(st.screen_y, 0, sh - 1))
        pyautogui.moveTo(int(st.screen_x), int(st.screen_y), _pause=False)

        if s.dwell_click_enabled:
            if self._maybe_dwell_click():
                return "dwell click"

        return hint

    def _apply_motion(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        s: PointerSettings,
    ) -> tuple[float, float]:
        dist = math.hypot(dx, dy)
        if dist < s.dead_zone_px:
            return x, y
        if dist > s.max_velocity_px:
            scale = s.max_velocity_px / dist
            dx *= scale
            dy *= scale
        return x + dx, y + dy

    def _maybe_dwell_click(self) -> bool:
        s = self.settings
        st = self.state
        now = time.monotonic()
        pos = (st.screen_x, st.screen_y)
        if st.dwell_anchor is None:
            st.dwell_anchor = pos
            st.dwell_start = now
            return False
        moved = math.hypot(pos[0] - st.dwell_anchor[0], pos[1] - st.dwell_anchor[1])
        if moved > s.dwell_move_threshold_px:
            st.dwell_anchor = pos
            st.dwell_start = now
            return False
        if st.dwell_start and (now - st.dwell_start) >= s.dwell_click_seconds:
            if not hasattr(self, "_last_dwell_click"):
                self._last_dwell_click = 0.0
            if now - self._last_dwell_click > 0.6:
                pyautogui.click(_pause=False)
                self._last_dwell_click = now
                st.dwell_start = None
                st.dwell_anchor = None
                return True
        return False

    def jump_to_aim(self, landmarks) -> None:
        """Absolute teleport to current aim position (optional gesture)."""
        raw_x, raw_y = self.aim_norm_from_landmarks(landmarks)
        tx, ty = self.norm_to_screen(raw_x, raw_y, landmarks)
        self.state.screen_x = tx
        self.state.screen_y = ty
        if self.filter_x and self.filter_y:
            self.filter_x.reset(raw_x)
            self.filter_y.reset(raw_y)
        pyautogui.moveTo(int(tx), int(ty), _pause=False)

    def reset(self) -> None:
        self.state = PointerState()
        sw, sh = pyautogui.size()
        self.state.screen_x = sw / 2
        self.state.screen_y = sh / 2
        self._reset_filters(30.0)
