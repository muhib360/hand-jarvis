"""Load/save user calibration and settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


@dataclass
class MouseCalibration:
    index_min_x: float = 0.12
    index_max_x: float = 0.88
    index_min_y: float = 0.12
    index_max_y: float = 0.88
    reference_hand_scale: float = 0.14
    smoothing: float = 0.28


@dataclass
class GestureCalibration:
    pinch_threshold: float = 0.045
    fist_tip_wrist_max: float = 0.17
    index_extend_margin: float = 0.02


@dataclass
class PointerSettings:
    mode: str = "relative"
    relative_gain: float = 2.2
    precision_gain: float = 0.28
    dead_zone_px: float = 4.0
    max_velocity_px: float = 48.0
    min_cutoff: float = 1.2
    beta: float = 0.05
    d_cutoff: float = 1.0
    dwell_click_seconds: float = 0.45
    dwell_click_enabled: bool = True
    dwell_move_threshold_px: float = 5.0
    dropout_frames: int = 3
    gesture_point_hold_frames: int = 6
    gesture_point_release_frames: int = 12
    landmark_smooth_window: int = 5


@dataclass
class AppConfig:
    palm_hold_seconds: float = 5.0
    mouse: MouseCalibration = field(default_factory=MouseCalibration)
    gestures: GestureCalibration = field(default_factory=GestureCalibration)
    pointer: PointerSettings = field(default_factory=PointerSettings)
    calibrated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AppConfig:
        mouse = MouseCalibration(**_filter_keys(MouseCalibration, data.get("mouse", {})))
        gestures = GestureCalibration(**_filter_keys(GestureCalibration, data.get("gestures", {})))
        pointer = PointerSettings(**_filter_keys(PointerSettings, data.get("pointer", {})))
        return cls(
            palm_hold_seconds=float(data.get("palm_hold_seconds", 5.0)),
            mouse=mouse,
            gestures=gestures,
            pointer=pointer,
            calibrated=bool(data.get("calibrated", False)),
        )


def _filter_keys(dc_type, data: dict) -> dict:
    names = {f.name for f in fields(dc_type)}
    return {k: v for k, v in data.items() if k in names}


def load_config() -> AppConfig:
    if not CONFIG_PATH.is_file():
        return AppConfig()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return AppConfig.from_dict(data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return AppConfig()


def save_config(config: AppConfig) -> None:
    CONFIG_PATH.write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
