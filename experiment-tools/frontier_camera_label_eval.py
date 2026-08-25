"""Reproduce the camera label reconstruction experiment from the March 2026 post."""

from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class Pose:
    pan: float
    tilt: float


def encode_vague(pose: Pose) -> Pose:
    pan = -24 if pose.pan < -8 else 24 if pose.pan > 8 else 0
    tilt = -14.5 if pose.tilt < -5 else 14.5 if pose.tilt > 5 else 0
    return Pose(pan, tilt)


def encode_precise(pose: Pose) -> Pose:
    return Pose(round(pose.pan), round(pose.tilt))


def axis_error(source: Pose, reconstructed: Pose) -> tuple[float, float]:
    return abs(source.pan - reconstructed.pan), abs(source.tilt - reconstructed.tilt)


def evaluate(encoder) -> dict[str, float]:
    poses = [
        Pose(pan_half_steps / 2, tilt_half_steps / 2)
        for pan_half_steps in range(-80, 81)
        for tilt_half_steps in range(-48, 49)
    ]
    errors = [axis_error(pose, encoder(pose)) for pose in poses]
    within_tolerance = sum(pan <= 2 and tilt <= 2 for pan, tilt in errors)
    return {
        "poses": len(poses),
        "pan_mae": fmean(pan for pan, _ in errors),
        "tilt_mae": fmean(tilt for _, tilt in errors),
        "within_two_degrees_pct": 100 * within_tolerance / len(poses),
    }


if __name__ == "__main__":
    for name, encoder in (("vague", encode_vague), ("precise", encode_precise)):
        print(name, evaluate(encoder))
