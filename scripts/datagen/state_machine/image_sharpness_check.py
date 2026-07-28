"""Measure how much detail survived into a recording's camera images.

Isaac Sim is not needed, and neither is the rest of this repository: the script reads a finished
recording -- an Isaac Lab HDF5 file or an mp4 from a LeRobot dataset -- and reports how sharp its
frames are. It exists because a rendering fault is easy to see and very hard to argue about, and
"the video looks blurry" is not something a reviewer can check.

What it reports, per camera, is the mean absolute difference between neighbouring pixels along
each axis::

    grad_x = mean |I(x+1, y) - I(x, y)|      horizontal detail
    grad_y = mean |I(x, y+1) - I(x, y)|      vertical detail
    ratio  = grad_y / grad_x                 how even the two are

Both fall when an image is blurred, so ``grad_y`` alone cannot say *why*. The ratio can, because
the usual suspects fail differently:

* **Video compression** takes a little off both axes and leaves the ratio alone or nudges it up.
  Re-encoding a good recording down to a sixth of its bitrate moved ``grad_y`` from 0.72 to 0.74
  and the ratio from 0.61 to 0.72 -- that is, not at all in the direction of a fault.
* **Renderer sample starvation** smears along one axis. A ``--num_envs 2`` recording measured
  ``grad_y`` 0.32 against 0.72 for the same scene at ``--num_envs 1``, with the ratio collapsing
  from 0.61 to 0.24, which no bitrate reproduces.

So a lopsided ratio points at the renderer and an even one points at the encoder. Neither number
means anything on its own -- both depend on the scene -- which is why the script wants two or more
recordings and prints the comparison itself. Record the same task twice, changing only the thing
under suspicion.

Usage::

    # the intended comparison: identical task, only --num_envs differs
    python scripts/datagen/state_machine/image_sharpness_check.py \\
        ./datasets/check_1env.hdf5 ./datasets/check_2env.hdf5

    # works on the mp4s of a LeRobot dataset too
    python scripts/datagen/state_machine/image_sharpness_check.py good.mp4 suspect.mp4

Requires numpy, plus h5py for HDF5 inputs and ffmpeg/ffprobe for mp4 inputs.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_SAMPLE_FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8)
"""Where in each episode to sample, as a fraction of its length.

Spread across the middle rather than taken at one instant, because sharpness varies with what the
arm happens to be doing, and clear of both ends, where the arm is parked and the frame is mostly
static background.
"""

_RATIO_DROP = 0.75
"""How far the ratio has to fall, relative to the baseline, to be called out as a rendering fault.

Compression moved the ratio *up* by 18% in the measurements above, so a fall of any real size
already points elsewhere; this leaves room for scene-to-scene variation before saying so.
"""


def _to_luma(frame: np.ndarray) -> np.ndarray:
    """Convert a frame to a 0-255 luminance plane, matching ffmpeg's ``gray`` conversion."""
    if frame.ndim == 2:
        gray = frame.astype(np.float64)
    else:
        gray = (
            0.299 * frame[..., 0].astype(np.float64)
            + 0.587 * frame[..., 1].astype(np.float64)
            + 0.114 * frame[..., 2].astype(np.float64)
        )
    # Float recordings are usually normalised; integer ones never are.
    if np.issubdtype(frame.dtype, np.floating) and float(gray.max()) <= 1.0:
        gray *= 255.0
    return gray


def _sharpness(frames: list[np.ndarray]) -> tuple[float, float]:
    """Mean horizontal and vertical neighbour difference over a set of frames."""
    horizontal = float(np.mean([np.abs(np.diff(frame, axis=1)).mean() for frame in frames]))
    vertical = float(np.mean([np.abs(np.diff(frame, axis=0)).mean() for frame in frames]))
    return horizontal, vertical


def _sample_indices(length: int) -> list[int]:
    """Frame indices to sample from a sequence of that length, without repeats."""
    return sorted({min(length - 1, max(0, int(length * fraction))) for fraction in _SAMPLE_FRACTIONS})


def _read_hdf5(path: Path) -> dict[str, list[np.ndarray]]:
    """Collect sample frames per camera from an Isaac Lab recording."""
    try:
        import h5py
    except ImportError:
        raise SystemExit(f"reading {path.name} needs h5py: pip install h5py")

    frames: dict[str, list[np.ndarray]] = {}
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise SystemExit(f"{path.name} has no 'data' group, so it is not an Isaac Lab recording")
        demos = sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[-1]))
        for demo in demos:
            observations = handle["data"][demo].get("obs")
            if observations is None:
                continue

            def collect(name, node, observations=observations):
                # An image stream is the only thing in here shaped (frames, height, width, colour).
                if not hasattr(node, "shape") or node.ndim != 4 or node.shape[-1] != 3:
                    return
                frames.setdefault(name, []).extend(
                    _to_luma(np.asarray(node[index])) for index in _sample_indices(node.shape[0])
                )

            observations.visititems(collect)
    if not frames:
        raise SystemExit(f"{path.name} holds no camera images -- was it recorded with --enable_cameras?")
    return frames


def _probe_video(path: Path) -> tuple[int, int, float]:
    """Width, height and duration of a video, via ffprobe."""
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    parsed = json.loads(output)
    stream = parsed["streams"][0]
    return int(stream["width"]), int(stream["height"]), float(parsed["format"]["duration"])


def _video_camera_name(path: Path) -> str:
    """Name the camera a video belongs to.

    A LeRobot dataset keeps the camera in a directory (``videos/observation.images.top/...``)
    rather than in the filename, so the filename alone would leave every camera called
    ``file-000`` and nothing would line up between two datasets.
    """
    for parent in path.parents:
        if parent.name.startswith("observation.images."):
            return parent.name[len("observation.images.") :]
    return path.stem


def _read_video(path: Path) -> dict[str, list[np.ndarray]]:
    """Collect sample frames from a video, decoding straight to luminance."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"reading {path.name} needs {tool} on PATH")

    width, height, duration = _probe_video(path)
    frames = []
    for fraction in _SAMPLE_FRACTIONS:
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{duration * fraction:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
        if len(raw) < width * height:
            continue
        frames.append(np.frombuffer(raw[: width * height], dtype=np.uint8).reshape(height, width).astype(np.float64))
    if not frames:
        raise SystemExit(f"could not decode any frame from {path.name}")
    return {_video_camera_name(path): frames}


def measure(path: Path) -> dict[str, tuple[float, float, int]]:
    """Return ``{camera: (grad_x, grad_y, frames_used)}`` for one recording."""
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    reader = _read_hdf5 if path.suffix.lower() in (".hdf5", ".h5") else _read_video
    return {camera: (*_sharpness(frames), len(frames)) for camera, frames in sorted(reader(path).items())}


def _pair_cameras(baseline: dict, cameras: dict) -> list[tuple[str, str]]:
    """Match each camera to its counterpart in the baseline, as ``(camera, baseline_camera)``.

    Names are used where they exist on both sides. Two single-camera recordings are paired
    regardless, because a lone mp4 is named after wherever it was saved rather than after the
    camera, and refusing to compare those would be pedantry.
    """
    if len(baseline) == 1 and len(cameras) == 1:
        return [(next(iter(cameras)), next(iter(baseline)))]
    return [(name, name) for name in cameras if name in baseline]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare how much detail survived into recorded camera images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The first recording given is treated as the baseline the others are compared against.",
    )
    parser.add_argument("recordings", nargs="+", type=Path, help="HDF5 recordings or mp4 videos to measure.")
    args = parser.parse_args()

    results = {path: measure(path) for path in args.recordings}

    print(f"{'recording':34s} {'camera':18s} {'grad_x':>7s} {'grad_y':>7s} {'ratio':>7s} {'frames':>7s}")
    for path, cameras in results.items():
        for camera, (horizontal, vertical, count) in cameras.items():
            ratio = vertical / horizontal if horizontal else float("nan")
            print(f"{path.name:34s} {camera:18s} {horizontal:7.2f} {vertical:7.2f} {ratio:7.3f} {count:7d}")

    if len(results) < 2:
        print("\nGive a second recording to compare against -- these numbers depend on the scene.")
        return 0

    baseline_path, baseline = next(iter(results.items()))
    print(f"\nagainst {baseline_path.name}:")
    compared = 0
    suspect = False
    for path, cameras in list(results.items())[1:]:
        pairs = _pair_cameras(baseline, cameras)
        for camera in cameras:
            if not any(camera == name for name, _ in pairs):
                print(f"  {path.name}: no camera matching '{camera}' in the baseline, skipping")
        for camera, baseline_camera in pairs:
            horizontal, vertical, _ = cameras[camera]
            base_h, base_v, _ = baseline[baseline_camera]
            base_ratio = base_v / base_h if base_h else float("nan")
            ratio = vertical / horizontal if horizontal else float("nan")
            change = ratio / base_ratio if base_ratio else float("nan")
            verdict = "rendering" if change < _RATIO_DROP else "no lopsided loss"
            compared += 1
            suspect = suspect or change < _RATIO_DROP
            print(
                f"  {path.name:32s} {camera:18s} grad_y {vertical / base_v:5.0%} of baseline, "
                f"ratio {ratio:.3f} vs {base_ratio:.3f} ({change:5.0%}) -> {verdict}"
            )

    if not compared:
        print(
            "\nNothing could be compared: no camera appears in both recordings. Name the cameras the same\n"
            "way in both, or pass one video per recording so they can be paired directly."
        )
        return 1

    if suspect:
        print(
            "\nDetail was lost along one axis far more than the other. Compression does not do that, so the\n"
            "frames were already smeared when they reached the recorder -- look at the rendering settings\n"
            "(--num_envs, --quality) rather than at the video encoder."
        )
    else:
        print(
            "\nWhat was lost came off both axes evenly, which is what compression does. If the recording\n"
            "still looks wrong, the encoder settings are the place to look."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
