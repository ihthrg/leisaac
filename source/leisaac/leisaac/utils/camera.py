from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass
from pxr import Gf, Usd

_PINHOLE_COEFFICIENT_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")
_FISHEYE_COEFFICIENT_NAMES = ("k1", "k2", "k3", "k4")


def _set_camera_attribute(camera_prim: Usd.Prim, name: str, value) -> None:
    attribute = camera_prim.GetAttribute(name)
    if not attribute.IsValid() or not attribute.Set(value):
        raise RuntimeError(f"Failed to set camera attribute '{name}' on '{camera_prim.GetPath()}'.")


@clone
def spawn_opencv_camera(
    prim_path: str,
    cfg: "OpenCvCameraCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD camera and apply an Isaac Sim OpenCV distortion schema."""
    if cfg.calibration_width <= 0 or cfg.calibration_height <= 0:
        raise ValueError("OpenCV camera calibration dimensions must be positive.")
    if cfg.fx <= 0 or cfg.fy <= 0:
        raise ValueError("OpenCV camera focal lengths must be positive.")

    if cfg.distortion_model == "pinhole":
        schema_name = "OmniLensDistortionOpenCvPinholeAPI"
        model_name = "opencvPinhole"
        attribute_prefix = "omni:lensdistortion:opencvPinhole"
        coefficient_names = _PINHOLE_COEFFICIENT_NAMES
    elif cfg.distortion_model == "fisheye":
        schema_name = "OmniLensDistortionOpenCvFisheyeAPI"
        model_name = "opencvFisheye"
        attribute_prefix = "omni:lensdistortion:opencvFisheye"
        coefficient_names = _FISHEYE_COEFFICIENT_NAMES
    else:
        raise ValueError(f"Unsupported OpenCV camera distortion model: {cfg.distortion_model}")

    if len(cfg.distortion_coefficients) > len(coefficient_names):
        raise ValueError(
            f"The {cfg.distortion_model} model accepts at most {len(coefficient_names)} distortion coefficients."
        )

    base_cfg = sim_utils.PinholeCameraCfg(
        visible=cfg.visible,
        semantic_tags=cfg.semantic_tags,
        copy_from_source=cfg.copy_from_source,
        projection_type=cfg.projection_type,
        clipping_range=cfg.clipping_range,
        focal_length=cfg.focal_length,
        focus_distance=cfg.focus_distance,
        f_stop=cfg.f_stop,
        horizontal_aperture=cfg.horizontal_aperture,
        vertical_aperture=cfg.vertical_aperture,
        horizontal_aperture_offset=cfg.horizontal_aperture_offset,
        vertical_aperture_offset=cfg.vertical_aperture_offset,
        lock_camera=cfg.lock_camera,
    )
    camera_prim = sim_utils.spawn_camera(
        prim_path,
        base_cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )

    if not camera_prim.ApplyAPI(schema_name):
        raise RuntimeError(f"Failed to apply camera schema '{schema_name}' on '{camera_prim.GetPath()}'.")
    _set_camera_attribute(camera_prim, "omni:lensdistortion:model", model_name)
    _set_camera_attribute(
        camera_prim, f"{attribute_prefix}:imageSize", Gf.Vec2i(cfg.calibration_width, cfg.calibration_height)
    )
    for attribute_name, value in (("cx", cfg.cx), ("cy", cfg.cy), ("fx", cfg.fx), ("fy", cfg.fy)):
        _set_camera_attribute(camera_prim, f"{attribute_prefix}:{attribute_name}", float(value))
    for coefficient_name, value in zip(coefficient_names, cfg.distortion_coefficients):
        _set_camera_attribute(camera_prim, f"{attribute_prefix}:{coefficient_name}", float(value))

    return camera_prim


@configclass
class OpenCvCameraCfg(sim_utils.PinholeCameraCfg):
    """USD camera configuration using native OpenCV calibration parameters."""

    func: Callable = spawn_opencv_camera

    calibration_width: int = MISSING
    calibration_height: int = MISSING
    fx: float = MISSING
    fy: float = MISSING
    cx: float = MISSING
    cy: float = MISSING
    distortion_model: Literal["pinhole", "fisheye"] = "pinhole"
    distortion_coefficients: tuple[float, ...] = ()
