"""Closed-form planar kinematic model of the SO-101 arm, calibrated from FK probes.

The SO-101 has five arm joints, so commanding it through a differential-IK action that takes a
full 6-DoF pose is over-constrained: the Jacobian is 6x5 and always has a left-null direction, so
the solver settles wherever the residual pose error lies entirely in that unreachable direction.
Measured in sim, that trap left the jaw 14 cm from its commanded point at the end of a 3 s
approach and let the arm drift 23 cm away from a *static* hold command. No amount of gain or
timing tuning fixes it, because it is a rank deficiency, not a transient.

This module removes the problem by solving the arm in joint space instead. Only four unknowns are
ever needed for this task -- ``shoulder_pan`` for the heading and ``shoulder_lift``/``elbow_flex``
/``wrist_flex`` for a top-down reach in the arm's own plane -- which is an exactly-determined
problem with a closed-form solution.

The model is *measured*, not assumed. :func:`calibrate_planar_arm` drives the arm to nine poses
and reads the resulting jaw positions; every parameter (link lengths, joint-angle offsets, joint
sign conventions, the pan axis, the arm plane) then follows from circle geometry, so nothing here
depends on a hand-derived URDF convention.
"""

import math

import torch

__all__ = ["PlanarArmModel", "calibrate_planar_arm", "wrap_to_pi"]

_WRIST_LIMIT_ITERATIONS = 4
"""Fixed-point passes used to pull ``wrist_flex`` back inside its limit.

Tilting the gripper also moves the wrist point, so the upstream joints shift a little and the
correction is not exact in one pass; the coupling is weak and a handful of passes settles it."""


def wrap_to_pi(angle):
    """Wrap ``angle`` (float or tensor, radians) into ``[-pi, pi]``."""
    if isinstance(angle, torch.Tensor):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    return math.atan2(math.sin(angle), math.cos(angle))


def _circle_center(p1, p2, p3):
    """Centre of the circle through three distinct, non-collinear 2-D points."""
    (ax, ay), (bx, by), (cx, cy) = p1, p2, p3
    det = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(det) < 1.0e-12:
        raise RuntimeError("planar-arm calibration: probe points are collinear, cannot fit a circle")
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    return (
        (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / det,
        (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / det,
    )


def _angle_about(center, point):
    return math.atan2(point[1] - center[1], point[0] - center[0])


def _project_to_plane(point, pan_axis, plane_dir):
    """Map a base-frame point to the arm plane's ``(forward, height)`` coordinates."""
    rel_x, rel_y = point[0] - pan_axis[0], point[1] - pan_axis[1]
    return (rel_x * plane_dir[0] + rel_y * plane_dir[1], point[2])


def _rotation_sign(center, point_zero, point_plus, joint_delta, joint_name):
    """Signed joint->rotation gain, read off a probe pair. Rounded to the nearest +-1.

    ``joint_delta`` is the joint displacement the arm *achieved* between the two probes, not the
    one that was asked for, so a probe pose the arm failed to hold exactly still calibrates
    correctly.
    """
    if abs(joint_delta) < 1.0e-3:
        raise RuntimeError(
            f"planar-arm calibration: '{joint_name}' barely moved between probes "
            f"({math.degrees(joint_delta):.2f} deg) -- the arm is not reaching the probe poses."
        )
    gain = wrap_to_pi(_angle_about(center, point_plus) - _angle_about(center, point_zero)) / joint_delta
    if abs(abs(gain) - 1.0) > 0.15:
        raise RuntimeError(
            f"planar-arm calibration: '{joint_name}' should swing the jaw one-for-one about its own "
            f"axis, but the measured gain is {gain:.3f}. Either the arm moves while the jaw is being "
            "read, or the joint is running into a limit."
        )
    return math.copysign(1.0, gain)


class PlanarArmModel:
    """Measured kinematics of the SO-101 arm, expressed as a pan joint plus a planar 3-link chain.

    All poses are in the *robot base frame*. ``shoulder_pan`` rotates the arm about a vertical
    axis; the remaining three pitch joints move the jaw inside a plane that is offset sideways
    from that axis by a fixed distance (the jaw sits off the arm's centre line).

    Inside the plane the chain is described by the absolute pitch of each link, measured from
    "horizontal, pointing away from the pan axis", positive upwards::

        pitch1 = a1 + k1 * shoulder_lift                              (upper arm)
        pitch2 = a2 + k1 * shoulder_lift + k2 * elbow_flex            (forearm)
        pitch3 = a3 + k1 * shoulder_lift + k2 * elbow_flex + k3 * wrist_flex   (gripper)

    Each joint carries through to every downstream link, which is why a change to ``shoulder_lift``
    alone tilts the forearm and the gripper too -- the reason ``wrist_flex = 0`` does *not* leave
    the gripper parallel to the forearm unless ``a3 == a2``.
    """

    def __init__(
        self,
        pan_axis: tuple[float, float],
        pan_sign: float,
        plane_dir: tuple[float, float],
        plane_offset: float,
        shoulder: tuple[float, float],
        lengths: tuple[float, float, float],
        pitch_offsets: tuple[float, float, float],
        pitch_signs: tuple[float, float, float],
        pan_zero: float = 0.0,
    ) -> None:
        self.pan_axis = pan_axis
        self.pan_sign = pan_sign
        self.plane_dir = plane_dir
        self.plane_offset = plane_offset
        self.shoulder = shoulder
        self.lengths = lengths
        self.pitch_offsets = pitch_offsets
        self.pitch_signs = pitch_signs
        self.pan_zero = pan_zero
        """``shoulder_pan`` angle the arm plane was measured at; solutions are offset from it."""

    # ------------------------------------------------------------------
    # Forward direction
    # ------------------------------------------------------------------

    def to_plane(self, point_b):
        """Map a base-frame point to this arm's ``(forward, height)`` plane coordinates."""
        return _project_to_plane(point_b, self.pan_axis, self.plane_dir)

    def describe(self) -> str:
        """Compact single-line summary of the measured parameters, for start-up logging."""
        return (
            "pan_axis=(%.4f, %.4f) pan_sign=%+.0f plane_dir=(%.4f, %.4f) lateral=%.4f "
            "shoulder=(%.4f, %.4f) lengths=(%.4f, %.4f, %.4f) "
            "pitch_offsets_deg=(%.2f, %.2f, %.2f) pitch_signs=(%+.0f, %+.0f, %+.0f)"
            % (
                *self.pan_axis,
                self.pan_sign,
                *self.plane_dir,
                self.plane_offset,
                *self.shoulder,
                *self.lengths,
                *(math.degrees(value) for value in self.pitch_offsets),
                *self.pitch_signs,
            )
        )

    def pitches_from_joints(self, lift, elbow, wrist_flex):
        """Absolute link pitches produced by the three pitch joints (radians)."""
        a1, a2, a3 = self.pitch_offsets
        k1, k2, k3 = self.pitch_signs
        pitch1 = a1 + k1 * lift
        pitch2 = a2 + k1 * lift + k2 * elbow
        pitch3 = a3 + k1 * lift + k2 * elbow + k3 * wrist_flex
        return pitch1, pitch2, pitch3

    def jaw_in_plane(self, lift, elbow, wrist_flex):
        """In-plane ``(forward, height)`` of the jaw for the given pitch joints."""
        pitch1, pitch2, pitch3 = self.pitches_from_joints(lift, elbow, wrist_flex)
        length1, length2, length3 = self.lengths
        forward = (
            self.shoulder[0] + length1 * math.cos(pitch1) + length2 * math.cos(pitch2) + length3 * math.cos(pitch3)
        )
        height = self.shoulder[1] + length1 * math.sin(pitch1) + length2 * math.sin(pitch2) + length3 * math.sin(pitch3)
        return forward, height

    # ------------------------------------------------------------------
    # Inverse direction
    # ------------------------------------------------------------------

    def joints_from_pitches(self, pitch1, pitch2, pitch3):
        """Invert :meth:`pitches_from_joints`. Accepts floats or tensors."""
        a1, a2, a3 = self.pitch_offsets
        k1, k2, k3 = self.pitch_signs
        lift = wrap_to_pi((pitch1 - a1) / k1)
        elbow = wrap_to_pi((pitch2 - a2 - k1 * lift) / k2)
        wrist_flex = wrap_to_pi((pitch3 - a3 - k1 * lift - k2 * elbow) / k3)
        return lift, elbow, wrist_flex

    def joints_for_link_pitches(self, pitch1_deg: float, pitch2_deg: float, pitch3_deg: float):
        """Joint angles (radians) that place each link at the requested absolute pitch.

        Used for the in-air hold pose, which is specified visually ("upper arm vertical, everything
        past the elbow horizontal") rather than as joint angles.
        """
        return self.joints_from_pitches(math.radians(pitch1_deg), math.radians(pitch2_deg), math.radians(pitch3_deg))

    def top_down_joints(
        self, jaw_pos_b: torch.Tensor, wrist_flex_limits: tuple[float, float] | None = None
    ) -> torch.Tensor:
        """Joint angles that put the jaw at ``jaw_pos_b`` with the gripper pointing straight down.

        Args:
            jaw_pos_b: ``(N, 3)`` jaw targets in the robot base frame.
            wrist_flex_limits: Optional ``(lower, upper)`` ``wrist_flex`` limits in radians. When
                given, a solution that would need the wrist past a limit is re-solved with the
                gripper pitched to whatever the limit does allow, instead of being clipped
                afterwards -- clipping moves the jaw off the target, which is the whole failure
                this class exists to avoid.

        Returns:
            ``(N, 4)`` tensor of ``(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex)`` in
            radians.

        Position is treated as the hard constraint and gripper pitch as the soft one. A strictly
        vertical gripper is not reachable everywhere -- it forces the wrist to sit a full link
        length directly above the target, which can push it outside the shoulder's reach or past
        the wrist limit at the far edge of the table -- and giving up centimetres of position to
        keep the gripper perfectly vertical is exactly the failure mode that made the fingers close
        on empty air. So where vertical is unavailable the gripper is tilted by the smallest angle
        that makes the target attainable, and the jaw still lands on it.
        """
        axis_x, axis_y = self.pan_axis
        dir_x, dir_y = self.plane_dir
        shoulder_f, shoulder_h = self.shoulder

        offset_x = jaw_pos_b[:, 0] - axis_x
        offset_y = jaw_pos_b[:, 1] - axis_y
        height = jaw_pos_b[:, 2]

        # The jaw is rigidly offset sideways from the arm plane, so the in-plane reach is the leg
        # of a right triangle whose hypotenuse is the horizontal distance to the pan axis.
        radius_sq = offset_x * offset_x + offset_y * offset_y
        forward = torch.sqrt(torch.clamp(radius_sq - self.plane_offset**2, min=1.0e-6))
        lateral = torch.full_like(forward, self.plane_offset)
        pan = wrap_to_pi(torch.atan2(offset_y, offset_x) - math.atan2(dir_y, dir_x) - torch.atan2(lateral, forward))
        pan = self.pan_zero + pan / self.pan_sign

        target_f = forward - shoulder_f
        target_h = height - shoulder_h
        pitch3 = torch.full_like(target_f, -0.5 * math.pi)
        for _ in range(_WRIST_LIMIT_ITERATIONS):
            pitch3 = self._reachable_gripper_pitch(target_f, target_h, pitch3)
            lift, elbow, wrist_flex = self._solve_pitch_joints(target_f, target_h, pitch3)
            if wrist_flex_limits is None:
                break
            allowed = torch.clamp(wrist_flex, wrist_flex_limits[0], wrist_flex_limits[1])
            excess = wrist_flex - allowed
            if float(excess.abs().max()) < 1.0e-6:
                break
            # ``pitch3`` moves one-for-one with ``wrist_flex`` at fixed upstream joints, so undoing
            # the excess pulls the wrist back inside its limit; re-solving then re-places the jaw.
            pitch3 = pitch3 - self.pitch_signs[2] * excess
        return torch.stack([pan, lift, elbow, wrist_flex], dim=-1)

    def _solve_pitch_joints(self, target_f: torch.Tensor, target_h: torch.Tensor, pitch3: torch.Tensor):
        """Two-link reach to the wrist implied by ``pitch3``, expressed as joint angles."""
        length1, length2, length3 = self.lengths
        wrist_f = target_f - length3 * torch.cos(pitch3)
        wrist_h = target_h - length3 * torch.sin(pitch3)
        reach_sq = wrist_f * wrist_f + wrist_h * wrist_h
        cos_elbow = torch.clamp(
            (reach_sq - length1 * length1 - length2 * length2) / (2.0 * length1 * length2), -1.0, 1.0
        )
        # Negative branch keeps the elbow bent the same way as the rest and hold poses; the
        # positive branch is the mirror-image configuration that folds the arm back on itself.
        interior = -torch.acos(cos_elbow)
        pitch1 = torch.atan2(wrist_h, wrist_f) - torch.atan2(
            length2 * torch.sin(interior), length1 + length2 * torch.cos(interior)
        )
        return self.joints_from_pitches(pitch1, pitch1 + interior, pitch3)

    def _reachable_gripper_pitch(
        self, target_f: torch.Tensor, target_h: torch.Tensor, preferred: torch.Tensor
    ) -> torch.Tensor:
        """Pitch closest to ``preferred`` that keeps the wrist inside the shoulder's reach.

        Args:
            target_f: In-plane forward offset of the jaw target from the shoulder.
            target_h: Height offset of the jaw target from the shoulder.
            preferred: Gripper pitch to stay at where it is attainable.
        """
        length1, length2, length3 = self.lengths
        max_reach = length1 + length2

        wrist_f = target_f - length3 * torch.cos(preferred)
        wrist_h = target_h - length3 * torch.sin(preferred)
        preferred_reach = torch.sqrt(wrist_f * wrist_f + wrist_h * wrist_h)
        distance = torch.sqrt(torch.clamp(target_f**2 + target_h**2, min=1.0e-12))

        # Pitches that put the wrist exactly on the reach boundary satisfy
        # |target - length3 * u(pitch)| = max_reach, i.e. cos(pitch - bearing) = ratio.
        bearing = torch.atan2(target_h, target_f)
        ratio = (distance**2 + length3**2 - max_reach**2) / (2.0 * length3 * distance)
        spread = torch.acos(torch.clamp(ratio, -1.0, 1.0))
        candidate_a = bearing + spread
        candidate_b = bearing - spread
        closer = torch.abs(wrap_to_pi(candidate_a - preferred)) <= torch.abs(wrap_to_pi(candidate_b - preferred))
        relaxed = torch.where(closer, candidate_a, candidate_b)

        # Only give up the preferred pitch where it is actually unreachable, and only where tilting
        # can help at all (``|ratio| > 1`` means no pitch brings the target inside the workspace).
        return torch.where(torch.logical_and(preferred_reach > max_reach, torch.abs(ratio) <= 1.0), relaxed, preferred)


def calibrate_planar_arm(probe, delta_deg: float = 45.0) -> PlanarArmModel:
    """Measure a :class:`PlanarArmModel` from forward-kinematics probes.

    Args:
        probe: Callable ``(pan_deg, lift_deg, elbow_deg, wrist_flex_deg)`` returning
            ``((x, y, z), (pan, lift, elbow, wrist_flex))`` -- the jaw position in the robot base
            frame, and the joint angles the arm *actually reached*, in radians. Working from the
            achieved angles rather than the requested ones keeps the result exact even when the
            arm cannot hold a probe pose perfectly. The caller is responsible for holding
            ``wrist_roll`` and the gripper at the values used during the episode -- both move the
            jaw, so calibrating at a different value silently biases every solution.
        delta_deg: Probe displacement. Large enough to keep the circle fits well conditioned,
            small enough to stay inside every joint limit.

    Returns:
        The calibrated model.
    """
    # Each joint rotates everything downstream of it about its own axis, so sweeping one joint
    # sends the jaw around a circle centred on that joint. Three points fix each circle exactly.
    at_zero, zero_joints = probe(0.0, 0.0, 0.0, 0.0)
    pan_minus, _ = probe(-delta_deg, 0.0, 0.0, 0.0)
    pan_plus, pan_plus_joints = probe(delta_deg, 0.0, 0.0, 0.0)
    lift_minus, _ = probe(0.0, -delta_deg, 0.0, 0.0)
    lift_plus, lift_plus_joints = probe(0.0, delta_deg, 0.0, 0.0)
    elbow_minus, _ = probe(0.0, 0.0, -delta_deg, 0.0)
    elbow_plus, elbow_plus_joints = probe(0.0, 0.0, delta_deg, 0.0)
    wrist_minus, _ = probe(0.0, 0.0, 0.0, -delta_deg)
    wrist_plus, wrist_plus_joints = probe(0.0, 0.0, 0.0, delta_deg)

    # 1. Pan axis: horizontal circle traced by the jaw as shoulder_pan sweeps.
    horizontal = {
        name: (point[0], point[1]) for name, point in (("zero", at_zero), ("minus", pan_minus), ("plus", pan_plus))
    }
    pan_axis = _circle_center(horizontal["minus"], horizontal["zero"], horizontal["plus"])
    pan_sign = _rotation_sign(
        pan_axis, horizontal["zero"], horizontal["plus"], pan_plus_joints[0] - zero_joints[0], "shoulder_pan"
    )

    # 2. Arm plane: every pan=0 probe lies in one vertical plane, so their horizontal positions
    #    lie on a line. Take its direction from the widest-separated pair for conditioning.
    in_plane_probes = [at_zero, lift_minus, lift_plus, elbow_minus, elbow_plus, wrist_minus, wrist_plus]
    best_pair, best_gap = None, 0.0
    for i, first in enumerate(in_plane_probes):
        for second in in_plane_probes[i + 1 :]:
            gap = math.hypot(second[0] - first[0], second[1] - first[1])
            if gap > best_gap:
                best_gap, best_pair = gap, (first, second)
    if best_gap < 1.0e-4:
        raise RuntimeError("planar-arm calibration: pan=0 probes did not move horizontally")
    plane_dir = ((best_pair[1][0] - best_pair[0][0]) / best_gap, (best_pair[1][1] - best_pair[0][1]) / best_gap)
    if (at_zero[0] - pan_axis[0]) * plane_dir[0] + (at_zero[1] - pan_axis[1]) * plane_dir[1] < 0.0:
        plane_dir = (-plane_dir[0], -plane_dir[1])
    perp_dir = (-plane_dir[1], plane_dir[0])

    def to_plane(point):
        return _project_to_plane(point, pan_axis, plane_dir)

    plane_offset = (at_zero[0] - pan_axis[0]) * perp_dir[0] + (at_zero[1] - pan_axis[1]) * perp_dir[1]

    # 3. Link circles, all in the (forward, height) plane.
    zero_p = to_plane(at_zero)
    shoulder = _circle_center(to_plane(lift_minus), zero_p, to_plane(lift_plus))
    elbow = _circle_center(to_plane(elbow_minus), zero_p, to_plane(elbow_plus))
    wrist = _circle_center(to_plane(wrist_minus), zero_p, to_plane(wrist_plus))

    pitch_signs = (
        _rotation_sign(shoulder, zero_p, to_plane(lift_plus), lift_plus_joints[1] - zero_joints[1], "shoulder_lift"),
        _rotation_sign(elbow, zero_p, to_plane(elbow_plus), elbow_plus_joints[2] - zero_joints[2], "elbow_flex"),
        _rotation_sign(wrist, zero_p, to_plane(wrist_plus), wrist_plus_joints[3] - zero_joints[3], "wrist_flex"),
    )
    lengths = (
        math.dist(elbow, shoulder),
        math.dist(wrist, elbow),
        math.dist(zero_p, wrist),
    )

    # The link pitches above were measured at whatever joint angles the reference probe reached,
    # which is not necessarily all-zero; subtract that configuration's contribution so the offsets
    # mean "pitch at zero joints", as ``pitches_from_joints`` assumes.
    sign1, sign2, sign3 = pitch_signs
    _, ref_lift, ref_elbow, ref_wrist_flex = zero_joints
    pitch_offsets = (
        _angle_about(shoulder, elbow) - sign1 * ref_lift,
        _angle_about(elbow, wrist) - sign1 * ref_lift - sign2 * ref_elbow,
        _angle_about(wrist, zero_p) - sign1 * ref_lift - sign2 * ref_elbow - sign3 * ref_wrist_flex,
    )

    return PlanarArmModel(
        pan_axis=pan_axis,
        pan_sign=pan_sign,
        plane_dir=plane_dir,
        plane_offset=plane_offset,
        shoulder=shoulder,
        lengths=lengths,
        pitch_offsets=pitch_offsets,
        pitch_signs=pitch_signs,
        pan_zero=zero_joints[0],
    )
