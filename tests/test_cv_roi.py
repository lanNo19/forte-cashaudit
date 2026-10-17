"""Per-video geometry (#7) without models or videos: synthetic textures for registration, plain boxes for anchors."""

import cv2
import numpy as np

from cashaudit.cv import roi
from cashaudit.cv.camera import Camera, load_camera


def texture(seed=0, w=464, h=848):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 200, np.uint8)
    for _ in range(400):  # rectangles and circles: plenty of ORB corners
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 40)), y + int(rng.integers(5, 40))), c, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(3, 20)), c, -1)
    return img


def refs_of(img):
    pts, desc = roi.orb_features(img)
    return {"s": (pts, desc)}


def test_registration_undoes_a_shift_and_rotation():
    ref = texture()
    h, w = ref.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.05)
    M[:, 2] += (20, -30)
    moved = cv2.warpAffine(ref, M, (w, h), borderValue=(200, 200, 200))
    H, info = roi.register(moved, refs_of(ref))
    assert H is not None and info["inliers"] >= roi.MIN_INLIERS and not info["mirrored"]
    # a point of the moved video maps back to where it was in the reference
    px = np.array([300.0, 500.0, 1.0])
    moved_px = M @ px
    cam = Camera(to_ref=H)
    rx, ry = cam.ref((moved_px[0] / w, moved_px[1] / h))
    assert abs(rx - 300 / w) < 0.01 and abs(ry - 500 / h) < 0.01


def test_registration_handles_mirror():
    ref = texture(1)
    H, info = roi.register(cv2.flip(ref, 1), refs_of(ref))
    assert H is not None and info["mirrored"]
    rx, ry = Camera(to_ref=H).ref((0.2, 0.5))
    assert abs(rx - 0.8) < 0.01 and abs(ry - 0.5) < 0.01


def test_other_room_does_not_register():
    H, info = roi.register(texture(2), refs_of(texture(3)))
    assert H is None and info["inliers"] < roi.MIN_INLIERS


def test_reference_file_ships_points_not_images():
    refs = roi.load_refs()
    assert len(refs) >= 5
    for pts, desc in refs.values():
        assert pts.shape[1] == 2 and pts.min() >= 0 and pts.max() <= 1
        assert desc.shape[1] == 32 and desc.dtype == np.uint8


def test_registered_camera_keeps_hand_polygons_and_takes_safe_from_anchor():
    base = load_camera("default")
    cam, _ = roi.build_camera(base, np.eye(3), {"safe": (0.5, (0.6, 0.4, 0.7, 0.5))})
    assert cam.room_floor == base.room_floor and cam.door == base.door
    assert cam.zone_of((0.65, 0.55), "B") == "safe"
    assert cam.zone_of((0.15, 0.45), "B") == "door"


def test_unregistered_camera_is_placed_from_handle_and_desk():
    found = {"handle": (0.4, (0.40, 0.30, 0.44, 0.32)), "desk": (0.5, (0.2, 0.42, 0.77, 0.62))}
    cam, info = roi.propose(found)
    assert cam.name == "auto" and "fallback" not in info
    assert cam.near_door((0.35, 0.40))  # just below and left of the handle
    assert not cam.in_room((0.3, 0.25)) and cam.in_room((0.5, 0.7))  # above the glass line / on the floor


def test_missing_anchors_keep_the_base_camera():
    base = load_camera("default")
    cam, info = roi.propose({"desk": (0.5, (0.1, 0.4, 0.6, 0.6))}, base)
    assert cam is base and info["fallback"] == "no handle"
