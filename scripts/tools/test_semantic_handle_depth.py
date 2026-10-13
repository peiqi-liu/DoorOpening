"""CPU geometry checks; run directly without importing IsaacLab or the package."""

import importlib.util
from pathlib import Path
import unittest

import torch


_PATH = Path(__file__).resolve().parents[2] / "source/DoorOpening/utils/camera_utils.py"
_SPEC = importlib.util.spec_from_file_location("standalone_camera_utils", _PATH)
camera = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(camera)


class SemanticHandleDepthTest(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "H": 15, "W": 15, "near_m": 0.1, "far_m": 4.0,
            "intrinsics": torch.tensor([[10.0, 0, 7.0], [0, 10.0, 7.0], [0, 0, 1.0]]),
        }
        self.pose = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1.0]])

    def points(self, pixels):
        # Identity camera points forward along +X, right along -Y, down along -Z.
        return torch.tensor(
            [[d, -(u + 0.5 - 7.0) * d / 10.0, -(v + 0.5 - 7.0) * d / 10.0]
             for u, v, d in pixels], dtype=torch.float32
        ).reshape(1, -1, 3)

    def render(self, background, handle, **kwargs):
        return camera.render_depth_with_handle_layer(
            self.points(background), self.points(handle), self.pose, self.spec, **kwargs
        )

    def test_empty_glass_retains_only_raw_handle_support(self):
        handle = [(u, 7, 1.0) for u in range(5, 10)]
        depth, _, owner = self.render([], handle, edge_drop_m=0.05)
        self.assertEqual(int(torch.isfinite(depth).sum()), len(handle))
        self.assertTrue(torch.equal(torch.isfinite(depth), owner))
        self.assertTrue(torch.all(depth[owner] == 1.0))

    def test_solid_panel_handle_does_not_expand_or_cut_hole(self):
        panel = [(u, v, 1.2) for v in range(15) for u in range(15)]
        handle = [(u, 7, 1.0) for u in range(5, 10)]
        depth, _, owner = self.render(panel, handle, edge_drop_m=0.05)
        self.assertTrue(torch.isfinite(depth).all())
        self.assertEqual(int(owner.sum()), len(handle))
        self.assertTrue(torch.all(depth[~owner] == 1.2))

    def test_edge_drop_cannot_reveal_occluded_handle(self):
        # With blur disabled, adjacent depths produce an edge exactly at the handle ray.
        depth, _, owner = self.render(
            [(7, 7, 0.7), (8, 7, 1.2)], [(7, 7, 1.0)],
            blur_kernel_px=0, edge_drop_m=0.05,
        )
        self.assertFalse(owner.any())
        self.assertTrue(torch.isinf(depth[0, 7, 7]))

    def test_window_hole_unchanged_outside_raw_handle(self):
        # Thick frame surrounds a window wider than the min filter's support.
        frame = [(u, v, 1.2) for v in range(15) for u in range(15)
                 if u < 2 or u > 12 or v < 2 or v > 12]
        baseline, _, _ = self.render(frame, [], edge_drop_m=0.05)
        depth, _, owner = self.render(frame, [(7, 7, 1.0)], edge_drop_m=0.05)
        self.assertEqual(int(owner.sum()), 1)
        self.assertTrue(torch.equal(depth[~owner], baseline[~owner]))
        self.assertTrue(torch.isinf(depth[0, 7, 6]))

    def test_nearer_background_wins_and_ties_stay_background(self):
        depth, _, owner = self.render(
            [(7, 7, 0.8), (10, 7, 1.0)], [(7, 7, 1.0), (10, 7, 1.0)],
            blur_kernel_px=0,
        )
        self.assertFalse(owner.any())
        self.assertAlmostEqual(float(depth[0, 7, 7]), 0.8, places=6)

    def test_handle_shadow_clears_panel_without_expanding_handle_returns(self):
        panel = [(u, v, 1.2) for v in range(15) for u in range(15)]
        depth, _, owner = self.render(
            panel, [(7, 7, 1.0)], blur_kernel_px=0, handle_occlusion_inflate_px=2
        )
        self.assertEqual(int(owner.sum()), 1)
        self.assertEqual(int(torch.isfinite(depth).sum()), 15 * 15 - 24)
        self.assertEqual(float(depth[0, 7, 7]), 1.0)
        self.assertTrue(torch.isinf(depth[0, 7, 6]))
        self.assertAlmostEqual(float(depth[0, 7, 4]), 1.2, places=6)

    def test_occluded_handle_does_not_cut_shadow_hole(self):
        depth, _, owner = self.render(
            [(7, 7, 0.8), (8, 7, 1.2)], [(7, 7, 1.0)],
            blur_kernel_px=0, handle_occlusion_inflate_px=2,
        )
        self.assertFalse(owner.any())
        self.assertAlmostEqual(float(depth[0, 7, 7]), 0.8, places=6)

    def test_empty_batches_and_nonfinite_padding(self):
        empty = torch.empty((2, 0, 3))
        pose = self.pose.repeat(2, 1)
        handle = torch.full((2, 3, 3), float("nan"))
        depth, _, owner = camera.render_depth_with_handle_layer(empty, handle, pose, self.spec)
        self.assertEqual(tuple(depth.shape), (2, 15, 15))
        self.assertTrue(torch.isinf(depth).all())
        self.assertFalse(owner.any())


if __name__ == "__main__":
    unittest.main()
