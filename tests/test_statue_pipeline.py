"""
Comprehensive Unit & Integration Test Suite for:
1. Statue Optimizer & Exporter (pipeline/statue_optimizer.py)
2. Master Statue Pipeline (pipeline/statue_pipeline.py)
3. Statue Database & Automation Settings (playground/database.py)
4. Fast API Endpoints (playground/server.py)
"""

import os
import sys
import unittest
import tempfile
import json
import numpy as np
import trimesh
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.statue_optimizer import (
    clean_and_repair_mesh,
    auto_ground_and_orient,
    add_statue_pedestal,
    decimate_mesh_for_statue,
    segment_statue_parts,
    export_all_statue_variants,
    STATUE_PALETTE
)
from playground import database

class TestStatueOptimizer(unittest.TestCase):
    def setUp(self):
        torso = trimesh.creation.cylinder(radius=0.3, height=0.8, sections=24)
        torso.apply_translation([0, 0.4, 0])
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.25)
        head.apply_translation([0, 1.0, 0])
        legs = trimesh.creation.cylinder(radius=0.2, height=0.6, sections=24)
        legs.apply_translation([0, -0.3, 0])
        self.sample_mesh = trimesh.util.concatenate([torso, head, legs])

    def test_clean_and_repair_mesh(self):
        cleaned = clean_and_repair_mesh(self.sample_mesh)
        self.assertIsNotNone(cleaned)
        self.assertGreater(len(cleaned.vertices), 0)
        self.assertGreater(len(cleaned.faces), 0)

    def test_auto_ground_and_orient(self):
        grounded = auto_ground_and_orient(
            self.sample_mesh,
            target_height=1.8,
            flatten_bottom=True
        )
        v = grounded.vertices
        min_y = v[:, 1].min()
        max_y = v[:, 1].max()
        center_x = (v[:, 0].max() + v[:, 0].min()) / 2.0
        center_z = (v[:, 2].max() + v[:, 2].min()) / 2.0

        self.assertAlmostEqual(min_y, 0.0, delta=0.02)
        self.assertAlmostEqual(max_y, 1.8, delta=0.05)
        self.assertAlmostEqual(center_x, 0.0, delta=0.02)
        self.assertAlmostEqual(center_z, 0.0, delta=0.02)

    def test_add_statue_pedestal(self):
        mesh_ped, ped = add_statue_pedestal(self.sample_mesh, shape="round", pedestal_height=0.06)
        self.assertIsNotNone(ped)
        self.assertGreater(len(mesh_ped.vertices), len(self.sample_mesh.vertices))

        mesh_sq, ped_sq = add_statue_pedestal(self.sample_mesh, shape="square", pedestal_height=0.06)
        self.assertIsNotNone(ped_sq)

        mesh_none, ped_none = add_statue_pedestal(self.sample_mesh, shape="none")
        self.assertIsNone(ped_none)

    def test_decimate_mesh_for_statue(self):
        orig_faces = len(self.sample_mesh.faces)
        target = min(200, orig_faces - 50)
        decimated = decimate_mesh_for_statue(self.sample_mesh, target_faces=target)
        self.assertLessEqual(len(decimated.faces), orig_faces)
        self.assertGreater(len(decimated.faces), 0)

    def test_segment_statue_parts(self):
        seg = segment_statue_parts(self.sample_mesh, has_pedestal=False)
        self.assertIn("submeshes", seg)
        self.assertIn("part_info", seg)
        self.assertIn("vertex_colors", seg)
        self.assertGreater(seg["num_parts_detected"], 0)
        self.assertEqual(len(seg["vertex_colors"]), len(self.sample_mesh.vertices))

    def test_export_all_statue_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            seg = segment_statue_parts(self.sample_mesh, has_pedestal=True)
            exported = export_all_statue_variants(
                base_mesh=self.sample_mesh,
                segmented_data=seg,
                output_dir=out_dir,
                stem="test_statue"
            )
            self.assertIn("plaster_glb", exported)
            self.assertIn("segmented_glb", exported)
            self.assertIn("id_colored_glb", exported)
            self.assertIn("manifest_json", exported)
            self.assertIn("package_zip", exported)

            for key, fpath in exported.items():
                p = Path(fpath)
                self.assertTrue(p.exists(), f"File {key} does not exist at {p}")
                self.assertGreater(p.stat().st_size, 0, f"File {key} is empty")


class TestStatueDatabase(unittest.TestCase):
    def setUp(self):
        database.init_db()

    def test_create_and_query_statue_job(self):
        test_id = "test_unit_job_001"
        job = database.create_statue_job(
            job_id=test_id,
            title="Unit Test Statue",
            input_filename="test.png",
            input_file_path="/tmp/test.png",
            generator_type="trellis",
            target_faces=25000,
            pedestal_shape="round",
            metadata={"test_key": "test_val"}
        )
        self.assertEqual(job["id"], test_id)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["target_faces"], 25000)

        updated = database.update_statue_job(
            test_id,
            status="completed",
            duration_sec=12.5,
            num_vertices=1000,
            num_faces=2000
        )
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["duration_sec"], 12.5)

        database.delete_statue_job(test_id)
        self.assertIsNone(database.get_statue_job(test_id))

    def test_automation_config(self):
        cfg = database.get_automation_config()
        self.assertIn("enabled", cfg)
        self.assertIn("input_folder", cfg)
        self.assertIn("webhook_url", cfg)

        database.update_automation_config({"webhook_url": "https://test.example.com/hook"})
        new_cfg = database.get_automation_config()
        self.assertEqual(new_cfg["webhook_url"], "https://test.example.com/hook")


class TestStatueAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playground.server import app
        from fastapi.testclient import TestClient
        cls.client = TestClient(app)

    def test_statue_html_page(self):
        resp = self.client.get("/statue")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("UniRig", resp.text)
        self.assertIn("Statue Studio", resp.text)
        self.assertIn("canvas-container", resp.text)

    def test_palette_api(self):
        resp = self.client.get("/api/statue/palette")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("palette", data)
        self.assertIn("material_presets", data)
        self.assertGreater(len(data["palette"]), 0)

    def test_automation_status_and_config_api(self):
        resp = self.client.get("/api/statue/automation/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("is_running", data)
        self.assertIn("config", data)

        cfg_resp = self.client.post("/api/statue/automation/config", json={"poll_interval_sec": 7})
        self.assertEqual(cfg_resp.status_code, 200)
        self.assertEqual(cfg_resp.json()["poll_interval_sec"], 7)

    def test_webhook_test_api(self):
        resp_bad = self.client.post("/api/statue/automation/test-webhook", json={})
        self.assertEqual(resp_bad.status_code, 400)


if __name__ == "__main__":
    unittest.main()
