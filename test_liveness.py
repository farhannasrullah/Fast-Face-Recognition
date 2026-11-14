# -*- coding: utf-8 -*-
"""Tes mesin keputusan liveness pasif (tanpa kamera). Jalankan: python -m unittest -v test_liveness"""
import unittest

import numpy as np

import liveness as lv


class StubPassive:
    """Pengganti model: mengembalikan probabilitas [kertas, asli, layar] yang kita atur."""
    def __init__(self, real):
        self.real, self.calls = real, 0

    def probs(self, frame, box):
        self.calls += 1
        r = self.real
        return np.array([(1 - r) / 2, r, (1 - r) / 2])


def make_engine(real=0.9):
    eng = lv.LivenessEngine(mode="pasif")
    eng.passive = StubPassive(real)
    eng.thresh, eng.spoof_thresh, eng.min_frames, eng.window, eng.min_px = 0.80, 0.50, 3, 8, 80
    return eng


FRAME = np.zeros((720, 1280, 3), np.uint8)
BOX = (500, 200, 200, 240)          # wajah 200 px di tengah frame


def feed(eng, st, n, real=None):
    if real is not None:
        eng.passive.real = real
    evs = []
    for _ in range(n):
        evs.append(eng.observe(st, FRAME, BOX, None, 0.0))
    return evs


class DecisionTests(unittest.TestCase):
    def test_asli_lolos_cepat(self):
        eng = make_engine(0.92)
        st = eng.new_state()
        feed(eng, st, 3)
        self.assertTrue(st.passed)

    def test_zona_ragu_tidak_dicap_palsu(self):
        eng = make_engine(0.65)                 # antara 0.50 dan 0.80
        st = eng.new_state()
        evs = feed(eng, st, 20)
        self.assertFalse(st.spoof, "wajah di zona abu-abu tidak boleh dicap PALSU")
        self.assertTrue(st.uncertain)
        self.assertFalse(st.passed, "syarat lolos tidak boleh dilonggarkan")
        self.assertNotIn("spoof", evs)
        self.assertIn("Belum yakin", st.prompt())

    def test_ragu_lalu_membaik_lolos(self):
        eng = make_engine(0.65)
        st = eng.new_state()
        feed(eng, st, 12)
        feed(eng, st, 10, real=0.95)            # pengguna memperbaiki cahaya / posisi
        self.assertTrue(st.passed)
        self.assertFalse(st.spoof or st.uncertain)

    def test_palsu_kuat_dicap_dan_dilaporkan_sekali(self):
        eng = make_engine(0.15)
        st = eng.new_state()
        evs = feed(eng, st, 20)
        self.assertTrue(st.spoof)
        self.assertEqual(evs.count("spoof"), 1)
        self.assertFalse(st.passed)

    def test_belum_cap_palsu_sebelum_jendela_penuh(self):
        eng = make_engine(0.15)
        st = eng.new_state()
        feed(eng, st, 7)
        self.assertFalse(st.spoof)
        feed(eng, st, 1)
        self.assertTrue(st.spoof)

    def test_pulih_dari_cap_palsu_bila_skor_naik(self):
        eng = make_engine(0.15)
        st = eng.new_state()
        feed(eng, st, 10)
        self.assertTrue(st.spoof)
        feed(eng, st, 10, real=0.95)
        self.assertTrue(st.passed)

    def test_foto_skor_rendah_tidak_pernah_lolos(self):
        for r in (0.05, 0.3, 0.6, 0.79):
            eng = make_engine(r)
            st = eng.new_state()
            feed(eng, st, 50)
            self.assertFalse(st.passed, f"skor {r} tidak boleh lolos")


class GateTests(unittest.TestCase):
    def test_wajah_kecil_tidak_dinilai(self):
        eng = make_engine(0.1)
        st = eng.new_state()
        for _ in range(30):
            eng.observe(st, FRAME, (600, 300, 50, 60), None, 0.0)
        self.assertEqual(eng.passive.calls, 0)
        self.assertFalse(st.spoof, "wajah kecil tidak boleh dicap palsu")
        self.assertEqual(st.prompt(), "Mendekat ke kamera")

    def test_wajah_terpotong_tepi_tidak_dinilai(self):
        eng = make_engine(0.1)
        st = eng.new_state()
        for _ in range(30):
            eng.observe(st, FRAME, (1150, 200, 200, 240), None, 0.0)      # keluar dari tepi kanan
        self.assertEqual(eng.passive.calls, 0)
        self.assertFalse(st.spoof)
        self.assertIn("di dalam layar", st.prompt())

    def test_gate_hilang_setelah_wajah_layak(self):
        eng = make_engine(0.92)
        st = eng.new_state()
        eng.observe(st, FRAME, (600, 300, 50, 60), None, 0.0)
        self.assertTrue(st.gate)
        feed(eng, st, 3)
        self.assertFalse(st.gate)
        self.assertTrue(st.passed)

    def test_min_px_nol_menonaktifkan_gate_ukuran(self):
        eng = make_engine(0.92)
        eng.min_px = 0
        st = eng.new_state()
        for _ in range(3):
            eng.observe(st, FRAME, (600, 300, 50, 60), None, 0.0)
        self.assertTrue(st.passed)


class RealModelSmoke(unittest.TestCase):
    def test_model_asli_berjalan_pada_gambar_acak(self):
        eng = lv.LivenessEngine(mode="pasif")
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        st = eng.new_state()
        for _ in range(9):
            eng.observe(st, img, BOX, None, 0.0)
        self.assertIsNotNone(st.last_probs)
        self.assertAlmostEqual(float(st.last_probs.sum()), 1.0, places=5)
        self.assertEqual(st.last_w, 200.0)


if __name__ == "__main__":
    unittest.main()