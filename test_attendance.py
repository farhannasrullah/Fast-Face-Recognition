# -*- coding: utf-8 -*-
"""Tes aturan absensi + keamanan konkurensi antar-proses.  Jalankan: python -m unittest -v"""
import contextlib
import csv
import multiprocessing as mp
import os
import tempfile
import unittest
from datetime import datetime

from attendance import AttendanceLogger

D = (2026, 11, 13)
def at(h, m=0, s=0, day=D):
    return datetime(*day, h, m, s)

def make(path, **kw):
    return AttendanceLogger(path, jam_terlambat="09:00:00", jam_pulang="11:00:00",
                            min_durasi_menit=kw.pop("min_durasi_menit", 120), use_cloud=False, **kw)


class DecideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "absensi.csv")
        self.lg = make(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_masuk_tepat_waktu_dan_terlambat(self):
        self.assertEqual(self.lg.try_log("Budi", "1", 0.9, at(8, 30)), ("masuk", "hadir"))
        self.assertEqual(self.lg.try_log("Sari", "2", 0.9, at(9, 0, 1)), ("masuk", "terlambat"))

    def test_scan_ulang_sebelum_jam_pulang_diabaikan(self):
        self.lg.try_log("Budi", "1", 0.9, at(8, 30))
        self.assertIsNone(self.lg.try_log("Budi", "1", 0.9, at(10, 59)))

    def test_pulang_normal_sekali_per_hari(self):
        self.lg.try_log("Budi", "1", 0.9, at(8, 0))
        self.assertEqual(self.lg.try_log("Budi", "1", 0.9, at(11, 5)), ("pulang", "pulang"))
        self.assertIsNone(self.lg.try_log("Budi", "1", 0.9, at(11, 6)))

    def test_REGRESI_orang_baru_datang_11_30_tidak_langsung_pulang(self):
        """Bug sebelumnya: masuk 11:30 lalu verifikasi ulang 60 dtk kemudian dicatat 'pulang'."""
        self.assertEqual(self.lg.try_log("Budi", "1", 0.9, at(11, 30)), ("masuk", "terlambat"))
        self.assertIsNone(self.lg.try_log("Budi", "1", 0.9, at(11, 31)))
        self.assertEqual(self.lg.try_log("Budi", "1", 0.9, at(13, 31)), ("pulang", "pulang"))

    def test_min_durasi_nol_mengembalikan_perilaku_lama(self):
        lg = make(os.path.join(self.tmp.name, "b.csv"), min_durasi_menit=0)
        lg.try_log("Budi", "1", 0.9, at(11, 30))
        self.assertEqual(lg.try_log("Budi", "1", 0.9, at(11, 31)), ("pulang", "pulang"))

    def test_reset_harian(self):
        self.lg.try_log("Budi", "1", 0.9, at(8, 0))
        self.lg.try_log("Budi", "1", 0.9, at(12, 0))
        self.assertEqual(self.lg.try_log("Budi", "1", 0.9, at(8, 5, day=(2026, 11, 14))), ("masuk", "hadir"))

    def test_state_dipulihkan_dari_csv_setelah_restart(self):
        self.lg.try_log("Budi", "1", 0.9, at(8, 0))
        baru = make(self.path)                      # proses baru membaca CSV yang sama
        self.assertIsNone(baru.try_log("Budi", "1", 0.9, at(8, 1)))
        self.assertEqual(baru.try_log("Budi", "1", 0.9, at(11, 5)), ("pulang", "pulang"))

    def test_baris_lama_5_kolom_tetap_terbaca(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["2026-11-13", "09:30:00", "Lama", "7", "0.8"])
        lg = make(self.path)
        self.assertEqual(lg.state["Lama"]["status"], "terlambat")
        self.assertIsNone(lg.try_log("Lama", "7", 0.9, at(9, 40)))


# ---------- konkurensi antar-proses ----------
NAMES = [f"Orang{i:03d}" for i in range(80)]

def _worker(path, barrier, no_lock):
    lg = make(path)
    if no_lock:
        lg._ipc = contextlib.nullcontext()          # simulasi kode LAMA (tanpa kunci antar-proses)
    barrier.wait()
    for n in NAMES:
        lg.try_log(n, "", 0.9, at(8, 0))

def run_race(path, procs=4, no_lock=False):
    ctx = mp.get_context("spawn")
    b = ctx.Barrier(procs)
    ps = [ctx.Process(target=_worker, args=(path, b, no_lock)) for _ in range(procs)]
    [p.start() for p in ps]
    [p.join(120) for p in ps]
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and r[0] != "tanggal"]
    masuk = [r[2] for r in rows if r[5] == "masuk"]
    return len(masuk), len(set(masuk))


class ConcurrencyTests(unittest.TestCase):
    def test_empat_proses_tidak_menghasilkan_baris_ganda(self):
        with tempfile.TemporaryDirectory() as d:
            total, unik = run_race(os.path.join(d, "absensi.csv"))
        self.assertEqual((total, unik), (len(NAMES), len(NAMES)))


if __name__ == "__main__":
    unittest.main(verbosity=2)