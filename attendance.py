# -*- coding: utf-8 -*-
"""
attendance.py - pencatat absensi (masuk / pulang). Dipisah dari main.py agar bisa diuji tanpa OpenCV.

Aturan (jam diatur di config.py):
  * Scan pertama hari itu sampai jam_terlambat        -> MASUK, status "hadir".
  * Scan pertama setelah jam_terlambat                -> MASUK, status "terlambat".
  * Scan setelah jam_pulang dari orang yang sudah masuk DAN sudah lewat min_durasi_menit sejak
    masuk                                             -> PULANG (sekali per hari).
  * Status di-reset otomatis tiap hari (dibandingkan per tanggal).

Keamanan konkurensi: urutan "sinkron -> putuskan -> tulis" dilindungi kunci antar-proses
(absensi.csv.lock), sehingga `main.py run` dan `dashboard.py` yang berjalan bersamaan tidak
menghasilkan baris ganda.
"""
import csv
import os
import threading
from datetime import datetime

from config import JAM_PULANG, JAM_TERLAMBAT, LOG_PATH, MIN_DURASI_MENIT
from ipclock import InterProcessLock

try:  # opsional: kirim absensi ke Firestore + Google Sheets
    import cloud_sync
except ImportError:
    cloud_sync = None


class AttendanceLogger:
    """Catat absen masuk (hadir/terlambat) dan pulang ke CSV.

    CSV dibaca inkremental (hanya baris baru) dan disinkronkan sebelum setiap keputusan.
    Setiap catatan baru juga diantrekan ke cloud_sync bila aktif.
    """
    HEADER = ["tanggal", "jam", "nama", "id", "skor", "tipe", "status"]
    LOCK_TIMEOUT = 5.0

    def __init__(self, path=LOG_PATH, jam_terlambat=None, jam_pulang=None,
                 min_durasi_menit=None, use_cloud=True):
        self.path = path
        self.jam_terlambat = jam_terlambat or JAM_TERLAMBAT
        self.jam_pulang = jam_pulang or JAM_PULANG
        self.min_durasi = MIN_DURASI_MENIT if min_durasi_menit is None else min_durasi_menit
        self.cloud = cloud_sync if use_cloud else None
        self.state = {}  # nama -> {"tgl": date, "masuk": datetime|None, "status": str, "pulang": datetime|None}
        self._pos = 0    # posisi byte terakhir yang sudah dibaca
        self._lock = threading.RLock()                                    # antar-thread
        self._ipc = InterProcessLock(path + ".lock", timeout=self.LOCK_TIMEOUT)   # antar-proses
        self.sync()
        if self.cloud is not None:   # mulai sinkronisasi otomatis + catch-up absensi.csv
            self.cloud.start()

    def _apply(self, nama, t, tipe, status):
        """Perbarui status harian satu orang. Hari yang berbeda = mulai dari nol (reset harian)."""
        s = self.state.get(nama)
        if s is None or s["tgl"] != t.date():
            if s is not None and t.date() < s["tgl"]:   # baris lama dari hari sebelumnya: abaikan
                return
            s = self.state[nama] = {"tgl": t.date(), "masuk": None, "status": "", "pulang": None}
        if tipe == "pulang":
            s["pulang"] = t
        elif s["masuk"] is None:
            s["masuk"], s["status"] = t, status

    def _ingest(self, r):
        if len(r) < 3 or r[0] == "tanggal":
            return
        try:
            t = datetime.strptime(f"{r[0]} {r[1]}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return
        tipe = r[5] if len(r) > 5 and r[5] else "masuk"
        status = r[6] if len(r) > 6 and r[6] else ("terlambat" if r[1] > self.jam_terlambat else "hadir")
        self._apply(r[2], t, tipe, status)

    def sync(self):
        """Baca baris baru yang ditulis proses lain."""
        with self._lock:
            try:
                size = os.stat(self.path).st_size
            except OSError:
                self.state.clear()
                self._pos = 0
                return
            if size == self._pos:
                return
            if size < self._pos:  # file diganti/dipotong: baca ulang dari awal
                self.state.clear()
                self._pos = 0
            with open(self.path, "rb") as f:
                f.seek(self._pos)
                data = f.read()
            end = data.rfind(b"\n")
            if end < 0:           # baris belum lengkap
                return
            self._pos += end + 1
            for r in csv.reader(data[:end + 1].decode("utf-8", "replace").splitlines()):
                self._ingest(r)

    def decide(self, name, now):
        """Tentukan catatan baru: ("masuk", "hadir"/"terlambat"), ("pulang", "pulang"), atau None."""
        s = self.state.get(name)
        if s is not None and s["tgl"] != now.date():
            s = None                                   # hari baru = reset otomatis
        jam = now.strftime("%H:%M:%S")
        if s is None or s["masuk"] is None:
            return "masuk", ("terlambat" if jam > self.jam_terlambat else "hadir")
        if (s["pulang"] is None and jam >= self.jam_pulang
                and (now - s["masuk"]).total_seconds() >= self.min_durasi * 60):
            return "pulang", "pulang"
        return None

    def log(self, name, pid, score, now, tipe, status):
        """Tulis satu baris. Internal: panggil lewat try_log agar terlindungi kunci antar-proses."""
        with self._lock:
            new = not os.path.exists(self.path)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(self.HEADER)
                w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), name, pid,
                            f"{score:.3f}", tipe, status])
            self._apply(name, now, tipe, status)
            if self.cloud is not None:   # non-blocking, tahan offline
                self.cloud.enqueue(name, pid, score, now, tipe, status)

    def try_log(self, name, pid, score, now):
        """Sinkron -> putuskan -> tulis sebagai SATU langkah atomik antar-proses.

        Kembalikan (tipe, status) bila tercatat, selain itu None. Bila kunci tidak didapat
        (jarang terjadi), catatan dilewati dan dicoba lagi pada verifikasi wajah berikutnya."""
        try:
            with self._ipc, self._lock:
                self.sync()
                d = self.decide(name, now)
                if d is None:
                    return None
                self.log(name, pid, score, now, *d)
                return d
        except TimeoutError as e:
            print(f"[ABSEN] PERINGATAN: {e}; dicoba lagi pada verifikasi berikutnya.")
            return None

    def count_today(self):
        today = datetime.now().date()
        with self._lock:
            return sum(1 for s in self.state.values() if s["tgl"] == today and s["masuk"])