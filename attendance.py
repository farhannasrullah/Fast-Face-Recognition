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

parse_row() adalah SATU-SATUNYA pembaca format baris absensi.csv; dipakai juga oleh main.py
(export), dashboard.py, dan cloud_sync.py (sebelumnya logika ini disalin di 4 tempat).
"""
import csv
import os
import threading
from datetime import date, datetime

from config import JAM_PULANG, JAM_TERLAMBAT, LOG_PATH, MIN_DURASI_MENIT
from ipclock import InterProcessLock

try:  # opsional: kirim absensi ke Firestore + Google Sheets
    import cloud_sync
except ImportError:
    cloud_sync = None

HEADER = ["tanggal", "jam", "nama", "id", "skor", "tipe", "status"]


def parse_row(r, jam_terlambat=JAM_TERLAMBAT):
    """Satu baris CSV -> dict, atau None untuk header/baris rusak.

    Baris lama 5 kolom tetap terbaca: tipe dianggap "masuk", status dihitung dari jamnya."""
    if len(r) < 3 or r[0] == "tanggal":
        return None
    try:
        t = datetime.strptime(f"{r[0]} {r[1]}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    tipe = r[5] if len(r) > 5 and r[5] else "masuk"
    status = r[6] if len(r) > 6 and r[6] else ("terlambat" if r[1] > jam_terlambat else "hadir")
    try:
        skor = float(r[4]) if len(r) > 4 and r[4] else 0.0
    except ValueError:
        skor = 0.0
    return {"tanggal": r[0], "jam": r[1], "nama": r[2], "id": r[3] if len(r) > 3 else "",
            "skor": skor, "tipe": tipe, "status": status, "t": t}


def read_log(path=LOG_PATH, jam_terlambat=JAM_TERLAMBAT):
    """Semua baris valid absensi.csv sebagai list dict (kosong bila file belum ada)."""
    out = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                d = parse_row(r, jam_terlambat)
                if d is not None:
                    out.append(d)
    except FileNotFoundError:
        pass
    return out


def log_lock(path=LOG_PATH, timeout=5.0):
    return InterProcessLock(path + ".lock", timeout=timeout)


def purge_before(cutoff, path=LOG_PATH, dry_run=False):
    """Hapus catatan dengan tanggal < cutoff (date) - retensi data (UU PDP Pasal 43).

    Ditulis ulang secara atomik di bawah kunci antar-proses. Mengembalikan (dihapus, tersisa)."""
    cut = cutoff.strftime("%Y-%m-%d") if isinstance(cutoff, (date, datetime)) else str(cutoff)
    with log_lock(path, timeout=30):
        if not os.path.exists(path):
            return 0, 0
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        keep, removed = [HEADER], 0
        for r in rows:
            if len(r) >= 1 and r[0] == "tanggal":
                continue
            if parse_row(r) is not None and r[0] < cut:
                removed += 1
            else:
                keep.append(r)
        if dry_run or not removed:
            return removed, len(keep) - 1
        tmp = path + ".retensi.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(keep)
        for i in range(20):                  # Windows: tunggu pembaca lain menutup file
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if i == 19:
                    raise
                threading.Event().wait(0.1)
    return removed, len(keep) - 1


class AttendanceLogger:
    """Catat absen masuk (hadir/terlambat) dan pulang ke CSV.

    CSV dibaca inkremental (hanya baris baru) dan disinkronkan sebelum setiap keputusan.
    Setiap catatan baru juga diantrekan ke cloud_sync bila aktif.
    """
    HEADER = HEADER
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
        self._ident = None   # (st_dev, st_ino): file diganti (mis. retensi) -> baca ulang dari awal
        self._lock = threading.RLock()                                    # antar-thread
        self._ipc = log_lock(path, timeout=self.LOCK_TIMEOUT)             # antar-proses
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
        d = parse_row(r, self.jam_terlambat)
        if d is not None:
            self._apply(d["nama"], d["t"], d["tipe"], d["status"])

    def sync(self):
        """Baca baris baru yang ditulis proses lain."""
        with self._lock:
            try:
                st = os.stat(self.path)
            except OSError:
                self.state.clear()
                self._pos, self._ident = 0, None
                return
            ident = (st.st_dev, st.st_ino)
            if ident != self._ident or st.st_size < self._pos:   # file diganti/dipotong: baca ulang
                self.state.clear()
                self._pos, self._ident = 0, ident
            if st.st_size == self._pos:
                return
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
            try:
                new = os.path.getsize(self.path) == 0     # file belum ada ATAU kosong -> tulis header
            except OSError:
                new = True
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
