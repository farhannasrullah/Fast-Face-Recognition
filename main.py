#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Fast Face Recognition System - Absensi Sederhana (versi optimal)
Berjalan penuh di CPU (0 VRAM) dan hemat RAM. Model: YuNet (deteksi) + SFace (embedding).

Perintah:
  python main.py register --nama "Budi" --id 2023001   # daftarkan wajah (5-10 sampel)
  python main.py run [--jarak dekat|sedang|jauh]       # mode absen otomatis (Q / ESC = keluar)
  python main.py export --format xlsx                  # ekspor rekap ke CSV atau Excel
  python main.py list                                  # daftar orang terdaftar
  python main.py delete --nama "Budi"                  # hapus data satu orang
  python main.py download-models                       # unduh model YuNet + SFace

Aturan absensi (atur di bagian KONFIGURASI):
  * Scan pertama hari itu sampai JAM_TERLAMBAT (09:00)  -> MASUK, status "hadir".
  * Scan pertama setelah JAM_TERLAMBAT                  -> MASUK, status "terlambat".
  * Scan setelah JAM_PULANG (11:00) dari orang yang sudah masuk (hadir/terlambat) dan sudah
    lewat MIN_DURASI_MENIT sejak masuk                  -> PULANG (sekali per hari).
  * Semua status di-reset otomatis setiap hari (dibandingkan per tanggal, tanpa restart).

Format absensi.csv: tanggal, jam, nama, id, skor, tipe, status
  (baris lama 5 kolom tetap terbaca: dianggap tipe "masuk", status dihitung dari jamnya).
"""
import argparse
import csv
import gc
import os
import sys
import threading
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

try:  # opsional: untuk menampilkan estimasi RAM
    import psutil
    _PROC = psutil.Process()
except ImportError:
    _PROC = None

# ============================ KONFIGURASI ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "faces_db.npz")   # embedding + nama + id
LOG_PATH = os.path.join(BASE_DIR, "absensi.csv")   # catatan absen (dibaca dashboard.py)
EXPORT_DIR = os.path.join(BASE_DIR, "export")

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
MODEL_URLS = {
    YUNET_FILE: [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_FILE,
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/" + YUNET_FILE,
    ],
    SFACE_FILE: [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/" + SFACE_FILE,
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/" + SFACE_FILE,
    ],
}
MODEL_MIN_BYTES = {YUNET_FILE: 100_000, SFACE_FILE: 10_000_000}  # deteksi file rusak/tidak lengkap

# Kamera
CAM_INDEX = 0
CAM_FPS = 30
CAM_MJPG = True            # MJPG meringankan USB dan menaikkan FPS pada banyak webcam

# Profil jarak (pilih lewat --jarak). Jarak pengenalan ditentukan oleh jumlah piksel wajah:
# wajah butuh >= ~50 px lebar (di frame kamera asli) agar dikenali andal. Untuk webcam 720p
# umum, lebar wajah (px) ~ 177 / jarak (m); untuk 480p kira-kira separuhnya.
# Format: (lebar kamera, tinggi kamera, lebar frame deteksi, lebar wajah minimum px, skor deteksi)
PROFIL = {
    "dekat":  (640, 480, 320, 80, 0.85),    # paling ringan, +-1 m
    "sedang": (1280, 720, 480, 50, 0.80),   # +-3 m
    "jauh":   (1280, 720, 640, 40, 0.70),   # +-4 m, akurasi menurun, butuh CPU lebih kuat
}
PROFIL_AKTIF = "sedang"

# Performa
DETECT_EVERY = 3           # deteksi tiap N frame; frame lain dilacak (optical flow)
CV_THREADS = 2             # batasi thread OpenCV agar CPU tidak penuh
GC_EVERY_FRAMES = 900      # bersihkan memori berkala

# Deteksi dan pengenalan
NMS_THRESH = 0.3
MATCH_THRESH = 0.40        # ambang skor orang (rata-rata 2 sampel terbaik; resmi SFace: 0.363)
MATCH_MARGIN = 0.05        # selisih minimum skor orang terbaik vs kedua (0 = nonaktif)
CONFIRM_MATCHES = 2        # kecocokan beruntun yang dibutuhkan sebelum dianggap dikenali
UNKNOWN_RETRY_DETIK = 0.4  # jeda ekstraksi embedding ulang untuk wajah yang belum dikenali
RECHECK_DETIK = 60         # verifikasi ulang wajah yang sudah dikenali (0 = tidak pernah)
TRACK_IOU_MIN = 0.25
TRACK_MAX_MISSED = 4       # siklus deteksi tanpa terdeteksi sebelum track dibuang
WEAK_TOLERANCE = 2         # skor buruk beruntun sebelum wajah yang sudah dikenali dianggap Unknown

# Absensi
JAM_TERLAMBAT = "09:00:00"  # masuk setelah jam ini = terlambat (format HH:MM:SS)
JAM_PULANG = "11:00:00"     # scan setelah jam ini (dan sudah masuk) = absen pulang
MIN_DURASI_MENIT = 30       # jarak minimum masuk -> pulang (mencegah masuk jam 11:05 langsung pulang)

# Registrasi
REG_SAMPLES = 8
REG_INTERVAL = 0.7         # detik antar pengambilan sampel
REG_MIN_FACE_PX = 90       # wajah harus cukup besar saat registrasi (kualitas embedding)
REG_PROMPTS = ["Hadap lurus ke kamera", "Tengok sedikit ke kiri", "Tengok sedikit ke kanan",
               "Dongak sedikit", "Tunduk sedikit", "Hadap lurus, rileks", "Senyum tipis",
               "Hadap lurus ke kamera", "Tengok sedikit ke kiri", "Tengok sedikit ke kanan"]

WIN_RUN = "Absensi Wajah (Q / ESC = keluar)"
WIN_REG = "Registrasi Wajah (Q / ESC = batal)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
# =====================================================================


def apply_profile(name):
    """Terapkan profil jarak ke konfigurasi global (panggil sebelum kamera/model dibuat)."""
    global CAM_WIDTH, CAM_HEIGHT, PROC_WIDTH, MIN_FACE_PX, DETECT_SCORE
    CAM_WIDTH, CAM_HEIGHT, PROC_WIDTH, MIN_FACE_PX, DETECT_SCORE = PROFIL[name]


apply_profile(PROFIL_AKTIF)


class CameraError(RuntimeError):
    pass


# ----------------------------- Model -----------------------------
def _download(urls, dest, min_bytes):
    err = None
    for url in urls:
        tmp = dest + ".part"
        try:
            print(f"  mengunduh: {url}")
            with urllib.request.urlopen(url, timeout=30) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(tmp) < min_bytes:
                raise ValueError("ukuran file terlalu kecil (bukan model yang valid)")
            os.replace(tmp, dest)
            return True
        except Exception as e:
            err = e
            if os.path.exists(tmp):
                os.remove(tmp)
    print(f"  gagal: {err}")
    return False


def ensure_models():
    """Pastikan kedua model ada; unduh otomatis jika belum."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = []
    for name, urls in MODEL_URLS.items():
        p = os.path.join(MODEL_DIR, name)
        if not (os.path.exists(p) and os.path.getsize(p) >= MODEL_MIN_BYTES[name]):
            print(f"Model {name} belum ada, mengunduh...")
            if not _download(urls, p, MODEL_MIN_BYTES[name]):
                raise RuntimeError(f"Model '{name}' tidak ditemukan dan gagal diunduh.\n"
                                   "Unduh manual dari salah satu tautan ini:\n  " + "\n  ".join(urls) +
                                   f"\nlalu simpan di folder: {MODEL_DIR}")
        paths.append(p)
    return paths


# ----------------------------- Kamera -----------------------------
def _cam_backend():
    if os.name == "nt":
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class Camera:
    """Baca kamera di thread terpisah dan hanya simpan frame terbaru (tanpa lag buffer)."""

    def __init__(self, index=CAM_INDEX):
        backend = _cam_backend()
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:   # cadangan: biarkan OpenCV memilih
            self.cap.release()
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            self.cap.release()
            raise CameraError(f"Kamera index {index} tidak terdeteksi. Periksa sambungan/izin kamera "
                              "atau ubah CAM_INDEX (opsi --kamera).")
        if CAM_MJPG:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.failed = False
        self._frame = None
        self._new = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        first = self.read(timeout=3.0)
        if first is None:
            self.close()
            raise CameraError("Kamera terbuka tetapi tidak mengirim gambar (dipakai aplikasi lain?).")
        h, w = first.shape[:2]
        print(f"Kamera aktif: {w}x{h}" + (f" (diminta {CAM_WIDTH}x{CAM_HEIGHT}; webcam tidak mendukung, "
                                           "jarak pengenalan jadi lebih pendek)" if w < CAM_WIDTH else ""))

    def _loop(self):
        fails = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 50:
                    self.failed = True
                    self._new.set()
                    return
                time.sleep(0.02)
                continue
            fails = 0
            self._frame = frame
            self._new.set()

    def read(self, timeout=1.0):
        """Kembalikan frame terbaru yang belum dibaca, atau None jika belum ada."""
        if not self._new.wait(timeout):
            return None
        self._new.clear()
        return None if self.failed else self._frame

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


def next_frame(cam):
    """Ambil frame berikutnya; lempar error bila kamera terputus."""
    frame = cam.read()
    if frame is None and cam.failed:
        raise CameraError("Kamera terputus.")
    return frame


# ----------------------------- Engine wajah -----------------------------
class FaceEngine:
    """Pembungkus YuNet (deteksi) dan SFace (embedding), keduanya di CPU."""

    def __init__(self, yunet_path, sface_path):
        try:
            self.det = cv2.FaceDetectorYN.create(
                yunet_path, "", (PROC_WIDTH, PROC_WIDTH), DETECT_SCORE, NMS_THRESH, 20,
                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
            self.rec = cv2.FaceRecognizerSF.create(
                sface_path, "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
        except (cv2.error, AttributeError) as e:
            raise RuntimeError("Gagal memuat model. Pastikan opencv-python >= 4.8 dan file model utuh "
                               f"(hapus folder models/ lalu unduh ulang). Detail: {e}")
        self._size = None

    def detect(self, small):
        """Deteksi wajah pada frame kecil; kembalikan list baris [x,y,w,h, 5 landmark, skor] (tanpa penyaringan ukuran)."""
        h, w = small.shape[:2]
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self.det.detect(small)
        return [] if faces is None else list(faces)

    def embed(self, full, row, scale):
        """Embedding 128-d ternormalisasi. Wajah di-crop dari frame resolusi penuh (lebih tajam)."""
        r = row.copy()
        r[0:14:2] *= scale[0]
        r[1:14:2] *= scale[1]
        feat = self.rec.feature(self.rec.alignCrop(full, r)).reshape(-1).astype(np.float32)
        n = np.linalg.norm(feat)
        return feat / n if n > 0 else feat


# ----------------------------- Database embedding -----------------------------
class Database:
    """Embedding di RAM sebagai matriks float32 (N,128).

    Pencocokan dilakukan per orang: skor orang = rata-rata 2 sampel paling mirip (1 sampel bila
    hanya ada satu). Seluruhnya vektor (satu perkalian matriks + reduceat), tanpa loop Python.
    """

    def __init__(self, path=DB_PATH):
        self.path = path
        self.names, self.ids = [], []
        self.embs = np.zeros((0, 128), np.float32)
        self._sig = None
        self._load()
        self._reindex()

    # ---- deteksi perubahan file ----
    def _sig_now(self):
        try:
            st = os.stat(self.path)
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def changed(self):
        """True jika file database berubah (misalnya diedit lewat dashboard)."""
        return self._sig_now() != self._sig

    def reload(self):
        self._load()
        self._reindex()

    def _load(self):
        self._sig = self._sig_now()
        if self._sig is None:
            return
        try:
            with np.load(self.path, allow_pickle=False) as d:  # 'with' agar file langsung ditutup
                names = [str(x) for x in d["names"]]
                embs = np.ascontiguousarray(d["embs"], dtype=np.float32)
                ids = [str(x) for x in d["ids"]] if "ids" in d.files else [""] * len(names)
            if not (len(names) == len(ids) == len(embs)):
                raise ValueError("jumlah nama, id, dan embedding tidak sama")
        except Exception as e:
            raise RuntimeError(f"Database wajah rusak/tidak terbaca ({self.path}): {e}")
        self.names, self.ids, self.embs = names, ids, embs

    def _reindex(self):
        """Susun indeks per orang (urutan sampel dikelompokkan) untuk pencocokan vektor."""
        if not self.names:
            self.uniq, self.uids = [], []
            self._counts = np.zeros(0, np.intp)
            self._starts = np.zeros(0, np.intp)
            self._es = np.zeros((0, 128), np.float32)
            return
        uniq, inv = np.unique(np.array(self.names, dtype=str), return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=len(uniq)).astype(np.intp)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.intp)
        ids = np.array(self.ids, dtype=str)
        self._counts, self._starts = counts, starts
        self._es = np.ascontiguousarray(self.embs[order])
        self.uniq = [str(x) for x in uniq]
        self.uids = [str(x) for x in ids[order[starts]]]

    def save(self):
        tmp = self.path + ".tmp.npz"  # tulis ke file sementara agar database tidak rusak saat crash
        np.savez_compressed(tmp, names=np.array(self.names, dtype=str),
                            ids=np.array(self.ids, dtype=str), embs=self.embs)
        os.replace(tmp, self.path)
        self._sig = self._sig_now()

    # ---- data ----
    def is_empty(self):
        return len(self.names) == 0

    def people(self):
        """{nama: [jumlah_sampel, id]}"""
        out = {}
        for n, i in zip(self.names, self.ids):
            out.setdefault(n, [0, i])[0] += 1
        return out

    def add(self, name, pid, samples):
        if pid:  # perbarui id untuk sampel lama dengan nama yang sama
            self.ids = [pid if n == name else i for n, i in zip(self.names, self.ids)]
        else:
            pid = next((i for n, i in zip(self.names, self.ids) if n == name), "")
        self.names += [name] * len(samples)
        self.ids += [pid] * len(samples)
        self.embs = np.vstack([self.embs, np.asarray(samples, np.float32)])
        self._reindex()

    def remove(self, name):
        keep = np.array([n != name for n in self.names], dtype=bool)
        removed = int(len(keep) - keep.sum())
        self.names = [n for n, k in zip(self.names, keep) if k]
        self.ids = [i for i, k in zip(self.ids, keep) if k]
        self.embs = np.ascontiguousarray(self.embs[keep])
        self._reindex()
        return removed

    def update(self, name, new_name=None, pid=None):
        """Ubah nama dan/atau ID semua sampel milik `name` (pid None = ID tidak diubah)."""
        if pid is not None:
            self.ids = [pid if n == name else i for n, i in zip(self.names, self.ids)]
        if new_name:
            self.names = [new_name if n == name else n for n in self.names]
        self._reindex()

    # ---- pencocokan ----
    def person_scores(self, emb):
        """Skor tiap orang (urutan sama dengan self.uniq)."""
        s = self._es @ emb
        m1 = np.maximum.reduceat(s, self._starts)
        if not (self._counts > 1).any():
            return m1
        s2 = np.where(s >= np.repeat(m1, self._counts), -1.0, s)   # buang sampel terbaik tiap orang
        m2 = np.maximum.reduceat(s2, self._starts)
        m2 = np.where(m2 < -0.5, m1, m2)                           # orang dengan 1 sampel / sampel kembar
        return np.where(self._counts > 1, (m1 + m2) / 2, m1)

    def best(self, emb):
        """Orang paling mirip tanpa aturan margin: (nama, id, skor, margin ke orang kedua)."""
        if not self.uniq:
            return None, "", 0.0, 0.0
        sc = self.person_scores(emb)
        i = int(sc.argmax())
        best = float(sc[i])
        margin = best - float(np.partition(sc, -2)[-2]) if len(sc) > 1 else 1.0
        return self.uniq[i], self.uids[i], best, margin

    def match(self, emb):
        """(nama, id, skor). nama None bila database kosong atau hasil ambigu (margin terlalu kecil)."""
        name, pid, score, margin = self.best(emb)
        if name is not None and score >= MATCH_THRESH and margin < MATCH_MARGIN:
            return None, "", score
        return name, pid, score


# ----------------------------- Pencatat absensi -----------------------------
class AttendanceLogger:
    """Catat absen masuk (hadir/terlambat) dan pulang ke CSV. Status di-reset otomatis tiap hari.

    CSV dibaca inkremental (hanya baris baru) dan disinkronkan sebelum setiap keputusan, sehingga
    aman dipakai bersama proses lain (dashboard, main.py kedua) tanpa catatan ganda.
    """
    HEADER = ["tanggal", "jam", "nama", "id", "skor", "tipe", "status"]

    def __init__(self, path=LOG_PATH):
        self.path = path
        self.state = {}  # nama -> {"tgl": date, "masuk": datetime|None, "status": str, "pulang": datetime|None}
        self._pos = 0    # posisi byte terakhir yang sudah dibaca
        self._lock = threading.RLock()
        self.sync()

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
        status = r[6] if len(r) > 6 and r[6] else ("terlambat" if r[1] > JAM_TERLAMBAT else "hadir")
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
            return "masuk", ("terlambat" if jam > JAM_TERLAMBAT else "hadir")
        if (s["pulang"] is None and jam >= JAM_PULANG
                and (now - s["masuk"]).total_seconds() >= MIN_DURASI_MENIT * 60):
            return "pulang", "pulang"
        return None

    def log(self, name, pid, score, now, tipe, status):
        with self._lock:
            new = not os.path.exists(self.path)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(self.HEADER)
                w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), name, pid,
                            f"{score:.3f}", tipe, status])
            self._apply(name, now, tipe, status)

    def try_log(self, name, pid, score, now):
        """Sinkron -> putuskan -> tulis, satu langkah. Kembalikan (tipe, status) bila tercatat, selain itu None."""
        with self._lock:
            self.sync()
            d = self.decide(name, now)
            if d is None:
                return None
            self.log(name, pid, score, now, *d)
            return d

    def count_today(self):
        today = datetime.now().date()
        with self._lock:
            return sum(1 for s in self.state.values() if s["tgl"] == today and s["masuk"])


# ----------------------------- Pelacak wajah -----------------------------
LK_PARAMS = dict(winSize=(21, 21), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def _iou(a, b):
    """IoU antar dua set kotak (x,y,w,h) -> matriks (len(a), len(b))."""
    iw = np.minimum((a[:, 0] + a[:, 2])[:, None], (b[:, 0] + b[:, 2])[None]) - \
        np.maximum(a[:, 0][:, None], b[:, 0][None])
    ih = np.minimum((a[:, 1] + a[:, 3])[:, None], (b[:, 1] + b[:, 3])[None]) - \
        np.maximum(a[:, 1][:, None], b[:, 1][None])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-6)


class Track:
    def __init__(self, tid, row):
        self.id = tid
        self.name = None      # nama terkonfirmasi
        self.pid = ""
        self.cand = None      # kandidat yang belum terkonfirmasi
        self.cand_n = 0
        self.weak = 0         # skor buruk beruntun pada wajah yang sudah dikenali
        self.score = 0.0
        self.next_embed = 0.0  # waktu paling cepat untuk ekstraksi embedding berikutnya
        self.update(row)

    def update(self, row):
        self.row = row                                   # baris deteksi segar
        self.box = row[:4].astype(np.float32)
        self.lm = row[4:14].reshape(5, 2).astype(np.float32)
        self.missed = 0
        self.fresh = True


class Tracker:
    """Pelacak ringan: optical flow pada 5 landmark di antara frame deteksi + pencocokan IoU."""

    def __init__(self):
        self.tracks = {}
        self._next = 1

    def advance(self, prev_gray, gray):
        """Geser kotak tiap wajah mengikuti gerakan landmark (tanpa menjalankan detektor)."""
        if not self.tracks:
            return
        tl = list(self.tracks.values())
        p0 = np.concatenate([t.lm for t in tl]).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
        if p1 is None:
            return
        p0, p1, ok = p0.reshape(-1, 2), p1.reshape(-1, 2), st.reshape(-1) == 1
        for i, t in enumerate(tl):
            s = slice(i * 5, i * 5 + 5)
            if ok[s].sum() >= 3:
                d = np.median(p1[s][ok[s]] - p0[s][ok[s]], axis=0)
                t.box[:2] += d
                t.lm += d

    def update(self, faces):
        """Cocokkan deteksi baru dengan track; kembalikan track yang mendapat deteksi segar."""
        tl = list(self.tracks.values())
        for t in tl:
            t.fresh = False
        used_t, used_d = set(), set()
        if faces and tl:
            iou = _iou(np.array([f[:4] for f in faces], np.float32),
                       np.array([t.box for t in tl], np.float32))
            order = np.argsort(-iou, axis=None)
            for di, ti in zip(*np.unravel_index(order, iou.shape)):
                di, ti = int(di), int(ti)
                if iou[di, ti] < TRACK_IOU_MIN:
                    break
                if di in used_d or ti in used_t:
                    continue
                tl[ti].update(faces[di].copy())
                used_d.add(di)
                used_t.add(ti)
        fresh = [tl[i] for i in used_t]
        for i, t in enumerate(tl):
            if i not in used_t:
                t.missed += 1
                if t.missed > TRACK_MAX_MISSED:
                    del self.tracks[t.id]
        for di, f in enumerate(faces):
            if di not in used_d:
                t = Track(self._next, f.copy())
                self._next += 1
                self.tracks[t.id] = t
                fresh.append(t)
        return fresh


# ----------------------------- Logika pengenalan -----------------------------
def _recheck_delay():
    """Jeda verifikasi ulang untuk wajah yang sudah dikenali (menentukan seberapa cepat absen pulang
    terdeteksi bagi orang yang tetap berdiri di depan kamera)."""
    return RECHECK_DETIK if RECHECK_DETIK > 0 else float("inf")


def apply_match(t, name, pid, score, now_t, logger, recheck):
    """Perbarui status track dari hasil pencocokan; catat absen bila sudah terkonfirmasi.

    Mengembalikan (datetime, tipe, status) bila ada catatan baru (masuk/pulang), selain itu None."""
    t.score = score
    if name is None or score < MATCH_THRESH:
        if t.name is not None:
            t.weak += 1
            if t.weak < WEAK_TOLERANCE:   # toleransi satu frame buruk pada wajah yang sudah dikenali
                t.next_embed = now_t + UNKNOWN_RETRY_DETIK
                return None
        t.name, t.cand, t.cand_n, t.weak = None, None, 0, 0
        t.next_embed = now_t + UNKNOWN_RETRY_DETIK
        return None
    t.weak = 0
    if name == t.name:
        t.cand, t.cand_n = name, CONFIRM_MATCHES
    else:
        t.name = None
        t.cand_n = t.cand_n + 1 if t.cand == name else 1
        t.cand = name
    if t.cand_n < CONFIRM_MATCHES:
        t.next_embed = now_t + UNKNOWN_RETRY_DETIK
        return None
    t.name, t.pid = name, pid
    t.next_embed = now_t + recheck
    dt = datetime.now()
    r = logger.try_log(name, pid, score, dt)
    if r:
        print(f"[{r[0].upper()}] {dt:%Y-%m-%d %H:%M:%S}  {name}  {r[1]}  (skor {score:.2f})")
        return dt, r[0], r[1]
    return None


def put_text(img, s, org, color, scale=0.55):
    big = img.shape[1] >= 900                 # font ikut membesar di resolusi tinggi
    scale, th = (scale * 1.4, 2) if big else (scale, 1)
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)  # outline agar terbaca
    cv2.putText(img, s, org, FONT, scale, color, th, cv2.LINE_AA)


def small_size(frame):
    h, w = frame.shape[:2]
    sw = min(PROC_WIDTH, w)
    sh = max(1, round(h * sw / w))
    return sw, sh, (w / sw, h / sh)


def _clean_field(value, label, limit):
    """Rapikan spasi dan tolak awalan yang bisa dibaca Excel sebagai rumus."""
    s = " ".join(str(value or "").split())
    if len(s) > limit:
        raise ValueError(f"{label} maksimal {limit} karakter.")
    if s[:1] in ("=", "+", "-", "@"):
        raise ValueError(f"{label} tidak boleh diawali = + - atau @ (aman untuk Excel).")
    return s


# ----------------------------- Mode registrasi -----------------------------
def register(name, pid, n_samples, cam_index):
    name = _clean_field(name, "Nama", 60)
    pid = _clean_field(pid, "ID", 40)
    if not name:
        raise ValueError("Nama tidak boleh kosong.")
    cv2.setNumThreads(CV_THREADS)
    engine = FaceEngine(*ensure_models())
    db = Database()
    cam = Camera(cam_index)
    samples, last = [], 0.0
    print(f"Registrasi '{name}': hadap kamera, {n_samples} sampel diambil otomatis. Q/ESC untuk batal.")
    try:
        while len(samples) < n_samples:
            frame = next_frame(cam)
            if frame is None:
                continue
            show = cv2.flip(frame, 1)    # pratinjau cermin; `frame` tetap bersih untuk embedding
            fw = frame.shape[1]
            sw, sh, scale = small_size(frame)
            faces = engine.detect(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA))
            faces = [f for f in faces if f[2] * scale[0] >= REG_MIN_FACE_PX * 0.5]  # abaikan wajah kecil di latar
            if len(faces) == 1:
                x, y, bw, bh = (int(v) for v in faces[0][:4] * np.array([*scale, *scale], np.float32))
                big = bw >= REG_MIN_FACE_PX
                msg = REG_PROMPTS[len(samples) % len(REG_PROMPTS)] if big else "Dekatkan wajah ke kamera"
                color = (0, 220, 0) if big else (0, 165, 255)
                if big and time.time() - last >= REG_INTERVAL:
                    samples.append(engine.embed(frame, faces[0], scale))   # dari frame bersih
                    last = time.time()
                mx = fw - x - bw                                           # koordinat pada pratinjau cermin
                cv2.rectangle(show, (mx, y), (mx + bw, y + bh), color, 2)
            elif not faces:
                msg, color = "Wajah belum terlihat / terlalu jauh", (0, 0, 255)
            else:
                msg, color = "Harus satu wajah saja di depan kamera", (0, 0, 255)
            put_text(show, f"{name}: sampel {len(samples)}/{n_samples}", (10, 24), (255, 255, 255), 0.7)
            put_text(show, msg, (10, show.shape[0] - 14), color, 0.7)
            cv2.imshow(WIN_REG, show)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()

    if len(samples) < n_samples:
        print("Registrasi dibatalkan, tidak ada data yang disimpan.")
        return
    mean = np.mean(samples, axis=0)
    mean /= np.linalg.norm(mean)
    if not db.is_empty():
        other, _, sc, _ = db.best(mean)
        if other != name and sc >= 0.5:
            print(f"PERINGATAN: wajah ini sangat mirip dengan '{other}' (skor {sc:.2f}). "
                  "Pastikan tidak terdaftar ganda dengan nama berbeda.")
    db.add(name, pid, samples)
    db.save()
    print(f"Tersimpan: {name} (+{n_samples} sampel). Total terdaftar: {len(db.people())} orang.")


# ----------------------------- Mode absen -----------------------------
def run_absen(cam_index, headless=False):
    cv2.setNumThreads(CV_THREADS)
    db = Database()
    if db.is_empty():
        print("PERINGATAN: database wajah kosong. Tambahkan wajah lewat dashboard (tab Kelola wajah) atau "
              'python main.py register --nama "Nama". Mode absen tetap berjalan dan memuat otomatis.')
    engine = FaceEngine(*ensure_models())
    logger, tracker = AttendanceLogger(), Tracker()
    recheck = _recheck_delay()
    cam = Camera(cam_index)
    print(f"Mode absen aktif ({len(db.people())} orang terdaftar). "
          f"Terlambat setelah {JAM_TERLAMBAT[:5]}, absen pulang mulai {JAM_PULANG[:5]}. "
          + ("Tekan Ctrl+C untuk berhenti." if headless else "Tekan Q / ESC untuk keluar."))

    small, gbuf, gi, have_prev = None, None, 0, False
    far = []  # wajah terdeteksi tetapi terlalu kecil (jauh)
    frame_no, fps, t_prev = 0, 0.0, time.perf_counter()
    ram_mb, ram_t, toast, toast_t = None, 0.0, "", 0.0
    db_t = 0.0  # waktu terakhir mengecek perubahan database / log
    try:
        while True:
            frame = next_frame(cam)
            if frame is None:
                continue
            frame_no += 1
            sw, sh, scale = small_size(frame)
            if small is None or small.shape[:2] != (sh, sw):  # alokasi buffer sekali, dipakai ulang
                small = np.empty((sh, sw, 3), np.uint8)
                gbuf = [np.empty((sh, sw), np.uint8), np.empty((sh, sw), np.uint8)]
                have_prev = False
                tracker.tracks.clear()
            cv2.resize(frame, (sw, sh), dst=small, interpolation=cv2.INTER_AREA)
            gray, prev = gbuf[gi], gbuf[1 - gi]
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY, dst=gray)
            if have_prev:
                tracker.advance(prev, gray)          # lacak posisi wajah di setiap frame
            gi, have_prev = 1 - gi, True

            now_t = time.time()
            if now_t - db_t > 2.0:                   # sinkron log dan muat ulang database bila berubah
                db_t = now_t
                logger.sync()
                if db.changed():
                    try:
                        db.reload()
                        print(f"Database dimuat ulang: {len(db.people())} orang terdaftar.")
                    except RuntimeError as e:
                        print(f"PERINGATAN: {e}")
                    for t in tracker.tracks.values():
                        t.next_embed = 0.0           # verifikasi ulang wajah yang sedang terlihat
            if frame_no % DETECT_EVERY == 0:         # deteksi hanya tiap N frame
                faces = engine.detect(small)
                near = [f for f in faces if f[2] * scale[0] >= MIN_FACE_PX]
                far = [f for f in faces if MIN_FACE_PX * 0.5 <= f[2] * scale[0] < MIN_FACE_PX]
                for t in tracker.update(near):
                    if now_t >= t.next_embed:        # embedding hanya jika perlu
                        res = apply_match(t, *db.match(engine.embed(frame, t.row, scale)),
                                          now_t, logger, recheck)
                        if res:
                            dt, tipe, st = res
                            toast, toast_t = f"{tipe.capitalize()} ({st}): {t.name} {dt:%H:%M:%S}", now_t

            t_now = time.perf_counter()
            inst = 1.0 / max(t_now - t_prev, 1e-6)
            fps, t_prev = (inst if fps == 0 else 0.9 * fps + 0.1 * inst), t_now
            if _PROC and now_t - ram_t > 1.0:
                ram_mb, ram_t = _PROC.memory_info().rss / 2 ** 20, now_t
            if frame_no % GC_EVERY_FRAMES == 0:
                gc.collect()

            if headless:
                time.sleep(0.001)
                continue
            k = np.array([*scale, *scale], np.float32)
            for f in far:                            # petunjuk: ada wajah, tetapi terlalu jauh
                x, y, bw, bh = (int(v) for v in f[:4] * k)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (150, 150, 150), 1)
                put_text(frame, f"Terlalu jauh ({bw}px, min {MIN_FACE_PX}px)", (x, max(y - 6, 14)),
                         (210, 210, 210), 0.5)
            for t in tracker.tracks.values():
                if t.missed > 1:
                    continue
                x, y, bw, bh = (int(v) for v in t.box * k)
                if t.name:
                    color, label = (0, 200, 0), f"{t.name} {t.score:.2f}"
                elif t.cand:
                    color, label = (0, 200, 255), f"{t.cand}? {t.score:.2f}"
                else:
                    color, label = (0, 0, 255), f"Unknown {t.score:.2f}"
                label += f" ({bw}px)"
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                put_text(frame, label, (x, max(y - 6, 14)), color)
            hud = f"FPS {fps:.0f}" + (f" | RAM {ram_mb:.0f} MB" if ram_mb else "") + \
                  f" | Hadir hari ini: {logger.count_today()}"
            if db.is_empty():
                hud += " | DB kosong"
            put_text(frame, hud, (10, 22), (255, 255, 255))
            if now_t - toast_t < 3.0:
                put_text(frame, toast, (10, frame.shape[0] - 14), (0, 255, 0), 0.7)
            cv2.imshow(WIN_RUN, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if frame_no % 30 == 0 and cv2.getWindowProperty(WIN_RUN, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        gc.collect()
        print("Mode absen dihentikan.")


# ----------------------------- Ekspor rekap -----------------------------
def export_rekap(dari, sampai, fmt):
    if not os.path.exists(LOG_PATH):
        raise RuntimeError("Belum ada data absensi (absensi.csv belum dibuat).")
    for s in (dari, sampai):
        if s:
            datetime.strptime(s, "%Y-%m-%d")  # validasi format tanggal
    grp = {}
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) < 3 or r[0] == "tanggal":
                continue
            tgl, jam, nama = r[0], r[1], r[2]
            pid = r[3] if len(r) > 3 else ""
            tipe = r[5] if len(r) > 5 and r[5] else "masuk"
            status = r[6] if len(r) > 6 and r[6] else ("terlambat" if jam > JAM_TERLAMBAT else "hadir")
            if (dari and tgl < dari) or (sampai and tgl > sampai):
                continue
            g = grp.setdefault((tgl, nama), {"id": pid, "masuk": "", "status": "", "pulang": ""})
            g["id"] = pid or g["id"]
            if tipe == "pulang":
                g["pulang"] = max(g["pulang"], jam)
            elif not g["masuk"] or jam < g["masuk"]:
                g["masuk"], g["status"] = jam, status
    if not grp:
        raise RuntimeError("Tidak ada data absensi pada rentang tanggal tersebut.")
    rows = [[t, n, g["id"], g["masuk"], g["status"], g["pulang"]] for (t, n), g in sorted(grp.items())]
    header = ["tanggal", "nama", "id", "jam_masuk", "status", "jam_pulang"]
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"rekap_{rows[0][0]}_{rows[-1][0]}.{fmt}")
    if fmt == "xlsx":
        try:
            from openpyxl import Workbook
        except ImportError:
            raise RuntimeError("Ekspor Excel butuh openpyxl: pip install openpyxl (atau pakai --format csv).")
        wb = Workbook()
        ws = wb.active
        ws.title = "Rekap"
        ws.append(header)
        for r in rows:
            ws.append(r)
        for col, width in zip("ABCDEF", (12, 26, 14, 13, 13, 13)):
            ws.column_dimensions[col].width = width
        ws.freeze_panes = "A2"
        wb.save(path)
    else:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig agar rapi di Excel
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    print(f"Rekap tersimpan: {path} ({len(rows)} baris)")


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Auto Fast Face Recognition - Absensi Sederhana (CPU only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("register", help="daftarkan wajah baru")
    p.add_argument("--nama", required=True)
    p.add_argument("--id", default="", help="NIM/NIP/ID (opsional)")
    p.add_argument("--jumlah", type=int, default=REG_SAMPLES, choices=range(5, 11), help="jumlah sampel (5-10)")
    p.add_argument("--kamera", type=int, default=CAM_INDEX)
    p.add_argument("--jarak", choices=list(PROFIL), default=PROFIL_AKTIF,
                   help="profil jarak: dekat (640x480, ringan), sedang (720p), jauh (720p, lebih sensitif)")

    p = sub.add_parser("run", help="mode absen otomatis")
    p.add_argument("--kamera", type=int, default=CAM_INDEX)
    p.add_argument("--jarak", choices=list(PROFIL), default=PROFIL_AKTIF,
                   help="profil jarak: dekat (640x480, ringan), sedang (720p), jauh (720p, lebih sensitif)")
    p.add_argument("--headless", action="store_true", help="tanpa jendela (paling hemat resource)")

    p = sub.add_parser("export", help="ekspor rekap absensi")
    p.add_argument("--dari", help="tanggal awal YYYY-MM-DD")
    p.add_argument("--sampai", help="tanggal akhir YYYY-MM-DD")
    p.add_argument("--format", choices=["csv", "xlsx"], default="csv")

    sub.add_parser("list", help="daftar orang terdaftar")
    p = sub.add_parser("delete", help="hapus data seseorang")
    p.add_argument("--nama", required=True)
    sub.add_parser("download-models", help="unduh model YuNet dan SFace")

    a = ap.parse_args()
    try:
        if a.cmd in ("register", "run"):
            apply_profile(a.jarak)
        if a.cmd == "register":
            register(a.nama.strip(), a.id.strip(), a.jumlah, a.kamera)
        elif a.cmd == "run":
            run_absen(a.kamera, a.headless)
        elif a.cmd == "export":
            export_rekap(a.dari, a.sampai, a.format)
        elif a.cmd == "list":
            people = Database().people()
            if not people:
                print("Belum ada wajah terdaftar.")
            for n, (c, i) in sorted(people.items()):
                print(f"- {n}" + (f" (ID {i})" if i else "") + f": {c} sampel")
        elif a.cmd == "delete":
            db = Database()
            removed = db.remove(a.nama.strip())
            if removed:
                db.save()
            print(f"{removed} sampel milik '{a.nama}' dihapus." if removed else f"Nama '{a.nama}' tidak ditemukan.")
        elif a.cmd == "download-models":
            print("Model siap:", *ensure_models(), sep="\n  ")
    except KeyboardInterrupt:
        print("\nDihentikan.")
    except (CameraError, RuntimeError, ValueError, OSError, cv2.error) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()