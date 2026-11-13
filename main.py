#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Fast Face Recognition System - Absensi Sederhana (versi optimal + RTSP always-on)
Berjalan penuh di CPU (0 VRAM) dan hemat RAM. Model: YuNet (deteksi) + SFace (embedding).

Perintah:
  python main.py register --nama "Budi" --id 2023001   # daftarkan wajah (5-10 sampel)
  python main.py run [--jarak dekat|sedang|jauh]       # mode absen otomatis (Q / ESC = keluar)
  python main.py run --rtsp "rtsp://user:pass@ip:554/stream" --headless   # CCTV selalu aktif
  python main.py export --format xlsx                  # ekspor rekap ke CSV atau Excel
  python main.py list                                  # daftar orang terdaftar
  python main.py delete --nama "Budi"                  # hapus data satu orang
  python main.py download-models                       # unduh model YuNet + SFace
  python main.py privasi status                        # ringkasan enkripsi, persetujuan, rekam jejak
  python main.py privasi --help                        # semua perintah privasi (UU PDP)

Aturan absensi (atur di config.py / config.json):
  * Scan pertama hari itu sampai JAM_TERLAMBAT (09:00)  -> MASUK, status "hadir".
  * Scan pertama setelah JAM_TERLAMBAT                  -> MASUK, status "terlambat".
  * Scan setelah JAM_PULANG (11:00) dari orang yang sudah masuk (hadir/terlambat) dan sudah
    lewat MIN_DURASI_MENIT sejak masuk                  -> PULANG (sekali per hari).
  * Semua status di-reset otomatis setiap hari (dibandingkan per tanggal, tanpa restart).

RTSP (CCTV): koneksi yang putus disambung ulang otomatis (backoff), program tidak berhenti.
URL RTSP juga bisa diberikan lewat environment variable RTSP_URL.

Cloud (opsional): jika cloud_sync.py ada di folder yang sama dan kredensial sudah diatur
(serviceAccount.json + cloud_config.json atau env), setiap absen juga dikirim ke Firebase Firestore
dan Google Sheets. absensi.csv tetap ditulis sebagai catatan lokal utama. Data biometrik TIDAK dikirim.

Liveness (liveness.py): absen hanya dicatat bila wajah lolos uji keaslian (config liveness_mode:
pasif = model MiniFASNet, aktif = tantangan acak tengok kiri-kanan / dongak-tunduk, ganda = keduanya).
Percobaan spoofing dicatat di spoof_log.csv dan audit_log.jsonl.

Evaluasi kuantitatif (FAR/FRR/EER, identifikasi open-set, APCER/BPCER/ACER, latensi): evaluate.py.

Privasi (privacy.py, UU No. 27/2022): faces_db.enc terenkripsi AES-256-GCM; pendaftaran wajib
persetujuan eksplisit (naskah ditampilkan, dicatat di persetujuan.json); wajah tanpa persetujuan
aktif tidak dikenali; semua aksi data dicatat di rekam jejak berantai hash.

Format absensi.csv: tanggal, jam, nama, id, skor, tipe, status
  (baris lama 5 kolom tetap terbaca: dianggap tipe "masuk", status dihitung dari jamnya).
"""
import argparse
import csv
import gc
import json
import os
import re
import sys
import threading
import time
import urllib.request
from datetime import datetime, timedelta

# Opsi FFmpeg untuk RTSP: TCP (stabil) + tanpa buffer. Harus diset sebelum VideoCapture pertama.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")

import cv2
import numpy as np

import config
import privacy
from attendance import AttendanceLogger, purge_before, read_log
from config import JAM_PULANG, JAM_TERLAMBAT
from liveness import LivenessEngine

try:  # opsional: untuk menampilkan estimasi RAM
    import psutil
    _PROC = psutil.Process()
except ImportError:
    _PROC = None

# ============================ KONFIGURASI ============================
# Lokasi file diatur terpusat di config.py (sebelumnya disalin di beberapa modul).
BASE_DIR = config.BASE_DIR
MODEL_DIR = config.MODEL_DIR
DB_PATH = config.DB_PATH           # embedding + nama + id, TERENKRIPSI (privacy.py)
LOG_PATH = config.LOG_PATH         # catatan absen (dibaca dashboard.py)
EXPORT_DIR = config.EXPORT_DIR

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
RTSP_RECONNECT_MIN = 1.0   # detik jeda sambung ulang pertama (CCTV putus)
RTSP_RECONNECT_MAX = 15.0  # jeda maksimum (backoff x2)

# Profil jarak (pilih lewat --jarak). Jarak pengenalan ditentukan oleh jumlah piksel wajah:
# wajah butuh >= ~50 px lebar (di frame kamera asli) agar dikenali andal. Untuk webcam 720p
# umum, lebar wajah (px) ~ 177 / jarak (m); untuk 480p kira-kira separuhnya.
# Format: (lebar kamera, tinggi kamera, lebar frame deteksi, lebar wajah minimum px, skor deteksi)
# Catatan: lebar/tinggi kamera hanya dipakai untuk kamera lokal; RTSP memakai resolusi stream.
PROFIL = {
    "dekat":  (640, 480, 320, 80, 0.85),    # paling ringan, +-1 m
    "sedang": (1280, 720, 480, 50, 0.80),   # +-3 m
    "jauh":   (1280, 720, 640, 40, 0.70),   # +-4 m, akurasi menurun, butuh CPU lebih kuat
}
PROFIL_AKTIF = "sedang"

# Performa
DETECT_EVERY = 3           # deteksi tiap N frame; frame lain dilacak (optical flow)
CV_THREADS = 2             # batasi thread OpenCV agar CPU tidak penuh
# (GC_EVERY_FRAMES dihapus: gc.collect() berkala tidak membebaskan buffer numpy/OpenCV - itu sudah
#  dilepas lewat reference counting - dan hanya menambah jeda sesaat di loop kamera.)

# Deteksi dan pengenalan
NMS_THRESH = 0.3
MATCH_THRESH = config.MATCH_THRESH   # ambang skor orang (rata-rata 2 sampel terbaik); kalibrasi: evaluate.py
MATCH_MARGIN = config.MATCH_MARGIN   # selisih minimum skor orang terbaik vs kedua (0 = nonaktif)
CONFIRM_MATCHES = 2        # kecocokan beruntun yang dibutuhkan sebelum dianggap dikenali
UNKNOWN_RETRY_DETIK = 0.4  # jeda ekstraksi embedding ulang untuk wajah yang belum dikenali
RECHECK_DETIK = 60         # verifikasi ulang wajah yang sudah dikenali (0 = tidak pernah)
TRACK_IOU_MIN = 0.25
TRACK_MAX_MISSED = 4       # siklus deteksi tanpa terdeteksi sebelum track dibuang
WEAK_TOLERANCE = 2         # skor buruk beruntun sebelum wajah yang sudah dikenali dianggap Unknown

# Absensi
# Jam terlambat / jam pulang / durasi minimum diatur terpusat di config.py (atau config.json / environment variable)

# Registrasi
REG_SAMPLES = 8
REG_INTERVAL = 0.7         # detik antar pengambilan sampel
REG_MIN_FACE_PX = 90       # wajah harus cukup besar saat registrasi (kualitas embedding)
REG_MIN_SHARP = 25.0       # variansi Laplacian minimum crop wajah 112x112 (di bawah ini = buram)
REG_BRIGHT = (40, 225)     # rentang rata-rata kecerahan crop wajah yang diterima
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


def is_rtsp(src):
    """True bila sumber berupa URL (rtsp://, http://, dst), False untuk index kamera lokal."""
    return isinstance(src, str) and "://" in src


def mask_url(src):
    """Sembunyikan password saat URL dicetak ke log."""
    return re.sub(r"//([^:@/]+):[^@/]*@", r"//\1:***@", str(src))


class Camera:
    """Baca kamera di thread terpisah dan hanya simpan frame terbaru (tanpa lag buffer).

    source: index kamera lokal (int) atau URL RTSP (str). Untuk RTSP, koneksi yang putus
    disambung ulang otomatis (backoff) tanpa menghentikan program.
    """

    def __init__(self, source=CAM_INDEX, reconnect=None):
        self.source = source
        self.rtsp = is_rtsp(source)
        self.reconnect = self.rtsp if reconnect is None else reconnect
        self.cap = None
        self.failed = False
        self.connected = False
        self._lost = False
        self._frame = None
        self._new = threading.Event()
        self._stop = threading.Event()
        if not self.rtsp:   # kamera lokal: gagal buka = langsung error
            self.cap = self._open()
            if not self.cap.isOpened():
                self.cap.release()
                raise CameraError(f"Kamera index {source} tidak terdeteksi. Periksa sambungan/izin kamera "
                                  "atau ubah CAM_INDEX (opsi --kamera).")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        first = self.read(timeout=10.0 if self.rtsp else 3.0)
        if first is None:
            if self.rtsp and self.reconnect:   # CCTV belum siap: tetap jalan, coba terus di latar belakang
                print(f"RTSP {mask_url(source)} belum tersambung, mencoba terus di latar belakang...")
                return
            self.close()
            raise CameraError("Kamera terbuka tetapi tidak mengirim gambar (dipakai aplikasi lain?).")
        h, w = first.shape[:2]
        if self.rtsp:
            print(f"RTSP aktif: {mask_url(source)} ({w}x{h})")
        else:
            print(f"Kamera aktif: {w}x{h}" + (f" (diminta {CAM_WIDTH}x{CAM_HEIGHT}; webcam tidak mendukung, "
                                               "jarak pengenalan jadi lebih pendek)" if w < CAM_WIDTH else ""))

    def _open(self):
        if self.rtsp:
            try:   # OpenCV >= 4.5.2: timeout agar read tidak menggantung saat jaringan putus
                return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG,
                                        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                                         cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])
            except TypeError:
                return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        backend = _cam_backend()
        cap = cv2.VideoCapture(self.source, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:   # cadangan: biarkan OpenCV memilih
            cap.release()
            cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            if CAM_MJPG:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _drop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.connected:
            print("Kamera terputus" + (", mencoba menyambung ulang..." if self.reconnect else "."))
            self._lost = True
        self.connected = False

    def _loop(self):
        fails, delay = 0, RTSP_RECONNECT_MIN
        try:
            while not self._stop.is_set():
                if self.cap is None:
                    self.cap = self._open()
                    if not self.cap.isOpened():
                        self._drop()
                        if not self.reconnect:
                            self.failed = True
                            self._new.set()
                            return
                        self._stop.wait(delay)
                        delay = min(delay * 2, RTSP_RECONNECT_MAX)
                        continue
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    fails, delay = 0, RTSP_RECONNECT_MIN
                    if not self.connected and self._lost:
                        print("Kamera tersambung kembali.")
                        self._lost = False
                    self.connected = True
                    self._frame = frame
                    self._new.set()
                    continue
                fails += 1
                if fails > (3 if self.rtsp else 50):
                    fails = 0
                    self._drop()
                    if not self.reconnect:
                        self.failed = True
                        self._new.set()
                        return
                    self._stop.wait(delay)
                    delay = min(delay * 2, RTSP_RECONNECT_MAX)
                else:
                    time.sleep(0.02)
        finally:
            self._drop()

    def read(self, timeout=1.0):
        """Kembalikan frame terbaru yang belum dibaca, atau None jika belum ada."""
        if not self._new.wait(timeout):
            return None
        self._new.clear()
        return None if self.failed else self._frame

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3.0)   # thread sendiri yang melepas kamera (finally di _loop)


def next_frame(cam):
    """Ambil frame berikutnya; lempar error bila kamera lokal terputus (RTSP: sambung ulang otomatis)."""
    frame = cam.read()
    if frame is None and cam.failed:
        raise CameraError("Kamera terputus.")
    return frame


# ----------------------------- Engine wajah -----------------------------
class FaceEngine:
    """Pembungkus YuNet (deteksi) dan SFace (embedding), keduanya di CPU."""

    def __init__(self, yunet_path, sface_path):
        self.yunet_path = yunet_path
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

    def align(self, full, row, scale):
        """Crop wajah 112x112 yang sudah disejajarkan, dari frame resolusi penuh (lebih tajam)."""
        r = row.copy()
        r[0:14:2] *= scale[0]
        r[1:14:2] *= scale[1]
        return self.rec.alignCrop(full, r)

    def feature(self, aligned):
        feat = self.rec.feature(aligned).reshape(-1).astype(np.float32)
        n = np.linalg.norm(feat)
        return feat / n if n > 0 else feat

    def embed(self, full, row, scale):
        """Embedding 128-d ternormalisasi."""
        return self.feature(self.align(full, row, scale))


def sample_quality(aligned):
    """Gerbang kualitas sampel registrasi. Kembalikan alasan penolakan (str) atau None bila layak.
    Sampel buram/gelap menghasilkan embedding yang buruk dan menaikkan salah tolak (FRR)."""
    g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    mean = float(g.mean())
    if mean < REG_BRIGHT[0]:
        return f"terlalu gelap (kecerahan {mean:.0f})"
    if mean > REG_BRIGHT[1]:
        return f"terlalu terang (kecerahan {mean:.0f})"
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if sharp < REG_MIN_SHARP:
        return f"buram (ketajaman {sharp:.0f}, minimal {REG_MIN_SHARP:.0f})"
    return None


# ----------------------------- Database embedding -----------------------------
class Database:
    """Embedding di RAM sebagai matriks float32 (N,128). File di disk TERENKRIPSI (privacy.py).

    Pencocokan dilakukan per orang: skor orang = rata-rata 2 sampel paling mirip (1 sampel bila
    hanya ada satu). Seluruhnya vektor (satu perkalian matriks + reduceat), tanpa loop Python.

    require_consent=True (mode absen): hanya orang dengan persetujuan AKTIF yang masuk indeks
    pencocokan (UU PDP Pasal 20). Data lain tetap ada untuk keperluan kelola (CRUD).
    path=None: database di memori saja (dipakai evaluate.py).
    """

    def __init__(self, path=DB_PATH, require_consent=False):
        self.path = path
        self.require_consent = require_consent
        self.names, self.ids = [], []
        self.embs = np.zeros((0, 128), np.float32)
        self.excluded = []
        self._sig = None
        if path:
            self._load()
        self._reindex()

    @classmethod
    def from_arrays(cls, names, ids, embs):
        db = cls(path=None)
        db.names, db.ids = list(names), list(ids)
        db.embs = np.ascontiguousarray(embs, np.float32).reshape(-1, 128)
        db._reindex()
        return db

    # ---- deteksi perubahan file (database + persetujuan) ----
    def _sig_now(self):
        if not self.path:
            return None
        try:
            st = os.stat(self.path)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
        if self.require_consent:
            return sig, privacy.consents().signature()
        return sig

    def changed(self):
        """True jika file database (atau persetujuan) berubah, misalnya diedit lewat dashboard."""
        return self._sig_now() != self._sig

    def reload(self):
        self._load()
        self._reindex()

    def _load(self):
        try:
            if self.path == config.DB_PATH:
                privacy.migrate_legacy(self.path)          # faces_db.npz lama -> terenkripsi
            self._sig = self._sig_now()
            names, ids, embs = privacy.load_db(self.path, migrate=False)
        except privacy.PrivacyError as e:
            raise RuntimeError(f"Database wajah tidak bisa dibuka ({self.path}): {e}")
        except Exception as e:
            raise RuntimeError(f"Database wajah rusak/tidak terbaca ({self.path}): {e}")
        self.names, self.ids, self.embs = names, ids, embs

    def _reindex(self):
        """Susun indeks per orang (urutan sampel dikelompokkan) untuk pencocokan vektor."""
        names, ids, embs = self.names, self.ids, self.embs
        self.excluded = []
        if self.require_consent and names:
            ok = privacy.consents().active_names()
            keep = np.array([n in ok for n in names], dtype=bool)
            self.excluded = sorted({n for n in names if n not in ok})
            names = [n for n, k in zip(names, keep) if k]
            ids = [i for i, k in zip(ids, keep) if k]
            embs = embs[keep]
        if not names:
            self.uniq, self.uids = [], []
            self._counts = np.zeros(0, np.intp)
            self._starts = np.zeros(0, np.intp)
            self._es = np.zeros((0, 128), np.float32)
            return
        uniq, inv = np.unique(np.array(names, dtype=str), return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv, minlength=len(uniq)).astype(np.intp)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.intp)
        ids = np.array(ids, dtype=str)
        self._counts, self._starts = counts, starts
        self._es = np.ascontiguousarray(embs[order])
        self.uniq = [str(x) for x in uniq]
        self.uids = [str(x) for x in ids[order[starts]]]

    def save(self):
        """Simpan terenkripsi (atomik, dikunci antar-proses agar dua penulis tidak bertabrakan)."""
        with privacy.db_lock(self.path):
            privacy.save_db(self.names, self.ids, self.embs, self.path)
        self._sig = self._sig_now()

    # ---- data ----
    def is_empty(self):
        return len(self.names) == 0

    def matchable(self):
        """Jumlah orang yang ikut dicocokkan (setelah filter persetujuan)."""
        return len(self.uniq)

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
        self.live = None       # liveness.LivenessState (dibuat saat pertama diproses)
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
    if t.live is not None and not t.live.passed:   # dikenali, tetapi BELUM lolos uji keaslian
        t.next_embed = now_t + UNKNOWN_RETRY_DETIK  # dicek lagi segera setelah liveness lolos
        return None
    t.next_embed = now_t + recheck
    dt = datetime.now()
    r = logger.try_log(name, pid, score, dt)
    if r:
        print(f"[{r[0].upper()}] {dt:%Y-%m-%d %H:%M:%S}  {name}  {r[1]}  (skor {score:.2f})")
        return dt, r[0], r[1]
    return None


def report_spoof(t, source="main"):
    """Catat percobaan spoofing (spoof_log.csv + rekam jejak) dan kembalikan event untuk UI."""
    now = datetime.now()
    cand = t.name or t.cand or ""
    skor = t.live.mean if t.live is not None and t.live.mean is not None else 0.0
    try:
        new = not os.path.exists(config.SPOOF_LOG_PATH)
        with open(config.SPOOF_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tanggal", "jam", "kandidat", "skor_live", "sumber"])
            w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), cand, f"{skor:.3f}", source])
    except OSError as e:
        print(f"[SPOOF] gagal menulis log: {e}")
    privacy.audit("spoof_terdeteksi", cand, {"skor_live": round(skor, 3), "sumber": source})
    print(f"[SPOOF] {now:%Y-%m-%d %H:%M:%S}  percobaan wajah palsu" + (f" (mirip {cand})" if cand else "")
          + f"  skor asli {skor:.2f}")
    return {"tipe": "spoof", "nama": cand, "id": "", "jam": now.strftime("%H:%M:%S"), "status": "ditolak",
            "skor_live": round(skor, 3)}


def process_faces(faces, frame, scale, now_t, tracker, engine, db, logger, recheck,
                  live=None, min_px=None, source="main"):
    """SATU pipeline per siklus deteksi: saring ukuran -> lacak -> liveness -> cocokkan -> catat.

    Dipakai `main.py run`, kamera server/CCTV dashboard, dan kamera browser dashboard (sebelumnya
    disalin di 3 tempat). Mengembalikan (track_segar, wajah_jauh, events)."""
    min_px = MIN_FACE_PX if min_px is None else min_px
    near = [f for f in faces if f[2] * scale[0] >= min_px]
    far = [f for f in faces if min_px * 0.5 <= f[2] * scale[0] < min_px]
    fresh = tracker.update(near)
    events = []
    k = np.array([*scale, *scale], np.float32)
    for t in fresh:
        if live is not None and live.enabled:
            if t.live is None:
                t.live = live.new_state()
            if not t.live.passed:
                lm = t.row[4:14].reshape(5, 2) * np.asarray(scale, np.float32)
                if live.observe(t.live, frame, t.row[:4] * k, lm, now_t) == "spoof":
                    events.append(report_spoof(t, source))
                if t.live.passed and t.name:      # baru lolos: catat sekarang, jangan tunggu jeda verifikasi
                    t.next_embed = 0.0
        if now_t >= t.next_embed:          # embedding hanya jika perlu
            res = apply_match(t, *db.match(engine.embed(frame, t.row, scale)), now_t, logger, recheck)
            if res:
                dt, tipe, st = res
                events.append({"nama": t.name, "id": t.pid, "jam": dt.strftime("%H:%M:%S"),
                               "tipe": tipe, "status": st})
    return fresh, far, events


def track_view(t):
    """(status, teks) satu track untuk tampilan: tercatat/hadir | liveness | spoof | cek | unknown."""
    lv = t.live
    if lv is not None and lv.spoof:
        return "spoof", f"PALSU? {t.name or t.cand or ''}".strip()
    if t.name and (lv is None or lv.passed):
        return "hadir", t.name
    if (t.name or t.cand) and lv is not None and not lv.passed:
        return "liveness", f"{t.name or t.cand}: {lv.prompt()}"
    if t.cand:
        return "cek", f"{t.cand}?"
    return "unknown", "Unknown"


_COLORS = {"hadir": (0, 200, 0), "liveness": (0, 200, 255), "cek": (0, 200, 255),
           "spoof": (0, 0, 255), "unknown": (0, 0, 255)}


def draw_tracks(frame, tracker, far, scale):
    """Gambar kotak + label semua wajah di frame (main.py run dan dashboard kamera server)."""
    k = np.array([*scale, *scale], np.float32)
    for f in far:                            # petunjuk: ada wajah, tetapi terlalu jauh
        x, y, bw, bh = (int(v) for v in f[:4] * k)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (150, 150, 150), 1)
        put_text(frame, f"Terlalu jauh ({bw}px, min {MIN_FACE_PX}px)", (x, max(y - 6, 14)), (210, 210, 210), 0.5)
    for t in tracker.tracks.values():
        if t.missed > 1:
            continue
        x, y, bw, bh = (int(v) for v in t.box * k)
        status, label = track_view(t)
        color = _COLORS[status]
        label = _ascii(label) + f" {t.score:.2f}"
        if t.live is not None and t.live.last is not None:
            label += f" L{t.live.last:.2f}"
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        put_text(frame, label, (x, max(y - 6, 14)), color)


def _ascii(s):
    """cv2.putText hanya mendukung ASCII."""
    return str(s).encode("ascii", "replace").decode()


def make_liveness(engine):
    """LivenessEngine sesuai config (None bila liveness_mode=off)."""
    live = LivenessEngine(yunet_path=engine.yunet_path)
    return live if live.enabled else None


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
def ask_consent(name, bukti=""):
    """Tampilkan naskah (Pasal 21) lalu minta persetujuan eksplisit (Pasal 20, 22).

    bukti: nomor/rujukan formulir persetujuan TERTULIS yang sudah ditandatangani (tanpa tanya).
    Mengembalikan True bila setuju."""
    if bukti:
        return True
    ver, text, _ = privacy.notice()
    print("\n" + "=" * 78 + "\n" + text + "=" * 78)
    if not sys.stdin.isatty():
        raise ValueError("Persetujuan butuh input interaktif. Bila persetujuan tertulis sudah ditandatangani, "
                         "pakai --bukti-persetujuan \"No. formulir\".")
    ans = input(f"Subjek data '{name}': ketik SETUJU untuk memberikan persetujuan (selain itu = menolak): ")
    return ans.strip().upper() == "SETUJU"


def register(name, pid, n_samples, cam_index, bukti="", petugas="", skip_live=False):
    name = _clean_field(name, "Nama", 60)
    pid = _clean_field(pid, "ID", 40)
    if not name:
        raise ValueError("Nama tidak boleh kosong.")
    if not ask_consent(name, bukti):
        privacy.audit("persetujuan_ditolak", name, {"metode": "cli"})
        print("Persetujuan tidak diberikan. Registrasi dibatalkan, tidak ada data wajah yang diproses.")
        return
    cv2.setNumThreads(CV_THREADS)
    engine = FaceEngine(*ensure_models())
    live = None if skip_live else make_liveness(engine)
    if live is not None and live.passive is None:
        live = None                          # registrasi diawasi petugas: cukup cek pasif (bila ada)
    if skip_live:
        privacy.audit("registrasi_tanpa_liveness", name, {"petugas": petugas or privacy.actor()})
    db = Database()
    reg = privacy.consents()
    already = reg.is_active(name)
    if not already:                          # persetujuan dicatat SEBELUM data wajah diproses
        reg.give(name, pid, "cli" if not bukti else "tertulis", petugas, bukti)
    cam = Camera(cam_index)
    samples, last, reject, lscores, t_start = [], 0.0, "", [], time.time()
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
                box = faces[0][:4] * np.array([*scale, *scale], np.float32)
                x, y, bw, bh = (int(v) for v in box)
                big = bw >= REG_MIN_FACE_PX
                msg = REG_PROMPTS[len(samples) % len(REG_PROMPTS)] if big else "Dekatkan wajah ke kamera"
                color = (0, 220, 0) if big else (0, 165, 255)
                if big and time.time() - last >= REG_INTERVAL:
                    last = time.time()
                    aligned = engine.align(frame, faces[0], scale)            # dari frame bersih
                    reject = sample_quality(aligned) or ""
                    if live is not None and not reject:                     # tolak foto/layar saat registrasi
                        lscores = (lscores + [live.passive.score(frame, box) or 0.0])[-3:]
                        if np.mean(lscores) < live.thresh:
                            reject = f"tidak lolos uji keaslian ({np.mean(lscores):.2f})"
                    if reject:
                        color = (0, 165, 255)
                    else:
                        samples.append(engine.feature(aligned))
                if reject:
                    msg = "Ditolak: " + reject
                    if "keaslian" in reject and time.time() - t_start > 15 and not samples:
                        msg += " - perbaiki cahaya; bila yakin asli: --lewati-liveness"
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
        if not already:
            reg.withdraw(name, "registrasi dibatalkan sebelum data wajah disimpan", status="batal")
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
    privacy.audit("wajah_didaftarkan", name, {"sampel": n_samples, "metode": "cli"})
    print(f"Tersimpan (terenkripsi): {name} (+{n_samples} sampel). Total terdaftar: {len(db.people())} orang.")


# ----------------------------- Mode absen -----------------------------
def run_absen(source, headless=False):
    """source: index kamera lokal (int) atau URL RTSP (str)."""
    cv2.setNumThreads(CV_THREADS)
    db = Database(require_consent=config.WAJIB_PERSETUJUAN)
    if db.is_empty():
        print("PERINGATAN: database wajah kosong. Tambahkan wajah lewat dashboard (tab Kelola wajah) atau "
              'python main.py register --nama "Nama". Mode absen tetap berjalan dan memuat otomatis.')
    warn_no_consent(db)
    engine = FaceEngine(*ensure_models())
    live = make_liveness(engine)
    logger, tracker = AttendanceLogger(), Tracker()
    recheck = _recheck_delay()
    cam = Camera(source)
    print(f"Mode absen aktif ({db.matchable()} orang dikenali). "
          f"Terlambat setelah {JAM_TERLAMBAT[:5]}, absen pulang mulai {JAM_PULANG[:5]}. "
          + (live.describe() if live else "PERINGATAN: liveness NONAKTIF (rentan foto/video).") + ". "
          + ("Tekan Ctrl+C untuk berhenti." if headless else "Tekan Q / ESC untuk keluar."))
    if live is not None and live.active and headless:
        print("PERINGATAN: tantangan aktif (geleng/angguk) butuh layar; pada --headless pengguna tidak "
              "melihat instruksi. Untuk CCTV pakai liveness_mode=pasif.")

    small, gbuf, gi, have_prev = None, None, 0, False
    far = []  # wajah terdeteksi tetapi terlalu kecil (jauh)
    frame_no, fps, t_prev = 0, 0.0, time.perf_counter()
    ram_mb, ram_t, toast, toast_t = None, 0.0, "", 0.0
    db_t = 0.0  # waktu terakhir mengecek perubahan database / log
    was_connected = True
    try:
        while True:
            frame = next_frame(cam)
            if frame is None:
                if was_connected and not cam.connected:   # RTSP putus: buang pelacak lama
                    was_connected = False
                    tracker.tracks.clear()
                    have_prev = False
                if not headless and not cam.connected:    # tetap tampilkan jendela + tangani tombol keluar
                    wait = np.zeros((240, 640, 3), np.uint8)
                    put_text(wait, "Kamera terputus, menyambung ulang...", (10, 120), (0, 165, 255), 0.7)
                    cv2.imshow(WIN_RUN, wait)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue
            was_connected = True
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
                        print(f"Database dimuat ulang: {db.matchable()} orang dikenali.")
                    except RuntimeError as e:
                        print(f"PERINGATAN: {e}")
                    for t in tracker.tracks.values():
                        t.next_embed = 0.0           # verifikasi ulang wajah yang sedang terlihat
            if frame_no % DETECT_EVERY == 0:         # deteksi hanya tiap N frame
                _, far, events = process_faces(engine.detect(small), frame, scale, now_t, tracker, engine,
                                               db, logger, recheck, live)
                for e in events:
                    if e["tipe"] == "spoof":
                        toast, toast_t = f"DITOLAK: wajah palsu terdeteksi {e['jam']}", now_t
                    else:
                        toast, toast_t = f"{e['tipe'].capitalize()} ({e['status']}): {e['nama']} {e['jam']}", now_t

            t_now = time.perf_counter()
            inst = 1.0 / max(t_now - t_prev, 1e-6)
            fps, t_prev = (inst if fps == 0 else 0.9 * fps + 0.1 * inst), t_now
            if _PROC and now_t - ram_t > 1.0:
                ram_mb, ram_t = _PROC.memory_info().rss / 2 ** 20, now_t

            if headless:
                time.sleep(0.001)
                continue
            draw_tracks(frame, tracker, far, scale)
            hud = f"FPS {fps:.0f}" + (f" | RAM {ram_mb:.0f} MB" if ram_mb else "") + \
                  f" | Hadir hari ini: {logger.count_today()}" + ("" if live else " | LIVENESS OFF")
            if db.is_empty():
                hud += " | DB kosong"
            put_text(frame, hud, (10, 22), (255, 255, 255))
            if now_t - toast_t < 3.0:
                put_text(frame, _ascii(toast), (10, frame.shape[0] - 14),
                         (0, 0, 255) if toast.startswith("DITOLAK") else (0, 255, 0), 0.7)
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
    for d in read_log(LOG_PATH):
        tgl, jam, nama, pid, tipe, status = d["tanggal"], d["jam"], d["nama"], d["id"], d["tipe"], d["status"]
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
    privacy.audit("ekspor_rekap", detail={"dari": dari or "", "sampai": sampai or "", "baris": len(rows),
                                          "file": os.path.basename(path)})


# ----------------------------- Privasi (UU PDP) -----------------------------
def warn_no_consent(db):
    if db.excluded:
        print(f"PERINGATAN: {len(db.excluded)} orang terdaftar TANPA persetujuan aktif dan tidak akan dikenali: "
              + ", ".join(db.excluded[:8]) + (" ..." if len(db.excluded) > 8 else "")
              + '. Catat persetujuan: python main.py privasi catat --nama "..." atau hapus datanya.')


def delete_person(name, alasan, status="dihapus"):
    """Hapus data biometrik seseorang + tandai persetujuan (Pasal 8, 9, 43). Kembalikan jumlah sampel."""
    db = Database()
    removed = db.remove(name)
    if removed:
        db.save()
    if privacy.consents().get(name):
        privacy.consents().withdraw(name, alasan, status=status)
    if removed:
        privacy.audit("wajah_dihapus", name, {"sampel": removed, "alasan": alasan})
    return removed


def privasi_cmd(a):
    reg = privacy.consents()
    c = a.aksi
    if c == "naskah":
        print(privacy.notice()[1])
    elif c == "status":
        db = Database()
        people = db.people()
        cons = reg.all()
        aktif = {n for n, r in cons.items() if r.get("status") == "aktif"}
        tanpa = sorted(n for n in people if n not in aktif)
        ok, n, msg = privacy.verify_audit()
        enc = os.path.exists(config.DB_PATH)
        print(f"Database wajah     : {config.DB_PATH} ({'TERENKRIPSI AES-256-GCM' if enc else 'belum ada'})")
        print("Kunci              : " + ("FACEDB_PASSPHRASE (kata sandi)" if os.environ.get("FACEDB_PASSPHRASE")
                                         else f"file {config.KEY_PATH}"))
        print(f"Orang terdaftar    : {len(people)} ({sum(c for c, _ in people.values())} sampel)")
        print(f"Persetujuan aktif  : {len(aktif)}   ditarik/batal/dihapus: {len(cons) - len(aktif)}")
        print(f"Tanpa persetujuan  : {len(tanpa)}" + (" -> " + ", ".join(tanpa[:10]) if tanpa else ""))
        print(f"Wajib persetujuan  : {'ya' if config.WAJIB_PERSETUJUAN else 'TIDAK (tidak sesuai UU PDP)'}")
        print(f"Liveness           : {config.LIVENESS_MODE}")
        print(f"Retensi absensi    : {config.RETENSI_HARI} hari")
        print(f"Rekam jejak        : {'UTUH' if ok else 'RUSAK'} - {msg}")
    elif c == "catat":
        name = a.nama.strip()
        db = Database()
        if name not in db.people():
            raise ValueError(f"'{name}' tidak terdaftar.")
        if reg.is_active(name):
            print(f"{name} sudah punya persetujuan aktif.")
            return
        if not ask_consent(name, a.bukti_persetujuan):
            privacy.audit("persetujuan_ditolak", name, {"metode": "cli"})
            print("Persetujuan tidak diberikan. Sesuai UU PDP, data wajahnya sebaiknya dihapus: "
                  f'python main.py privasi tarik --nama "{name}"')
            return
        reg.give(name, db.people()[name][1], "tertulis" if a.bukti_persetujuan else "cli", a.petugas,
                 a.bukti_persetujuan)
        print(f"Persetujuan {name} dicatat.")
    elif c == "tarik":
        name = a.nama.strip()
        removed = delete_person(name, a.alasan or "persetujuan ditarik subjek data", status="ditarik")
        if not removed and not reg.get(name):
            raise ValueError(f"'{name}' tidak ditemukan.")
        print(f"Persetujuan {name} ditarik; {removed} sampel wajah dihapus dari database terenkripsi. "
              "Riwayat absensi tetap (hapus lewat retensi bila tidak diperlukan lagi).")
    elif c == "akses":
        name = a.nama.strip()
        db = Database()
        people = db.people()
        rows = [{k: v for k, v in d.items() if k != "t"} for d in read_log(LOG_PATH) if d["nama"] == name]
        if name not in people and not rows and not reg.get(name):
            raise ValueError(f"Tidak ada data untuk '{name}'.")
        out = {"subjek": name, "dibuat": datetime.now().isoformat(timespec="seconds"),
               "pengendali": config.PENGENDALI, "kontak_pdp": config.KONTAK_PDP,
               "data_biometrik": {"jumlah_sampel_embedding": people.get(name, [0, ""])[0],
                                  "id": people.get(name, [0, ""])[1],
                                  "keterangan": "template wajah 128 angka per sampel, tersimpan terenkripsi; "
                                                "foto tidak disimpan"},
               "persetujuan": reg.get(name), "absensi": rows,
               "rekam_jejak": privacy.audit_entries(name)}
        path = a.keluar or os.path.join(EXPORT_DIR, f"data_subjek_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        privacy.audit("akses_data_subjek", name, {"file": os.path.basename(path), "baris_absensi": len(rows)})
        print(f"Salinan data pribadi {name} (Pasal 7): {path}")
    elif c == "retensi":
        hari = a.hari or config.RETENSI_HARI
        cutoff = (datetime.now() - timedelta(days=hari)).date()
        n, sisa = purge_before(cutoff, LOG_PATH, dry_run=not a.ya)
        if not a.ya:
            print(f"[SIMULASI] {n} catatan absensi lebih tua dari {cutoff} ({hari} hari) akan dihapus; {sisa} tersisa. "
                  "Jalankan ulang dengan --ya untuk menghapus.")
            return
        privacy.audit("retensi_absensi", detail={"sebelum": str(cutoff), "dihapus": n})
        print(f"{n} catatan absensi sebelum {cutoff} dihapus ({sisa} tersisa). "
              "Salinan di Firestore: python cloud_sync.py retensi --hari " + str(hari))
    elif c == "audit":
        ok, n, msg = privacy.verify_audit()
        print(("UTUH: " if ok else "PERINGATAN, RUSAK: ") + msg)
        if a.nama:
            for e in privacy.audit_entries(a.nama.strip()):
                print(f"  {e['waktu']}  {e['aksi']:<24} {e.get('pelaku', '')}  {e.get('detail', {})}")
        if not ok:
            sys.exit(2)
    elif c == "rotasi-kunci":
        pw = None
        if a.passphrase:
            import getpass
            pw = getpass.getpass("Kata sandi baru: ")
            if pw != getpass.getpass("Ulangi kata sandi: ") or len(pw) < 12:
                raise ValueError("Kata sandi tidak sama atau kurang dari 12 karakter.")
        n = privacy.rotate_key(pw)
        print(f"Database ({n} sampel) dienkripsi ulang dengan kunci baru."
              + (" Mulai sekarang set FACEDB_PASSPHRASE sebelum menjalankan program." if pw else ""))
    elif c == "hapus-tanpa-persetujuan":
        db = Database()
        tanpa = sorted(n for n in db.people() if not reg.is_active(n))
        if not tanpa:
            print("Semua wajah terdaftar memiliki persetujuan aktif.")
            return
        print(f"{len(tanpa)} orang tanpa persetujuan: " + ", ".join(tanpa))
        if not a.ya:
            print("Jalankan ulang dengan --ya untuk menghapus data wajah mereka.")
            return
        for n in tanpa:
            delete_person(n, "tidak ada persetujuan aktif (pembersihan)")
        print(f"Data wajah {len(tanpa)} orang dihapus.")


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
    p.add_argument("--bukti-persetujuan", default="",
                   help="rujukan formulir persetujuan TERTULIS yang sudah ditandatangani (lewati tanya SETUJU)")
    p.add_argument("--petugas", default="", help="nama petugas yang mendaftarkan (rekam jejak)")
    p.add_argument("--lewati-liveness", action="store_true",
                   help="matikan cek keaslian saat registrasi (diawasi petugas; dicatat di rekam jejak)")

    p = sub.add_parser("run", help="mode absen otomatis")
    p.add_argument("--kamera", type=int, default=CAM_INDEX)
    p.add_argument("--rtsp", default=os.environ.get("RTSP_URL", ""),
                   help="URL RTSP CCTV (selalu aktif, sambung ulang otomatis); mengalahkan --kamera")
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

    p = sub.add_parser("privasi", help="perlindungan data pribadi (UU No. 27/2022)",
                       description="naskah | status | catat | tarik | akses | retensi | audit | rotasi-kunci | "
                                   "hapus-tanpa-persetujuan")
    p.add_argument("aksi", choices=["naskah", "status", "catat", "tarik", "akses", "retensi", "audit",
                                    "rotasi-kunci", "hapus-tanpa-persetujuan"],
                   help="naskah: tampilkan pemberitahuan (Pasal 21); status: ringkasan; catat: rekam persetujuan "
                        "orang yang sudah terdaftar; tarik: tarik persetujuan + hapus wajah (Pasal 8-9); akses: "
                        "salinan data subjek (Pasal 7); retensi: hapus absensi lama (Pasal 43); audit: verifikasi "
                        "rekam jejak (Pasal 31); rotasi-kunci: enkripsi ulang dengan kunci baru")
    p.add_argument("--nama", default="")
    p.add_argument("--alasan", default="")
    p.add_argument("--bukti-persetujuan", default="", help="rujukan formulir persetujuan tertulis")
    p.add_argument("--petugas", default="")
    p.add_argument("--keluar", default="", help="file keluaran untuk 'akses'")
    p.add_argument("--hari", type=int, default=0, help="masa retensi (hari), bawaan dari config")
    p.add_argument("--ya", action="store_true", help="benar-benar jalankan aksi penghapusan")
    p.add_argument("--passphrase", action="store_true", help="rotasi-kunci: pakai kata sandi, bukan file kunci")

    a = ap.parse_args()
    try:
        if a.cmd in ("register", "run"):
            apply_profile(a.jarak)
        if a.cmd == "privasi":
            if a.aksi in ("catat", "tarik", "akses") and not a.nama.strip():
                raise ValueError(f"privasi {a.aksi} butuh --nama")
            privasi_cmd(a)
        elif a.cmd == "register":
            register(a.nama.strip(), a.id.strip(), a.jumlah, a.kamera, a.bukti_persetujuan.strip(), a.petugas,
                     a.lewati_liveness)
        elif a.cmd == "run":
            run_absen(a.rtsp.strip() or a.kamera, a.headless)
        elif a.cmd == "export":
            export_rekap(a.dari, a.sampai, a.format)
        elif a.cmd == "list":
            people = Database().people()
            reg = privacy.consents()
            if not people:
                print("Belum ada wajah terdaftar.")
            for n, (c, i) in sorted(people.items()):
                print(f"- {n}" + (f" (ID {i})" if i else "") + f": {c} sampel, persetujuan: "
                      + ("aktif" if reg.is_active(n) else "BELUM ADA"))
        elif a.cmd == "delete":
            removed = delete_person(a.nama.strip(), "dihapus admin (main.py delete)")
            print(f"{removed} sampel milik '{a.nama}' dihapus." if removed else f"Nama '{a.nama}' tidak ditemukan.")
        elif a.cmd == "download-models":
            print("Model siap:", *ensure_models(), sep="\n  ")
    except KeyboardInterrupt:
        print("\nDihentikan.")
    except (CameraError, RuntimeError, ValueError, OSError, cv2.error, TimeoutError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()