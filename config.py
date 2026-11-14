# -*- coding: utf-8 -*-
"""
config.py - SATU sumber konfigurasi (dipakai main.py, dashboard.py, cloud_sync.py, evaluate.py).

Urutan prioritas: environment variable (huruf besar)  >  config.json di folder ini  >  bawaan.

Aturan absensi
  jam_terlambat        masuk setelah jam ini = "terlambat"                          (09:00:00)
  jam_pulang           scan setelah jam ini oleh yang sudah masuk = pulang          (11:00:00)
  min_durasi_menit     jarak minimum masuk -> pulang, dalam menit                   (120)

Pengenalan wajah (kalibrasi lewat `python evaluate.py pengenalan`)
  match_thresh         ambang skor kosinus orang (rata-rata 2 sampel terbaik)       (0.40)
  match_margin         selisih minimum skor orang terbaik vs kedua (0 = nonaktif)    (0.05)

Liveness / anti-spoofing (kalibrasi lewat `python evaluate.py liveness`)
  liveness_mode        off | pasif | aktif | ganda                                  (pasif)
                       pasif = model MiniFASNet (tanpa interaksi, cocok CCTV/headless)
                       aktif = tantangan gerak kepala acak (geleng/angguk), butuh layar
                       ganda = keduanya harus lolos (paling aman untuk kios absen)
  liveness_thresh      ambang rata-rata probabilitas "asli" model pasif             (0.80)
  liveness_min_frames  jumlah frame minimum sebelum diputuskan "asli"                (3)
  liveness_window      jumlah frame jendela keputusan                               (8)
  liveness_spoof_thresh  rata-rata di BAWAH ini (setelah jendela penuh) = dinyatakan PALSU.
                       Antara spoof_thresh dan liveness_thresh = "belum yakin": tidak dicatat,
                       tidak dicap palsu, pengguna diminta mendekat/memperbaiki cahaya  (0.50)
  liveness_min_px      lebar wajah minimum (px, frame asli) agar model pasif mau menilai.
                       Wajah lebih kecil/terpotong tepi layar TIDAK dinilai (0 = nonaktif) (80)
  tantangan_ambang     rentang koordinat afin hidung minimum agar tantangan lolos    (0.28)
  tantangan_timeout    detik untuk menyelesaikan tantangan                          (8)
  tantangan_min_px     lebar wajah minimum (px, frame asli) untuk tantangan aktif    (100)

Privasi (UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi)
  wajib_persetujuan    true = wajah tanpa persetujuan aktif TIDAK dikenali           (true)
  retensi_hari         masa simpan catatan absensi (hari) untuk `privasi retensi`    (365)
  pengendali           nama instansi pengendali data (ditampilkan di naskah)
  kontak_pdp           kontak petugas pelindungan data (ditampilkan di naskah)
  Kunci enkripsi faces_db: env FACEDB_PASSPHRASE (kata sandi) ATAU file kunci
  (env FACEDB_KEY_FILE, bawaan: kunci/facedb.key, dibuat otomatis).

Contoh config.json:
  {"jam_terlambat": "08:15", "jam_pulang": "16:00", "min_durasi_menit": 240,
   "liveness_mode": "ganda", "pengendali": "Universitas Contoh", "kontak_pdp": "pdp@contoh.ac.id"}

Nilai yang salah format menghentikan program dengan pesan jelas (fail fast), bukan diam-diam
memakai nilai bawaan.
"""
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------- lokasi file (satu tempat) -----------------------------
LOG_PATH = os.path.join(BASE_DIR, "absensi.csv")              # catatan absen
DB_PATH = os.path.join(BASE_DIR, "faces_db.enc")              # embedding wajah TERENKRIPSI (AES-256-GCM)
LEGACY_DB_PATH = os.path.join(BASE_DIR, "faces_db.npz")       # format lama (polos), dimigrasi otomatis
CONSENT_PATH = os.path.join(BASE_DIR, "persetujuan.json")     # catatan persetujuan subjek data
AUDIT_PATH = os.path.join(BASE_DIR, "audit_log.jsonl")        # rekam jejak berantai hash (Pasal 31)
SPOOF_LOG_PATH = os.path.join(BASE_DIR, "spoof_log.csv")      # percobaan spoofing yang terdeteksi
NOTICE_PATH = os.path.join(BASE_DIR, "naskah_persetujuan.txt")  # opsional: naskah persetujuan kustom
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXPORT_DIR = os.path.join(BASE_DIR, "export")

_DEFAULTS = {
    "jam_terlambat": "09:00:00", "jam_pulang": "11:00:00", "min_durasi_menit": "120",
    "match_thresh": "0.40", "match_margin": "0.05",
    "liveness_mode": "pasif", "liveness_thresh": "0.80",
    "liveness_min_frames": "3", "liveness_window": "8",
    "liveness_spoof_thresh": "0.50", "liveness_min_px": "80",
    "tantangan_ambang": "0.28", "tantangan_timeout": "8", "tantangan_min_px": "100",
    "wajib_persetujuan": "true", "retensi_hari": "365",
    "pengendali": "(isi nama instansi di config.json: \"pengendali\")",
    "kontak_pdp": "(isi kontak petugas pelindungan data di config.json: \"kontak_pdp\")",
}


def _load_file():
    try:
        with open(os.path.join(BASE_DIR, "config.json"), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


_FILE = _load_file()


def _get(key):
    v = os.environ.get(key.upper())
    if v is None or not str(v).strip():
        v = _FILE.get(key, _DEFAULTS[key])
    return str(v).strip()


def _jam(key):
    raw = _get(key)
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(raw, fmt).strftime("%H:%M:%S")   # selalu HH:MM:SS (aman dibandingkan sebagai string)
        except ValueError:
            pass
    raise ValueError(f"Konfigurasi '{key}' tidak valid: {raw!r} (gunakan HH:MM atau HH:MM:SS).")


def _int(key, lo=0, hi=None):
    raw = _get(key)
    try:
        n = int(raw)
    except ValueError:
        n = None
    if n is None or n < lo or (hi is not None and n > hi):
        rng = f">= {lo}" if hi is None else f"{lo}..{hi}"
        raise ValueError(f"Konfigurasi '{key}' tidak valid: {raw!r} (harus bilangan bulat {rng}).")
    return n


def _float(key, lo, hi):
    raw = _get(key)
    try:
        x = float(raw)
    except ValueError:
        x = None
    if x is None or not lo <= x <= hi:
        raise ValueError(f"Konfigurasi '{key}' tidak valid: {raw!r} (harus angka {lo}..{hi}).")
    return x


def _choice(key, options):
    raw = _get(key).lower()
    if raw not in options:
        raise ValueError(f"Konfigurasi '{key}' tidak valid: {raw!r} (pilih salah satu: {', '.join(options)}).")
    return raw


def _bool(key):
    raw = _get(key).lower()
    if raw in ("1", "true", "ya", "yes", "on"):
        return True
    if raw in ("0", "false", "tidak", "no", "off"):
        return False
    raise ValueError(f"Konfigurasi '{key}' tidak valid: {raw!r} (true/false).")


JAM_TERLAMBAT = _jam("jam_terlambat")
JAM_PULANG = _jam("jam_pulang")
MIN_DURASI_MENIT = _int("min_durasi_menit")

MATCH_THRESH = _float("match_thresh", 0.0, 1.0)
MATCH_MARGIN = _float("match_margin", 0.0, 1.0)

LIVENESS_MODE = _choice("liveness_mode", ("off", "pasif", "aktif", "ganda"))
LIVENESS_THRESH = _float("liveness_thresh", 0.0, 1.0)
LIVENESS_MIN_FRAMES = _int("liveness_min_frames", 1, 100)
LIVENESS_WINDOW = max(_int("liveness_window", 1, 200), LIVENESS_MIN_FRAMES)
LIVENESS_SPOOF_THRESH = _float("liveness_spoof_thresh", 0.0, 1.0)
LIVENESS_MIN_PX = _int("liveness_min_px", 0, 2000)
if LIVENESS_SPOOF_THRESH > LIVENESS_THRESH:
    raise ValueError(f"Konfigurasi 'liveness_spoof_thresh' ({LIVENESS_SPOOF_THRESH}) tidak boleh lebih besar "
                     f"dari 'liveness_thresh' ({LIVENESS_THRESH}).")
TANTANGAN_AMBANG = _float("tantangan_ambang", 0.01, 2.0)
TANTANGAN_TIMEOUT = _int("tantangan_timeout", 2, 120)
TANTANGAN_MIN_PX = _int("tantangan_min_px", 20, 2000)

WAJIB_PERSETUJUAN = _bool("wajib_persetujuan")
RETENSI_HARI = _int("retensi_hari", 1)
PENGENDALI = _get("pengendali")
KONTAK_PDP = _get("kontak_pdp")
KEY_PATH = os.environ.get("FACEDB_KEY_FILE") or os.path.join(BASE_DIR, "kunci", "facedb.key")