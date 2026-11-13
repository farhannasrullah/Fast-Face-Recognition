# -*- coding: utf-8 -*-
"""
config.py - SATU sumber konfigurasi aturan absensi (dipakai main.py, dashboard.py, cloud_sync.py).

Urutan prioritas: environment variable (huruf besar)  >  config.json di folder ini  >  bawaan.

  jam_terlambat     masuk setelah jam ini = "terlambat"                 (bawaan 09:00:00)
  jam_pulang        scan setelah jam ini oleh yang sudah masuk = pulang (bawaan 11:00:00)
  min_durasi_menit  jarak minimum masuk -> pulang, dalam menit          (bawaan 120)

Contoh config.json:
  {"jam_terlambat": "08:15", "jam_pulang": "16:00", "min_durasi_menit": 240}

Nilai yang salah format menghentikan program dengan pesan jelas (fail fast), bukan diam-diam
memakai nilai bawaan.
"""
import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "absensi.csv")

_DEFAULTS = {"jam_terlambat": "09:00:00", "jam_pulang": "11:00:00", "min_durasi_menit": "120"}


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


def _menit(key):
    raw = _get(key)
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError(f"Konfigurasi '{key}' tidak valid: {raw!r} (harus bilangan bulat >= 0).")
    return n


JAM_TERLAMBAT = _jam("jam_terlambat")
JAM_PULANG = _jam("jam_pulang")
MIN_DURASI_MENIT = _menit("min_durasi_menit")