# -*- coding: utf-8 -*-
"""
cloud_backup.py - cadangan data penting ke Firestore + pemulihan otomatis saat server mulai.

Untuk hosting yang disknya hilang tiap deploy (Railway tanpa Volume). Yang dicadangkan:
  faces_db.enc       database wajah, SUDAH terenkripsi AES-256-GCM oleh privacy.py (hanya dicadangkan
                     bila FACEDB_PASSPHRASE diisi; file kunci TIDAK PERNAH dikirim ke cloud)
  persetujuan.json   catatan persetujuan subjek data
  audit_log.jsonl    rekam jejak berantai hash
  absensi.csv        catatan absensi

Cara kerja: restore() mengunduh file yang BELUM ada di disk lokal; start() menyalakan thread yang
mengunggah file yang berubah tiap BACKUP_DETIK (bawaan 60). Bila pemulihan gagal, unggahan TIDAK
dinyalakan agar cadangan bagus tidak tertimpa file kosong. Matikan dengan env BACKUP_FIRESTORE=false.
"""
import base64
import hashlib
import os
import threading
import time
from datetime import datetime, timezone

import config

COLLECTION = os.environ.get("BACKUP_COLLECTION", "backup_data")
CHUNK = 600_000            # karakter base64 per dokumen (batas dokumen Firestore ~1 MiB)
INTERVAL = int(os.environ.get("BACKUP_DETIK") or 60)
FILES = ["faces_db.enc", "persetujuan.json", "audit_log.jsonl", "absensi.csv"]

_last = {}                 # nama file -> sha256 terakhir yang sudah diunggah
_ok_to_upload = False


def _enabled():
    return os.environ.get("BACKUP_FIRESTORE", "true").strip().lower() not in ("0", "false", "no", "off", "tidak")


def _skip(name):
    return name == "faces_db.enc" and not os.environ.get("FACEDB_PASSPHRASE", "").strip()


def _client():
    import cloud_sync
    s = cloud_sync.get_sync()
    if not s.fs_on:
        return None
    return s.fs_client()


def _path(name):
    return os.path.join(config.DATA_DIR, name)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _upload(db, name, data):
    sha = _sha(data)
    b64 = base64.b64encode(data).decode("ascii")
    parts = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)] or [""]
    col = db.collection(COLLECTION)
    for i, p in enumerate(parts):
        col.document(f"{name}__{i}").set({"data": p, "sha": sha})
    col.document(name).set({"parts": len(parts), "sha": sha, "bytes": len(data),
                            "ts": datetime.now(timezone.utc).isoformat()})   # manifes ditulis TERAKHIR
    _last[name] = sha


def _download(db, name):
    col = db.collection(COLLECTION)
    man = col.document(name).get()
    if not man.exists:
        return None
    m = man.to_dict()
    b64 = ""
    for i in range(int(m["parts"])):
        d = col.document(f"{name}__{i}").get()
        if not d.exists:
            raise RuntimeError(f"bagian {i} dari {name} hilang di Firestore")
        b64 += d.to_dict()["data"]
    data = base64.b64decode(b64)
    if _sha(data) != m["sha"]:
        raise RuntimeError(f"checksum {name} tidak cocok")
    return data


def restore():
    """Unduh file yang belum ada di disk lokal. Mengembalikan True bila aman untuk mulai mengunggah."""
    global _ok_to_upload
    if not _enabled():
        return False
    try:
        db = _client()
        if db is None:
            print("[BACKUP] nonaktif: Firestore belum aktif.")
            return False
        for name in FILES:
            if _skip(name):
                print(f"[BACKUP] {name} dilewati: isi FACEDB_PASSPHRASE agar database wajah bisa dicadangkan.")
                continue
            p = _path(name)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
            data = _download(db, name)
            if data is None:
                continue
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".restore.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
            _last[name] = _sha(data)
            print(f"[BACKUP] dipulihkan dari Firestore: {name} ({len(data)} byte)")
        _ok_to_upload = True
        print(f"[BACKUP] siap: cek tiap {INTERVAL} dtk, koleksi Firestore '{COLLECTION}'.")
    except Exception as e:      # jangan menjatuhkan dashboard karena cadangan
        print(f"[BACKUP] pemulihan GAGAL ({str(e)[:200]}); unggahan cadangan ditunda.")
        _ok_to_upload = False
    return _ok_to_upload


def backup_once():
    db = _client()
    if db is None:
        return
    for name in FILES:
        if _skip(name):
            continue
        p = _path(name)
        try:
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            with open(p, "rb") as f:
                data = f.read()
            if _sha(data) != _last.get(name):
                _upload(db, name, data)
                print(f"[BACKUP] {name} dicadangkan ke Firestore ({len(data)} byte).")
        except Exception as e:
            print(f"[BACKUP] gagal mencadangkan {name}: {str(e)[:200]}")


def _loop():
    while True:
        time.sleep(INTERVAL)
        try:
            backup_once()
        except Exception as e:
            print(f"[BACKUP] error: {str(e)[:200]}")


def start():
    if _enabled() and _ok_to_upload:
        threading.Thread(target=_loop, name="cloud-backup", daemon=True).start()