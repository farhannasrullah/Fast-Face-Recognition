# -*- coding: utf-8 -*-
"""
privacy.py - pelindungan data biometrik sesuai UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi.

Embedding wajah adalah DATA BIOMETRIK = Data Pribadi yang bersifat spesifik (Pasal 4 ayat (2) huruf b).
Modul ini menyediakan kontrol teknis yang dipakai main.py dan dashboard.py:

  1. Enkripsi saat disimpan (Pasal 35, 36): faces_db.enc dienkripsi AES-256-GCM (terautentikasi;
     file yang diubah/rusak ditolak). Kunci: env FACEDB_PASSPHRASE (diturunkan dengan scrypt) atau
     file kunci acak 256-bit (bawaan kunci/facedb.key, dibuat otomatis, izin 600).
     faces_db.npz lama (tidak terenkripsi) dimigrasi otomatis lalu ditimpa nol dan dihapus.
  2. Persetujuan eksplisit tertulis/terekam (Pasal 20 ayat (2) huruf a, Pasal 22) beserta
     informasi wajib sebelum persetujuan (Pasal 21): persetujuan.json mencatat siapa, kapan,
     versi + hash naskah, metode, petugas. Penarikan persetujuan (Pasal 9) menghapus data biometrik.
  3. Rekam jejak pemrosesan (Pasal 31): audit_log.jsonl berantai hash SHA-256 sehingga
     penghapusan/pengubahan baris di tengah dapat dideteksi (`python main.py privasi audit`).
  4. Hak akses & salinan (Pasal 7) dan penghapusan (Pasal 8, 43): lihat perintah `privasi` di main.py.

Catatan jujur: enkripsi melindungi file database (cadangan, salinan, pencurian disk). Saat program
berjalan, embedding tetap ada di RAM dalam bentuk terdekripsi; dan bila file kunci disimpan di folder
yang sama dengan database, siapa pun yang menyalin SELURUH folder tetap bisa mendekripsi. Simpan kunci
di lokasi terpisah (FACEDB_KEY_FILE) atau pakai FACEDB_PASSPHRASE untuk perlindungan penuh.
"""
import base64
import getpass
import hashlib
import io
import json
import os
import secrets
import socket
import threading
from datetime import datetime

import numpy as np

import config
from ipclock import InterProcessLock

try:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:          # dilaporkan dengan pesan jelas saat dibutuhkan
    AESGCM = None
    InvalidTag = Exception

MAGIC = b"FDBENC01"
KDF_KEYFILE, KDF_SCRYPT = 0, 1
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1
KEYFILE_HEADER = "FACEDB-KEY-V1"
NOTICE_VERSION = "1.0"


class PrivacyError(RuntimeError):
    pass


def _need_crypto():
    if AESGCM is None:
        raise PrivacyError("Enkripsi database wajah butuh paket 'cryptography': pip install cryptography")


def actor():
    """Siapa yang melakukan aksi (untuk rekam jejak)."""
    try:
        user = getpass.getuser()
    except Exception:
        user = "?"
    return f"{user}@{socket.gethostname()}"


def _atomic_write(path, data, mode=0o600):
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), mode)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    for i in range(10):                      # Windows: file tujuan bisa sedang dibaca proses lain
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 9:
                os.remove(tmp)
                raise
            threading.Event().wait(0.1)


def wipe_file(path):
    """Timpa isi file dengan nol lalu hapus (upaya terbaik; pada SSD/jurnal tidak dijamin 100%)."""
    try:
        n = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.write(b"\0" * n)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ============================== kunci ==============================
_key_cache = {}
_key_lock = threading.Lock()


def _keyfile_key(create):
    path = config.KEY_PATH
    if not os.path.exists(path):
        if not create:
            raise PrivacyError(f"File kunci tidak ditemukan: {path}. Tanpa kunci ini database wajah tidak "
                               "bisa dibuka. Kembalikan dari cadangan atau set FACEDB_KEY_FILE.")
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        key = secrets.token_bytes(32)
        body = f"{KEYFILE_HEADER}\n{base64.b64encode(key).decode()}\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(body)
        print(f"[PRIVASI] Kunci enkripsi baru dibuat: {path}\n"
              "          CADANGKAN file ini di tempat terpisah. Tanpa kunci, data wajah tidak bisa dibuka.")
        audit("kunci_dibuat", detail={"lokasi": path})
        return key
    with open(path, encoding="ascii") as f:
        lines = f.read().split()
    if len(lines) != 2 or lines[0] != KEYFILE_HEADER:
        raise PrivacyError(f"Format file kunci tidak dikenal: {path}")
    key = base64.b64decode(lines[1])
    if len(key) != 32:
        raise PrivacyError(f"Panjang kunci salah di {path}")
    return key


def _key(kdf, salt, create=False):
    """Kunci AES-256 untuk (kdf, salt). Dicache agar scrypt tidak dihitung tiap muat ulang."""
    with _key_lock:
        ck = (kdf, salt, config.KEY_PATH if kdf == KDF_KEYFILE else None)
        if ck in _key_cache:
            return _key_cache[ck]
        if kdf == KDF_SCRYPT:
            pw = os.environ.get("FACEDB_PASSPHRASE", "")
            if not pw:
                raise PrivacyError("Database wajah dienkripsi dengan kata sandi: set environment variable "
                                   "FACEDB_PASSPHRASE.")
            key = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                                 maxmem=128 * 1024 * 1024, dklen=32)
        else:
            key = _keyfile_key(create)
        _key_cache[ck] = key
        return key


def key_id(key):
    return hashlib.sha256(b"facedb-key-id" + key).digest()[:8]


def encrypt_bytes(plain):
    """Header: MAGIC | kdf(1) | salt(16) | key_id(8) | nonce(12) ; lalu ciphertext+tag. Header = AAD."""
    _need_crypto()
    if os.environ.get("FACEDB_PASSPHRASE"):
        kdf, salt = KDF_SCRYPT, secrets.token_bytes(16)
    else:
        kdf, salt = KDF_KEYFILE, b"\0" * 16
    key = _key(kdf, salt, create=True)
    nonce = secrets.token_bytes(12)
    header = MAGIC + bytes([kdf]) + salt + key_id(key) + nonce
    return header + AESGCM(key).encrypt(nonce, plain, header)


def is_encrypted(blob):
    return blob[:len(MAGIC)] == MAGIC


def decrypt_bytes(blob, source="database"):
    _need_crypto()
    if not is_encrypted(blob) or len(blob) < len(MAGIC) + 1 + 16 + 8 + 12 + 16:
        raise PrivacyError(f"{source} bukan file terenkripsi yang valid.")
    p = len(MAGIC)
    kdf, salt = blob[p], blob[p + 1:p + 17]
    kid, nonce = blob[p + 17:p + 25], blob[p + 25:p + 37]
    header = blob[:p + 37]
    key = _key(kdf, salt)
    if key_id(key) != kid:
        raise PrivacyError(f"Kunci tidak cocok untuk {source} (" +
                           ("FACEDB_PASSPHRASE salah" if kdf == KDF_SCRYPT else f"file kunci {config.KEY_PATH} berbeda")
                           + ").")
    try:
        return AESGCM(key).decrypt(nonce, blob[p + 37:], header)
    except InvalidTag:
        raise PrivacyError(f"{source} rusak atau telah diubah (verifikasi integritas AES-GCM gagal).")


# ============================== database wajah ==============================
def _pack(names, ids, embs):
    buf = io.BytesIO()
    np.savez_compressed(buf, names=np.array(names, dtype=str), ids=np.array(ids, dtype=str),
                        embs=np.asarray(embs, np.float32).reshape(-1, 128))
    return buf.getvalue()


def _unpack(raw, source):
    with np.load(io.BytesIO(raw), allow_pickle=False) as d:
        names = [str(x) for x in d["names"]]
        embs = np.ascontiguousarray(d["embs"], dtype=np.float32).reshape(-1, 128)
        ids = [str(x) for x in d["ids"]] if "ids" in d.files else [""] * len(names)
    if not (len(names) == len(ids) == len(embs)):
        raise ValueError(f"{source}: jumlah nama, id, dan embedding tidak sama")
    return names, ids, embs


def db_lock(path=None):
    return InterProcessLock((path or config.DB_PATH) + ".lock", timeout=15)


def save_db(names, ids, embs, path=None):
    """Simpan database wajah terenkripsi secara atomik."""
    path = path or config.DB_PATH
    _atomic_write(path, encrypt_bytes(_pack(names, ids, embs)))


def migrate_legacy(path=None, legacy=None):
    """faces_db.npz polos -> faces_db.enc terenkripsi. Mengembalikan True bila ada migrasi."""
    path, legacy = path or config.DB_PATH, legacy or config.LEGACY_DB_PATH
    if os.path.exists(path) or not os.path.exists(legacy):
        return False
    with db_lock(path):
        if os.path.exists(path) or not os.path.exists(legacy):   # proses lain sudah memigrasi
            return False
        with open(legacy, "rb") as f:
            raw = f.read()
        names, ids, embs = _unpack(raw, legacy)
        save_db(names, ids, embs, path)
        if load_db(path, migrate=False)[0] != names:              # verifikasi sebelum menghapus yang lama
            raise PrivacyError("Verifikasi migrasi gagal; file lama tidak dihapus.")
        wipe_file(legacy)
    print(f"[PRIVASI] {os.path.basename(legacy)} (tidak terenkripsi) dimigrasi ke "
          f"{os.path.basename(path)} (AES-256-GCM) dan file lama dihapus.")
    audit("migrasi_enkripsi", detail={"sampel": len(names), "orang": len(set(names))})
    return True


def load_db(path=None, migrate=True):
    """(names, ids, embs). Database belum ada -> kosong."""
    path = path or config.DB_PATH
    if migrate and path == config.DB_PATH:
        migrate_legacy(path)
    if not os.path.exists(path):
        return [], [], np.zeros((0, 128), np.float32)
    with open(path, "rb") as f:
        blob = f.read()
    return _unpack(decrypt_bytes(blob, os.path.basename(path)), path)


def rotate_key(new_passphrase=None):
    """Enkripsi ulang database dengan kunci baru (file kunci baru, atau kata sandi baru)."""
    with db_lock():
        names, ids, embs = load_db(migrate=False)
        old_env = os.environ.get("FACEDB_PASSPHRASE")
        if new_passphrase is not None:
            os.environ["FACEDB_PASSPHRASE"] = new_passphrase
        else:
            os.environ.pop("FACEDB_PASSPHRASE", None)
            if os.path.exists(config.KEY_PATH):
                bak = config.KEY_PATH + datetime.now().strftime(".lama-%Y%m%d%H%M%S")
                os.replace(config.KEY_PATH, bak)
                print(f"[PRIVASI] Kunci lama dipindah ke {bak} (hapus setelah memastikan semua berjalan).")
        _key_cache.clear()
        try:
            save_db(names, ids, embs)
        except Exception:
            if old_env is not None:
                os.environ["FACEDB_PASSPHRASE"] = old_env
            raise
    audit("rotasi_kunci", detail={"metode": "passphrase" if new_passphrase else "file_kunci"})
    return len(names)


# ============================== naskah persetujuan (Pasal 21) ==============================
NOTICE_TEMPLATE = """\
PEMBERITAHUAN DAN PERSETUJUAN PEMROSESAN DATA BIOMETRIK WAJAH
(UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi) - versi {versi}

Pengendali data : {pengendali}
Kontak PDP      : {kontak}

1. Legalitas: pemrosesan dilakukan atas dasar PERSETUJUAN EKSPLISIT Anda (Pasal 20 ayat (2) huruf a).
   Data wajah termasuk data biometrik = data pribadi bersifat spesifik (Pasal 4 ayat (2) huruf b).
2. Tujuan: HANYA untuk mencatat kehadiran (masuk/pulang) secara otomatis. Tidak dipakai untuk
   tujuan lain, tidak dijual, dan tidak dibagikan ke pihak ketiga.
3. Jenis data: (a) template wajah berupa vektor angka 128 dimensi (embedding) yang diturunkan dari
   foto saat pendaftaran - FOTO TIDAK DISIMPAN; (b) nama dan nomor induk; (c) catatan absensi
   (tanggal, jam, skor kecocokan, status). Saat absen, kamera memeriksa keaslian wajah (liveness)
   tanpa menyimpan gambar.
4. Keamanan: template wajah disimpan terenkripsi (AES-256-GCM) di komputer pengendali dan tidak
   dikirim ke layanan awan. Catatan absensi (bukan data biometrik) dapat disinkronkan ke
   Firebase/Google Sheets milik pengendali.
5. Masa simpan: template wajah disimpan selama Anda terdaftar aktif dan dihapus saat Anda menarik
   persetujuan atau tidak lagi terdaftar. Catatan absensi disimpan {retensi} hari.
6. Hak Anda: mendapat informasi (Pasal 5), memperbarui data (Pasal 6), mengakses dan memperoleh
   salinan (Pasal 7), meminta penghapusan (Pasal 8), MENARIK PERSETUJUAN kapan saja (Pasal 9) -
   template wajah Anda segera dihapus - serta mengajukan keberatan atas keputusan otomatis (Pasal 10).
   Tanpa persetujuan, Anda tetap dapat absen dengan cara manual yang disediakan pengendali.
7. Bila terjadi kegagalan pelindungan data, Anda diberi tahu paling lambat 3 x 24 jam (Pasal 46).

Dengan menyatakan SETUJU, Anda menyatakan telah membaca dan memahami informasi di atas dan
memberikan persetujuan secara sukarela.
"""


def notice():
    """(versi, teks, sha256). Naskah kustom dibaca dari naskah_persetujuan.txt bila ada."""
    try:
        with open(config.NOTICE_PATH, encoding="utf-8") as f:
            text = f.read()
        text = text.format(pengendali=config.PENGENDALI, kontak=config.KONTAK_PDP,
                           retensi=config.RETENSI_HARI, versi="kustom")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"kustom-{digest[:8]}", text, digest
    except (OSError, KeyError, ValueError, IndexError):
        pass
    text = NOTICE_TEMPLATE.format(versi=NOTICE_VERSION, pengendali=config.PENGENDALI,
                                  kontak=config.KONTAK_PDP, retensi=config.RETENSI_HARI)
    return NOTICE_VERSION, text, hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================== persetujuan (Pasal 20, 22, 9) ==============================
class ConsentRegistry:
    """persetujuan.json: {"versi": 1, "subjek": {nama: catatan}}. Aman dipakai lintas proses."""

    def __init__(self, path=None):
        self.path = path or config.CONSENT_PATH
        self._lock = InterProcessLock(self.path + ".lock", timeout=10)
        self._sig = None
        self._data = {"versi": 1, "subjek": {}}

    def _sig_now(self):
        try:
            st = os.stat(self.path)
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    def signature(self):
        return self._sig_now()

    def _read(self):
        sig = self._sig_now()
        if sig == self._sig:
            return self._data
        data = {"versi": 1, "subjek": {}}
        if sig is not None:
            try:
                with open(self.path, encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict) and isinstance(d.get("subjek"), dict):
                    data = d
            except (OSError, ValueError) as e:
                raise PrivacyError(f"{self.path} rusak: {e}")
        self._sig, self._data = sig, data
        return data

    def _write(self, data):
        _atomic_write(self.path, json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"), 0o640)
        self._sig, self._data = self._sig_now(), data

    def all(self):
        return dict(self._read()["subjek"])

    def get(self, nama):
        return self._read()["subjek"].get(nama)

    def is_active(self, nama):
        r = self.get(nama)
        return bool(r and r.get("status") == "aktif")

    def active_names(self):
        return {n for n, r in self._read()["subjek"].items() if r.get("status") == "aktif"}

    def give(self, nama, pid, metode, petugas, bukti=""):
        ver, _, digest = notice()
        with self._lock:
            self._sig = None
            data = self._read()
            rec = {"nama": nama, "id": pid or "", "status": "aktif",
                   "diberikan": datetime.now().isoformat(timespec="seconds"),
                   "versi_naskah": ver, "hash_naskah": digest, "metode": metode,
                   "petugas": petugas or actor(), "bukti": bukti, "ditarik": None, "alasan_tarik": ""}
            prev = data["subjek"].get(nama)
            if prev:
                rec["riwayat"] = (prev.get("riwayat") or []) + [{k: prev.get(k) for k in
                                                                 ("status", "diberikan", "ditarik", "versi_naskah")}]
            data["subjek"][nama] = rec
            self._write(data)
        audit("persetujuan_diberikan", nama, {"versi_naskah": ver, "metode": metode, "bukti": bukti},
              pelaku=rec["petugas"])
        return rec

    def withdraw(self, nama, alasan="ditarik subjek data", status="ditarik", pelaku=None):
        with self._lock:
            self._sig = None
            data = self._read()
            rec = data["subjek"].get(nama)
            if rec is None:
                return None
            rec.update(status=status, ditarik=datetime.now().isoformat(timespec="seconds"), alasan_tarik=alasan)
            self._write(data)
        audit("persetujuan_ditarik", nama, {"alasan": alasan, "status": status}, pelaku=pelaku)
        return rec

    def rename(self, old, new):
        with self._lock:
            self._sig = None
            data = self._read()
            if old in data["subjek"] and old != new:
                rec = data["subjek"].pop(old)
                rec["nama"] = new
                rec.setdefault("nama_lama", []).append(old)
                data["subjek"][new] = rec
                self._write(data)

    def set_id(self, nama, pid):
        with self._lock:
            self._sig = None
            data = self._read()
            if nama in data["subjek"]:
                data["subjek"][nama]["id"] = pid
                self._write(data)


_registry = None


def consents():
    global _registry
    if _registry is None:
        _registry = ConsentRegistry()
    return _registry


# ============================== rekam jejak berantai hash (Pasal 31) ==============================
_audit_lock = InterProcessLock(config.AUDIT_PATH + ".lock", timeout=10)
GENESIS = "0" * 64


def _entry_hash(prev, body):
    return hashlib.sha256((prev + json.dumps(body, ensure_ascii=False, sort_keys=True)).encode("utf-8")).hexdigest()


def _last_hash(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - 8192))
            lines = f.read().splitlines()
        for line in reversed(lines):
            if line.strip():
                return json.loads(line)["hash"]
    except (OSError, ValueError, KeyError):
        pass
    return GENESIS


def audit(aksi, subjek="", detail=None, pelaku=None):
    """Tambah satu entri rekam jejak. Tidak pernah melempar error (log tidak boleh mematikan absensi)."""
    try:
        with _audit_lock:
            prev = _last_hash(config.AUDIT_PATH)
            body = {"waktu": datetime.now().isoformat(timespec="seconds"), "aksi": aksi, "subjek": subjek,
                    "pelaku": pelaku or actor(), "detail": detail or {}, "prev": prev}
            body["hash"] = _entry_hash(prev, body)
            with open(config.AUDIT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(body, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[PRIVASI] gagal menulis rekam jejak: {e}")


def verify_audit(path=None):
    """(ok, jumlah_entri, pesan). Mendeteksi entri yang diubah, dihapus, atau disisipkan."""
    path = path or config.AUDIT_PATH
    if not os.path.exists(path):
        return True, 0, "Belum ada rekam jejak."
    prev, n = GENESIS, 0
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                h = e.pop("hash")
            except (ValueError, KeyError):
                return False, n, f"baris {no}: format rusak"
            if e.get("prev") != prev:
                return False, n, f"baris {no}: rantai terputus (ada entri yang dihapus/disisipkan sebelum baris ini)"
            if _entry_hash(prev, e) != h:
                return False, n, f"baris {no}: isi entri telah diubah"
            prev, n = h, n + 1
    return True, n, f"Rantai utuh: {n} entri, hash terakhir {prev[:16]}..."


def audit_entries(subjek=None):
    out = []
    try:
        with open(config.AUDIT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if subjek is None or e.get("subjek") == subjek:
                    out.append(e)
    except OSError:
        pass
    return out
