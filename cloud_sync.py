#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cloud_sync.py - sinkronisasi OTOMATIS absensi ke Firebase Firestore dan Google Sheets.

Tidak perlu menjalankan perintah apa pun. Cukup jalankan `python main.py run` atau
`python dashboard.py`; semua absen otomatis terkirim.

Cara kerja
  1. Absen baru: main.py (AttendanceLogger.log) memanggil enqueue(). Catatan masuk antrean dan
     dikirim oleh thread latar (tidak memperlambat pengenalan wajah).
  2. Catch-up otomatis: saat program mulai, lalu tiap 5 menit, seluruh absensi.csv dibandingkan dengan
     cloud_state.json. Baris yang belum pernah terkirim (data lama, atau saat internet mati / cloud
     belum dikonfigurasi) dikirim otomatis.
  3. Gagal kirim (internet putus, jam komputer salah, dll): dicoba lagi otomatis dengan jeda
     bertahap, termasuk setelah program dimulai ulang.
  4. Tanpa data ganda: Firestore memakai ID dokumen deterministik; Google Sheets dicek lewat kolom
     "key" sebelum menambah baris.

Data di Firestore
  absensi/{tanggal}_{nama}_{tipe}   -> satu dokumen per kejadian (masuk / pulang)
  rekap_harian/{tanggal}_{nama}     -> satu dokumen per orang per hari (jam_masuk, status, jam_pulang)

Kolom di Google Sheets (tab "Absensi"):
  tanggal | jam | nama | id | skor | tipe | status | key

Konfigurasi: environment variable (huruf besar) ATAU file cloud_config.json di folder ini.
  credentials          path service account JSON (bawaan: serviceAccount.json)    GOOGLE_CREDENTIALS
  firestore_enabled    true/false (bawaan: true)                                  FIRESTORE_ENABLED
  firestore_collection nama koleksi log (bawaan: absensi)                         FIRESTORE_COLLECTION
  firestore_daily      nama koleksi rekap harian (bawaan: rekap_harian)           FIRESTORE_DAILY
  gsheet_id            ID spreadsheet (kosong = Google Sheets nonaktif)           GSHEET_ID
  gsheet_tab           nama tab (bawaan: Absensi)                                 GSHEET_TAB
  (jam_terlambat kini diatur di config.py / config.json / env JAM_TERLAMBAT)

Perintah (opsional, untuk pengecekan manual):
  python cloud_sync.py test       # cek koneksi Firestore dan Google Sheets
  python cloud_sync.py backfill   # paksa catch-up sekarang (sinkron, sampai selesai)
  python cloud_sync.py retensi --hari 365 [--ya]   # hapus dokumen Firestore lebih tua dari N hari (UU PDP Pasal 43)

Privasi: yang dikirim hanya catatan absensi (nama, id, jam, skor, status). Data biometrik (embedding
wajah) TIDAK PERNAH dikirim ke cloud. Google Sheets tidak dihapus otomatis oleh `retensi`; hapus baris
lama secara manual (filter kolom tanggal) sesuai masa retensi yang sama.

Perbaikan efisiensi:
  * catch-up berkala dilewati bila absensi.csv tidak berubah sejak catch-up terakhir (sebelumnya seluruh
    CSV dibaca ulang tiap 5 menit walau tidak ada data baru);
  * kunci baris Google Sheets di-cache di memori (sebelumnya SELURUH kolom key diunduh setiap kali
    mengirim satu absen -> lambat dan boros kuota API saat sheet membesar).
"""
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from config import LOG_PATH, DATA_DIR   # noqa: E402  (lokasi file terpusat di config.py)
STATE_PATH = os.path.join(DATA_DIR, "cloud_state.json")   # kunci catatan yang sudah terkirim
SHEET_HEADER = ["tanggal", "jam", "nama", "id", "skor", "tipe", "status", "key"]
FS_BATCH = 200            # 2 tulisan per catatan -> 400 (batas Firestore 500)
SHEET_CHUNK = 500
POLL_DETIK = 30           # cek antrean berkala walau tidak ada catatan baru
CATCHUP_DETIK = 300       # cocokkan absensi.csv dengan state tiap N detik
RETRY_MAX = 300           # jeda maksimum antar percobaan ulang (detik)


def _file_cfg():
    try:
        with open(os.path.join(BASE_DIR, "cloud_config.json"), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


_FILE = _file_cfg()


def cfg(key, default=""):
    v = os.environ.get(key.upper())
    if v is None:
        v = _FILE.get(key, default)
    return str(v).strip()


def _truthy(v):
    return str(v).lower() in ("1", "true", "ya", "yes", "on")


from config import JAM_TERLAMBAT   # satu sumber konfigurasi: config.py


def name_key(nama):
    """Kunci aman untuk ID dokumen: slug nama + hash pendek (nama unik, unicode aman)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", nama).strip("-").lower()[:40] or "x"
    try:   # md5 hanya untuk ID dokumen (bukan keamanan); keluaran sama dengan versi lama -> tanpa duplikat
        h = hashlib.md5(nama.encode("utf-8"), usedforsecurity=False)
    except TypeError:
        h = hashlib.md5(nama.encode("utf-8"))
    return f"{slug}-{h.hexdigest()[:6]}"


def make_record(nama, pid, skor, now, tipe, status):
    tgl = now.strftime("%Y-%m-%d")
    return {"key": f"{tgl}_{name_key(nama)}_{tipe}", "tanggal": tgl, "jam": now.strftime("%H:%M:%S"),
            "nama": nama, "id": pid or "", "skor": round(float(skor), 3), "tipe": tipe, "status": status,
            "ts": now.isoformat(timespec="seconds"), "fs": False, "gs": False}


def read_csv_records():
    """Baca absensi.csv menjadi daftar catatan (kunci ganda dibuang, yang pertama dipakai)."""
    from attendance import read_log   # impor di sini: attendance.py juga mengimpor modul ini
    out, seen = [], set()
    for d in read_log(LOG_PATH, JAM_TERLAMBAT):
        rec = make_record(d["nama"], d["id"], d["skor"], d["t"], d["tipe"], d["status"])
        if rec["key"] not in seen:
            seen.add(rec["key"])
            out.append(rec)
    return out


def _csv_sig():
    try:
        st = os.stat(LOG_PATH)
        return st.st_mtime_ns, st.st_size, st.st_ino
    except OSError:
        return None


class CloudSync:
    def __init__(self, start_thread=True):
        self.cred = (os.environ.get("GOOGLE_CREDENTIALS") or str(_FILE.get("credentials") or "")
                     or "serviceAccount.json")
        # Railway: isi JSON service account lewat env GOOGLE_CREDENTIALS_JSON (tanpa file di repo)
        _cj = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
        if _cj:
            try:
                json.loads(_cj)
                self.cred = os.path.join(DATA_DIR, "serviceAccount.json")
                with open(self.cred, "w", encoding="utf-8") as _f:
                    _f.write(_cj)
                os.chmod(self.cred, 0o600)
            except ValueError:
                print("[CLOUD] GOOGLE_CREDENTIALS_JSON bukan JSON yang valid; diabaikan.")
        if not os.path.isabs(self.cred):
            self.cred = os.path.join(BASE_DIR, self.cred)
        has_cred = os.path.exists(self.cred)
        self.fs_col = cfg("firestore_collection", "absensi")
        self.fs_daily = cfg("firestore_daily", "rekap_harian")
        self.sheet_id = cfg("gsheet_id")
        self.sheet_tab = cfg("gsheet_tab", "Absensi")
        self.fs_on = has_cred and _truthy(cfg("firestore_enabled", "true"))
        self.gs_on = has_cred and bool(self.sheet_id)
        self.enabled = self.fs_on or self.gs_on
        self.source = os.path.splitext(os.path.basename(sys.argv[0] or "main"))[0] or "main"
        # satu outbox per program (main.py / dashboard.py) agar dua proses tidak saling menimpa
        self.outbox = os.path.join(DATA_DIR, f"cloud_outbox_{self.source}.jsonl")
        self.pending = []
        self.state = {"fs": set(), "gs": set()}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._fs = None
        self._ws = None
        self._gs_keys = None      # cache kolom key Google Sheets (anti-duplikat tanpa unduh ulang)
        self._cu_sig = None       # tanda absensi.csv saat catch-up terakhir
        if not has_cred:
            print(f"[CLOUD] nonaktif: file kredensial tidak ditemukan ({self.cred}).")
        elif not self.enabled:
            print("[CLOUD] nonaktif: Firestore dimatikan dan gsheet_id kosong.")
        else:
            print("[CLOUD] aktif (otomatis): " + ", ".join(
                n for n, on in (("Firestore", self.fs_on), ("Google Sheets", self.gs_on)) if on))
        if self.enabled:
            self._state_merge()
        if self.enabled and start_thread:
            self._load_outbox()
            self.th = threading.Thread(target=self._worker, daemon=True, name="cloud-sync")
            self.th.start()
            self._wake.set()     # catch-up pertama langsung jalan

    # ---------------- state: kunci yang sudah terkirim ----------------
    def _state_merge(self):
        """Gabungkan state di file dengan memori (aman bila dua proses berjalan bersamaan)."""
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                d = json.load(f)
            with self._lock:
                for k in ("fs", "gs"):
                    self.state[k] |= set(map(str, d.get(k, [])))
        except (OSError, ValueError, AttributeError):
            pass

    def _state_save(self):
        try:
            with self._lock:
                data = {k: sorted(v) for k, v in self.state.items()}
            tmp = STATE_PATH + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, STATE_PATH)
        except OSError:
            pass     # kegagalan menyimpan state hanya berarti pengiriman ulang (aman, anti-duplikat)

    # ---------------- antrean + outbox ----------------
    def _load_outbox(self):
        try:
            with open(self.outbox, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if isinstance(r, dict) and "key" in r:
                            r["fs"] = r.get("fs") or r["key"] in self.state["fs"]
                            r["gs"] = r.get("gs") or r["key"] in self.state["gs"]
                            self.pending.append(r)
                    except ValueError:
                        pass
        except OSError:
            return
        if self.pending:
            print(f"[CLOUD] {len(self.pending)} catatan belum terkirim dari sesi sebelumnya, dikirim ulang.")

    def _rewrite_outbox(self):
        with self._lock:
            if not self.pending:
                if os.path.exists(self.outbox):
                    os.remove(self.outbox)
                return
            tmp = self.outbox + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r in self.pending:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, self.outbox)

    def enqueue(self, nama, pid, skor, now, tipe, status):
        """Dipanggil dari AttendanceLogger.log. Cepat dan tidak pernah melempar error."""
        if not self.enabled:
            return
        try:
            r = make_record(nama, pid, skor, now, tipe, status)
            with self._lock:
                self.pending.append(r)
                with open(self.outbox, "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._wake.set()
        except Exception as e:
            print(f"[CLOUD] gagal mengantre: {e}")

    # ---------------- catch-up otomatis dari absensi.csv ----------------
    def _catchup(self, force=False):
        sig = _csv_sig()
        if not force and sig is not None and sig == self._cu_sig:
            return 0               # CSV tidak berubah: tidak perlu dibaca ulang
        try:
            recs = read_csv_records()
        except Exception as e:
            print(f"[CLOUD] gagal membaca absensi.csv: {e}")
            return 0
        self._state_merge()
        added = 0
        with self._lock:
            have = {r["key"] for r in self.pending}
            for r in recs:
                if r["key"] in have:
                    continue
                r["fs"] = r["key"] in self.state["fs"]
                r["gs"] = r["key"] in self.state["gs"]
                if (self.fs_on and not r["fs"]) or (self.gs_on and not r["gs"]):
                    self.pending.append(r)
                    added += 1
        if added:
            print(f"[CLOUD] catch-up: {added} catatan di absensi.csv belum ada di cloud, akan dikirim.")
        self._cu_sig = sig
        return added

    # ---------------- thread pengirim ----------------
    def _worker(self):
        delay, next_cu = 0, 0.0
        while not self._stop.is_set():
            self._wake.wait(delay or POLL_DETIK)
            self._wake.clear()
            if time.time() >= next_cu:
                self._catchup()
                next_cu = time.time() + CATCHUP_DETIK
            ok = self._cycle()
            delay = 0 if ok else min(max(delay * 2, 5), RETRY_MAX)

    def _cycle(self):
        """Kirim semua catatan yang belum terkirim. True bila semuanya berhasil."""
        with self._lock:
            fs_todo = [r for r in self.pending if self.fs_on and not r["fs"]]
            gs_todo = [r for r in self.pending if self.gs_on and not r["gs"]]
        ok = True
        if fs_todo:
            ok &= self._try("fs", fs_todo)
        if gs_todo:
            ok &= self._try("gs", gs_todo)
        if fs_todo or gs_todo:
            with self._lock:
                self.pending = [r for r in self.pending
                                if (self.fs_on and not r["fs"]) or (self.gs_on and not r["gs"])]
            self._rewrite_outbox()
        return ok

    def _try(self, kind, recs):
        label = "Firestore" if kind == "fs" else "Google Sheets"
        try:
            (self.fs_write if kind == "fs" else self.gs_write)(recs)
        except Exception as e:
            if kind == "gs":
                self._ws = None          # buka ulang koneksi pada percobaan berikutnya
                self._gs_keys = None     # dan baca ulang kolom key (bisa saja sebagian sudah terkirim)
            msg = str(e)
            print(f"[CLOUD] {label} gagal ({len(recs)} catatan, dicoba lagi otomatis): {msg[:300]}")
            if "invalid_grant" in msg or "Invalid JWT" in msg:
                print("[CLOUD] PETUNJUK: jam komputer tidak akurat. Aktifkan 'Set time automatically' di "
                      "Windows lalu klik 'Sync now'.")
            return False
        with self._lock:
            for r in recs:
                r[kind] = True
            self.state[kind].update(r["key"] for r in recs)
        self._state_save()
        print(f"[CLOUD] {label}: {len(recs)} catatan terkirim.")
        return True

    # ---------------- Firestore ----------------
    def fs_client(self):
        if self._fs is None:
            import firebase_admin
            from firebase_admin import credentials, firestore
            if not firebase_admin._apps:
                firebase_admin.initialize_app(credentials.Certificate(self.cred))
            self._fs = firestore.client()
        return self._fs

    def fs_write(self, recs):
        db = self.fs_client()
        for i in range(0, len(recs), FS_BATCH):
            batch = db.batch()
            for r in recs[i:i + FS_BATCH]:
                batch.set(db.collection(self.fs_col).document(r["key"]), {
                    "tanggal": r["tanggal"], "jam": r["jam"], "nama": r["nama"], "id": r["id"],
                    "skor": r["skor"], "tipe": r["tipe"], "status": r["status"],
                    "waktu": datetime.fromisoformat(r["ts"]).astimezone(),   # waktu lokal -> timestamp
                    "sumber": self.source})
                daily = {"tanggal": r["tanggal"], "nama": r["nama"], "id": r["id"]}
                if r["tipe"] == "pulang":
                    daily["jam_pulang"] = r["jam"]
                else:
                    daily.update(jam_masuk=r["jam"], status=r["status"])
                dkey = f"{r['tanggal']}_{name_key(r['nama'])}"
                batch.set(db.collection(self.fs_daily).document(dkey), daily, merge=True)
            batch.commit()

    # ---------------- Google Sheets ----------------
    def worksheet(self):
        if self._ws is None:
            import gspread
            gc = gspread.service_account(filename=self.cred)
            sh = gc.open_by_key(self.sheet_id)
            try:
                ws = sh.worksheet(self.sheet_tab)
            except gspread.WorksheetNotFound:
                ws = sh.add_worksheet(self.sheet_tab, rows=1000, cols=len(SHEET_HEADER))
            if not ws.row_values(1):
                ws.append_row(SHEET_HEADER)
                ws.freeze(rows=1)
            self._ws = ws
        return self._ws

    @staticmethod
    def sheet_row(r):
        # apostrof di depan ID agar angka 0 di depan tidak hilang (USER_ENTERED)
        return [r["tanggal"], r["jam"], r["nama"], ("'" + r["id"]) if r["id"] else "",
                r["skor"], r["tipe"], r["status"], r["key"]]

    def gs_write(self, recs):
        ws = self.worksheet()
        if self._gs_keys is None:                             # kolom key dibaca SEKALI, lalu di-cache
            self._gs_keys = set(ws.col_values(len(SHEET_HEADER)))
        new = [r for r in recs if r["key"] not in self._gs_keys]
        for i in range(0, len(new), SHEET_CHUNK):
            chunk = new[i:i + SHEET_CHUNK]
            ws.append_rows([self.sheet_row(r) for r in chunk], value_input_option="USER_ENTERED")
            self._gs_keys.update(r["key"] for r in chunk)
            if i + SHEET_CHUNK < len(new):
                time.sleep(1.2)                               # jaga kuota API Sheets

    # ---------------- retensi (UU PDP Pasal 43) ----------------
    def fs_purge_before(self, cutoff, dry_run=True):
        """Hapus dokumen Firestore dengan tanggal < cutoff (YYYY-MM-DD) di kedua koleksi."""
        db = self.fs_client()
        total = {}
        for col in (self.fs_col, self.fs_daily):
            q = db.collection(col)
            try:
                from google.cloud.firestore_v1.base_query import FieldFilter
                q = q.where(filter=FieldFilter("tanggal", "<", cutoff))
            except ImportError:
                q = q.where("tanggal", "<", cutoff)
            if dry_run:
                try:
                    n = int(q.count().get()[0][0].value)                 # agregasi di server
                except Exception:
                    n = sum(1 for _ in q.select([]).stream())
            else:
                n = 0
                while True:
                    docs = list(q.limit(400).stream())
                    if not docs:
                        break
                    batch = db.batch()
                    for d in docs:
                        batch.delete(d.reference)
                    batch.commit()
                    n += len(docs)
            total[col] = n
        return total


# ---------------- singleton untuk main.py ----------------
_sync = None
_init_lock = threading.Lock()


def get_sync():
    global _sync
    with _init_lock:
        if _sync is None:
            _sync = CloudSync()
        return _sync


def start():
    """Opsional: panggil lebih awal agar catch-up langsung berjalan saat program dimulai."""
    return get_sync()


def enqueue(nama, pid, skor, now, tipe, status):
    get_sync().enqueue(nama, pid, skor, now, tipe, status)


# ---------------- CLI ----------------
def _cli():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("test", "backfill", "retensi"):
        raise SystemExit(__doc__)
    s = CloudSync(start_thread=False)
    if not s.enabled:
        raise SystemExit("Tidak ada tujuan aktif. Periksa serviceAccount.json dan gsheet_id.")
    if cmd == "retensi":
        import argparse
        from datetime import timedelta
        from config import RETENSI_HARI
        ap = argparse.ArgumentParser(prog="cloud_sync.py retensi")
        ap.add_argument("--hari", type=int, default=RETENSI_HARI)
        ap.add_argument("--ya", action="store_true")
        a = ap.parse_args(sys.argv[2:])
        if not s.fs_on:
            raise SystemExit("Firestore nonaktif.")
        cutoff = (datetime.now() - timedelta(days=a.hari)).strftime("%Y-%m-%d")
        res = s.fs_purge_before(cutoff, dry_run=not a.ya)
        print(("Dihapus" if a.ya else "[SIMULASI] Akan dihapus") + f" (tanggal < {cutoff}): " +
              ", ".join(f"{k}={v}" for k, v in res.items()) + ("" if a.ya else ". Tambah --ya untuk menghapus."))
        if a.ya:
            try:
                import privacy
                privacy.audit("retensi_firestore", detail={"sebelum": cutoff, **res})
            except Exception:
                pass
        if s.gs_on:
            print(f"Google Sheets: hapus manual baris dengan tanggal < {cutoff}.")
        return
    if cmd == "test":
        if s.fs_on:
            s.fs_client().collection(s.fs_col).limit(1).get()
            print("Firestore OK.")
        if s.gs_on:
            ws = s.worksheet()
            print(f"Google Sheets OK: tab '{ws.title}' di '{ws.spreadsheet.title}'.")
        return
    n = s._catchup(force=True)
    if not n:
        print("Semua catatan di absensi.csv sudah ada di cloud.")
        return
    print("Selesai." if s._cycle() else "Ada yang gagal; akan dicoba lagi otomatis saat main.py/dashboard.py berjalan.")


if __name__ == "__main__":
    _cli()