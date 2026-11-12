"""
Dashboard Admin Absensi Wajah (versi optimal) - gaya UI selaras dengan chat RAG (netral hangat, aksen ungu).

Halaman  : Ringkasan | Kamera absen | Kelola wajah (CRUD) | Rekap
Tema     : Otomatis (ikut sistem) / Terang / Gelap, pilihan tersimpan di browser.
Membaca/menulis absensi.csv dan faces_db.npz milik main.py (letakkan di folder yang sama).

Aturan absensi (sama dengan main.py): masuk setelah JAM_TERLAMBAT = terlambat; scan setelah JAM_PULANG
dari orang yang sudah masuk = absen pulang; semua status di-reset otomatis tiap hari.

Kamera absen punya dua sumber:
  * Kamera server  : server membaca kamera komputer ini langsung lewat OpenCV dan memakai alur yang
                     sama dengan 'main.py run' (deteksi tiap 3 frame + optical flow). Paling cepat.
                     Tidak bisa dipakai bersamaan dengan 'main.py run' (rebutan kamera).
  * Kamera browser : kamera perangkat yang membuka halaman (HP/PC lain). Browser hanya mengizinkan kamera
                     di localhost atau HTTPS.

Jalankan : python dashboard.py
Buka     : http://127.0.0.1:8000  (otomatis terbuka di browser)
Opsi     : --port 8000
           --host 0.0.0.0   (agar bisa dibuka dari HP/PC lain di jaringan yang sama)
           --no-browser     (jangan buka browser otomatis)
           --pin 1234       (wajib jika dibuka dari jaringan; melindungi kamera absen, pengelolaan
                             wajah, DAN pembacaan data absensi)

PIN dikirim apa adanya lewat HTTP biasa; di jaringan yang tidak tepercaya pakai HTTPS.
"""
import argparse
import base64
import csv
import hmac
import io
import json
import os
import re
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE, "absensi.csv")
DB_FILE = os.path.join(BASE, "faces_db.npz")
JAM_TERLAMBAT = "09:00:00"   # HARUS sama dengan main.py; dipakai untuk baris lama tanpa kolom status
_cache = {}
_build_cache = {}


def _sig(path):
    try:
        st = os.stat(path)
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def cached(path, loader):
    """Baca ulang file hanya jika isinya berubah (hemat CPU dan RAM)."""
    key = _sig(path)
    hit = _cache.get((path, loader.__name__))
    if hit and hit[0] == key:
        return hit[1]
    val = loader(path if key else None)
    _cache[(path, loader.__name__)] = (key, val)
    return val


def read_rows(path):
    """Baris absensi: (tanggal, jam, nama, tipe, status). Baris lama 5 kolom dianggap tipe 'masuk'."""
    rows = []
    if path:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) >= 3 and r[0] != "tanggal":
                    tipe = r[5] if len(r) > 5 and r[5] else "masuk"
                    st = r[6] if len(r) > 6 and r[6] else ("terlambat" if r[1] > JAM_TERLAMBAT else "hadir")
                    rows.append((r[0], r[1], r[2], tipe, st))
    return rows


def read_names(path):
    if not path:
        return []
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as d:
            return sorted({str(n) for n in d["names"]})
    except Exception:
        return []


def _build(date):
    rows = cached(LOG_FILE, read_rows)
    registered = cached(DB_FILE, read_names)

    masuk, pulang = {}, {}                  # status hari terpilih
    for d, j, n, tipe, st in rows:
        if d != date:
            continue
        if tipe == "pulang":
            pulang[n] = max(pulang.get(n, ""), j)
        elif n not in masuk or j < masuk[n][0]:
            masuk[n] = (j, st)
    present = sorted(({"jam": j, "nama": n, "status": st, "pulang": pulang.get(n, "")}
                      for n, (j, st) in masuk.items()), key=lambda p: p["jam"], reverse=True)
    here = set(masuk)

    end = datetime.strptime(date, "%Y-%m-%d")
    days = [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    per_day, first, recap = {}, {}, {}
    for d, j, n, tipe, st in rows:
        r = recap.setdefault(n, {"nama": n, "total": 0, "days": 0, "late": 0, "last": ""})
        r["total"] += 1
        r["last"] = max(r["last"], f"{d} {j}")
        if tipe == "masuk" and (n, d) not in first:      # catatan masuk pertama per orang per hari
            first[(n, d)] = st
            per_day.setdefault(d, set()).add(n)
    for (n, d), st in first.items():
        recap[n]["days"] += 1
        recap[n]["late"] += st == "terlambat"

    total = len(registered)
    return {
        "date": date,
        "registered": total,
        "present": present,
        "late": sum(1 for p in present if p["status"] == "terlambat"),
        "absent": [n for n in registered if n not in here],
        "pct": min(100, round(100 * len(here) / total)) if total else 0,
        "trend": [{"date": d, "count": len(per_day.get(d, ()))} for d in days],
        "work_days": len(per_day),
        "recap": sorted(recap.values(), key=lambda r: (-r["total"], r["nama"])),
    }


def build(date):
    """Statistik per tanggal; hasil di-memo selama CSV dan database tidak berubah."""
    key = (_sig(LOG_FILE), _sig(DB_FILE))
    hit = _build_cache.get(date)
    if not hit or hit[0] != key:
        if len(_build_cache) > 16:
            _build_cache.clear()
        hit = _build_cache[date] = (key, _build(date))
    return {**hit[1], "updated": datetime.now().strftime("%H:%M:%S")}


# ----------------------------- Kelola wajah (CRUD) + kamera absen -----------------------------
JSON = "application/json; charset=utf-8"
MIN_SAMPLES = 3            # minimal foto valid untuk orang baru
MAX_IMAGES = 12            # maksimal foto per permintaan
MAX_BODY = 40 * 1024 * 1024
MAX_FRAME = 4 * 1024 * 1024
PIN = ""                   # diisi dari --pin
EDIT_ENABLED = True        # False jika dashboard terbuka di jaringan tanpa --pin
EDIT_LOCK = threading.Lock()
ENGINE_LOCK = threading.Lock()   # model OpenCV tidak aman dipakai bersamaan
SCAN_LOCK = threading.Lock()
INIT_LOCK = threading.RLock()
SCAN_DETECT_W = 640        # lebar frame untuk deteksi (embedding tetap dari frame penuh)
SCAN_MIN_FACE_PX = 48      # lebar wajah minimum (px, pada frame kiriman browser, maks 1280)
SCAN_SESSION_TTL = 30.0    # detik: sesi kamera tanpa frame baru dibuang
_core = {}
_sessions = {}


class ApiError(Exception):
    def __init__(self, code, msg, rejected=None):
        super().__init__(msg)
        self.code, self.msg, self.rejected = code, msg, rejected or []


def get_main():
    """Muat main.py (OpenCV) sekali saja, hanya saat pertama kali dibutuhkan."""
    with INIT_LOCK:
        if "main" not in _core:
            os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(40_000_000))  # tolak gambar raksasa
            sys.path.insert(0, BASE)
            import main
            main.cv2.setNumThreads(main.CV_THREADS)
            _core["main"] = main
        return _core["main"]


def get_engine():
    main = get_main()
    with INIT_LOCK:
        if "engine" not in _core:
            _core["engine"] = main.FaceEngine(*main.ensure_models())
        return _core["engine"]


def get_logger(main):
    """Satu logger bersama; membaca CSV hanya bila ada baris baru."""
    with INIT_LOCK:
        if "logger" not in _core:
            _core["logger"] = main.AttendanceLogger()
        return _core["logger"]


def read_people(path):
    """{nama: [jumlah_sampel, id]} dari faces_db.npz (hanya numpy)."""
    if not path:
        return {}
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as d:
            names = [str(n) for n in d["names"]]
            ids = [str(i) for i in d["ids"]] if "ids" in d.files else [""] * len(names)
    except Exception:
        return {}
    out = {}
    for n, i in zip(names, ids):
        out.setdefault(n, [0, i])[0] += 1
    return out


def people_info():
    people = cached(DB_FILE, read_people)
    return {"people": [{"nama": n, "id": v[1], "sampel": v[0]}
                       for n, v in sorted(people.items(), key=lambda x: x[0].casefold())],
            "editable": EDIT_ENABLED, "pin_required": bool(PIN),
            "min_samples": MIN_SAMPLES, "max_images": MAX_IMAGES}


def clean_text(s, label, limit, required=False):
    s = " ".join(str(s if s is not None else "").split())
    if required and not s:
        raise ApiError(400, f"{label} wajib diisi.")
    if len(s) > limit:
        raise ApiError(400, f"{label} maksimal {limit} karakter.")
    if s[:1] in ("=", "+", "-", "@"):  # cegah formula injection saat diekspor ke Excel
        raise ApiError(400, f"{label} tidak boleh diawali = + - atau @.")
    return s


def check_images(images, minimum=1):
    if not isinstance(images, list) or not all(isinstance(x, str) for x in images):
        raise ApiError(400, "Daftar foto tidak valid.")
    if not minimum <= len(images) <= MAX_IMAGES:
        raise ApiError(400, f"Kirim {minimum}-{MAX_IMAGES} foto.")


def embed_image(main, engine, data_url):
    """Satu foto -> embedding. Lempar ValueError berisi alasan jika foto tidak layak dipakai."""
    np, cv2 = main.np, main.cv2
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        raise ValueError("bukan gambar yang valid")
    h, w = img.shape[:2]
    if max(h, w) > 1600:                       # batasi ukuran agar hemat RAM
        f = 1600 / max(h, w)
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    sw = min(w, 640)
    sh = max(1, round(h * sw / w))
    small = img if sw == w else cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    faces = engine.detect(small)
    if not faces:
        raise ValueError("wajah tidak terdeteksi")
    if len(faces) > 1:
        raise ValueError("lebih dari satu wajah")
    scale = (w / sw, h / sh)
    width = int(faces[0][2] * scale[0])
    if width < main.REG_MIN_FACE_PX:
        raise ValueError(f"wajah terlalu kecil ({width}px, minimal {main.REG_MIN_FACE_PX}px)")
    return engine.embed(img, faces[0], scale)


def process_images(images):
    main, engine = get_main(), get_engine()
    good, rejected = [], []
    for i, d in enumerate(images, 1):
        try:
            with ENGINE_LOCK:
                good.append(embed_image(main, engine, d))
        except ValueError as e:
            rejected.append(f"foto {i}: {e}")
    return good, rejected


def similar_warning(db, good, name):
    if db.is_empty():
        return ""
    np = get_main().np
    mean = np.mean(good, axis=0)
    mean /= np.linalg.norm(mean)
    other, _, sc, _ = db.best(mean)
    if other and other.casefold() != name.casefold() and sc >= 0.5:
        return f"Wajah ini sangat mirip dengan '{other}' (skor {sc:.2f}). Pastikan tidak terdaftar ganda."
    return ""


def api_create(d):
    name, pid = clean_text(d.get("nama"), "Nama", 60, True), clean_text(d.get("id"), "ID", 40)
    check_images(d.get("images"), MIN_SAMPLES)
    db = get_main().Database(DB_FILE)
    if any(n.casefold() == name.casefold() for n in db.people()):
        raise ApiError(409, f"Nama '{name}' sudah terdaftar. Pakai 'Tambah sampel' untuk menambah foto.")
    good, rejected = process_images(d["images"])
    if len(good) < MIN_SAMPLES:
        raise ApiError(422, f"Hanya {len(good)} foto valid, minimal {MIN_SAMPLES}.", rejected)
    warn = similar_warning(db, good, name)
    db.add(name, pid, good)
    db.save()
    return {"message": f"{name} ditambahkan ({len(good)} sampel).", "rejected": rejected, "warning": warn}


def api_add_samples(d):
    name = str(d.get("nama") or "")
    check_images(d.get("images"))
    db = get_main().Database(DB_FILE)
    if name not in db.people():
        raise ApiError(404, f"'{name}' tidak ditemukan.")
    good, rejected = process_images(d["images"])
    if not good:
        raise ApiError(422, "Tidak ada foto yang valid.", rejected)
    warn = similar_warning(db, good, name)
    db.add(name, "", good)
    db.save()
    return {"message": f"{len(good)} sampel ditambahkan untuk {name}.", "rejected": rejected, "warning": warn}


def api_update(d):
    old = str(d.get("nama") or "")
    new = clean_text(d.get("nama_baru") or old, "Nama", 60, True)
    pid = clean_text(d["id"], "ID", 40) if d.get("id") is not None else None
    db = get_main().Database(DB_FILE)
    if old not in db.people():
        raise ApiError(404, f"'{old}' tidak ditemukan.")
    if new != old and any(n.casefold() == new.casefold() for n in db.people()):
        raise ApiError(409, f"Nama '{new}' sudah dipakai.")
    db.update(old, new, pid)
    db.save()
    return {"message": f"Data {new} diperbarui. Riwayat absensi lama tetap memakai nama sebelumnya."}


def api_delete(d):
    name = str(d.get("nama") or "")
    db = get_main().Database(DB_FILE)
    removed = db.remove(name)
    if not removed:
        raise ApiError(404, f"'{name}' tidak ditemukan.")
    db.save()
    return {"message": f"{name} dihapus ({removed} sampel). Riwayat absensi tidak ikut terhapus."}


# ----------------------------- Kamera absen: frame dari browser -----------------------------
def scan_db(main):
    """Database untuk pemindaian; dimuat ulang otomatis bila file berubah."""
    db = _core.get("db")
    if db is None:
        db = _core["db"] = main.Database(DB_FILE)
    elif db.changed():
        db.reload()
        for s in _sessions.values():          # verifikasi ulang wajah yang sedang terlihat
            for t in s["tracker"].tracks.values():
                t.next_embed = 0.0
    return db


def get_session(main, cam, now):
    """Satu pelacak per sesi kamera browser; sesi yang lama tidak aktif dibuang."""
    for k in [k for k, v in _sessions.items() if now - v["ts"] > SCAN_SESSION_TTL]:
        del _sessions[k]
    s = _sessions.get(cam)
    if s is None:
        s = _sessions[cam] = {"tracker": main.Tracker(), "ts": now}
    s["ts"] = now
    return s


def api_scan(raw, cam):
    main, engine = get_main(), get_engine()
    np, cv2 = main.np, main.cv2
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(400, "Frame bukan gambar yang valid.")
    h, w = img.shape[:2]
    sw = min(w, SCAN_DETECT_W)
    sh = max(1, round(h * sw / w))
    small = img if sw == w else cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    scale = (w / sw, h / sh)
    faces_out, events = [], []
    now = time.time()
    with SCAN_LOCK:
        db, logger = scan_db(main), get_logger(main)
        tracker = get_session(main, cam, now)["tracker"]
        recheck = main._recheck_delay()
        with ENGINE_LOCK:
            rows = engine.detect(small)
            near = [f for f in rows if f[2] * scale[0] >= SCAN_MIN_FACE_PX]
            far = [f for f in rows if SCAN_MIN_FACE_PX * 0.5 <= f[2] * scale[0] < SCAN_MIN_FACE_PX]
            fresh = tracker.update(near)
            for t in fresh:
                if now >= t.next_embed:       # embedding hanya bila perlu (sama seperti main.py)
                    res = main.apply_match(t, *db.match(engine.embed(img, t.row, scale)),
                                           now, logger, recheck)
                    if res:
                        dt, tipe, st = res
                        t.flash = now + 2.0
                        events.append({"nama": t.name, "id": t.pid, "jam": dt.strftime("%H:%M:%S"),
                                       "tipe": tipe, "status": st})
        for f in far:
            faces_out.append({"box": [float(f[0] / sw), float(f[1] / sh), float(f[2] / sw), float(f[3] / sh)],
                              "status": "far"})
        for t in fresh:
            if t.name:
                status = "tercatat" if now < getattr(t, "flash", 0.0) else "hadir"
            elif t.cand:
                status = "cek"
            else:
                status = "unknown"
            item = {"box": [float(t.box[0] / sw), float(t.box[1] / sh), float(t.box[2] / sw), float(t.box[3] / sh)],
                    "status": status, "skor": round(float(t.score), 2)}
            if t.name or t.cand:
                item["nama"] = t.name or t.cand
            faces_out.append(item)
        db_empty = db.is_empty()
    return {"faces": faces_out, "events": events, "db_empty": db_empty}


# ----------------------------- Kamera absen: kamera server (seperti main.py) -----------------------------
class ServerScanner:
    """Baca kamera langsung di server dan jalankan alur yang sama dengan main.py run."""
    IDLE_STOP = 30.0   # detik tanpa permintaan frame dari browser -> kamera dilepas

    def __init__(self):
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.running, self.jpeg, self.err = False, None, ""
        self.fps, self.events, self.last_req = 0.0, [], 0.0
        self.th = None

    def start(self, cam_index):
        with self.lock:
            if self.running:
                self.last_req = time.time()
                return
            main, engine = get_main(), get_engine()
            cam = main.Camera(cam_index)          # lempar CameraError bila gagal
            self.stop_ev.clear()
            self.err, self.jpeg, self.events = "", None, []
            self.last_req, self.running = time.time(), True
            self.th = threading.Thread(target=self._loop, args=(main, engine, cam), daemon=True)
            self.th.start()

    def stop(self):
        self.stop_ev.set()
        th = self.th
        if th:
            th.join(timeout=3.0)

    def status(self):
        with self.lock:
            ev, self.events = self.events, []
            return {"running": self.running, "fps": round(self.fps), "err": self.err, "events": ev}

    def _loop(self, main, engine, cam):
        np, cv2 = main.np, main.cv2
        try:
            db, logger, tracker = main.Database(DB_FILE), get_logger(main), main.Tracker()
            recheck = main._recheck_delay()
            small = gbuf = None
            gi, have_prev, far = 0, False, []
            n, fps, t_prev, db_t = 0, 0.0, time.perf_counter(), 0.0
            while not self.stop_ev.is_set():
                now_t = time.time()
                if now_t - self.last_req > self.IDLE_STOP:
                    break
                frame = main.next_frame(cam)
                if frame is None:
                    continue
                n += 1
                sw, sh, scale = main.small_size(frame)
                if small is None or small.shape[:2] != (sh, sw):
                    small = np.empty((sh, sw, 3), np.uint8)
                    gbuf = [np.empty((sh, sw), np.uint8), np.empty((sh, sw), np.uint8)]
                    have_prev = False
                    tracker.tracks.clear()
                cv2.resize(frame, (sw, sh), dst=small, interpolation=cv2.INTER_AREA)
                gray, prev = gbuf[gi], gbuf[1 - gi]
                cv2.cvtColor(small, cv2.COLOR_BGR2GRAY, dst=gray)
                if have_prev:
                    tracker.advance(prev, gray)
                gi, have_prev = 1 - gi, True

                if now_t - db_t > 2.0:                       # sinkron log dan muat ulang database
                    db_t = now_t
                    logger.sync()
                    if db.changed():
                        try:
                            db.reload()
                        except RuntimeError:
                            pass
                        for t in tracker.tracks.values():
                            t.next_embed = 0.0
                if n % main.DETECT_EVERY == 0:
                    with ENGINE_LOCK:
                        faces = engine.detect(small)
                        near = [f for f in faces if f[2] * scale[0] >= main.MIN_FACE_PX]
                        far = [f for f in faces
                               if main.MIN_FACE_PX * 0.5 <= f[2] * scale[0] < main.MIN_FACE_PX]
                        for t in tracker.update(near):
                            if now_t >= t.next_embed:
                                res = main.apply_match(t, *db.match(engine.embed(frame, t.row, scale)),
                                                       now_t, logger, recheck)
                                if res:
                                    dt, tipe, st = res
                                    with self.lock:
                                        self.events = (self.events + [{
                                            "nama": t.name, "tipe": tipe, "status": st,
                                            "jam": dt.strftime("%H:%M:%S")}])[-20:]

                # gambar kotak langsung di frame (sama seperti main.py)
                k = np.array([*scale, *scale], np.float32)
                for f in far:
                    x, y, bw, bh = (int(v) for v in f[:4] * k)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (150, 150, 150), 1)
                    main.put_text(frame, "Terlalu jauh", (x, max(y - 6, 14)), (210, 210, 210), 0.5)
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
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                    main.put_text(frame, label, (x, max(y - 6, 14)), color)
                t_now = time.perf_counter()
                inst = 1.0 / max(t_now - t_prev, 1e-6)
                fps, t_prev = (inst if fps == 0 else 0.9 * fps + 0.1 * inst), t_now
                self.fps = fps
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    self.jpeg = buf.tobytes()
        except Exception as e:
            self.err = str(e)
        finally:
            cam.close()
            with self.lock:
                self.running, self.jpeg = False, None


scanner = ServerScanner()


def api_cam_start(d):
    try:
        scanner.start(int(d.get("kamera") or 0))
    except Exception as e:
        raise ApiError(500, str(e))
    return {"message": "Kamera server aktif."}


def api_cam_stop(d):
    scanner.stop()
    return {"message": "Kamera server dihentikan."}


ROUTES = {("POST", "/api/people"): api_create, ("POST", "/api/people/samples"): api_add_samples,
          ("PUT", "/api/people"): api_update, ("DELETE", "/api/people"): api_delete,
          ("POST", "/api/server-cam/start"): api_cam_start,
          ("POST", "/api/server-cam/stop"): api_cam_stop}


def pin_ok(headers):
    return hmac.compare_digest(headers.get("X-Pin", "").encode(), PIN.encode())


PAGE = r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Absensi Wajah · Admin</title>
<script>try{var t=localStorage.getItem('theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* tema: terang (bawaan) | gelap | otomatis mengikuti sistem (tanpa atribut data-theme) */
:root{color-scheme:light;--bg:#faf9f7;--side:#f3f1ec;--panel:#ffffff;--ink:#1f1d1a;--mute:#5c574f;--line:#e5e0d8;--acc:#7c3aed;--acc2:#c4a5fb;--soft:#efe8fd;--on-acc:#ffffff;--ok:#15803d;--ok-soft:#dcf3e3;--warn:#b45309;--warn-soft:#fdecc8;--bad:#b42318;--bad-soft:#fde4e1;--shadow:0 1px 2px rgba(31,29,26,.06),0 8px 24px rgba(31,29,26,.07)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1b1916;--side:#22201c;--panel:#2a2723;--ink:#f1ede6;--mute:#b3ab9f;--line:#3a362f;--acc:#a78bfa;--acc2:#6d3fd6;--soft:#352b52;--on-acc:#1b1916;--ok:#6ee7a0;--ok-soft:#1f3a2a;--warn:#fbbf24;--warn-soft:#3d2f12;--bad:#f97066;--bad-soft:#40201d;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.28)}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1b1916;--side:#22201c;--panel:#2a2723;--ink:#f1ede6;--mute:#b3ab9f;--line:#3a362f;--acc:#a78bfa;--acc2:#6d3fd6;--soft:#352b52;--on-acc:#1b1916;--ok:#6ee7a0;--ok-soft:#1f3a2a;--warn:#fbbf24;--warn-soft:#3d2f12;--bad:#f97066;--bad-soft:#40201d;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.28)}
/* token layout: satu skala jarak, radius, dan tinggi kontrol dipakai di seluruh halaman */
:root{--s1:4px;--s2:8px;--s3:12px;--s4:16px;--s5:20px;--s6:24px;--s7:32px;--gap:16px;--pad:20px;--r-card:20px;--r-inner:14px;--r-ctl:10px;--ctl:38px}
@media (max-width:600px){:root{--gap:12px;--pad:16px}}
*{box-sizing:border-box}
[hidden]{display:none!important}
p{margin:0}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Plus Jakarta Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;transition:background .2s,color .2s}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.ic{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex:none}
.app{display:grid;grid-template-columns:264px minmax(0,1fr);min-height:100vh}

/* sidebar */
.side{background:var(--side);border-right:1px solid var(--line);padding:var(--s5) var(--s4);display:flex;flex-direction:column;gap:var(--s1);position:sticky;top:0;height:100vh}
.brand{display:flex;align-items:center;gap:var(--s3);padding:var(--s1) var(--s3) var(--s5);font-weight:700;font-size:16px}
.logo{width:34px;height:34px;border-radius:11px;background:var(--acc);color:var(--on-acc);display:grid;place-items:center}
.nav{display:flex;flex-direction:column;gap:var(--s1)}
.nav a{display:flex;align-items:center;gap:var(--s3);height:var(--ctl);padding:0 var(--s3);border-radius:var(--r-ctl);color:var(--mute);text-decoration:none;font-weight:500}
.nav a:hover{background:color-mix(in srgb,var(--acc) 7%,transparent)}
.nav a.on{background:var(--soft);color:var(--acc);font-weight:650}
.sfoot{margin-top:auto;padding:var(--s4) var(--s3) 0;border-top:1px solid var(--line);font-size:13px;color:var(--mute)}
.sfoot label{display:block;margin-bottom:var(--s3)}
.sfoot input{margin-top:var(--s1);width:100%}
.seg{display:flex;gap:var(--s1);padding:var(--s1);border:1px solid var(--line);border-radius:var(--r-ctl);background:var(--panel);margin-bottom:var(--s3)}
.seg button{flex:1;display:grid;place-items:center;height:30px;border:0;border-radius:7px;background:transparent;color:var(--mute);cursor:pointer}
.seg button:hover{color:var(--ink)}
.seg button[aria-pressed=true]{background:var(--soft);color:var(--acc)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--mute);margin-right:var(--s2)}
.dot.on{background:var(--ok)}
.top{display:none}
.scrim{display:none}
@media (max-width:860px){
  .app{display:block}
  .side{position:fixed;inset:0 auto 0 0;width:282px;transform:translateX(-102%);transition:transform .25s;z-index:30;height:100dvh}
  .side.open{transform:none;box-shadow:var(--shadow)}
  .scrim.open{display:block;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:20}
  .top{display:flex;align-items:center;gap:var(--s3);height:56px;padding:0 var(--s4);background:var(--bg);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;font-weight:700}
}

/* kerangka halaman */
.main{padding:var(--s7) clamp(16px,4vw,48px) 56px;max-width:1120px;margin:0 auto}
.hello{display:flex;flex-wrap:wrap;gap:var(--gap);align-items:flex-end;justify-content:space-between;margin-bottom:var(--s6)}
h1{font-family:Lora,Georgia,serif;font-weight:500;font-size:clamp(26px,4vw,34px);line-height:1.2;margin:0}
.sub{margin-top:var(--s1);color:var(--mute)}
h2{margin:0;font-size:16px;line-height:1.3;font-weight:650}
.mute{color:var(--mute)}
.meta{color:var(--mute);font-size:13px;white-space:nowrap}
.note{margin-bottom:var(--gap)}
.note:empty{display:none}
.err{color:var(--bad)}.okmsg{color:var(--ok)}

/* kontrol: tinggi seragam */
input,select,.btn{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:var(--r-ctl);height:var(--ctl);padding:0 var(--s3)}
input::placeholder{color:var(--mute)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:var(--s2);text-decoration:none;cursor:pointer;white-space:nowrap}
.btn:hover{border-color:var(--acc)}
button.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.pri{background:var(--acc);border-color:var(--acc);color:var(--on-acc);font-weight:600}
.btn.sm{height:30px;padding:0 var(--s3);font-size:13px;border-radius:8px}
.btn.bad{color:var(--bad);border-color:var(--bad)}

/* kartu: satu pola (padding, radius, header) untuk semua */
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r-card);padding:var(--pad);box-shadow:var(--shadow);min-width:0}
.card-h{display:flex;align-items:center;justify-content:space-between;gap:var(--s3);min-height:var(--ctl);margin-bottom:var(--s3)}
.card-h input{width:min(190px,55%)}

/* saran cepat: sejajar dengan grid di bawahnya */
.sugs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--gap);margin-bottom:var(--gap)}
.sug{display:flex;gap:var(--s3);align-items:center;padding:var(--s3) var(--s4);border:1px solid var(--line);border-radius:var(--r-inner);background:var(--panel);color:var(--ink);text-decoration:none;font-weight:600;cursor:pointer;font-size:14px;text-align:left;min-width:0}
.sug:hover{border-color:var(--acc);background:var(--soft)}
.sug .ico{width:36px;height:36px;border-radius:11px;background:var(--soft);color:var(--acc);display:grid;place-items:center;flex:none}
.sug small{display:block;color:var(--mute);font-weight:400;line-height:1.35}
@media (max-width:760px){.sugs{grid-auto-flow:column;grid-template-columns:none;grid-auto-columns:minmax(230px,80%);overflow-x:auto;scroll-snap-type:x mandatory}.sug{scroll-snap-align:start}}

/* grid bento: ring utama > statistik > daftar */
.bento{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:var(--gap)}
.hero{grid-column:span 5;grid-row:span 2;display:flex;flex-direction:column;background:linear-gradient(165deg,var(--soft) 0%,var(--panel) 62%)}
.hero-body{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:var(--s2)}
.ringbig{position:relative;width:min(220px,100%);aspect-ratio:1;margin-bottom:var(--s2)}
.ringbig svg{position:absolute;inset:0;width:100%;height:100%}
.ringbig .mid{position:absolute;inset:0;display:grid;place-content:center}
.ringbig b{font-family:Lora,Georgia,serif;font-weight:500;font-size:clamp(40px,6vw,52px);line-height:1;font-variant-numeric:tabular-nums}
.ringbig small{color:var(--mute);font-size:13px}
.hero .cap{font-size:16px;font-weight:650}
.hero .last{font-size:13px;color:var(--mute)}
.rot{transform:rotate(-90deg)}
.rt{fill:none;stroke:var(--line)}
.rp{fill:none;stroke:var(--acc);stroke-linecap:round;transition:stroke-dasharray .9s cubic-bezier(.2,.8,.2,1),opacity .2s}
#h-arc{stroke:url(#g1)}
.trio{grid-column:span 7;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--gap)}
.stat{display:flex;flex-direction:column;align-items:flex-start}
.stat b{display:block;font-size:28px;line-height:1.1;font-weight:650;font-variant-numeric:tabular-nums;margin-top:var(--s3)}
.stat .l{color:var(--mute);font-size:13px;margin-top:var(--s1)}
.st-ic{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:var(--soft);color:var(--acc)}
.st-ic.ok{background:var(--ok-soft);color:var(--ok)}
.st-ic.warn{background:var(--warn-soft);color:var(--warn)}
.weekc{grid-column:span 7}
.week{display:flex;justify-content:space-between;gap:var(--s2)}
.wd{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;gap:var(--s1);font-size:12px;line-height:1.3;color:var(--mute)}
.wd .mr{margin-bottom:var(--s1)}
.wd .rp{stroke:color-mix(in srgb,var(--acc) 50%,transparent)}
.wd.sel .rp{stroke:var(--acc)}
.wd.sel{color:var(--ink);font-weight:600}
.c7{grid-column:span 7}
.c5{grid-column:span 5}
@media (max-width:1000px){.hero,.trio,.weekc,.c7,.c5{grid-column:1/-1;grid-row:auto}}
@media (max-width:520px){.st-ic{display:none}.stat b{margin-top:0}}

/* cincin mini */
.mr{position:relative;display:inline-grid;place-items:center;width:var(--s);height:var(--s);flex:none}
.mr svg{position:absolute;inset:0;width:100%;height:100%}
.mr i{position:relative;font-style:normal;font-weight:650;font-size:calc(var(--s)*.3);font-variant-numeric:tabular-nums}
.mr.ok .rp{stroke:var(--ok)}
.mr.warn .rp{stroke:var(--warn)}
.mr.low .rp{stroke:var(--bad)}

/* tabel */
table{width:100%;border-collapse:collapse}
th,td{padding:var(--s3) var(--s2);border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
th:first-child,td:first-child{padding-left:0}
th:last-child,td:last-child{padding-right:0}
tr:last-child td{border-bottom:0}
th{font-size:13px;color:var(--mute);font-weight:500;position:sticky;top:0;background:var(--panel);z-index:1}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{max-height:340px;overflow:auto}
.scroll.tall{max-height:none}
.act{text-align:right;white-space:nowrap}
.who{display:flex;align-items:center;gap:var(--s3)}
.av{width:30px;height:30px;border-radius:50%;background:var(--soft);color:var(--acc);display:grid;place-items:center;font-weight:700;font-size:13px;flex:none}
#absent{display:flex;flex-wrap:wrap;gap:var(--s2)}
.chip{display:inline-block;padding:var(--s1) var(--s3);border-radius:999px;background:var(--soft);color:var(--acc);font-size:14px}
.chip.late{background:var(--warn-soft);color:var(--warn)}
.chip.ok{background:var(--ok-soft);color:var(--ok)}
.pcell{display:flex;align-items:center;gap:var(--s3)}

/* kamera */
.camgrid{display:grid;grid-template-columns:minmax(0,2fr) minmax(250px,1fr);gap:var(--gap);align-items:start}
@media (max-width:900px){.camgrid{grid-template-columns:1fr}}
.stage{position:relative;background:#0f0e0c;border-radius:var(--r-card);overflow:hidden;aspect-ratio:16/9;box-shadow:var(--shadow)}
.stage video,.stage canvas{position:absolute;inset:0;width:100%;height:100%}
.stage video{object-fit:contain;transform:scaleX(-1)}
.stage:fullscreen{border-radius:0;aspect-ratio:auto}
.idle{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:#e9e4da;padding:var(--s5)}
.idle .logo{margin:0 auto var(--s3);width:52px;height:52px;border-radius:16px}
.pill{position:absolute;top:var(--s4);left:var(--s4);max-width:calc(100% - 32px);padding:var(--s2) var(--s4);border-radius:999px;background:rgba(20,18,15,.62);color:#fff;font-size:13px;font-weight:600;backdrop-filter:blur(8px)}
.pill.err{background:rgba(180,35,24,.85);color:#fff}
.toast{position:absolute;left:50%;bottom:var(--s4);transform:translate(-50%,12px);opacity:0;transition:.25s;display:flex;gap:var(--s3);align-items:center;padding:var(--s2) var(--s5) var(--s2) var(--s2);border-radius:999px;background:var(--panel);color:var(--ink);box-shadow:var(--shadow);font-weight:600;white-space:nowrap;max-width:94%}
.toast.show{opacity:1;transform:translate(-50%,0)}
.ctrl{display:flex;flex-wrap:wrap;gap:var(--s3);align-items:center;margin-top:var(--gap)}
.ctrl select{max-width:240px}
.feed{display:flex;flex-direction:column;gap:var(--s3)}
.feed .row2{display:flex;justify-content:space-between;align-items:center;gap:var(--s3)}
.camsum{display:flex;align-items:center;gap:var(--s4);padding-bottom:var(--s4);margin-bottom:var(--s4);border-bottom:1px solid var(--line)}
.camsum b{display:block;font-size:16px}
.tips{margin:var(--s4) 0 0;padding:var(--s4) 0 0;border-top:1px solid var(--line);list-style:none;color:var(--mute);font-size:13px}
.tips li+li{margin-top:var(--s1)}

/* dialog */
dialog{border:1px solid var(--line);border-radius:var(--r-card);background:var(--panel);color:var(--ink);padding:var(--s6);width:min(560px,94vw);box-shadow:var(--shadow)}
dialog::backdrop{background:rgba(0,0,0,.45)}
dialog h2{margin:0 0 var(--s4);font-family:Lora,Georgia,serif;font-weight:500;font-size:22px}
dialog label:not(.btn){display:block;margin:0 0 var(--s3);font-size:13.5px;font-weight:500}
dialog input:not([type=file]){display:block;width:100%;margin-top:var(--s1);font-weight:400}
.row{display:flex;gap:var(--s2);flex-wrap:wrap;align-items:center;margin:var(--s2) 0}
.row.end{justify-content:flex-end;margin-top:var(--s5)}
video.pv{width:100%;max-height:260px;background:#000;border-radius:var(--r-inner)}
.thumbs{display:flex;flex-wrap:wrap;gap:var(--s2);margin:var(--s2) 0}
.th{position:relative}
.th img{width:64px;height:64px;object-fit:cover;border-radius:var(--r-ctl);display:block}
.th button{position:absolute;top:-6px;right:-6px;width:20px;height:20px;padding:0;border-radius:50%;border:0;background:var(--bad);color:#fff;cursor:pointer}
</style>
</head>
<body>
<div class="top">
  <button class="btn sm" id="menu" type="button" aria-label="Buka menu"><svg class="ic" viewBox="0 0 24 24"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
  Absensi Wajah
</div>
<div class="scrim" id="scrim"></div>
<div class="app">
  <aside class="side" id="side">
    <div class="brand"><span class="logo"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="9" r="4"/><path d="M4.5 20c1-4 4-6 7.5-6s6.5 2 7.5 6"/></svg></span>Absensi Wajah</div>
    <nav class="nav">
      <a href="#ringkasan" id="n-ringkasan"><svg class="ic" viewBox="0 0 24 24"><path d="M3 11l9-8 9 8M5 10v10h14V10"/></svg>Ringkasan</a>
      <a href="#kamera" id="n-kamera"><svg class="ic" viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.5"/></svg>Kamera absen</a>
      <a href="#kelola" id="n-kelola"><svg class="ic" viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.8-3.6 3.4-5.5 6.5-5.5s5.7 1.9 6.5 5.5M16 5a3.5 3.5 0 010 7M18 14.8c1.9.7 3.2 2.5 3.5 5.2"/></svg>Kelola wajah</a>
      <a href="#rekap" id="n-rekap"><svg class="ic" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg>Rekap kehadiran</a>
    </nav>
    <div class="sfoot">
      <label id="pinbox" hidden>PIN admin <input type="password" id="pin" autocomplete="off"></label>
      <div class="seg" role="group" aria-label="Tema tampilan">
        <button type="button" data-theme-btn="auto" title="Otomatis (ikut sistem)" aria-label="Tema otomatis"><svg class="ic" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/></svg></button>
        <button type="button" data-theme-btn="light" title="Terang" aria-label="Tema terang"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6L7 7M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"/></svg></button>
        <button type="button" data-theme-btn="dark" title="Gelap" aria-label="Tema gelap"><svg class="ic" viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z"/></svg></button>
      </div>
      <span class="dot" id="dot"></span><span id="status">Menghubungkan...</span>
    </div>
  </aside>

  <main class="main">
    <!-- RINGKASAN -->
    <section id="v-ringkasan">
      <div class="hello">
        <div><h1 id="greet">Selamat datang</h1><p class="sub" id="s-sub">Memuat data...</p></div>
        <label class="mute">Tanggal <input type="date" id="date"></label>
      </div>
      <div class="sugs">
        <a class="sug" href="#kamera"><span class="ico"><svg class="ic" viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.5"/></svg></span><span>Buka kamera absen<small>Pindai wajah langsung di sini</small></span></a>
        <a class="sug" href="#kelola" id="q-add"><span class="ico"><svg class="ic" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></span><span>Tambah orang baru<small>Daftarkan wajah dengan foto atau kamera</small></span></a>
        <a class="sug" href="#" id="export"><span class="ico"><svg class="ic" viewBox="0 0 24 24"><path d="M12 4v11M7 11l5 5 5-5M5 20h14"/></svg></span><span>Unduh CSV<small>Absensi pada tanggal terpilih</small></span></a>
      </div>

      <div class="bento">
        <div class="card hero">
          <div class="card-h"><h2 id="h-title">Kehadiran hari ini</h2></div>
          <div class="hero-body">
          <div class="ringbig" role="img" aria-label="Persentase kehadiran">
            <svg viewBox="0 0 120 120" class="rot">
              <defs><linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0" style="stop-color:var(--acc2)"/><stop offset="1" style="stop-color:var(--acc)"/></linearGradient></defs>
              <circle class="rt" cx="60" cy="60" r="50" pathLength="100" style="stroke-width:11"/>
              <circle class="rp" id="h-arc" cx="60" cy="60" r="50" pathLength="100" stroke-dasharray="0 100" style="stroke-width:11;opacity:0"/>
            </svg>
            <div class="mid"><b id="s-pct">0%</b><small>hadir</small></div>
          </div>
          <p class="cap" id="h-cap">Memuat...</p>
          <p class="last" id="h-last"></p>
          </div>
        </div>

        <div class="trio">
          <div class="card stat"><span class="st-ic ok"><svg class="ic" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg></span><b id="s-hadir">0</b><span class="l">Sudah hadir</span></div>
          <div class="card stat"><span class="st-ic warn"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/></svg></span><b id="s-belum">0</b><span class="l">Belum hadir</span></div>
          <div class="card stat"><span class="st-ic"><svg class="ic" viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.8-3.6 3.4-5.5 6.5-5.5s5.7 1.9 6.5 5.5M16 5a3.5 3.5 0 010 7M18 14.8c1.9.7 3.2 2.5 3.5 5.2"/></svg></span><b id="s-total">0</b><span class="l">Wajah terdaftar</span></div>
        </div>

        <div class="card weekc">
          <div class="card-h"><h2>7 hari terakhir</h2><span class="meta">Orang hadir per hari</span></div>
          <div class="week" id="trend"></div>
        </div>

        <div class="card c7">
          <div class="card-h"><h2>Kedatangan terbaru</h2><input type="search" id="q" placeholder="Cari nama" aria-label="Cari nama"></div>
          <div class="scroll"><table><thead><tr><th>Nama</th><th>Masuk</th><th>Status</th><th>Pulang</th></tr></thead><tbody id="present"></tbody></table></div>
        </div>
        <div class="card c5"><div class="card-h"><h2>Belum hadir</h2><span class="meta" id="a-count"></span></div><div id="absent"></div></div>
      </div>
    </section>

    <!-- KAMERA ABSEN -->
    <section id="v-kamera" hidden>
      <div class="hello">
        <div><h1>Kamera absen</h1><p class="sub">Berdiri di depan kamera. Wajah dikenali otomatis dan kehadiran langsung tercatat.</p></div>
      </div>
      <div class="camgrid">
        <div>
          <div class="stage" id="stage">
            <video id="c-video" playsinline muted></video>
            <canvas id="c-canvas"></canvas>
            <div class="idle" id="c-idle"><div><span class="logo"><svg class="ic" viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.5"/></svg></span>Kamera belum aktif<br><span class="mute">Tekan Mulai kamera untuk memulai absen otomatis</span></div></div>
            <div class="pill" id="c-pill" hidden></div>
            <div class="toast" id="c-toast"></div>
          </div>
          <div class="ctrl">
            <select id="c-src" aria-label="Sumber kamera">
              <option value="server">Kamera server (seperti main.py)</option>
              <option value="browser">Kamera browser</option>
            </select>
            <button class="btn pri" id="c-toggle" type="button">Mulai kamera</button>
            <select id="c-dev" aria-label="Pilih kamera" hidden></select>
            <button class="btn" id="c-full" type="button"><svg class="ic" viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>Layar penuh</button>
          </div>
        </div>
        <div class="card">
          <div class="card-h"><h2>Baru saja hadir</h2></div>
          <div class="camsum"><span id="c-ring"></span><div><b id="c-num"></b><span class="mute" id="c-count"></span></div></div>
          <div class="feed" id="c-feed"></div>
          <ul class="tips">
            <li>Jarak ideal 0,5 sampai 3 meter dari kamera.</li>
            <li>Pastikan cahaya cukup dan wajah menghadap lurus.</li>
            <li>Kotak kuning berarti sedang memverifikasi, hijau berarti dikenali.</li>
            <li>Kamera server memakai kamera komputer ini dan tidak bisa berjalan bersamaan dengan main.py run.</li>
          </ul>
        </div>
      </div>
    </section>

    <!-- KELOLA WAJAH -->
    <section id="v-kelola" hidden>
      <div class="hello">
        <div><h1>Kelola wajah</h1><p class="sub">Tambah, ubah, atau hapus wajah terdaftar. Foto tidak disimpan, hanya data pengenal wajah (embedding).</p></div>
        <button class="btn pri" id="k-add" type="button"><svg class="ic" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>Tambah orang</button>
      </div>
      <p id="k-msg" role="status" class="note"></p>
      <div class="card">
        <div class="card-h"><h2>Orang terdaftar</h2><span class="meta" id="k-count"></span></div>
        <div class="scroll tall"><table>
        <thead><tr><th>Nama</th><th>ID</th><th>Sampel</th><th></th></tr></thead>
        <tbody id="k-rows"></tbody>
      </table></div></div>
    </section>

    <!-- REKAP -->
    <section id="v-rekap" hidden>
      <div class="hello">
        <div><h1>Rekap kehadiran</h1><p class="sub" id="r-sub">Persentase = hari hadir dibanding total hari yang punya catatan absensi.</p></div>
      </div>
      <div class="card">
        <div class="card-h"><h2>Per orang</h2><input type="search" id="r-q" placeholder="Cari nama" aria-label="Cari nama"></div>
        <div class="scroll tall"><table>
        <thead><tr><th>Nama</th><th>Kehadiran</th><th>Terlambat</th><th>Total catatan</th><th>Terakhir hadir</th></tr></thead>
        <tbody id="recap"></tbody>
      </table></div></div>
    </section>
  </main>
</div>

<dialog id="dlg" aria-labelledby="dlg-title">
  <h2 id="dlg-title">Tambah orang</h2>
  <label>Nama <input id="d-nama" maxlength="60" autocomplete="off"></label>
  <label>ID (opsional) <input id="d-id" maxlength="40" autocomplete="off"></label>
  <div id="d-media">
    <div class="row">
      <label class="btn">Unggah foto <input type="file" id="d-file" accept="image/*" multiple hidden></label>
      <button class="btn" id="d-cam" type="button">Pakai kamera browser</button>
    </div>
    <div id="d-camwrap" hidden>
      <video class="pv" id="d-video" playsinline muted></video>
      <p id="d-prompt" class="mute">&nbsp;</p>
      <div class="row">
        <button class="btn" id="d-auto" type="button">Ambil 8 foto otomatis</button>
        <button class="btn" id="d-snap" type="button">Ambil 1 foto</button>
      </div>
    </div>
    <div class="thumbs" id="d-thumbs"></div>
    <p class="mute" id="d-count"></p>
  </div>
  <p class="err" id="d-err" role="alert"></p>
  <div class="row end">
    <button class="btn" id="d-cancel" type="button">Batal</button>
    <button class="btn pri" id="d-save" type="button">Simpan</button>
  </div>
</dialog>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const today = () => new Date().toLocaleDateString('sv-SE');
const ini = n => (String(n).trim()[0] || '?').toUpperCase();
const sleep = ms => new Promise(r => setTimeout(r, ms));
const VIEWS = ['ringkasan', 'kamera', 'kelola', 'rekap'];
const dateEl = $('date');
dateEl.value = today();
const INFO_DEFAULT = {people: [], editable: true, pin_required: false, min_samples: 3, max_images: 12};
let view = 'ringkasan', data = null, info = {...INFO_DEFAULT}, pendingAdd = false;

const pinHeader = () => { const p = sessionStorage.getItem('pin'); return p ? {'X-Pin': p} : {}; };
const needPin = () => { $('pinbox').hidden = false; };
let pinTimer = null;
$('pin').value = sessionStorage.getItem('pin') || '';
$('pin').addEventListener('input', e => {
  sessionStorage.setItem('pin', e.target.value);
  clearTimeout(pinTimer);
  pinTimer = setTimeout(() => { load(); if (view === 'kelola') loadPeople(); }, 400);
});

// ---------- tema ----------
function applyTheme(t, save) {
  const el = document.documentElement;
  if (t === 'light' || t === 'dark') el.dataset.theme = t; else { delete el.dataset.theme; t = 'auto'; }
  document.querySelectorAll('[data-theme-btn]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.themeBtn === t)));
  if (save) { try { t === 'auto' ? localStorage.removeItem('theme') : localStorage.setItem('theme', t); } catch (e) { /* abaikan */ } }
}
document.querySelectorAll('[data-theme-btn]').forEach(b => b.onclick = () => applyTheme(b.dataset.themeBtn, true));
applyTheme(document.documentElement.dataset.theme || 'auto');

// ---------- cincin progres ----------
function ring(pct, size, sw, label, cls) {
  const p = Math.max(0, Math.min(100, Math.round(pct)));
  return `<span class="mr ${cls || ''}" style="--s:${size}px" role="img" aria-label="${p}%"><svg class="rot" viewBox="0 0 100 100">` +
    `<circle class="rt" cx="50" cy="50" r="42" pathLength="100" style="stroke-width:${sw}"/>` +
    `<circle class="rp" cx="50" cy="50" r="42" pathLength="100" stroke-dasharray="${p} 100" style="stroke-width:${sw}${p ? '' : ';opacity:0'}"/></svg>` +
    `<i>${label}</i></span>`;
}
const tone = p => p >= 80 ? 'ok' : p >= 50 ? 'warn' : 'low';

// ---------- navigasi ----------
function closeNav() { $('side').classList.remove('open'); $('scrim').classList.remove('open'); }
$('menu').onclick = () => { $('side').classList.add('open'); $('scrim').classList.add('open'); };
$('scrim').onclick = closeNav;

function route() {
  const h = location.hash.slice(1);
  const next = VIEWS.includes(h) ? h : 'ringkasan';
  if (view === 'kamera' && next !== 'kamera') { camStop(); srvStop(); }
  view = next;
  VIEWS.forEach(v => { $('v-' + v).hidden = v !== view; $('n-' + v).classList.toggle('on', v === view); });
  closeNav();
  if (view === 'kelola' || view === 'kamera') loadPeople();
  load();
}

// ---------- data absensi ----------
async function load() {
  if (!dateEl.value) dateEl.value = today();
  const date = view === 'kamera' ? today() : dateEl.value;
  try {
    const r = await fetch('/api/data?date=' + date, {cache: 'no-store', headers: pinHeader()});
    if (r.status === 401) { needPin(); $('status').textContent = 'Masukkan PIN admin'; $('dot').className = 'dot'; return; }
    if (!r.ok) throw new Error(r.status);
    data = await r.json();
    draw();
    $('status').textContent = 'Diperbarui ' + data.updated;
    $('dot').className = 'dot on';
  } catch (e) {
    $('status').textContent = 'Server tidak terjangkau';
    $('dot').className = 'dot';
  }
}

function greeting() {
  const h = new Date().getHours();
  return h < 11 ? 'Selamat pagi' : h < 15 ? 'Selamat siang' : h < 18 ? 'Selamat sore' : 'Selamat malam';
}

function draw() {
  drawSummary(); drawRecap(); drawFeed();
}

function drawSummary() {
  const n = data.present.length, t = data.registered;
  const d = new Date(dateEl.value + 'T00:00');
  const label = d.toLocaleDateString('id-ID', {weekday: 'long', day: 'numeric', month: 'long', year: 'numeric'});
  $('greet').textContent = greeting() + ', Admin';
  $('s-sub').textContent = label + '.' + (t ? '' : ' Belum ada wajah terdaftar, tambahkan lewat Kelola wajah.');
  $('h-title').textContent = dateEl.value === today() ? 'Kehadiran hari ini' : 'Kehadiran ' + d.toLocaleDateString('id-ID', {day: 'numeric', month: 'short'});
  $('s-pct').textContent = data.pct + '%';
  const arc = $('h-arc');
  arc.setAttribute('stroke-dasharray', data.pct + ' 100'); arc.style.opacity = data.pct ? 1 : 0;
  $('h-cap').textContent = t ? `${n} dari ${t} orang sudah hadir` + (data.late ? `, ${data.late} terlambat` : '') : 'Belum ada wajah terdaftar';
  $('h-last').textContent = n ? `Terakhir: ${data.present[0].nama} pukul ${data.present[0].jam}` : 'Belum ada kedatangan';
  $('s-hadir').textContent = n;
  $('s-belum').textContent = t ? data.absent.length : '-';
  $('s-total').textContent = t;
  $('a-count').textContent = t ? `${data.absent.length} orang` : '';
  drawPresent();
  $('absent').innerHTML = !t ? '<p class="mute">Daftar belum tersedia.</p>'
    : data.absent.length ? data.absent.map(x => `<span class="chip">${esc(x)}</span>`).join('')
    : '<p class="mute">Semua orang terdaftar sudah hadir.</p>';
  const den = t || Math.max(1, ...data.trend.map(x => x.count));
  $('trend').innerHTML = data.trend.map((x, i) => {
    const dd = new Date(x.date + 'T00:00');
    return `<div class="wd${i === data.trend.length - 1 ? ' sel' : ''}">${ring(100 * x.count / den, 46, 12, x.count)}` +
      `<span>${esc(dd.toLocaleDateString('id-ID', {weekday: 'short'}))}</span><span>${dd.getDate()}</span></div>`;
  }).join('');
}

function drawPresent() {
  const q = $('q').value.trim().toLowerCase();
  const rows = data.present.filter(p => p.nama.toLowerCase().includes(q));
  $('present').innerHTML = rows.map(p =>
    `<tr><td><div class="who"><span class="av">${esc(ini(p.nama))}</span>${esc(p.nama)}</div></td>` +
    `<td class="num">${esc(p.jam)}</td>` +
    `<td><span class="chip ${p.status === 'terlambat' ? 'late' : 'ok'}">${p.status === 'terlambat' ? 'Terlambat' : 'Hadir'}</span></td>` +
    `<td class="num">${p.pulang ? esc(p.pulang) : '<span class="mute">-</span>'}</td></tr>`).join('')
    || `<tr><td colspan="4" class="mute">${q ? 'Nama tidak ditemukan.' : 'Belum ada yang hadir pada tanggal ini.'}</td></tr>`;
}

function drawRecap() {
  const q = $('r-q').value.trim().toLowerCase();
  const wd = Math.max(1, data.work_days || 0);
  $('r-sub').textContent = `Persentase = hari hadir dibanding ${data.work_days || 0} hari yang punya catatan absensi.`;
  const rows = data.recap.filter(r => r.nama.toLowerCase().includes(q));
  $('recap').innerHTML = rows.map(r => {
    const p = Math.min(100, Math.round(100 * r.days / wd));
    return `<tr><td><div class="who"><span class="av">${esc(ini(r.nama))}</span>${esc(r.nama)}</div></td>` +
      `<td><div class="pcell">${ring(p, 42, 12, p + '%', tone(p))}<span class="mute num">${r.days} hari</span></div></td>` +
      `<td class="num">${r.late ? `<span class="chip late">${r.late} hari</span>` : '<span class="mute">-</span>'}</td>` +
      `<td class="num">${r.total}</td><td class="num">${esc(r.last)}</td></tr>`;
  }).join('') || '<tr><td colspan="5" class="mute">Belum ada data absensi.</td></tr>';
}

function drawFeed() {
  const n = data.present.length, t = data.registered;
  $('c-ring').innerHTML = ring(data.pct, 64, 11, data.pct + '%', 'ok');
  $('c-num').textContent = t ? `${n} dari ${t} orang` : `${n} orang`;
  $('c-count').textContent = 'hadir hari ini';
  $('c-feed').innerHTML = data.present.slice(0, 8).map(p =>
    `<div class="row2"><div class="who"><span class="av">${esc(ini(p.nama))}</span>${esc(p.nama)}</div><span class="mute num">${esc(p.jam)}${p.pulang ? ' → ' + esc(p.pulang) : ''}</span></div>`).join('')
    || '<p class="mute">Belum ada yang hadir.</p>';
}

dateEl.addEventListener('change', load);
$('q').addEventListener('input', () => data && drawPresent());
$('r-q').addEventListener('input', () => data && drawRecap());
setInterval(() => { if (!document.hidden && view !== 'kelola') load(); }, 4000);

// unduh CSV lewat fetch agar PIN ikut terkirim
$('export').onclick = async e => {
  e.preventDefault();
  try {
    const r = await fetch('/export?date=' + dateEl.value, {headers: pinHeader(), cache: 'no-store'});
    if (r.status === 401) { needPin(); throw new Error('Masukkan PIN admin di sidebar.'); }
    if (!r.ok) throw new Error('Gagal mengunduh (' + r.status + ')');
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement('a');
    a.href = url; a.download = 'absensi_' + dateEl.value + '.csv';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (err) { $('status').textContent = err.message; }
};

// ---------- kamera absen ----------
const cam = {stream: null, on: false, tok: 0};
const srv = {on: false, tok: 0};
const COL = {tercatat: '#16a34a', hadir: '#16a34a', cek: '#d97706', unknown: '#dc2626', far: '#78716c'};
const grab = document.createElement('canvas');
let toastTimer = null, reloadTimer = null;

function setPill(text, err) {
  const p = $('c-pill');
  p.hidden = !text; p.textContent = text || ''; p.className = 'pill' + (err ? ' err' : '');
}

async function listDevices() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    const cur = cam.stream && cam.stream.getVideoTracks()[0].getSettings().deviceId;
    $('c-dev').innerHTML = devs.map((d, i) => `<option value="${esc(d.deviceId)}"${d.deviceId === cur ? ' selected' : ''}>${esc(d.label || 'Kamera ' + (i + 1))}</option>`).join('');
    $('c-dev').hidden = devs.length < 2;
  } catch (e) { /* abaikan */ }
}

// --- sumber: kamera browser (frame dikirim ke server) ---
async function camStart() {
  if (!info.editable) { setPill('Kamera absen nonaktif: jalankan dashboard dengan --pin jika dibuka dari jaringan.', true); return; }
  if (!navigator.mediaDevices) { setPill('Kamera browser hanya tersedia di localhost atau HTTPS.', true); return; }
  camStop(true);
  try {
    const dev = $('c-dev').value;
    cam.stream = await navigator.mediaDevices.getUserMedia({video: dev && !$('c-dev').hidden ? {deviceId: {exact: dev}} : {width: {ideal: 1280}, height: {ideal: 720}}});
    const v = $('c-video'); v.srcObject = cam.stream; await v.play();
  } catch (e) { setPill('Kamera tidak bisa dibuka (izin ditolak atau sedang dipakai aplikasi lain).', true); return; }
  cam.on = true; const tok = ++cam.tok;
  const id = Math.random().toString(36).slice(2, 12);   // satu sesi pelacak di server per kamera aktif
  $('c-idle').hidden = true; $('c-toggle').textContent = 'Hentikan kamera';
  setPill('Menunggu wajah...');
  await listDevices();
  scanLoop(tok, id);
}

function camStop(keepUi) {
  if (!cam.on && !cam.stream) return;
  cam.on = false; cam.tok++;
  if (cam.stream) cam.stream.getTracks().forEach(t => t.stop());
  cam.stream = null; $('c-video').srcObject = null;
  const c = $('c-canvas'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
  if (!keepUi) { $('c-idle').hidden = false; setPill(''); $('c-toggle').textContent = 'Mulai kamera'; }
}

async function scanLoop(tok, id) {
  while (cam.on && cam.tok === tok) {
    if (document.hidden) { await sleep(500); continue; }
    const v = $('c-video');
    if (v.videoWidth) {
      const w = Math.min(1280, v.videoWidth), h = Math.round(v.videoHeight * w / v.videoWidth);
      grab.width = w; grab.height = h;
      grab.getContext('2d').drawImage(v, 0, 0, w, h);
      const blob = await new Promise(r => grab.toBlob(r, 'image/jpeg', 0.72));
      try {
        const r = await fetch('/api/scan?cam=' + id, {method: 'POST', headers: {'Content-Type': 'image/jpeg', ...pinHeader()}, body: blob});
        const j = await r.json().catch(() => ({message: 'Respons server tidak valid'}));
        if (!r.ok) throw new Error(j.message || r.status);
        if (!cam.on || cam.tok !== tok) return;
        drawFaces(j.faces);
        j.events.forEach(showToast);
        if (j.events.length) { clearTimeout(reloadTimer); reloadTimer = setTimeout(load, 300); }
        setPill(j.db_empty ? 'Database wajah kosong. Tambahkan orang di Kelola wajah.'
          : j.faces.length ? `${j.faces.length} wajah terdeteksi` : 'Menunggu wajah...', j.db_empty);
      } catch (e) {
        if (cam.tok !== tok) return;
        setPill(e.message, true);
        await sleep(1500);
      }
    }
    await sleep(80);
  }
}

function layout() {
  const c = $('c-canvas'), v = $('c-video'), dpr = window.devicePixelRatio || 1;
  const cw = c.clientWidth, ch = c.clientHeight;
  if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) { c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr); }
  const vw = v.videoWidth || 16, vh = v.videoHeight || 9, s = Math.min(cw / vw, ch / vh);
  const dw = vw * s, dh = vh * s;
  return {dpr, cw, ch, ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw, dh};
}

function drawFaces(list) {
  const L = layout(), g = $('c-canvas').getContext('2d');
  g.setTransform(L.dpr, 0, 0, L.dpr, 0, 0); g.clearRect(0, 0, L.cw, L.ch);
  g.font = '600 13px "Plus Jakarta Sans",system-ui,sans-serif'; g.textBaseline = 'middle';
  for (const f of list) {
    const [bx, by, bw, bh] = f.box;
    const x = L.ox + (1 - bx - bw) * L.dw, y = L.oy + by * L.dh, w = bw * L.dw, h = bh * L.dh;   // video dicerminkan
    const col = COL[f.status] || COL.unknown;
    const text = f.status === 'tercatat' ? `✓ ${f.nama} · Tercatat`
      : f.status === 'hadir' ? `${f.nama} · Sudah dikenali`
      : f.status === 'cek' ? `${f.nama}?`
      : f.status === 'far' ? 'Terlalu jauh' : 'Tidak dikenal';
    g.strokeStyle = col; g.lineWidth = 2.5; g.beginPath();
    if (g.roundRect) g.roundRect(x, y, w, h, 14); else g.rect(x, y, w, h);
    g.stroke();
    const tw = g.measureText(text).width + 18, ly = Math.max(y - 28, 2);
    g.fillStyle = col; g.beginPath();
    if (g.roundRect) g.roundRect(x, ly, tw, 24, 12); else g.rect(x, ly, tw, 24);
    g.fill();
    g.fillStyle = '#fff'; g.fillText(text, x + 9, ly + 12.5);
  }
}

function showToast(e) {
  const t = $('c-toast');
  const aksi = e.tipe === 'pulang' ? 'Pulang' : (e.status === 'terlambat' ? 'Terlambat' : 'Hadir');
  t.innerHTML = `<span class="av">${esc(ini(e.nama))}</span>${aksi}: ${esc(e.nama)} · ${esc(e.jam)}`;
  t.classList.add('show'); clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
}

// --- sumber: kamera server (server membaca kamera, sama seperti main.py) ---
async function srvStart() {
  if (!info.editable) { setPill('Kamera server nonaktif: jalankan dashboard dengan --pin jika dibuka dari jaringan.', true); return; }
  setPill('Membuka kamera server...');
  try { await api('POST', '/api/server-cam/start', {kamera: 0}); }
  catch (e) { setPill(e.message, true); return; }
  srv.on = true; const tok = ++srv.tok;
  $('c-idle').hidden = true; $('c-toggle').textContent = 'Hentikan kamera';
  setPill('Menunggu wajah...');
  srvLoop(tok);
}

async function srvStop() {
  if (!srv.on) return;
  srv.on = false; srv.tok++;
  try { await api('POST', '/api/server-cam/stop', {}); } catch (e) { /* abaikan */ }
  const c = $('c-canvas'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
  $('c-idle').hidden = false; setPill(''); $('c-toggle').textContent = 'Mulai kamera';
}

async function srvLoop(tok) {
  let lastStatus = 0;
  while (srv.on && srv.tok === tok) {
    if (document.hidden) { await sleep(500); continue; }
    try {
      const r = await fetch('/api/server-cam/frame', {cache: 'no-store', headers: pinHeader()});
      if (r.status === 401 || r.status === 403) throw new Error('PIN salah atau kamera server nonaktif.');
      if (r.ok && r.status !== 204) {
        const bmp = await createImageBitmap(await r.blob());
        if (!srv.on || srv.tok !== tok) { bmp.close(); return; }
        const L = layout(), g = $('c-canvas').getContext('2d');
        const s = Math.min(L.cw / bmp.width, L.ch / bmp.height);
        const dw = bmp.width * s, dh = bmp.height * s;
        g.setTransform(L.dpr, 0, 0, L.dpr, 0, 0); g.clearRect(0, 0, L.cw, L.ch);
        g.drawImage(bmp, (L.cw - dw) / 2, (L.ch - dh) / 2, dw, dh); bmp.close();
      }
      if (Date.now() - lastStatus > 1000) {
        lastStatus = Date.now();
        const j = await (await fetch('/api/server-cam/status', {cache: 'no-store', headers: pinHeader()})).json();
        j.events.forEach(showToast);
        if (j.events.length) { clearTimeout(reloadTimer); reloadTimer = setTimeout(load, 300); }
        if (!j.running) {
          srv.on = false; setPill(j.err || 'Kamera server berhenti.', true);
          $('c-idle').hidden = false; $('c-toggle').textContent = 'Mulai kamera'; return;
        }
        setPill(`Kamera server aktif · ${j.fps} FPS`);
      }
    } catch (e) {
      if (srv.tok !== tok) return;
      setPill(e.message, true); await sleep(1500);
    }
    await sleep(40);
  }
}

$('c-toggle').onclick = () => {
  if (cam.on) camStop();
  else if (srv.on) srvStop();
  else if ($('c-src').value === 'server') srvStart(); else camStart();
};
$('c-dev').onchange = () => cam.on && camStart();
$('c-full').onclick = () => { const s = $('stage'); document.fullscreenElement ? document.exitFullscreen() : s.requestFullscreen && s.requestFullscreen(); };

// ---------- kelola wajah ----------
let mode = null, target = null, imgs = [], stream = null, busy = false;
const PROMPTS = ['Hadap lurus', 'Tengok sedikit ke kiri', 'Tengok sedikit ke kanan', 'Dongak sedikit',
                 'Tunduk sedikit', 'Hadap lurus, rileks', 'Senyum tipis', 'Hadap lurus'];

function say(t, err) { const m = $('k-msg'); m.textContent = t; m.className = 'note ' + (err ? 'err' : 'okmsg'); }

async function api(method, path, body) {
  const r = await fetch(path, {method, headers: {'Content-Type': 'application/json', ...pinHeader()}, body: JSON.stringify(body)});
  const j = await r.json().catch(() => ({message: 'Respons server tidak valid'}));
  if (!r.ok) { const e = new Error(j.message || r.status); e.rejected = j.rejected || []; throw e; }
  return j;
}

async function loadPeople() {
  try {
    const r = await fetch('/api/people', {cache: 'no-store', headers: pinHeader()});
    if (r.status === 401) { needPin(); say('Masukkan PIN admin di sidebar.', true); return; }
    info = {...info, ...(await r.json())};
  } catch (e) { say('Server tidak terjangkau', true); return; }
  $('pinbox').hidden = !info.pin_required;
  $('k-add').disabled = !info.editable;
  $('k-count').textContent = `${info.people.length} orang`;
  $('k-rows').innerHTML = info.people.map((p, i) =>
    `<tr><td><div class="who"><span class="av">${esc(ini(p.nama))}</span>${esc(p.nama)}</div></td><td>${esc(p.id) || '<span class="mute">-</span>'}</td><td class="num">${p.sampel}</td>` +
    `<td class="act"><button class="btn sm" data-a="samples" data-i="${i}">Tambah sampel</button> ` +
    `<button class="btn sm" data-a="edit" data-i="${i}">Ubah</button> ` +
    `<button class="btn sm bad" data-a="del" data-i="${i}">Hapus</button></td></tr>`).join('')
    || '<tr><td colspan="4" class="mute">Belum ada wajah terdaftar.</td></tr>';
  if (!info.editable) say('Pengelolaan nonaktif: jalankan dashboard dengan --pin untuk mengaktifkannya dari jaringan.', true);
  else if ($('k-msg').classList.contains('err')) say('');
  if (pendingAdd) { pendingAdd = false; if (info.editable && view === 'kelola') openDlg('add'); }
}

function renderThumbs() {
  $('d-thumbs').innerHTML = imgs.map((s, i) =>
    `<span class="th"><img src="${s}" alt="Foto ${i + 1}"><button type="button" data-rm="${i}" aria-label="Hapus foto ${i + 1}">&times;</button></span>`).join('');
  $('d-count').textContent = `${imgs.length} foto dipilih. Disarankan 5-8 foto, wajah menghadap kamera, cahaya cukup.`;
}

function openDlg(m, p) {
  mode = m; target = p || null; imgs = [];
  $('dlg-title').textContent = m === 'add' ? 'Tambah orang' : m === 'samples' ? `Tambah sampel: ${p.nama}` : `Ubah: ${p.nama}`;
  $('d-nama').value = p ? p.nama : ''; $('d-id').value = p ? p.id : '';
  $('d-nama').disabled = $('d-id').disabled = m === 'samples';
  $('d-media').hidden = m === 'edit';
  $('d-save').textContent = m === 'edit' ? 'Simpan perubahan' : 'Simpan';
  $('d-err').textContent = ''; $('d-camwrap').hidden = true; $('d-prompt').innerHTML = '&nbsp;';
  renderThumbs();
  $('dlg').showModal();
}

async function fileToJpeg(f) {
  const bmp = await createImageBitmap(f);
  const k = Math.min(1, 1280 / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height); bmp.close();
  return c.toDataURL('image/jpeg', 0.9);
}

async function addFiles(files) {
  for (const f of files) {
    if (imgs.length >= info.max_images) { $('d-err').textContent = `Maksimal ${info.max_images} foto.`; break; }
    try { imgs.push(await fileToJpeg(f)); } catch (e) { $('d-err').textContent = `${f.name}: bukan gambar yang bisa dibaca.`; }
  }
  renderThumbs();
}

async function startCam() {
  $('d-err').textContent = '';
  if (!navigator.mediaDevices) { $('d-err').textContent = 'Kamera browser hanya tersedia di localhost atau HTTPS. Gunakan unggah foto.'; return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 1280}, height: {ideal: 720}}});
    $('d-video').srcObject = stream; await $('d-video').play(); $('d-camwrap').hidden = false;
  } catch (e) { $('d-err').textContent = 'Kamera tidak bisa dibuka (izin ditolak atau sedang dipakai). Gunakan unggah foto.'; }
}

function stopCam() { if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; }

function snap() {
  const v = $('d-video');
  if (!v.videoWidth || imgs.length >= info.max_images) return;
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0); imgs.push(c.toDataURL('image/jpeg', 0.9)); renderThumbs();
}

async function autoSnap() {
  if (busy) return;
  busy = true; $('d-auto').disabled = true;
  for (let i = 0; i < 8 && imgs.length < info.max_images && stream; i++) {
    $('d-prompt').textContent = PROMPTS[i];
    await sleep(900); snap();
  }
  $('d-prompt').textContent = 'Selesai. Periksa foto, lalu Simpan.';
  busy = false; $('d-auto').disabled = false;
}

async function save() {
  const nama = $('d-nama').value.trim(), id = $('d-id').value.trim();
  if (mode === 'add' && imgs.length < info.min_samples) { $('d-err').textContent = `Minimal ${info.min_samples} foto.`; return; }
  if (mode === 'samples' && !imgs.length) { $('d-err').textContent = 'Pilih minimal 1 foto.'; return; }
  $('d-err').textContent = ''; $('d-save').disabled = true; $('d-save').textContent = 'Memproses...';
  try {
    const j = mode === 'add' ? await api('POST', '/api/people', {nama, id, images: imgs})
      : mode === 'samples' ? await api('POST', '/api/people/samples', {nama: target.nama, images: imgs})
      : await api('PUT', '/api/people', {nama: target.nama, nama_baru: nama, id});
    $('dlg').close();
    say(j.message + (j.warning ? ' ' + j.warning : '') + (j.rejected && j.rejected.length ? ' Ditolak: ' + j.rejected.join('; ') + '.' : ''));
    loadPeople();
  } catch (e) {
    $('d-err').textContent = e.message + (e.rejected && e.rejected.length ? ' (' + e.rejected.join('; ') + ')' : '');
  }
  $('d-save').disabled = false; $('d-save').textContent = mode === 'edit' ? 'Simpan perubahan' : 'Simpan';
}

async function del(p) {
  if (!confirm(`Hapus ${p.nama} beserta ${p.sampel} sampel wajahnya? Riwayat absensi tidak ikut terhapus.`)) return;
  try { say((await api('DELETE', '/api/people', {nama: p.nama})).message); } catch (e) { say(e.message, true); }
  loadPeople();
}

window.addEventListener('hashchange', route);
$('q-add').onclick = e => { e.preventDefault(); pendingAdd = true; if (view === 'kelola') loadPeople(); else location.hash = '#kelola'; };
$('k-add').onclick = () => openDlg('add');
$('k-rows').onclick = e => {
  const b = e.target.closest('button[data-a]'); if (!b) return;
  const p = info.people[+b.dataset.i];
  if (b.dataset.a === 'del') del(p); else openDlg(b.dataset.a, p);
};
$('d-file').onchange = e => { addFiles([...e.target.files]); e.target.value = ''; };
$('d-cam').onclick = startCam; $('d-snap').onclick = snap; $('d-auto').onclick = autoSnap;
$('d-thumbs').onclick = e => { const b = e.target.closest('button[data-rm]'); if (b) { imgs.splice(+b.dataset.rm, 1); renderThumbs(); } };
$('d-save').onclick = save; $('d-cancel').onclick = () => $('dlg').close();
$('dlg').addEventListener('close', stopCam);
fetch('/api/info', {cache: 'no-store'}).then(r => r.json()).then(j => { info = {...info, ...j}; $('pinbox').hidden = !j.pin_required; }).catch(() => {});
route();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype, extra=None):
        b = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def send_bytes(self, code, b, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self.send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/info":      # publik: hanya status PIN dan mode edit
            return self.send(200, json.dumps({"pin_required": bool(PIN), "editable": EDIT_ENABLED}), JSON)
        if u.path not in ("/api/people", "/api/data", "/export",
                          "/api/server-cam/frame", "/api/server-cam/status"):
            return self.send(404, "Halaman tidak ditemukan", "text/plain; charset=utf-8")
        if PIN and not pin_ok(self.headers):   # data absensi ikut dilindungi PIN
            return self.send(401, json.dumps({"ok": False, "message": "PIN diperlukan."}), JSON)
        if u.path == "/api/server-cam/frame":
            if not EDIT_ENABLED:
                return self.send(403, "Nonaktif", "text/plain; charset=utf-8")
            scanner.last_req = time.time()
            jpg = scanner.jpeg
            if jpg is None:
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            return self.send_bytes(200, jpg, "image/jpeg")
        if u.path == "/api/server-cam/status":
            if scanner.running:
                scanner.last_req = time.time()
            return self.send(200, json.dumps(scanner.status()), JSON)
        if u.path == "/api/people":
            return self.send(200, json.dumps(people_info()), JSON)
        date = parse_qs(u.query).get("date", [datetime.now().strftime("%Y-%m-%d")])[0]
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return self.send(400, "Tanggal tidak valid", "text/plain; charset=utf-8")
        if u.path == "/api/data":
            self.send(200, json.dumps(build(date)), JSON)
        else:
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["tanggal", "jam", "nama", "tipe", "status"])
            w.writerows((d, j, n, t, s) for d, j, n, t, s in cached(LOG_FILE, read_rows) if d == date)
            self.send(200, "\ufeff" + buf.getvalue(), "text/csv; charset=utf-8",   # BOM agar rapi di Excel
                      {"Content-Disposition": f'attachment; filename="absensi_{date}.csv"'})

    def guard(self, ctype, limit):
        """Cek keamanan (aktif, origin, host, content-type, PIN, ukuran) lalu kembalikan isi body."""
        host, origin = self.headers.get("Host", ""), self.headers.get("Origin")
        if not EDIT_ENABLED:
            raise ApiError(403, "Fitur ini nonaktif: jalankan dashboard dengan --pin jika dibuka dari jaringan.")
        if origin and urlparse(origin).netloc != host:
            raise ApiError(403, "Permintaan lintas situs ditolak.")
        if not PIN and host.rsplit(":", 1)[0].strip("[]") not in ("127.0.0.1", "localhost", "::1"):
            raise ApiError(403, "Host tidak diizinkan.")
        if not self.headers.get("Content-Type", "").startswith(ctype):
            raise ApiError(415, f"Content-Type harus {ctype}.")
        if not pin_ok(self.headers):
            raise ApiError(403, "PIN salah atau belum diisi.")
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= limit:
            raise ApiError(413, "Ukuran data tidak valid atau terlalu besar.")
        return self.rfile.read(n)

    def route(self, method):
        """Aksi CRUD: cek keamanan -> baca JSON -> jalankan di bawah kunci -> kirim hasil."""
        action = ROUTES.get((method, urlparse(self.path).path))
        if action is None:
            return self.send(404, "Halaman tidak ditemukan", "text/plain; charset=utf-8")
        try:
            raw = self.guard("application/json", MAX_BODY)
            try:
                data = json.loads(raw)
            except ValueError:
                raise ApiError(400, "JSON tidak valid.")
            if not isinstance(data, dict):
                raise ApiError(400, "JSON tidak valid.")
            with EDIT_LOCK:
                out = action(data)
            print(f"[KELOLA] {out['message']}")
            self.send(200, json.dumps({"ok": True, **out}), JSON)
        except ApiError as e:
            self.send(e.code, json.dumps({"ok": False, "message": e.msg, "rejected": e.rejected}), JSON)
        except Exception as e:  # model/file bermasalah
            self.send(500, json.dumps({"ok": False, "message": f"Gagal: {e}"}), JSON)

    def scan(self):
        """Terima satu frame JPEG dari kamera browser, kembalikan wajah terdeteksi dan absen baru."""
        q = parse_qs(urlparse(self.path).query)
        cam = re.sub(r"[^A-Za-z0-9_-]", "", (q.get("cam") or ["default"])[0])[:40] or "default"
        try:
            out = api_scan(self.guard("image/jpeg", MAX_FRAME), cam)
            self.send(200, json.dumps({"ok": True, **out}), JSON)
        except ApiError as e:
            self.send(e.code, json.dumps({"ok": False, "message": e.msg}), JSON)
        except Exception as e:
            self.send(500, json.dumps({"ok": False, "message": f"Gagal memindai: {e}"}), JSON)

    def do_POST(self):
        if urlparse(self.path).path == "/api/scan":
            return self.scan()
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")

    def do_DELETE(self):
        self.route("DELETE")


def main():
    global PIN, EDIT_ENABLED
    ap = argparse.ArgumentParser(description="Dashboard Admin Absensi Wajah")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--pin", default="", help="PIN admin untuk kamera absen, pengelolaan wajah, dan data absensi (wajib jika --host bukan localhost)")
    a = ap.parse_args()
    PIN = a.pin
    EDIT_ENABLED = a.host in ("127.0.0.1", "localhost", "::1") or bool(PIN)
    if not EDIT_ENABLED:
        print("PERINGATAN: dashboard terbuka di jaringan tanpa --pin, jadi kamera absen dan pengelolaan wajah dinonaktifkan "
              "(hanya baca). Data absensi bisa dibaca siapa pun di jaringan; pakai --pin untuk melindunginya.")

    server = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"Dashboard berjalan di {url}  (Ctrl+C untuk berhenti)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        scanner.stop()
        server.server_close()


if __name__ == "__main__":
    main()