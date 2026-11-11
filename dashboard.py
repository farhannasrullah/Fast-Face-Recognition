"""
Dashboard Admin Absensi Wajah - gaya UI selaras dengan chat RAG (netral hangat, aksen ungu).

Halaman  : Ringkasan | Kamera absen (built-in di browser) | Kelola wajah (CRUD) | Rekap
Membaca/menulis absensi.csv dan faces_db.npz milik main.py (letakkan di folder yang sama).

Jalankan : python dashboard.py
Buka     : http://127.0.0.1:8000  (otomatis terbuka di browser)
Opsi     : --port 8000
           --host 0.0.0.0   (agar bisa dibuka dari HP/PC lain di jaringan yang sama)
           --no-browser     (jangan buka browser otomatis)
           --pin 1234       (wajib jika dibuka dari jaringan agar kamera absen dan pengelolaan wajah aktif)

Kamera absen memakai kamera browser. Browser hanya mengizinkan kamera di localhost atau HTTPS,
jadi untuk perangkat lain di jaringan gunakan HTTPS (mis. reverse proxy) atau buka dari komputer ini.
Boleh berjalan bersamaan dengan 'main.py run': perubahan wajah dimuat ulang otomatis.
"""
import argparse
import base64
import csv
import hmac
import io
import json
import os
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
_cache = {}


def cached(path, loader):
    """Baca ulang file hanya jika isinya berubah (hemat CPU dan RAM)."""
    try:
        st = os.stat(path)
    except OSError:
        return loader(None)
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get((path, loader.__name__))
    if not hit or hit[0] != key:
        hit = (key, loader(path))
        _cache[(path, loader.__name__)] = hit
    return hit[1]


def read_rows(path):
    rows = []
    if path:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.reader(f):
                if len(r) >= 3 and r[0] != "tanggal":
                    rows.append((r[0], r[1], r[2]))
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


def build(date):
    rows = cached(LOG_FILE, read_rows)
    registered = cached(DB_FILE, read_names)
    present = sorted(((j, n) for d, j, n in rows if d == date), reverse=True)
    here = {n for _, n in present}

    end = datetime.strptime(date, "%Y-%m-%d")
    days = [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
    per_day = {}
    recap = {}
    for d, j, n in rows:
        per_day.setdefault(d, set()).add(n)
        r = recap.setdefault(n, {"nama": n, "total": 0, "last": ""})
        r["total"] += 1
        r["last"] = max(r["last"], f"{d} {j}")

    total = len(registered)
    return {
        "date": date,
        "registered": total,
        "present": [{"jam": j, "nama": n} for j, n in present],
        "absent": [n for n in registered if n not in here],
        "pct": min(100, round(100 * len(here) / total)) if total else 0,
        "trend": [{"date": d, "count": len(per_day.get(d, ()))} for d in days],
        "recap": sorted(recap.values(), key=lambda r: (-r["total"], r["nama"])),
        "updated": datetime.now().strftime("%H:%M:%S"),
    }


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
SCAN_MIN_FACE_PX = 56      # lebar wajah minimum (px, pada frame kiriman browser)
SCAN_STREAK_GAP = 2.5      # detik maksimum antar kecocokan beruntun
RECHECK_LOG = 5.0          # jeda cek ulang catatan absen untuk orang yang sama
_core = {}
_streak, _checked = {}, {}


class ApiError(Exception):
    def __init__(self, code, msg, rejected=None):
        super().__init__(msg)
        self.code, self.msg, self.rejected = code, msg, rejected or []


def get_main():
    """Muat main.py (OpenCV) sekali saja, hanya saat pertama kali dibutuhkan."""
    if "main" not in _core:
        os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(40_000_000))  # tolak gambar raksasa
        sys.path.insert(0, BASE)
        import main
        main.cv2.setNumThreads(main.CV_THREADS)
        _core["main"] = main
    return _core["main"]


def get_engine():
    main = get_main()
    if "engine" not in _core:
        _core["engine"] = main.FaceEngine(*main.ensure_models())
    return _core["engine"]


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
    other, _, sc = db.match(mean)
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


# ----------------------------- Kamera absen (frame dari browser) -----------------------------
def scan_db(main):
    """Database untuk pemindaian; dimuat ulang otomatis bila file berubah."""
    db = _core.get("db")
    if db is None:
        db = _core["db"] = main.Database(DB_FILE)
    elif db.changed():
        db.reload()
    return db


def log_once(main, name, pid, score, now, events):
    """Catat absen sesuai aturan cooldown main.py. Kembalikan 'tercatat' atau 'hadir'."""
    chk = _checked.get(name)
    if chk and now - chk[0] < RECHECK_LOG:
        return chk[1]
    lg = main.AttendanceLogger()        # baca ulang CSV agar sinkron dengan 'main.py run'
    dt = datetime.now()
    if lg.should_log(name, dt):
        lg.log(name, pid, score, dt)
        events.append({"nama": name, "id": pid, "jam": dt.strftime("%H:%M:%S")})
        print(f"[ABSEN] {dt:%Y-%m-%d %H:%M:%S}  {name}  (skor {score:.2f}, kamera web)")
        status = "tercatat"
    else:
        status = "hadir"
    _checked[name] = (now, status)
    return status


def api_scan(raw):
    main, engine = get_main(), get_engine()
    np, cv2 = main.np, main.cv2
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(400, "Frame bukan gambar yang valid.")
    h, w = img.shape[:2]
    sw = min(w, 640)
    sh = max(1, round(h * sw / w))
    small = img if sw == w else cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    scale = (w / sw, h / sh)
    faces_out, events = [], []
    with SCAN_LOCK:
        db = scan_db(main)
        found = []
        with ENGINE_LOCK:
            for row in engine.detect(small):
                big = row[2] * scale[0] >= SCAN_MIN_FACE_PX
                found.append((row, engine.embed(img, row, scale) if big else None))
        now = time.time()
        for k in [k for k, v in _streak.items() if now - v[1] > SCAN_STREAK_GAP]:
            del _streak[k]
        for row, emb in found:
            box = [float(row[0] / sw), float(row[1] / sh), float(row[2] / sw), float(row[3] / sh)]
            if emb is None:
                faces_out.append({"box": box, "status": "far"})
                continue
            name, pid, score = db.match(emb)
            if name is None or score < main.MATCH_THRESH:
                faces_out.append({"box": box, "status": "unknown", "skor": round(score, 2)})
                continue
            st = _streak.get(name)
            n = st[0] + 1 if st else 1
            _streak[name] = (n, now)
            if n < main.CONFIRM_MATCHES:
                faces_out.append({"box": box, "status": "cek", "nama": name, "skor": round(score, 2)})
                continue
            status = log_once(main, name, pid, score, now, events)
            faces_out.append({"box": box, "status": status, "nama": name, "skor": round(score, 2)})
        db_empty = db.is_empty()
    return {"faces": faces_out, "events": events, "db_empty": db_empty}


ROUTES = {("POST", "/api/people"): api_create, ("POST", "/api/people/samples"): api_add_samples,
          ("PUT", "/api/people"): api_update, ("DELETE", "/api/people"): api_delete}


PAGE = r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Absensi Wajah · Admin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{color-scheme:light dark;--bg:#faf9f7;--side:#f3f1ec;--panel:#ffffff;--ink:#1f1d1a;--mute:#5c574f;--line:#e5e0d8;--acc:#7c3aed;--soft:#efe8fd;--on-acc:#ffffff;--ok:#15803d;--ok-soft:#dcf3e3;--bad:#b42318;--bad-soft:#fde4e1;--shadow:0 1px 2px rgba(31,29,26,.06),0 8px 24px rgba(31,29,26,.07)}
@media (prefers-color-scheme:dark){:root{--bg:#1b1916;--side:#22201c;--panel:#2a2723;--ink:#f1ede6;--mute:#b3ab9f;--line:#3a362f;--acc:#a78bfa;--soft:#352b52;--on-acc:#1b1916;--ok:#6ee7a0;--ok-soft:#1f3a2a;--bad:#f97066;--bad-soft:#40201d;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.28)}}
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Plus Jakarta Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.ic{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;flex:none}
.app{display:grid;grid-template-columns:264px minmax(0,1fr);min-height:100vh}

/* sidebar */
.side{background:var(--side);border-right:1px solid var(--line);padding:18px 14px;display:flex;flex-direction:column;gap:4px;position:sticky;top:0;height:100vh}
.brand{display:flex;align-items:center;gap:10px;padding:4px 8px 16px;font-weight:700;font-size:16px}
.logo{width:34px;height:34px;border-radius:11px;background:var(--acc);color:var(--on-acc);display:grid;place-items:center}
.nav{display:flex;flex-direction:column;gap:2px}
.nav a{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:12px;color:var(--mute);text-decoration:none;font-weight:500}
.nav a:hover{background:color-mix(in srgb,var(--acc) 7%,transparent)}
.nav a.on{background:var(--soft);color:var(--acc);font-weight:650}
.sfoot{margin-top:auto;padding:12px 8px 0;border-top:1px solid var(--line);font-size:13px;color:var(--mute)}
.sfoot label{display:block;margin-bottom:10px}
.sfoot input{margin-top:4px;width:100%}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--mute);margin-right:6px}
.dot.on{background:var(--ok)}
.top{display:none}
.scrim{display:none}
@media (max-width:860px){
  .app{display:block}
  .side{position:fixed;inset:0 auto 0 0;width:282px;transform:translateX(-102%);transition:transform .25s;z-index:30;height:100dvh}
  .side.open{transform:none;box-shadow:var(--shadow)}
  .scrim.open{display:block;position:fixed;inset:0;background:rgba(0,0,0,.4);z-index:20}
  .top{display:flex;align-items:center;gap:10px;padding:10px 14px;background:var(--bg);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;font-weight:700}
}

/* konten */
.main{padding:36px clamp(16px,4vw,48px) 56px;max-width:1120px;margin:0 auto}
.hello{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;justify-content:space-between;margin-bottom:22px}
h1{font-family:Lora,Georgia,serif;font-weight:500;font-size:clamp(26px,4vw,36px);line-height:1.2;margin:0}
.sub{margin:6px 0 0;color:var(--mute)}
h2{margin:0 0 12px;font-size:16px;font-weight:650}
.mute{color:var(--mute)}
input,select,.btn{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:7px 12px}
input::placeholder{color:var(--mute)}
.btn{display:inline-flex;align-items:center;gap:8px;text-decoration:none;cursor:pointer}
.btn:hover{border-color:var(--acc)}
button.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.pri{background:var(--acc);border-color:var(--acc);color:var(--on-acc);font-weight:600}
.btn.sm{padding:3px 10px;font-size:13px;border-radius:8px}
.btn.bad{color:var(--bad);border-color:var(--bad)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:20px;box-shadow:var(--shadow)}

/* kartu saran (seperti saran pertanyaan di chat) */
.sugs{display:flex;gap:12px;overflow-x:auto;scroll-snap-type:x mandatory;padding:2px 2px 14px;margin:0 -2px 8px}
.sug{flex:1 0 230px;scroll-snap-align:start;display:flex;gap:12px;align-items:center;padding:14px 16px;border:1px solid var(--line);border-radius:16px;background:var(--panel);color:var(--ink);text-decoration:none;font-weight:600;cursor:pointer;font-family:inherit;font-size:14px;text-align:left}
.sug:hover{border-color:var(--acc);background:var(--soft)}
.sug .ico{width:36px;height:36px;border-radius:11px;background:var(--soft);color:var(--acc);display:grid;place-items:center;flex:none}
.sug small{display:block;color:var(--mute);font-weight:400}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
.stat{padding:16px 18px;border-radius:18px;background:var(--panel);border:1px solid var(--line)}
.stat b{display:block;font-size:30px;line-height:1.1;font-weight:650;font-variant-numeric:tabular-nums}
.stat span{color:var(--mute);font-size:13px}
.meter{height:6px;border-radius:3px;background:var(--line);overflow:hidden;margin-top:10px}
.meter i{display:block;height:100%;width:0;background:var(--acc);transition:width .4s}
.cols{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:18px;align-items:start}
@media (max-width:900px){.cols{grid-template-columns:1fr}}
.cols>div>.card+.card{margin-top:18px}
.head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:8px}
.head h2{margin:0}
table{width:100%;border-collapse:collapse}
td,th{padding:9px 6px;border-bottom:1px solid var(--line);text-align:left}
tr:last-child td{border-bottom:0}
th{font-size:13px;color:var(--mute);font-weight:500}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{max-height:400px;overflow:auto}
.act{text-align:right;white-space:nowrap}
.who{display:flex;align-items:center;gap:10px}
.av{width:30px;height:30px;border-radius:50%;background:var(--soft);color:var(--acc);display:grid;place-items:center;font-weight:700;font-size:13px;flex:none}
.chip{display:inline-block;margin:0 6px 6px 0;padding:3px 12px;border-radius:999px;background:var(--soft);color:var(--acc);font-size:14px}
.trend{display:flex;gap:8px;align-items:flex-end}
.bar{flex:1;text-align:center;font-size:12px;color:var(--mute);font-variant-numeric:tabular-nums}
.fill{height:96px;display:flex;align-items:flex-end}
.fill i{display:block;width:100%;min-height:2px;background:var(--soft);border-radius:5px 5px 0 0}
.bar.sel .fill i{background:var(--acc)}
.bar.sel{color:var(--ink);font-weight:600}
.err{color:var(--bad)}.okmsg{color:var(--ok)}
.bar2{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:14px}

/* kamera */
.camgrid{display:grid;grid-template-columns:minmax(0,2fr) minmax(250px,1fr);gap:18px;align-items:start}
@media (max-width:900px){.camgrid{grid-template-columns:1fr}}
.stage{position:relative;background:#0f0e0c;border-radius:24px;overflow:hidden;aspect-ratio:16/9;box-shadow:var(--shadow)}
.stage video,.stage canvas{position:absolute;inset:0;width:100%;height:100%}
.stage video{object-fit:contain;transform:scaleX(-1)}
.stage:fullscreen{border-radius:0;aspect-ratio:auto}
.idle{position:absolute;inset:0;display:grid;place-items:center;text-align:center;color:#e9e4da;padding:20px}
.idle .logo{margin:0 auto 12px;width:52px;height:52px;border-radius:16px}
.pill{position:absolute;top:14px;left:14px;max-width:calc(100% - 28px);padding:6px 14px;border-radius:999px;background:rgba(20,18,15,.62);color:#fff;font-size:13px;font-weight:600;backdrop-filter:blur(8px)}
.pill.err{background:rgba(180,35,24,.85);color:#fff}
.toast{position:absolute;left:50%;bottom:18px;transform:translate(-50%,12px);opacity:0;transition:.25s;display:flex;gap:10px;align-items:center;padding:8px 18px 8px 8px;border-radius:999px;background:var(--panel);color:var(--ink);box-shadow:var(--shadow);font-weight:600;white-space:nowrap;max-width:94%}
.toast.show{opacity:1;transform:translate(-50%,0)}
.ctrl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:14px}
.ctrl select{max-width:240px}
.feed{display:flex;flex-direction:column;gap:12px}
.feed .row2{display:flex;justify-content:space-between;align-items:center;gap:10px}
.tips{margin:14px 0 0;padding:0;list-style:none;color:var(--mute);font-size:13px}
.tips li+li{margin-top:4px}

/* dialog */
dialog{border:1px solid var(--line);border-radius:22px;background:var(--panel);color:var(--ink);padding:22px;width:min(560px,94vw);box-shadow:var(--shadow)}
dialog::backdrop{background:rgba(0,0,0,.45)}
dialog h2{margin-top:0;font-family:Lora,Georgia,serif;font-weight:500;font-size:22px}
dialog label:not(.btn){display:block;margin:0 0 10px}
dialog input:not([type=file]){display:block;width:100%;margin-top:4px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
.row.end{justify-content:flex-end;margin-top:16px}
video.pv{width:100%;max-height:260px;background:#000;border-radius:14px}
.thumbs{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.th{position:relative}
.th img{width:64px;height:64px;object-fit:cover;border-radius:10px;display:block}
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
    <div class="brand"><span class="logo"><svg class="ic" viewBox="0 0 24 24"><circle cx="12" cy="9" r="4"/><path d="M4.5 20c1-4 4-6 7.5-6s6.500 2 7.500 6"/></svg></span>Absensi Wajah</div>
    <nav class="nav">
      <a href="#ringkasan" id="n-ringkasan"><svg class="ic" viewBox="0 0 24 24"><path d="M3 11l9-8 9 8M5 10v10h14V10"/></svg>Ringkasan</a>
      <a href="#kamera" id="n-kamera"><svg class="ic" viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.500"/></svg>Kamera absen</a>
      <a href="#kelola" id="n-kelola"><svg class="ic" viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.500"/><path d="M2.500 20c.8-3.600 3.400-5.500 6.500-5.500s5.700 1.900 6.500 5.500M16 5a3.500 3.500 0 010 7M18 14.800c1.900.7 3.200 2.500 3.500 5.200"/></svg>Kelola wajah</a>
      <a href="#rekap" id="n-rekap"><svg class="ic" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg>Rekap kehadiran</a>
    </nav>
    <div class="sfoot">
      <label id="pinbox" hidden>PIN admin <input type="password" id="pin" autocomplete="off"></label>
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
        <a class="sug" href="#kamera"><span class="ico"><svg class="ic" viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.500"/></svg></span><span>Buka kamera absen<small>Pindai wajah langsung di sini</small></span></a>
        <a class="sug" href="#kelola" id="q-add"><span class="ico"><svg class="ic" viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg></span><span>Tambah orang baru<small>Daftarkan wajah dengan foto atau kamera</small></span></a>
        <a class="sug" href="#" id="export"><span class="ico"><svg class="ic" viewBox="0 0 24 24"><path d="M12 4v11M7 11l5 5 5-5M5 20h14"/></svg></span><span>Unduh CSV<small>Absensi pada tanggal terpilih</small></span></a>
      </div>
      <div class="stats">
        <div class="stat"><b id="s-hadir">0</b><span>Sudah hadir</span></div>
        <div class="stat"><b id="s-belum">0</b><span>Belum hadir</span></div>
        <div class="stat"><b id="s-total">0</b><span>Wajah terdaftar</span></div>
        <div class="stat"><b id="s-pct">0%</b><span>Tingkat kehadiran</span><div class="meter" role="img" aria-label="Persentase kehadiran"><i id="meter"></i></div></div>
      </div>
      <div class="cols">
        <div class="card">
          <div class="head"><h2>Kedatangan terbaru</h2><input type="search" id="q" placeholder="Cari nama" aria-label="Cari nama"></div>
          <div class="scroll"><table><thead><tr><th>Nama</th><th>Jam</th></tr></thead><tbody id="present"></tbody></table></div>
        </div>
        <div>
          <div class="card"><h2>Belum hadir</h2><div id="absent"></div></div>
          <div class="card"><h2>7 hari terakhir</h2><div class="trend" id="trend"></div></div>
        </div>
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
            <div class="idle" id="c-idle"><div><span class="logo"><svg class="ic" viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.500"/></svg></span>Kamera belum aktif<br><span class="mute">Tekan Mulai kamera untuk memulai absen otomatis</span></div></div>
            <div class="pill" id="c-pill" hidden></div>
            <div class="toast" id="c-toast"></div>
          </div>
          <div class="ctrl">
            <button class="btn pri" id="c-toggle" type="button">Mulai kamera</button>
            <select id="c-dev" aria-label="Pilih kamera" hidden></select>
            <button class="btn" id="c-full" type="button"><svg class="ic" viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>Layar penuh</button>
          </div>
        </div>
        <div class="card">
          <h2>Baru saja hadir</h2>
          <p class="mute" id="c-count" style="margin:-6px 0 12px"></p>
          <div class="feed" id="c-feed"></div>
          <ul class="tips">
            <li>Jarak ideal 0,5 sampai 2 meter dari kamera.</li>
            <li>Pastikan cahaya cukup dan wajah menghadap lurus.</li>
            <li>Kotak ungu berarti sedang memverifikasi, hijau berarti dikenali.</li>
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
      <p id="k-msg" role="status"></p>
      <div class="card"><div class="scroll"><table>
        <thead><tr><th>Nama</th><th>ID</th><th>Sampel</th><th></th></tr></thead>
        <tbody id="k-rows"></tbody>
      </table></div></div>
    </section>

    <!-- REKAP -->
    <section id="v-rekap" hidden>
      <div class="hello">
        <div><h1>Rekap kehadiran</h1><p class="sub">Total hari hadir setiap orang berdasarkan absensi.csv.</p></div>
        <input type="search" id="r-q" placeholder="Cari nama" aria-label="Cari nama">
      </div>
      <div class="card"><div class="scroll" style="max-height:none"><table>
        <thead><tr><th>Nama</th><th>Total catatan</th><th>Terakhir hadir</th></tr></thead>
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
let view = 'ringkasan', data = null, info = null, pendingAdd = false;

const pinHeader = () => { const p = sessionStorage.getItem('pin'); return p ? {'X-Pin': p} : {}; };
$('pin').value = sessionStorage.getItem('pin') || '';
$('pin').addEventListener('input', e => sessionStorage.setItem('pin', e.target.value));

// ---------- navigasi ----------
function closeNav() { $('side').classList.remove('open'); $('scrim').classList.remove('open'); }
$('menu').onclick = () => { $('side').classList.add('open'); $('scrim').classList.add('open'); };
$('scrim').onclick = closeNav;

function route() {
  const h = location.hash.slice(1);
  const next = VIEWS.includes(h) ? h : 'ringkasan';
  if (view === 'kamera' && next !== 'kamera') camStop();
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
  $('export').href = '/export?date=' + dateEl.value;
  try {
    const r = await fetch('/api/data?date=' + date, {cache: 'no-store'});
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
  const label = new Date(dateEl.value + 'T00:00').toLocaleDateString('id-ID', {weekday: 'long', day: 'numeric', month: 'long', year: 'numeric'});
  $('greet').textContent = greeting() + ', Admin';
  $('s-sub').textContent = t ? `${label}. ${n} dari ${t} orang sudah hadir.` : `${label}. Belum ada wajah terdaftar, tambahkan lewat Kelola wajah.`;
  $('s-hadir').textContent = n;
  $('s-belum').textContent = t ? data.absent.length : '-';
  $('s-total').textContent = t;
  $('s-pct').textContent = data.pct + '%';
  $('meter').style.width = data.pct + '%';
  drawPresent();
  $('absent').innerHTML = !t ? '<p class="mute">Daftar belum tersedia.</p>'
    : data.absent.length ? data.absent.map(x => `<span class="chip">${esc(x)}</span>`).join('')
    : '<p class="mute">Semua orang terdaftar sudah hadir.</p>';
  const max = Math.max(1, ...data.trend.map(x => x.count));
  $('trend').innerHTML = data.trend.map((x, i) =>
    `<div class="bar${i === data.trend.length - 1 ? ' sel' : ''}"><div>${x.count}</div>` +
    `<div class="fill"><i style="height:${x.count / max * 100}%"></i></div><div>${x.date.slice(5)}</div></div>`).join('');
}

function drawPresent() {
  const q = $('q').value.trim().toLowerCase();
  const rows = data.present.filter(p => p.nama.toLowerCase().includes(q));
  $('present').innerHTML = rows.map(p =>
    `<tr><td><div class="who"><span class="av">${esc(ini(p.nama))}</span>${esc(p.nama)}</div></td><td class="num">${esc(p.jam)}</td></tr>`).join('')
    || `<tr><td colspan="2" class="mute">${q ? 'Nama tidak ditemukan.' : 'Belum ada yang hadir pada tanggal ini.'}</td></tr>`;
}

function drawRecap() {
  const q = $('r-q').value.trim().toLowerCase();
  const rows = data.recap.filter(r => r.nama.toLowerCase().includes(q));
  $('recap').innerHTML = rows.map(r =>
    `<tr><td><div class="who"><span class="av">${esc(ini(r.nama))}</span>${esc(r.nama)}</div></td><td class="num">${r.total}</td><td class="num">${esc(r.last)}</td></tr>`).join('')
    || '<tr><td colspan="3" class="mute">Belum ada data absensi.</td></tr>';
}

function drawFeed() {
  const n = data.present.length, t = data.registered;
  $('c-count').textContent = t ? `${n} dari ${t} orang hadir hari ini` : `${n} orang hadir hari ini`;
  $('c-feed').innerHTML = data.present.slice(0, 8).map(p =>
    `<div class="row2"><div class="who"><span class="av">${esc(ini(p.nama))}</span>${esc(p.nama)}</div><span class="mute num">${esc(p.jam)}</span></div>`).join('')
    || '<p class="mute">Belum ada yang hadir.</p>';
}

dateEl.addEventListener('change', load);
$('q').addEventListener('input', () => data && drawPresent());
$('r-q').addEventListener('input', () => data && drawRecap());
setInterval(() => { if (!document.hidden && view !== 'kelola') load(); }, 4000);

// ---------- kamera absen ----------
const cam = {stream: null, on: false, tok: 0};
const COL = {tercatat: '#16a34a', hadir: '#16a34a', cek: '#7c3aed', unknown: '#dc2626', far: '#78716c'};
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

async function camStart() {
  if (!info || !info.editable) { setPill('Kamera absen nonaktif: jalankan dashboard dengan --pin jika dibuka dari jaringan.', true); return; }
  if (!navigator.mediaDevices) { setPill('Kamera browser hanya tersedia di localhost atau HTTPS.', true); return; }
  camStop(true);
  try {
    const dev = $('c-dev').value;
    cam.stream = await navigator.mediaDevices.getUserMedia({video: dev && !$('c-dev').hidden ? {deviceId: {exact: dev}} : {width: {ideal: 1280}, height: {ideal: 720}}});
    const v = $('c-video'); v.srcObject = cam.stream; await v.play();
  } catch (e) { setPill('Kamera tidak bisa dibuka (izin ditolak atau sedang dipakai aplikasi lain).', true); return; }
  cam.on = true; const tok = ++cam.tok;
  $('c-idle').hidden = true; $('c-toggle').textContent = 'Hentikan kamera';
  setPill('Menunggu wajah...');
  await listDevices();
  scanLoop(tok);
}

function camStop(keepUi) {
  cam.on = false; cam.tok++;
  if (cam.stream) cam.stream.getTracks().forEach(t => t.stop());
  cam.stream = null; $('c-video').srcObject = null;
  const c = $('c-canvas'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
  if (!keepUi) { $('c-idle').hidden = false; setPill(''); $('c-toggle').textContent = 'Mulai kamera'; }
}

async function scanLoop(tok) {
  while (cam.on && cam.tok === tok) {
    const v = $('c-video');
    if (v.videoWidth) {
      const w = Math.min(800, v.videoWidth), h = Math.round(v.videoHeight * w / v.videoWidth);
      grab.width = w; grab.height = h;
      grab.getContext('2d').drawImage(v, 0, 0, w, h);
      const blob = await new Promise(r => grab.toBlob(r, 'image/jpeg', 0.75));
      try {
        const r = await fetch('/api/scan', {method: 'POST', headers: {'Content-Type': 'image/jpeg', ...pinHeader()}, body: blob});
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
    await sleep(120);
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
      : f.status === 'hadir' ? `${f.nama} · Sudah hadir`
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
  t.innerHTML = `<span class="av">${esc(ini(e.nama))}</span>Tercatat: ${esc(e.nama)} · ${esc(e.jam)}`;
  t.classList.add('show'); clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
}

$('c-toggle').onclick = () => cam.on ? camStop() : camStart();
$('c-dev').onchange = () => cam.on && camStart();
$('c-full').onclick = () => { const s = $('stage'); document.fullscreenElement ? document.exitFullscreen() : s.requestFullscreen && s.requestFullscreen(); };

// ---------- kelola wajah ----------
let mode = null, target = null, imgs = [], stream = null, busy = false;
const PROMPTS = ['Hadap lurus', 'Tengok sedikit ke kiri', 'Tengok sedikit ke kanan', 'Dongak sedikit',
                 'Tunduk sedikit', 'Hadap lurus, rileks', 'Senyum tipis', 'Hadap lurus'];

function say(t, err) { const m = $('k-msg'); m.textContent = t; m.className = err ? 'err' : 'okmsg'; }

async function api(method, path, body) {
  const r = await fetch(path, {method, headers: {'Content-Type': 'application/json', ...pinHeader()}, body: JSON.stringify(body)});
  const j = await r.json().catch(() => ({message: 'Respons server tidak valid'}));
  if (!r.ok) { const e = new Error(j.message || r.status); e.rejected = j.rejected || []; throw e; }
  return j;
}

async function loadPeople() {
  try {
    info = await (await fetch('/api/people', {cache: 'no-store'})).json();
  } catch (e) { say('Server tidak terjangkau', true); return; }
  $('pinbox').hidden = !info.pin_required;
  $('k-add').disabled = !info.editable;
  $('k-rows').innerHTML = info.people.map((p, i) =>
    `<tr><td><div class="who"><span class="av">${esc(ini(p.nama))}</span>${esc(p.nama)}</div></td><td>${esc(p.id) || '<span class="mute">-</span>'}</td><td class="num">${p.sampel}</td>` +
    `<td class="act"><button class="btn sm" data-a="samples" data-i="${i}">Tambah sampel</button> ` +
    `<button class="btn sm" data-a="edit" data-i="${i}">Ubah</button> ` +
    `<button class="btn sm bad" data-a="del" data-i="${i}">Hapus</button></td></tr>`).join('')
    || '<tr><td colspan="4" class="mute">Belum ada wajah terdaftar.</td></tr>';
  if (!info.editable) say('Pengelolaan nonaktif: jalankan dashboard dengan --pin untuk mengaktifkannya dari jaringan.', true);
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
fetch('/api/people', {cache: 'no-store'}).then(r => r.json()).then(j => { info = j; $('pinbox').hidden = !j.pin_required; }).catch(() => {});
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

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/people":
            return self.send(200, json.dumps(people_info()), JSON)
        date = parse_qs(u.query).get("date", [datetime.now().strftime("%Y-%m-%d")])[0]
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return self.send(400, "Tanggal tidak valid", "text/plain; charset=utf-8")

        if u.path == "/":
            self.send(200, PAGE, "text/html; charset=utf-8")
        elif u.path == "/api/data":
            self.send(200, json.dumps(build(date)), "application/json; charset=utf-8")
        elif u.path == "/export":
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["tanggal", "jam", "nama"])
            w.writerows(r for r in cached(LOG_FILE, read_rows) if r[0] == date)
            self.send(200, buf.getvalue(), "text/csv; charset=utf-8",
                      {"Content-Disposition": f'attachment; filename="absensi_{date}.csv"'})
        else:
            self.send(404, "Halaman tidak ditemukan", "text/plain; charset=utf-8")

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
        if not hmac.compare_digest(self.headers.get("X-Pin", "").encode(), PIN.encode()):
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
        try:
            out = api_scan(self.guard("image/jpeg", MAX_FRAME))
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
    ap.add_argument("--pin", default="", help="PIN admin untuk kamera absen dan pengelolaan wajah (wajib jika --host bukan localhost)")
    a = ap.parse_args()
    PIN = a.pin
    EDIT_ENABLED = a.host in ("127.0.0.1", "localhost", "::1") or bool(PIN)
    if not EDIT_ENABLED:
        print("PERINGATAN: dashboard terbuka di jaringan tanpa --pin, jadi kamera absen dan pengelolaan wajah dinonaktifkan (hanya baca).")

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
        server.server_close()


if __name__ == "__main__":
    main()