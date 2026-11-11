"""
Dashboard Absensi Wajah - ringan. Tab 'Kelola wajah': tambah, ubah, dan hapus wajah (CRUD).
Membaca absensi.csv dan faces_db.npz milik main.py (letakkan di folder yang sama).

Jalankan : python dashboard.py
Buka     : http://127.0.0.1:8000  (otomatis terbuka di browser)
Opsi     : --port 8000
           --host 0.0.0.0   (agar bisa dibuka dari HP/PC lain di jaringan yang sama)
           --no-browser     (jangan buka browser otomatis)
           --pin 1234       (wajib jika dibuka dari jaringan agar wajah bisa dikelola; dikirim lewat HTTP biasa)
Boleh berjalan bersamaan dengan mode run: perubahan wajah dimuat ulang otomatis oleh mode run.
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


# ----------------------------- Kelola wajah (CRUD) -----------------------------
JSON = "application/json; charset=utf-8"
MIN_SAMPLES = 3            # minimal foto valid untuk orang baru
MAX_IMAGES = 12            # maksimal foto per permintaan
MAX_BODY = 40 * 1024 * 1024
PIN = ""                   # diisi dari --pin
EDIT_ENABLED = True        # False jika dashboard terbuka di jaringan tanpa --pin
EDIT_LOCK = threading.Lock()
_core = {}


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


ROUTES = {("POST", "/api/people"): api_create, ("POST", "/api/people/samples"): api_add_samples,
          ("PUT", "/api/people"): api_update, ("DELETE", "/api/people"): api_delete}


PAGE = r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dashboard Absensi Wajah</title>
<style>
:root{color-scheme:light dark;--bg:#f3f4f1;--panel:#fbfbf9;--ink:#1c2024;--mute:#646b73;--line:#dcdfd8;--ok:#0b6e8a;--ok-soft:#cfe6ed;--late:#9a4d00;--late-soft:#f6e3c8}
@media (prefers-color-scheme:dark){:root{--bg:#15181b;--panel:#1b1f23;--ink:#e6e9ec;--mute:#98a0a8;--line:#2c3237;--ok:#5cc3de;--ok-soft:#1d3a44;--late:#f0b765;--late-soft:#3b2d17}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",system-ui,-apple-system,Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
header{border-bottom:1px solid var(--line);background:var(--panel)}
header .wrap{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding-top:14px;padding-bottom:14px}
.brand{font-weight:650;font-size:16px}
.ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
input,.btn{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:5px 10px;text-decoration:none}
:focus-visible{outline:2px solid var(--ok);outline-offset:2px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--mute);margin-right:6px}
.dot.on{background:var(--ok)}
.live{color:var(--mute);font-size:13px}
.sum{padding:28px 0 20px}
.sum h1{margin:0;font-size:30px;line-height:1.2;font-weight:600;font-variant-numeric:tabular-nums}
.sum p{margin:4px 0 14px;color:var(--mute)}
.meter{height:6px;border-radius:3px;background:var(--line);overflow:hidden}
.meter i{display:block;height:100%;width:0;background:var(--ok);transition:width .4s}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:32px;padding-bottom:28px}
@media (max-width:760px){.grid{grid-template-columns:1fr;gap:24px}}
section{min-width:0}
h2{margin:0 0 10px;font-size:16px;font-weight:650}
.head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px}
.head h2{margin:0}
table{width:100%;border-collapse:collapse}
td,th{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left}
th{font-size:13px;color:var(--mute);font-weight:500}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{max-height:380px;overflow:auto}
.chip{display:inline-block;margin:0 6px 6px 0;padding:2px 10px;border-radius:999px;background:var(--late-soft);color:var(--late);font-size:14px}
.mute{color:var(--mute)}
.trend{display:flex;gap:8px;align-items:flex-end;margin-top:6px}
.bar{flex:1;text-align:center;font-size:12px;color:var(--mute);font-variant-numeric:tabular-nums}
.fill{height:96px;display:flex;align-items:flex-end}
.fill i{display:block;width:100%;min-height:2px;background:var(--ok-soft);border-radius:3px 3px 0 0}
.bar.sel .fill i{background:var(--ok)}
.bar.sel{color:var(--ink);font-weight:600}
.side>section+section{margin-top:28px}
footer{border-top:1px solid var(--line);padding:16px 0 40px}

:root{--bad:#b42318;--on-ok:#fff}
@media (prefers-color-scheme:dark){:root{--bad:#f97066;--on-ok:#0b1d24}}
[hidden]{display:none!important}
.tabs{display:flex;gap:4px}
.tabs a{padding:5px 12px;border-radius:6px;color:var(--mute);text-decoration:none}
.tabs a.on{background:var(--bg);color:var(--ink);font-weight:600}
button.btn,label.btn{cursor:pointer}
button.btn:disabled{opacity:.5;cursor:not-allowed}
.btn.pri{background:var(--ok);border-color:var(--ok);color:var(--on-ok)}
.btn.sm{padding:2px 8px;font-size:13px}
.btn.bad{color:var(--bad);border-color:var(--bad)}
.bar2{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px}
.act{text-align:right;white-space:nowrap}
.err{color:var(--bad)}.okmsg{color:var(--ok)}
dialog{border:1px solid var(--line);border-radius:10px;background:var(--panel);color:var(--ink);padding:20px;width:min(560px,94vw)}
dialog::backdrop{background:rgba(0,0,0,.45)}
dialog h2{margin-top:0}
dialog label:not(.btn){display:block;margin:0 0 10px}
dialog input:not([type=file]){display:block;width:100%;margin-top:4px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
.row.end{justify-content:flex-end;margin-top:14px}
video{width:100%;max-height:260px;background:#000;border-radius:6px}
.thumbs{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
.th{position:relative}
.th img{width:64px;height:64px;object-fit:cover;border-radius:6px;display:block}
.th button{position:absolute;top:-6px;right:-6px;width:20px;height:20px;padding:0;border-radius:50%;border:0;background:var(--bad);color:#fff;cursor:pointer}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="brand">Absensi Wajah</div>
  <nav class="tabs"><a href="#absensi" id="t-abs" class="on">Absensi</a><a href="#kelola" id="t-kelola">Kelola wajah</a></nav>
  <div class="ctl" id="ctl">
    <span class="live"><span class="dot" id="dot"></span><span id="status">Menghubungkan...</span></span>
    <label>Tanggal <input type="date" id="date"></label>
    <a class="btn" id="export" href="#">Unduh CSV</a>
  </div>
</div></header>

<div class="wrap" id="v-abs">
  <div class="sum">
    <h1 id="sum">Memuat data...</h1>
    <p id="sub">&nbsp;</p>
    <div class="meter" role="img" aria-label="Persentase kehadiran"><i id="meter"></i></div>
  </div>

  <div class="grid">
    <section>
      <div class="head"><h2>Kedatangan terbaru</h2><input type="search" id="q" placeholder="Cari nama" aria-label="Cari nama"></div>
      <div class="scroll"><table>
        <thead><tr><th>Jam</th><th>Nama</th></tr></thead>
        <tbody id="present"></tbody>
      </table></div>
    </section>
    <div class="side">
      <section><h2>Belum hadir</h2><div id="absent"></div></section>
      <section><h2>7 hari terakhir</h2><div class="trend" id="trend"></div></section>
    </div>
  </div>

  <section>
    <h2>Rekap kehadiran</h2>
    <div class="scroll"><table>
      <thead><tr><th>Nama</th><th>Total hari hadir</th><th>Terakhir hadir</th></tr></thead>
      <tbody id="recap"></tbody>
    </table></div>
  </section>
</div>
<div class="wrap" id="v-kelola" hidden>
  <div class="sum">
    <h1>Kelola wajah</h1>
    <p>Tambah, ubah, atau hapus wajah terdaftar. Foto tidak disimpan, hanya data pengenal wajah (embedding). Mode absen memuat perubahan otomatis dalam beberapa detik.</p>
  </div>
  <div class="bar2">
    <button class="btn pri" id="k-add" type="button">Tambah orang</button>
    <label id="k-pinbox" hidden>PIN admin <input type="password" id="k-pin" autocomplete="off"></label>
    <span id="k-msg" role="status"></span>
  </div>
  <div class="scroll"><table>
    <thead><tr><th>Nama</th><th>ID</th><th>Sampel</th><th></th></tr></thead>
    <tbody id="k-rows"></tbody>
  </table></div>
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
      <video id="d-video" playsinline muted></video>
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
<footer><div class="wrap mute">Data dibaca langsung dari absensi.csv dan diperbarui otomatis tiap 4 detik.</div></footer>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const today = () => new Date().toLocaleDateString('sv-SE');
const dateEl = $('date');
dateEl.value = today();
let data = null;

async function load() {
  if (!dateEl.value) dateEl.value = today();
  $('export').href = '/export?date=' + dateEl.value;
  try {
    const r = await fetch('/api/data?date=' + dateEl.value, {cache: 'no-store'});
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

function draw() {
  const n = data.present.length, t = data.registered;
  $('sum').textContent = t ? `${n} dari ${t} orang sudah hadir` : `${n} orang sudah hadir`;
  $('sub').textContent = t
    ? `${data.pct}% hadir, ${data.absent.length} belum hadir`
    : 'Belum ada wajah terdaftar. Jalankan: python absen_wajah.py enroll "Nama"';
  $('meter').style.width = data.pct + '%';
  drawPresent();
  $('absent').innerHTML = !t ? '<p class="mute">Daftar belum tersedia.</p>'
    : data.absent.length ? data.absent.map(x => `<span class="chip">${esc(x)}</span>`).join('')
    : '<p class="mute">Semua orang terdaftar sudah hadir.</p>';
  const max = Math.max(1, ...data.trend.map(x => x.count));
  $('trend').innerHTML = data.trend.map((x, i) =>
    `<div class="bar${i === data.trend.length - 1 ? ' sel' : ''}"><div>${x.count}</div>` +
    `<div class="fill"><i style="height:${x.count / max * 100}%"></i></div><div>${x.date.slice(5)}</div></div>`).join('');
  $('recap').innerHTML = data.recap.map(r =>
    `<tr><td>${esc(r.nama)}</td><td class="num">${r.total}</td><td class="num">${esc(r.last)}</td></tr>`).join('')
    || '<tr><td colspan="3" class="mute">Belum ada data absensi.</td></tr>';
}

function drawPresent() {
  const q = $('q').value.trim().toLowerCase();
  const rows = data.present.filter(p => p.nama.toLowerCase().includes(q));
  $('present').innerHTML = rows.map(p =>
    `<tr><td class="num">${esc(p.jam)}</td><td>${esc(p.nama)}</td></tr>`).join('')
    || `<tr><td colspan="2" class="mute">${q ? 'Nama tidak ditemukan.' : 'Belum ada yang hadir pada tanggal ini.'}</td></tr>`;
}

dateEl.addEventListener('change', load);
$('q').addEventListener('input', () => data && drawPresent());
setInterval(() => { if (!document.hidden && view === 'abs') load(); }, 4000);
load();

// ---------- Kelola wajah ----------
let view = 'abs', info = null, mode = null, target = null, imgs = [], stream = null, busy = false;
const PROMPTS = ['Hadap lurus', 'Tengok sedikit ke kiri', 'Tengok sedikit ke kanan', 'Dongak sedikit',
                 'Tunduk sedikit', 'Hadap lurus, rileks', 'Senyum tipis', 'Hadap lurus'];

function route() {
  view = location.hash === '#kelola' ? 'kelola' : 'abs';
  $('v-abs').hidden = view !== 'abs';
  $('v-kelola').hidden = view !== 'kelola';
  $('ctl').hidden = view !== 'abs';
  $('t-abs').classList.toggle('on', view === 'abs');
  $('t-kelola').classList.toggle('on', view === 'kelola');
  if (view === 'kelola') loadPeople(); else load();
}

function say(t, err) { const m = $('k-msg'); m.textContent = t; m.className = err ? 'err' : 'okmsg'; }

async function api(method, path, body) {
  const h = {'Content-Type': 'application/json'};
  if ($('k-pin').value) h['X-Pin'] = $('k-pin').value;
  const r = await fetch(path, {method, headers: h, body: JSON.stringify(body)});
  const j = await r.json().catch(() => ({message: 'Respons server tidak valid'}));
  if (!r.ok) { const e = new Error(j.message || r.status); e.rejected = j.rejected || []; throw e; }
  return j;
}

async function loadPeople() {
  try {
    info = await (await fetch('/api/people', {cache: 'no-store'})).json();
  } catch (e) { say('Server tidak terjangkau', true); return; }
  $('k-pinbox').hidden = !info.pin_required;
  $('k-add').disabled = !info.editable;
  $('k-rows').innerHTML = info.people.map((p, i) =>
    `<tr><td>${esc(p.nama)}</td><td>${esc(p.id) || '<span class="mute">-</span>'}</td><td class="num">${p.sampel}</td>` +
    `<td class="act"><button class="btn sm" data-a="samples" data-i="${i}">Tambah sampel</button> ` +
    `<button class="btn sm" data-a="edit" data-i="${i}">Ubah</button> ` +
    `<button class="btn sm bad" data-a="del" data-i="${i}">Hapus</button></td></tr>`).join('')
    || '<tr><td colspan="4" class="mute">Belum ada wajah terdaftar.</td></tr>';
  if (!info.editable) say('Pengelolaan nonaktif: jalankan dashboard dengan --pin untuk mengaktifkannya dari jaringan.', true);
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
  } catch (e) { $('d-err').textContent = 'Kamera tidak bisa dibuka (izin ditolak atau sedang dipakai mode absen). Gunakan unggah foto.'; }
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
    await new Promise(r => setTimeout(r, 900)); snap();
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

    def route(self, method):
        """Aksi CRUD: cek keamanan -> baca JSON -> jalankan di bawah kunci -> kirim hasil."""
        action = ROUTES.get((method, urlparse(self.path).path))
        if action is None:
            return self.send(404, "Halaman tidak ditemukan", "text/plain; charset=utf-8")
        try:
            host, origin = self.headers.get("Host", ""), self.headers.get("Origin")
            if not EDIT_ENABLED:
                raise ApiError(403, "Pengelolaan wajah nonaktif: jalankan dashboard dengan --pin jika dibuka dari jaringan.")
            if origin and urlparse(origin).netloc != host:
                raise ApiError(403, "Permintaan lintas situs ditolak.")
            if not PIN and host.rsplit(":", 1)[0].strip("[]") not in ("127.0.0.1", "localhost", "::1"):
                raise ApiError(403, "Host tidak diizinkan.")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ApiError(415, "Content-Type harus application/json.")
            if not hmac.compare_digest(self.headers.get("X-Pin", "").encode(), PIN.encode()):
                raise ApiError(403, "PIN salah atau belum diisi.")
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_BODY:
                raise ApiError(413, "Ukuran data tidak valid atau terlalu besar.")
            try:
                data = json.loads(self.rfile.read(n))
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

    def do_POST(self):
        self.route("POST")

    def do_PUT(self):
        self.route("PUT")

    def do_DELETE(self):
        self.route("DELETE")


def main():
    global PIN, EDIT_ENABLED
    ap = argparse.ArgumentParser(description="Dashboard Absensi Wajah")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--pin", default="", help="PIN admin untuk mengelola wajah (wajib jika --host bukan localhost)")
    a = ap.parse_args()
    PIN = a.pin
    EDIT_ENABLED = a.host in ("127.0.0.1", "localhost", "::1") or bool(PIN)
    if not EDIT_ENABLED:
        print("PERINGATAN: dashboard terbuka di jaringan tanpa --pin, jadi pengelolaan wajah dinonaktifkan (hanya baca).")

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
