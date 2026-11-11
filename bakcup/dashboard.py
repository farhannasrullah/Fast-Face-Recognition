"""
Dashboard Absensi Wajah - sederhana, ringan, tanpa library tambahan.
Membaca absensi.csv dan faces_db.npz yang dibuat oleh absen_wajah.py.

Jalankan : python dashboard.py
Buka     : http://127.0.0.1:8000  (otomatis terbuka di browser)
Opsi     : --port 8000
           --host 0.0.0.0   (agar bisa dibuka dari HP/PC lain di jaringan yang sama)
           --no-browser     (jangan buka browser otomatis)
Jalankan di folder yang sama dengan absen_wajah.py, boleh bersamaan dengan mode run.
"""
import argparse
import csv
import io
import json
import os
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

LOG_FILE = "absensi.csv"
DB_FILE = "faces_db.npz"
_cache = {}


def cached(path, loader):
    """Baca ulang file hanya jika isinya berubah (hemat CPU dan RAM)."""
    try:
        st = os.stat(path)
    except OSError:
        return loader(None)
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if not hit or hit[0] != key:
        hit = (key, loader(path))
        _cache[path] = hit
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
        return sorted({str(n) for n in np.load(path, allow_pickle=False)["names"]})
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
</style>
</head>
<body>
<header><div class="wrap">
  <div class="brand">Absensi Wajah</div>
  <div class="ctl">
    <span class="live"><span class="dot" id="dot"></span><span id="status">Menghubungkan...</span></span>
    <label>Tanggal <input type="date" id="date"></label>
    <a class="btn" id="export" href="#">Unduh CSV</a>
  </div>
</div></header>

<div class="wrap">
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
setInterval(() => { if (!document.hidden) load(); }, 4000);
load();
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


def main():
    ap = argparse.ArgumentParser(description="Dashboard Absensi Wajah")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

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
