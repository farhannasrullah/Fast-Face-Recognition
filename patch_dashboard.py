"""
Patch UI/UX dashboard.py: login PIN lewat popup + perbaikan kemudahan pakai admin.

Pemakaian (di folder yang sama dengan dashboard.py):
    python patch_dashboard.py
Skrip membuat cadangan dashboard.py.bak, lalu menerapkan perubahan.
Jika ada bagian yang tidak cocok (file sudah berubah), skrip berhenti tanpa mengubah apa pun.
"""
import shutil
import sys

FILE = "dashboard.py"
src = open(FILE, encoding="utf-8").read()
EDITS = []


def edit(old, new):
    EDITS.append((old, new))


# ---------------------------------------------------------------- SERVER: endpoint cek PIN
edit(r'''        if u.path not in ("/api/people", "/api/data", "/export",
                          "/api/server-cam/frame", "/api/server-cam/status"):''',
     r'''        if u.path not in ("/api/people", "/api/data", "/export", "/api/auth",
                          "/api/server-cam/frame", "/api/server-cam/status"):''')

edit(r'''        if u.path == "/api/server-cam/frame":
            if not EDIT_ENABLED:''',
     r'''        if u.path == "/api/auth":      # lolos pin_denied = PIN benar (dipakai popup login)
            return self.send(200, json.dumps({"ok": True}), JSON)
        if u.path == "/api/server-cam/frame":
            if not EDIT_ENABLED:''')

# ---------------------------------------------------------------- CSS
edit(r'''</style>
</head>''', r'''/* login popup, konfirmasi, dan toast */
.app.locked{filter:blur(10px);pointer-events:none;user-select:none}
#login{width:min(400px,92vw);text-align:center}
#login::backdrop{background:rgba(20,18,15,.6);backdrop-filter:blur(6px)}
#login .logo{margin:0 auto var(--s3);width:52px;height:52px;border-radius:16px}
#login h2{margin-bottom:var(--s1)}
#login label{text-align:left;margin-top:var(--s4)}
#login label.chk{font-weight:500;margin-top:var(--s3)}
#login input:not([type=checkbox]){height:44px;font-size:16px}
#login .err{min-height:20px;margin-top:var(--s2);font-size:13.5px;text-align:left}
#login .btn.pri{width:100%;height:44px;margin-top:var(--s2)}
#cf{width:min(420px,92vw)}
.btn.pri.danger{background:var(--bad);border-color:var(--bad);color:#fff}
.sfoot #logout{width:100%;margin-bottom:var(--s3)}
.gtoast{position:fixed;left:50%;bottom:var(--s6);transform:translate(-50%,16px);opacity:0;pointer-events:none;transition:.25s;z-index:100;max-width:min(560px,92vw);padding:var(--s3) var(--s5);border-radius:14px;background:var(--ink);color:var(--bg);font-weight:600;box-shadow:var(--shadow)}
.gtoast.show{opacity:1;transform:translate(-50%,0)}
.gtoast.bad{background:var(--bad);color:#fff}
</style>
</head>''')

# ---------------------------------------------------------------- HTML: sidebar (PIN diganti tombol Keluar)
edit(r'''      <label id="pinbox" hidden>PIN admin <input type="password" id="pin" autocomplete="off"></label>''',
     r'''      <button class="btn sm" id="logout" type="button" hidden><svg class="ic" viewBox="0 0 24 24"><path d="M9 4H5v16h4M16 8l4 4-4 4M20 12H9"/></svg>Keluar</button>''')

# ---------------------------------------------------------------- HTML: dialog login + konfirmasi + toast
edit(r'''<dialog id="dlg" aria-labelledby="dlg-title">''', r'''<dialog id="login" aria-labelledby="lg-title">
  <span class="logo"><svg class="ic" viewBox="0 0 24 24"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 018 0v3"/></svg></span>
  <h2 id="lg-title">Masuk Admin</h2>
  <p class="mute">Masukkan PIN admin untuk membuka dashboard absensi.</p>
  <label>PIN admin <input type="password" id="lg-pin" autocomplete="current-password" autofocus></label>
  <label class="chk"><input type="checkbox" id="lg-show"> Tampilkan PIN</label>
  <p class="err" id="lg-err" role="alert"></p>
  <button class="btn pri" id="lg-go" type="button">Masuk</button>
</dialog>

<dialog id="cf" aria-labelledby="cf-t">
  <h2 id="cf-t"></h2>
  <p class="mute" id="cf-m"></p>
  <div class="row end">
    <button class="btn" id="cf-no" type="button">Batal</button>
    <button class="btn pri danger" id="cf-yes" type="button">Ya</button>
  </div>
</dialog>
<div class="gtoast" id="gt" role="status" aria-live="polite"></div>

<dialog id="dlg" aria-labelledby="dlg-title">''')

# ---------------------------------------------------------------- JS: logika login / toast / konfirmasi
edit(r'''const needPin = () => { $('pinbox').hidden = false; };
let pinTimer = null;
$('pin').value = sessionStorage.getItem('pin') || '';
$('pin').addEventListener('input', e => {
  sessionStorage.setItem('pin', e.target.value);
  clearTimeout(pinTimer);
  pinTimer = setTimeout(() => { load(); if (view === 'kelola') loadPeople(); }, 400);
});''', r'''
// ---------- notifikasi & konfirmasi ----------
let gtT = null;
function toast(msg, err) {
  const t = $('gt'); t.textContent = msg; t.className = 'gtoast show' + (err ? ' bad' : '');
  clearTimeout(gtT); gtT = setTimeout(() => t.classList.remove('show'), err ? 6000 : 4500);
}
function ask(title, msg, yes) {
  return new Promise(res => {
    const d = $('cf');
    $('cf-t').textContent = title; $('cf-m').textContent = msg; $('cf-yes').textContent = yes || 'Ya';
    const end = v => { d.close(); res(v); };
    $('cf-yes').onclick = () => end(true); $('cf-no').onclick = () => end(false); d.oncancel = () => res(false);
    d.showModal(); $('cf-no').focus();
  });
}

// ---------- login PIN lewat popup ----------
const appEl = document.querySelector('.app');
function syncAuthUi() { $('logout').hidden = !(info.pin_required && sessionStorage.getItem('pin')); }
function needPin(msg) {
  info.pin_required = true;
  const d = $('login');
  appEl.classList.add('locked'); $('logout').hidden = true;
  if (!d.open) { $('lg-pin').value = ''; $('lg-err').textContent = msg || ''; d.showModal(); }
  else if (msg) $('lg-err').textContent = msg;
  $('lg-pin').focus();
}
async function doLogin() {
  const pin = $('lg-pin').value;
  if (!pin) { $('lg-err').textContent = 'Masukkan PIN terlebih dahulu.'; $('lg-pin').focus(); return; }
  $('lg-go').disabled = true; $('lg-go').textContent = 'Memeriksa...';
  try {
    const r = await fetch('/api/auth', {cache: 'no-store', headers: {'X-Pin': pin}});
    if (r.ok) {
      sessionStorage.setItem('pin', pin);
      $('login').close(); appEl.classList.remove('locked'); syncAuthUi();
      toast('Berhasil masuk. Selamat datang, Admin.');
      load(); if (view === 'kelola' || view === 'kamera') loadPeople(); if (view === 'kamera') autoWatch();
    } else {
      const j = await r.json().catch(() => ({}));
      $('lg-err').textContent = j.message || 'PIN salah.';
      $('lg-pin').select();
    }
  } catch (e) { $('lg-err').textContent = 'Server tidak terjangkau.'; }
  $('lg-go').disabled = false; $('lg-go').textContent = 'Masuk';
}
$('lg-go').onclick = doLogin;
$('lg-pin').addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
$('lg-show').onchange = e => { $('lg-pin').type = e.target.checked ? 'text' : 'password'; };
$('login').addEventListener('cancel', e => e.preventDefault());   // popup login tidak bisa ditutup dengan Esc
$('logout').onclick = async () => {
  camStop(); await srvStop();
  sessionStorage.removeItem('pin'); data = null; needPin();
};''')

# ---------------------------------------------------------------- JS: 401/429 membuka popup
edit(r'''    if (r.status === 401) { needPin(); $('status').textContent = 'Masukkan PIN admin'; $('dot').className = 'dot'; return; }
    if (r.status === 429) { needPin(); $('status').textContent = (await r.json()).message; $('dot').className = 'dot'; return; }''',
     r'''    if (r.status === 401 || r.status === 429) {
      const m = (await r.json().catch(() => ({}))).message || '';
      sessionStorage.removeItem('pin');
      needPin(m === 'PIN diperlukan.' ? '' : m);
      $('status').textContent = 'Perlu masuk'; $('dot').className = 'dot'; return;
    }''')

edit(r'''setInterval(() => { if (!document.hidden && view !== 'kelola') load(); }, 4000);''',
     r'''setInterval(() => { if (!document.hidden && view !== 'kelola' && !$('login').open) load(); }, 4000);''')

edit(r'''    if (r.status === 401) { needPin(); throw new Error('Masukkan PIN admin di sidebar.'); }''',
     r'''    if (r.status === 401) { needPin(); throw new Error('Masuk terlebih dahulu.'); }''')

edit(r'''    if (r.status === 401) { needPin(); say('Masukkan PIN admin di sidebar.', true); return; }
    info = {...info, ...(await r.json())};''',
     r'''    if (r.status === 401 || r.status === 429) {
      const m = (await r.json().catch(() => ({}))).message || '';
      sessionStorage.removeItem('pin'); needPin(m === 'PIN diperlukan.' ? '' : m); return;
    }
    info = {...info, ...(await r.json())};''')

edit(r'''  $('pinbox').hidden = !info.pin_required;
  $('k-add').disabled = !info.editable;''', r'''  syncAuthUi();
  $('k-add').disabled = !info.editable;''')

edit(r'''  info = {...info, ...j}; $('pinbox').hidden = !j.pin_required;
  applyRtspUi(); if (view === 'kamera') autoWatch();''',
     r'''  info = {...info, ...j}; syncAuthUi();
  if (j.pin_required && !sessionStorage.getItem('pin')) needPin();
  applyRtspUi(); if (view === 'kamera') autoWatch();''')

# sesi berakhir saat aksi (PIN berubah / diblokir): kembali ke popup login
edit(r'''  if (!r.ok) { const e = new Error(j.message || r.status); e.rejected = j.rejected || []; throw e; }''',
     r'''  if (!r.ok) {
    if ((r.status === 403 || r.status === 429) && /PIN/.test(j.message || '')) { sessionStorage.removeItem('pin'); needPin('Sesi berakhir. Masuk kembali.'); }
    const e = new Error(j.message || r.status); e.rejected = j.rejected || []; throw e;
  }''')

# ---------------------------------------------------------------- JS: umpan balik aksi lewat toast
edit(r'''    say(j.message + (j.warning ? ' ' + j.warning : '') + (j.rejected && j.rejected.length ? ' Ditolak: ' + j.rejected.join('; ') + '.' : ''));
    loadPeople();''', r'''    toast(j.message + (j.warning ? ' ' + j.warning : '') + (j.rejected && j.rejected.length ? ' Ditolak: ' + j.rejected.join('; ') + '.' : ''),
          !!(j.warning || (j.rejected && j.rejected.length)));
    loadPeople();''')

edit(r'''  if (!confirm(`Hapus ${p.nama} beserta ${p.sampel} sampel wajahnya? Data biometrik dimusnahkan dan persetujuan ditandai dihapus. Riwayat absensi tidak ikut terhapus.`)) return;
  try { say((await api('DELETE', '/api/people', {nama: p.nama})).message); } catch (e) { say(e.message, true); }''',
     r'''  if (!await ask(`Hapus ${p.nama}?`, `${p.sampel} sampel wajah akan dimusnahkan permanen dan persetujuan ditandai dihapus. Riwayat absensi tidak ikut terhapus.`, 'Ya, hapus')) return;
  try { toast((await api('DELETE', '/api/people', {nama: p.nama})).message); } catch (e) { toast(e.message, true); }''')

# ---------------------------------------------------------------- JS: kemudahan form (fokus + Enter)
edit(r'''  $('dlg').showModal();
}''', r'''  $('dlg').showModal();
  if (m === 'add' || m === 'edit') $('d-nama').focus();
}''')

edit(r'''$('dlg').addEventListener('close', stopCam);''', r'''$('dlg').addEventListener('close', stopCam);
$('dlg').addEventListener('keydown', e => {      // Enter di kolom teks = Simpan
  if (e.key === 'Enter' && e.target.tagName === 'INPUT' && !['checkbox', 'file'].includes(e.target.type)) { e.preventDefault(); save(); }
});''')

# ---------------------------------------------------------------- terapkan
for i, (old, _) in enumerate(EDITS, 1):
    n = src.count(old)
    if n != 1:
        sys.exit(f"GAGAL: bagian #{i} ditemukan {n}x (harus 1x). Tidak ada yang diubah.\n{old[:90]!r}")

shutil.copyfile(FILE, FILE + ".bak")
for old, new in EDITS:
    src = src.replace(old, new)
open(FILE, "w", encoding="utf-8").write(src)
print(f"Selesai: {len(EDITS)} perubahan diterapkan. Cadangan: {FILE}.bak")