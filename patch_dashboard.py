"""
Patch produksi untuk dashboard.py (UI/UX mobile + kesiapan deploy).

Pakai : letakkan file ini di folder yang sama dengan dashboard.py, lalu jalankan
        python patch_dashboard.py
Aman  : setiap anchor harus cocok TEPAT satu kali. Jika ada yang tidak cocok, tidak ada yang diubah.
        Cadangan asli disimpan sebagai dashboard.py.bak
"""
import py_compile
import shutil
import sys

TARGET = "dashboard.py"
P = []   # daftar (anchor_lama, pengganti)


def sub(old, new):
    P.append((old, new))


# ============================ 1. HEAD: viewport, theme-color, favicon ============================
sub(r'''<meta name="viewport" content="width=device-width, initial-scale=1">''',
    r'''<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#faf9f7" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#1b1916" media="(prefers-color-scheme: dark)">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Absensi">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='9' fill='%237c3aed'/%3E%3Ccircle cx='16' cy='12' r='5' fill='none' stroke='white' stroke-width='2'/%3E%3Cpath d='M7 27c1.5-5 5-7 9-7s7.5 2 9 7' fill='none' stroke='white' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E">''')

# ============================ 2. CSS tambahan ============================
sub(r'''.gtoast.bad{background:var(--bad);color:#fff}''',
    r'''.gtoast.bad{background:var(--bad);color:#fff}

/* ===== produksi: dasar ===== */
html{-webkit-text-size-adjust:100%;scroll-padding-top:72px}
a,button,summary,select,input{touch-action:manipulation;-webkit-tap-highlight-color:transparent}
h1:focus{outline:none}
dialog{overscroll-behavior:contain;max-height:92vh;max-height:92dvh;overflow:auto}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}

/* skeleton saat data pertama dimuat */
@keyframes sh{to{background-position:-200% 0}}
.is-loading .stat b,.is-loading #s-pct,.is-loading #h-cap{color:transparent!important;border-radius:8px;min-width:48px;background:linear-gradient(90deg,var(--line) 25%,var(--soft) 50%,var(--line) 75%);background-size:200% 100%;animation:sh 1.2s linear infinite}

/* banner koneksi terputus */
.netbar{position:sticky;top:0;z-index:12;margin:0 0 var(--gap);padding:var(--s2) var(--s4);border-radius:var(--r-ctl);background:var(--bad-soft);color:var(--bad);font-weight:600;font-size:13.5px}

/* navigasi bawah (hanya mobile) */
.tabbar{display:none}

/* target sentuh nyaman di layar sentuh */
@media (pointer:coarse){
  :root{--ctl:44px}
  .btn.sm{height:36px;padding:0 var(--s4)}
  .seg button{height:36px}
}

@media (max-width:860px){
  input,select,textarea{font-size:16px}                 /* cegah iOS zoom otomatis saat fokus */
  .nav a[href^="#"]{display:none}                       /* menu utama pindah ke tab bawah; drawer: tema, keluar, privasi */
  .side{padding-bottom:calc(var(--s5) + env(safe-area-inset-bottom))}
  .top{padding-top:env(safe-area-inset-top);height:calc(56px + env(safe-area-inset-top))}
  .netbar{top:calc(56px + env(safe-area-inset-top))}
  html{scroll-padding-top:calc(72px + env(safe-area-inset-top))}
  .main{padding-bottom:calc(92px + env(safe-area-inset-bottom))}
  .gtoast{bottom:calc(84px + env(safe-area-inset-bottom))}
  .tabbar{display:grid;grid-template-columns:repeat(4,1fr);gap:var(--s1);position:fixed;left:0;right:0;bottom:0;z-index:15;
    padding:6px 8px calc(6px + env(safe-area-inset-bottom));background:color-mix(in srgb,var(--panel) 92%,transparent);
    -webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);border-top:1px solid var(--line)}
  .tabbar a{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;min-height:48px;border-radius:12px;
    color:var(--mute);text-decoration:none;font-size:11.5px;font-weight:600}
  .tabbar a.on{color:var(--acc);background:var(--soft)}
}

@media (max-width:600px){
  .hello>label{display:flex;align-items:center;gap:var(--s2);width:100%}
  .hello>label input{flex:1;min-width:0}
  .ringbig{width:min(180px,100%)}
  .ctrl{display:grid;grid-template-columns:1fr 1fr}
  .ctrl select{max-width:none;grid-column:1/-1}
  .ctrl #c-toggle{grid-column:1/-1;height:48px}
  .card{padding:var(--s4)}

  /* tabel Kelola wajah dan Rekap menjadi kartu per orang */
  #v-kelola table,#v-rekap table{display:block}
  #v-kelola thead,#v-rekap thead{display:none}
  #v-kelola tbody,#v-rekap tbody{display:grid;gap:var(--s3)}
  #k-rows tr,#recap tr{display:grid;grid-template-columns:1fr 1fr;gap:var(--s2) var(--s3);padding:var(--s4);
    border:1px solid var(--line);border-radius:var(--r-inner);background:var(--bg)}
  #k-rows td,#recap td{display:flex;flex-direction:column;align-items:flex-start;gap:2px;padding:0!important;border:0;font-size:14px;min-width:0}
  #k-rows td:first-child,#recap td:first-child{grid-column:1/-1}
  #k-rows td:last-child{grid-column:1/-1;flex-direction:row;flex-wrap:wrap;justify-content:flex-start;gap:var(--s2);
    padding-top:var(--s3)!important;border-top:1px solid var(--line)}
  #k-rows td::before,#recap td::before{font-size:12px;font-weight:500;color:var(--mute)}
  #k-rows td:nth-child(2)::before{content:"ID"}
  #k-rows td:nth-child(3)::before{content:"Sampel"}
  #k-rows td:nth-child(4)::before{content:"Persetujuan"}
  #recap td:nth-child(2)::before{content:"Kehadiran"}
  #recap td:nth-child(3)::before{content:"Terlambat"}
  #recap td:nth-child(4)::before{content:"Total catatan"}
  #recap td:nth-child(5)::before{content:"Terakhir hadir"}

  /* dialog menjadi bottom sheet */
  dialog{margin:auto 0 0;width:100%;max-width:100%;border-radius:var(--r-card) var(--r-card) 0 0;
    padding:var(--s5) var(--s4) calc(var(--s4) + env(safe-area-inset-bottom))}
  #login,#cf{margin:auto;width:min(400px,92vw);border-radius:var(--r-card);padding:var(--s6) var(--s5)}
  dialog .row.end .btn{flex:1}
}
@media (max-width:480px){
  #present .av{display:none}
  #present th,#present td{padding:var(--s3) var(--s1)}
  .wd .mr{--s:34px!important}                            /* 7 cincin muat di layar 360px */
}''')

# ============================ 3. HTML: tab bawah, banner koneksi, skeleton ============================
sub(r'''<div class="scrim" id="scrim"></div>''',
    r'''<div class="scrim" id="scrim"></div>
<nav class="tabbar" aria-label="Navigasi utama">
  <a href="#ringkasan" data-v="ringkasan"><svg class="ic" viewBox="0 0 24 24"><path d="M3 11l9-8 9 8M5 10v10h14V10"/></svg>Ringkasan</a>
  <a href="#kamera" data-v="kamera"><svg class="ic" viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v12H4z"/><circle cx="12" cy="13" r="3.5"/></svg>Kamera</a>
  <a href="#kelola" data-v="kelola"><svg class="ic" viewBox="0 0 24 24"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.8-3.6 3.4-5.5 6.5-5.5s5.7 1.9 6.5 5.5M16 5a3.5 3.5 0 010 7M18 14.8c1.9.7 3.2 2.5 3.5 5.2"/></svg>Wajah</a>
  <a href="#rekap" data-v="rekap"><svg class="ic" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg>Rekap</a>
</nav>''')

sub(r'''<main class="main">''',
    r'''<main class="main">
    <div class="netbar" id="net" role="status" hidden>Koneksi ke server terputus. Mencoba lagi otomatis...</div>''')

sub(r'''<div class="bento">''', r'''<div class="bento is-loading" id="bento">''')

# ============================ 4. JS: drawer, tab aktif, fokus, status koneksi ============================
sub(r'''function closeNav() { $('side').classList.remove('open'); $('scrim').classList.remove('open'); }''',
    r'''function setNav(o) { $('side').classList.toggle('open', o); $('scrim').classList.toggle('open', o); $('menu').setAttribute('aria-expanded', String(o)); }
function closeNav() { setNav(false); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeNav(); });''')

sub(r'''$('menu').onclick = () => { $('side').classList.add('open'); $('scrim').classList.add('open'); };''',
    r'''$('menu').onclick = () => setNav(true);''')

sub(r'''VIEWS.forEach(v => { $('v-' + v).hidden = v !== view; $('n-' + v).classList.toggle('on', v === view); });''',
    r'''VIEWS.forEach(v => { $('v-' + v).hidden = v !== view; $('n-' + v).classList.toggle('on', v === view); });
  document.querySelectorAll('.tabbar a').forEach(a => { const on = a.dataset.v === view; a.classList.toggle('on', on); on ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current'); });
  const h1 = $('v-' + view).querySelector('h1'); if (h1) { h1.tabIndex = -1; h1.focus({preventScroll: true}); }
  window.scrollTo(0, 0);''')

sub(r'''$('dot').className = 'dot on';''', r'''$('dot').className = 'dot on'; $('net').hidden = true;''')
sub(r'''$('status').textContent = 'Server tidak terjangkau';''', r'''$('status').textContent = 'Server tidak terjangkau'; $('net').hidden = false;''')
sub(r'''drawSummary(); drawRecap(); drawFeed();''', r'''$('bento').classList.remove('is-loading'); drawSummary(); drawRecap(); drawFeed();''')

# ============================ 5. JS: kamera di HP ============================
sub(r'''if (!info.rtsp) return;''',
    r'''if (!info.rtsp && !['localhost', '127.0.0.1', '[::1]', '::1'].includes(location.hostname)) {   // di-hosting di cloud: tidak ada kamera server
    const o = $('c-src').querySelector('option[value="server"]'); if (o) o.remove();
    $('c-src').value = 'browser'; $('c-tip-srv').hidden = true; syncSrcUi();
  }
  if (!info.rtsp) return;''')

sub(r'''const grab = document.createElement('canvas');''',
    r'''const grab = document.createElement('canvas');
let wl = null;   // layar HP/tablet tidak mati selama kamera absen aktif
async function keepAwake(on) {
  try {
    if (on && 'wakeLock' in navigator) { if (!wl) wl = await navigator.wakeLock.request('screen'); }
    else if (wl) { await wl.release(); wl = null; }
  } catch (e) { wl = null; }
}
document.addEventListener('visibilitychange', () => { if (!document.hidden && cam.on) { wl = null; keepAwake(true); } });''')

sub(r'''cam.on = true; const tok = ++cam.tok;''', r'''cam.on = true; const tok = ++cam.tok; keepAwake(true);''')
sub(r'''cam.on = false; cam.tok++;''', r'''cam.on = false; cam.tok++; keepAwake(false);''')
sub(r'''grab.toBlob(r, 'image/jpeg', 0.95)''', r'''grab.toBlob(r, 'image/jpeg', 0.85)''')   # hemat data seluler

sub(r'''$('c-full').onclick = () => { const s = $('stage'); document.fullscreenElement ? document.exitFullscreen() : s.requestFullscreen && s.requestFullscreen(); };''',
    r'''$('c-full').onclick = () => { const s = $('stage'); document.fullscreenElement ? document.exitFullscreen() : s.requestFullscreen && s.requestFullscreen(); };
if (!$('stage').requestFullscreen) $('c-full').hidden = true;   // iPhone Safari tidak mendukung fullscreen elemen''')

# ============================ 6. Server: deploy di belakang proxy (Railway dkk.) ============================
sub(r'''def pin_blocked(ip):''',
    r'''TRUST_PROXY = os.environ.get("TRUST_PROXY") == "1"   # set 1 bila di belakang reverse proxy (Railway, Nginx, Cloudflare)


def client_ip(handler):
    """IP klien asli. Di belakang proxy, client_address selalu IP proxy, jadi satu penyerang akan mengunci SEMUA admin."""
    if TRUST_PROXY:
        xff = handler.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[-1].strip()    # entri paling kanan = yang dicatat proxy tepercaya (tidak bisa dipalsukan klien)
    return handler.client_address[0]


def pin_blocked(ip):''')
sub(r'''ip = handler.client_address[0]''', r'''ip = client_ip(handler)''')
sub(r'''wait = pin_blocked(self.client_address[0])''', r'''wait = pin_blocked(client_ip(self))''')
sub(r'''data["_ip"] = self.client_address[0]''', r'''data["_ip"] = client_ip(self)''')
sub(r'''pelaku=f"dashboard({self.client_address[0]})")''', r'''pelaku=f"dashboard({client_ip(self)})")''')

sub(r'''"Referrer-Policy": "no-referrer"}''',
    r'''"Referrer-Policy": "no-referrer",
                    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()"}''')

sub(r'''return self.send(200, PAGE, "text/html; charset=utf-8")''',
    r'''return self.send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/healthz":       # untuk health check platform hosting (tanpa data sensitif)
            return self.send(200, "ok", "text/plain; charset=utf-8")''')


def main():
    try:
        src = open(TARGET, encoding="utf-8").read()
    except OSError as e:
        sys.exit(f"Tidak bisa membuka {TARGET}: {e}")
    out, bad = src, []
    for i, (old, new) in enumerate(P, 1):
        n = out.count(old)
        if n != 1:
            bad.append((i, n, old[:70].replace("\n", " ")))
        else:
            out = out.replace(old, new)
    if bad:
        print("DIBATALKAN, tidak ada yang diubah. Anchor tidak cocok tepat 1x (dashboard.py mungkin sudah berubah/di-patch):")
        for i, n, o in bad:
            print(f"  #{i}: ditemukan {n}x -> {o}")
        sys.exit(1)
    shutil.copyfile(TARGET, TARGET + ".bak")
    open(TARGET, "w", encoding="utf-8").write(out)
    try:
        py_compile.compile(TARGET, doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copyfile(TARGET + ".bak", TARGET)
        sys.exit(f"Syntax error setelah patch, dikembalikan ke asli: {e}")
    print(f"OK: {len(P)} perubahan diterapkan. Cadangan: {TARGET}.bak")


if __name__ == "__main__":
    main()