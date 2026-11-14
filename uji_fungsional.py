import os, sys, json, tempfile, shutil, subprocess, io
BASE = os.path.dirname(os.path.abspath(__file__))   # letakkan file ini di folder proyek (sejajar main.py)
sys.path.insert(0, BASE)
import numpy as np
import config, privacy
from privacy import PrivacyError

tmp = tempfile.mkdtemp()
config.KEY_PATH = os.path.join(tmp, "k", "facedb.key")
config.AUDIT_PATH = os.path.join(tmp, "audit.jsonl")
os.environ.pop("FACEDB_PASSPHRASE", None)
privacy._key_cache.clear()

R = []
def rec(kode, uji, harapan, ok, bukti=""):
    R.append((kode, uji, harapan, "LULUS" if ok else "GAGAL", bukti))

rng = np.random.default_rng(1)
E = rng.normal(size=(6,128)).astype(np.float32); E /= np.linalg.norm(E,axis=1,keepdims=True)
names=["A","A","B","B","C","C"]; ids=["1","1","2","2","3","3"]
db = os.path.join(tmp,"f.enc")

# E1 roundtrip
privacy.save_db(names,ids,E,db); n,i,e = privacy.load_db(db,migrate=False)
rec("E1","Simpan lalu muat database terenkripsi","Isi identik", n==names and i==ids and np.allclose(e,E), f"{len(n)} sampel")
# E2 plaintext tidak bocor
blob=open(db,"rb").read()
rec("E2","Embedding tidak muncul sebagai teks polos di file","Byte embedding tidak ditemukan", E[0].tobytes() not in blob and b"A" != blob[:1], f"header={blob[:8]}")
# E3 tamper
b=bytearray(blob); b[-20]^=1; open(db,"wb").write(bytes(b))
try: privacy.load_db(db,migrate=False); ok=False; m=""
except PrivacyError as ex: ok=True; m=str(ex)[:60]
rec("E3","Ubah 1 byte ciphertext","Ditolak (integritas GCM)", ok, m)
open(db,"wb").write(blob)
# E4 kunci salah
os.replace(config.KEY_PATH, config.KEY_PATH+".bak")
privacy._key_cache.clear()
privacy._keyfile_key(True); privacy._key_cache.clear()
try: privacy.load_db(db,migrate=False); ok=False; m=""
except PrivacyError as ex: ok=True; m=str(ex)[:60]
rec("E4","Dekripsi dengan file kunci berbeda","Ditolak (kunci tidak cocok)", ok, m)
os.replace(config.KEY_PATH+".bak", config.KEY_PATH); privacy._key_cache.clear()
# E5 passphrase
os.environ["FACEDB_PASSPHRASE"]="sandi-uji-123"; privacy._key_cache.clear()
db2=os.path.join(tmp,"p.enc"); privacy.save_db(names,ids,E,db2)
ok1 = privacy.load_db(db2,migrate=False)[0]==names
os.environ["FACEDB_PASSPHRASE"]="salah"; privacy._key_cache.clear()
try: privacy.load_db(db2,migrate=False); ok2=False
except PrivacyError: ok2=True
os.environ.pop("FACEDB_PASSPHRASE"); privacy._key_cache.clear()
rec("E5","Mode passphrase (scrypt): benar vs salah","Benar terbuka, salah ditolak", ok1 and ok2)
# E6 rotasi
config.DB_PATH=db; privacy.rotate_key(); 
ok = privacy.load_db(db,migrate=False)[0]==names
rec("E6","Rotasi kunci","Data tetap terbaca dengan kunci baru", ok)

# A1-A3 audit
for k in range(5): privacy.audit("uji",f"s{k}",{"k":k})
ok,nn,msg = privacy.verify_audit(config.AUDIT_PATH)
rec("A1","Rantai audit utuh","Terverifikasi", ok and nn>=5, msg[:50])
lines=open(config.AUDIT_PATH).read().splitlines()
p2=os.path.join(tmp,"a2.jsonl"); L=lines[:]; d=json.loads(L[2]); d["subjek"]="DIUBAH"; L[2]=json.dumps(d)
open(p2,"w").write("\n".join(L)+"\n"); ok,_,m=privacy.verify_audit(p2)
rec("A2","Ubah isi satu entri audit","Terdeteksi", not ok, m)
p3=os.path.join(tmp,"a3.jsonl"); open(p3,"w").write("\n".join(lines[:2]+lines[3:])+"\n"); ok,_,m=privacy.verify_audit(p3)
rec("A3","Hapus satu entri di tengah","Terdeteksi", not ok, m)

# C1-C3 persetujuan
reg = privacy.ConsentRegistry(os.path.join(tmp,"c.json"))
reg.give("Budi","9","tertulis","uji","F-001")
a=reg.is_active("Budi"); reg.withdraw("Budi"); b=reg.is_active("Budi")
rec("C1","Beri lalu tarik persetujuan","aktif -> tidak aktif", a and not b)
reg.give("Budi","9","tertulis","uji","F-002"); r=reg.get("Budi")
rec("C2","Beri persetujuan ulang setelah ditarik","Riwayat tersimpan", len(r.get("riwayat",[]))==1 and r["status"]=="aktif")
rec("C3","Catatan persetujuan menyimpan hash naskah","hash_naskah 64 hex", len(r["hash_naskah"])==64)

# L1-L4 liveness fungsi murni
import liveness as lv
lm=np.array([[30,40],[70,40],[50,60],[35,80],[65,80]],float)
uv0=lv.affine_coords(lm)
th=np.deg2rad(25); Rm=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
lm2=(lm@Rm.T)*1.7+np.array([13,-7]); lm2=lm2@np.array([[1,0.2],[0,1]]).T
uv1=lv.affine_coords(lm2)
rec("L1","Koordinat afin invarian terhadap rotasi+skala+geser+shear","Selisih (u,v) < 1e-6", max(abs(uv0[0]-uv1[0]),abs(uv0[1]-uv1[1]))<1e-6, f"selisih={max(abs(uv0[0]-uv1[0]),abs(uv0[1]-uv1[1])):.2e}")
flat=[0.0]*20
rec("L2","Tantangan: deret datar (foto datar)","Tidak lolos", not lv.challenge_passed(flat,0.28)[0])
osc=list(np.r_[np.zeros(3),np.linspace(0,.2,6),np.linspace(.2,-.2,8),np.linspace(-.2,0,5)])
rec("L3","Tantangan: gerak dua arah ±0.2","Lolos", lv.challenge_passed(osc,0.28)[0])
one=list(np.r_[np.zeros(3),np.linspace(0,.4,10)])
rec("L4","Tantangan: gerak satu arah saja","Tidak lolos (harus dua arah)", not lv.challenge_passed(one,0.28)[0])

# D1-D5 pencocokan (LOGIKA dengan embedding sintetis; BUKAN pengukuran akurasi)
import main as M
def unit(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
base={k:unit(rng.normal(size=128)) for k in "ABC"}
S={k:unit(base[k]+0.07*rng.normal(size=(3,128))) for k in "ABC"}      # sampel satu orang saling mirip (~0.6-0.7)
nms=[];idl=[];em=[]
for k,i in zip("ABC","123"):
    for v in S[k][:2]: nms.append(k);idl.append(i);em.append(v)
d=M.Database.from_arrays(nms,idl,np.array(em))
nm,pid,sc=d.match(S["A"][2])
rec("D1","Sampel baru milik A (tidak ikut terdaftar)","Dikenali sebagai A", nm=="A" and sc>=M.MATCH_THRESH, f"skor={sc:.2f}")
q2=unit(rng.normal(size=128)); nm,pid,sc=d.match(q2)
rec("D2","Vektor acak (bukan siapa pun)","Skor di bawah ambang", sc<M.MATCH_THRESH, f"skor={sc:.2f}")
mix=unit(base["A"]+base["B"]); nm,_,sc=d.match(mix); _,_,sb,mg=d.best(mix)
rec("D3","Wajah ambigu A/B (campuran)","Ditolak aturan margin", nm is None, f"skor={sb:.2f}, margin={mg:.3f}, batas margin={M.MATCH_MARGIN}")
d1=M.Database.from_arrays(["X"],["7"],S["A"][:1]); nm,_,sc=d1.match(S["A"][1])
rec("D4","Orang dengan 1 sampel","Tetap dikenali", nm=="X", f"skor={sc:.2f}")
d0=M.Database.from_arrays([],[],np.zeros((0,128),np.float32)); nm,_,sc=d0.match(S["A"][0])
rec("D5","Database kosong","Tidak error, hasil None", nm is None)

# K1 config fail-fast
for var,val,kode in [("MATCH_THRESH","abc","K1"),("LIVENESS_MODE","xyz","K2"),("JAM_PULANG","25:99","K3")]:
    env={**os.environ, var:val}
    r=subprocess.run([sys.executable,"-c","import config"],cwd=BASE,env=env,capture_output=True,text=True)
    rec(kode,f"Config {var}={val}","Berhenti dengan pesan jelas", r.returncode!=0 and "tidak valid" in r.stderr, r.stderr.strip().splitlines()[-1][:70] if r.stderr else "")

# retensi
import attendance
from datetime import date
p=os.path.join(tmp,"abs.csv")
open(p,"w").write("tanggal,jam,nama,id,skor,tipe,status\n2025-01-01,08:00:00,A,1,0.9,masuk,hadir\n2026-11-10,08:00:00,B,2,0.9,masuk,hadir\n")
rem,left=attendance.purge_before(date(2026,1,1),p)
rec("R1","Retensi: hapus catatan sebelum cutoff","1 dihapus, 1 tersisa", (rem,left)==(1,1), f"{rem}/{left}")

print("| Kode | Pengujian | Hasil diharapkan | Status | Bukti |\n|---|---|---|---|---|")
for r in R: print("| "+" | ".join(str(x).replace("|","/") for x in r)+" |")
print(f"\nTotal {len(R)}: lulus {sum(r[3]=='LULUS' for r in R)}, gagal {sum(r[3]=='GAGAL' for r in R)}")
shutil.rmtree(tmp)