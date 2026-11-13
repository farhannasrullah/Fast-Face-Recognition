<div align="center">

# Absensi Wajah

**Sistem absensi pengenalan wajah berbasis CPU dengan anti-spoofing (liveness), evaluasi kuantitatif,
dan pelindungan data biometrik sesuai UU No. 27 Tahun 2022 (PDP).**

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8)
![CPU only](https://img.shields.io/badge/GPU-tidak%20perlu-success)
![Liveness](https://img.shields.io/badge/anti--spoofing-MiniFASNet%20%2B%20tantangan-orange)
![UU PDP](https://img.shields.io/badge/privasi-UU%2027%2F2022-7c3aed)

</div>

---

## Daftar isi

1. [Fitur](#1-fitur)
2. [Arsitektur](#2-arsitektur)
3. [Pipeline pengenalan](#3-pipeline-pengenalan)
4. [Instalasi](#4-instalasi)
5. [Mulai cepat](#5-mulai-cepat)
6. [Konfigurasi](#6-konfigurasi)
7. [Penggunaan CLI (`main.py`)](#7-penggunaan-cli-mainpy)
8. [Dashboard web (`dashboard.py`)](#8-dashboard-web-dashboardpy)
9. [Liveness / anti-spoofing](#9-liveness--anti-spoofing)
10. [Privasi & UU PDP](#10-privasi--uu-pdp)
11. [Evaluasi kuantitatif (`evaluate.py`)](#11-evaluasi-kuantitatif-evaluatepy)
12. [Sinkronisasi cloud (opsional)](#12-sinkronisasi-cloud-opsional)
13. [Struktur proyek & file data](#13-struktur-proyek--file-data)
14. [Pengujian](#14-pengujian)
15. [Pemecahan masalah](#15-pemecahan-masalah)
16. [Batasan](#16-batasan)
17. [Atribusi & lisensi model](#17-atribusi--lisensi-model)

---

## 1. Fitur

| Area | Fitur |
|---|---|
| **Pengenalan** | Deteksi YuNet + embedding SFace 128-d. Pencocokan per orang memakai rata-rata 2 sampel terbaik, ambang, dan margin; harus cocok 2× berturut-turut. Pelacak optical flow sehingga deteksi cukup tiap 3 frame |
| **Absensi** | Masuk (hadir/terlambat) dan pulang (setelah jam pulang + durasi minimum), reset harian otomatis, aman dijalankan beberapa proses sekaligus (kunci antar-proses) |
| **Sumber kamera** | Webcam lokal, CCTV RTSP (selalu aktif, sambung ulang otomatis), kamera browser (HP/PC lain) |
| **Anti-spoofing** | Pasif: MiniFASNetV2 + V1SE. Aktif: tantangan acak tengok kiri-kanan / dongak-tunduk berbasis koordinat afin. Percobaan palsu dicatat |
| **Privasi** | Database wajah AES-256-GCM, persetujuan eksplisit + naskah (Pasal 21), rekam jejak berantai hash, hak akses/tarik/hapus, retensi, rotasi kunci |
| **Dashboard** | Ringkasan, kamera absen, kelola wajah (CRUD + persetujuan), rekap. PIN + anti brute-force, tema terang/gelap |
| **Evaluasi** | FAR/FRR/EER/ROC, identifikasi open-set (Rank-1, TPIR, FPIR, F1), APCER/BPCER/ACER, latensi & FPS, uji lapangan |
| **Cloud** | Firestore + Google Sheets dengan antrean offline dan catch-up otomatis. **Data biometrik tidak pernah dikirim** |

---

## 2. Arsitektur

```mermaid
flowchart LR
    classDef src fill:#efe8fd,stroke:#7c3aed,color:#1f1d1a
    classDef app fill:#ffffff,stroke:#5c574f,color:#1f1d1a
    classDef core fill:#fdecc8,stroke:#b45309,color:#1f1d1a
    classDef store fill:#dcf3e3,stroke:#15803d,color:#1f1d1a
    classDef secret fill:#fde4e1,stroke:#b42318,color:#1f1d1a
    classDef cloud fill:#e0f2fe,stroke:#0369a1,color:#1f1d1a

    subgraph SUMBER["Sumber kamera"]
        CAM["Webcam lokal"]:::src
        RTSP["CCTV RTSP<br/>(selalu aktif)"]:::src
        BRW["Kamera browser<br/>(HP / PC lain)"]:::src
    end

    subgraph APLIKASI["Aplikasi"]
        RUN["main.py run<br/>CLI / headless"]:::app
        DASH["dashboard.py<br/>admin web + PIN"]:::app
        EVAL["evaluate.py<br/>evaluasi kuantitatif"]:::app
    end

    subgraph INTI["Inti pemrosesan (main.py)"]
        PF["process_faces()<br/>satu pipeline bersama"]:::core
        FE["FaceEngine<br/>YuNet + SFace"]:::core
        LV["LivenessEngine<br/>MiniFASNet + tantangan"]:::core
        DB["Database<br/>filter persetujuan"]:::core
        AL["AttendanceLogger<br/>kunci antar-proses"]:::core
    end

    subgraph DATA["Penyimpanan lokal"]
        ENC[("faces_db.enc<br/>AES-256-GCM")]:::secret
        KEY[("kunci/facedb.key<br/>atau passphrase")]:::secret
        CON[("persetujuan.json")]:::store
        AUD[("audit_log.jsonl<br/>berantai hash")]:::store
        CSV[("absensi.csv")]:::store
        SPF[("spoof_log.csv")]:::store
    end

    subgraph CLOUD["Cloud (opsional)"]
        CS["cloud_sync.py<br/>antrean + catch-up"]:::cloud
        FS["Firestore"]:::cloud
        GS["Google Sheets"]:::cloud
    end

    CAM --> RUN
    RTSP --> RUN
    RTSP --> DASH
    CAM --> DASH
    BRW --> DASH
    RUN --> PF
    DASH --> PF
    EVAL -.-> FE
    EVAL -.-> LV
    PF --> FE
    PF --> LV
    PF --> DB
    PF --> AL
    DB <--> ENC
    KEY -. dekripsi .-> ENC
    DB -. hanya persetujuan aktif .-> CON
    AL --> CSV
    LV -. percobaan palsu .-> SPF
    PF -. aksi data .-> AUD
    AL --> CS
    CS --> FS
    CS --> GS
```

<sub>Gambar untuk dokumen: [PNG](docs/img/01_arsitektur.png) · [SVG](docs/img/01_arsitektur.svg)</sub>

Ketiga sumber kamera memakai **satu pipeline yang sama** (`main.process_faces`), sehingga aturan
pengenalan, liveness, dan pencatatan selalu identik di CLI maupun dashboard.

| Modul | Tanggung jawab |
|---|---|
| `main.py` | CLI, `FaceEngine` (YuNet + SFace), `Database` (terenkripsi, filter persetujuan), `Tracker`, `process_faces()`, perintah `privasi` |
| `liveness.py` | `PassiveLiveness` (MiniFASNet), `LandmarkRefiner`, `LivenessState`, `LivenessEngine` |
| `privacy.py` | Enkripsi, kunci, migrasi, naskah, `ConsentRegistry`, rekam jejak |
| `attendance.py` | Aturan masuk/pulang, `parse_row` (pembaca CSV tunggal), retensi |
| `dashboard.py` | Server web admin |
| `evaluate.py` | Evaluasi kuantitatif |
| `cloud_sync.py` | Firestore + Google Sheets |
| `config.py` | Satu sumber konfigurasi dan lokasi file |
| `ipclock.py` | Kunci file antar-proses (Windows/Linux/macOS) |

---

## 3. Pipeline pengenalan

```mermaid
flowchart TD
    classDef step fill:#ffffff,stroke:#5c574f,color:#1f1d1a
    classDef dec fill:#efe8fd,stroke:#7c3aed,color:#1f1d1a
    classDef live fill:#fdecc8,stroke:#b45309,color:#1f1d1a
    classDef ok fill:#dcf3e3,stroke:#15803d,color:#1f1d1a
    classDef bad fill:#fde4e1,stroke:#b42318,color:#1f1d1a

    F["Frame kamera<br/>(thread terpisah, hanya frame terbaru)"]:::step --> R["Resize ke lebar deteksi<br/>+ grayscale"]:::step
    R --> N{"frame ke-N?<br/>(DETECT_EVERY = 3)"}:::dec
    N -- tidak --> OF["Optical flow LK<br/>geser kotak & landmark"]:::step --> F
    N -- ya --> D["Deteksi YuNet<br/>kotak + 5 landmark"]:::step
    D --> SZ{"lebar wajah >= MIN_FACE_PX?"}:::dec
    SZ -- tidak --> FAR["Label 'terlalu jauh'"]:::step
    SZ -- ya --> T["Tracker: cocokkan IoU<br/>track lama / track baru"]:::step
    T --> LQ{"Liveness lolos?"}:::dec
    LQ -- belum --> LP["Pasif (mode pasif/ganda): crop 2,7x & 4,0x<br/>MiniFASNetV2 + V1SE<br/>rata-rata jendela frame"]:::live
    LP --> LA["Aktif (mode aktif/ganda):<br/>landmark resolusi penuh<br/>koordinat afin hidung (u, v)"]:::live
    LA --> SP{"rata-rata < ambang<br/>setelah 8 frame?"}:::dec
    SP -- ya --> SPOOF["DITOLAK: wajah palsu<br/>spoof_log.csv + audit_log"]:::bad
    SP -- tidak --> E
    LQ -- sudah --> E{"waktu verifikasi?<br/>(0,4 dtk / 60 dtk)"}:::dec
    E -- belum --> F
    E -- ya --> EMB["alignCrop resolusi penuh<br/>embedding SFace 128-d"]:::step
    EMB --> M["Cocokkan ke database<br/>(hanya persetujuan aktif)<br/>skor = rata-rata 2 sampel terbaik"]:::step
    M --> TH{"skor >= match_thresh<br/>dan margin >= match_margin?"}:::dec
    TH -- tidak --> U["Unknown"]:::bad
    TH -- ya --> C{"cocok 2x berturut-turut?"}:::dec
    C -- belum --> F
    C -- ya --> G{"liveness lolos?"}:::dec
    G -- belum --> W["Kotak kuning + instruksi<br/>(cek lagi 0,4 dtk)"]:::live
    G -- ya --> DEC{"Aturan absensi"}:::dec
    DEC -- "pertama hari ini" --> MS["MASUK: hadir / terlambat"]:::ok
    DEC -- "sudah masuk, >= jam pulang,<br/>>= durasi minimum" --> PL["PULANG"]:::ok
    DEC -- "lainnya" --> NO["Tidak dicatat ulang"]:::step
    MS --> WR["Tulis absensi.csv<br/>(kunci antar-proses)"]:::ok
    PL --> WR
    WR --> Q["Antrean cloud_sync<br/>(non-blocking)"]:::step
```

<sub>Gambar untuk dokumen: [PNG](docs/img/02_pipeline.png) · [SVG](docs/img/02_pipeline.svg)</sub>

Alur pencatatan absen:

```mermaid
sequenceDiagram
    autonumber
    participant K as Kamera (thread)
    participant P as process_faces()
    participant LV as LivenessEngine
    participant DB as Database (RAM)
    participant AL as AttendanceLogger
    participant CSV as absensi.csv
    participant CS as cloud_sync

    K->>P: frame terbaru
    P->>P: YuNet + Tracker (IoU / optical flow)
    P->>LV: observe(track, frame)
    LV-->>P: skor pasif / progres tantangan
    P->>DB: match(embedding SFace)
    DB-->>P: nama, skor, margin
    Note over P: dikenali 2x berturut-turut DAN liveness lolos
    P->>AL: try_log(nama, waktu)
    AL->>AL: kunci antar-proses (absensi.csv.lock)
    AL->>CSV: sync baris baru dari proses lain
    AL->>AL: decide: masuk / pulang / tidak
    AL->>CSV: tulis baris
    AL-)CS: enqueue (non-blocking)
    CS-)CS: kirim Firestore / Sheets, coba ulang bila offline
```

<sub>Gambar untuk dokumen: [PNG](docs/img/08_alur_absen.png) · [SVG](docs/img/08_alur_absen.svg)</sub>

---

## 4. Instalasi

**Kebutuhan:** Python 3.9+, webcam atau CCTV RTSP. Tidak perlu GPU.

```bash
git clone https://github.com/<akun>/<repo>.git
cd <repo>
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py download-models          # YuNet + SFace (~39 MB); model liveness sudah ada di models/
```

| Paket | Wajib? | Untuk |
|---|---|---|
| `opencv-python`, `numpy` | ya | deteksi, pengenalan, liveness |
| `cryptography` | ya | enkripsi database wajah |
| `matplotlib` | opsional | grafik evaluasi |
| `openpyxl` | opsional | ekspor `.xlsx` |
| `psutil` | opsional | tampilan RAM |
| `firebase-admin`, `gspread` | opsional | sinkron cloud |

> Bila unduhan model gagal (jaringan kampus/proxy), unduh manual dari
> [opencv_zoo](https://github.com/opencv/opencv_zoo) dan simpan ke folder `models/`.

---

## 5. Mulai cepat

```bash
# 1) atur jam kerja & identitas pengendali data
cp config.contoh.json config.json        # Windows: copy config.contoh.json config.json

# 2) daftarkan wajah (naskah persetujuan tampil, subjek mengetik SETUJU)
python main.py register --nama "Budi Santoso" --id 2023001 --petugas "Admin"

# 3) mulai absen
python main.py run                       # jendela kamera; Q / ESC untuk keluar

# 4) atau lewat dashboard
python dashboard.py                      # buka http://127.0.0.1:8000
```

> **Kunci enkripsi.** Saat pertama menyimpan wajah, sistem membuat `kunci/facedb.key`.
> **Cadangkan file ini di tempat terpisah.** Tanpa kunci, database wajah tidak bisa dibuka.

> **Pengguna versi lama.** `faces_db.npz` otomatis dienkripsi menjadi `faces_db.enc`. Orang yang sudah
> terdaftar sebelumnya belum punya persetujuan, sehingga **tidak dikenali** sampai persetujuannya
> dicatat (lihat [§10](#10-privasi--uu-pdp)).

---

## 6. Konfigurasi

Urutan prioritas: **environment variable (huruf besar) > `config.json` > bawaan**. Nilai yang salah
format menghentikan program dengan pesan yang jelas.

```json
{
  "jam_terlambat": "08:00",
  "jam_pulang": "16:00",
  "min_durasi_menit": 240,
  "match_thresh": 0.40,
  "liveness_mode": "ganda",
  "pengendali": "Universitas Contoh - Fakultas Teknik",
  "kontak_pdp": "pdp@contoh.ac.id"
}
```

| Kunci | Bawaan | Keterangan |
|---|---|---|
| `jam_terlambat` | `09:00:00` | Masuk setelah jam ini = terlambat |
| `jam_pulang` | `11:00:00` | Scan setelah jam ini (oleh yang sudah masuk) = pulang |
| `min_durasi_menit` | `120` | Jarak minimum masuk → pulang |
| `match_thresh` | `0.40` | Ambang skor kosinus; **kalibrasi** dengan `evaluate.py pengenalan` |
| `match_margin` | `0.05` | Selisih minimum skor orang terbaik vs kedua (0 = nonaktif) |
| `liveness_mode` | `pasif` | `off` / `pasif` / `aktif` / `ganda` |
| `liveness_thresh` | `0.80` | Ambang probabilitas "asli" model pasif |
| `liveness_min_frames` / `liveness_window` | `3` / `8` | Frame minimum untuk lolos / jendela keputusan spoof |
| `tantangan_ambang` | `0.28` | Rentang koordinat afin minimum untuk tantangan aktif |
| `tantangan_timeout` | `8` | Detik untuk menyelesaikan tantangan |
| `tantangan_min_px` | `100` | Lebar wajah minimum untuk tantangan aktif |
| `wajib_persetujuan` | `true` | Wajah tanpa persetujuan aktif tidak dikenali |
| `retensi_hari` | `365` | Masa simpan catatan absensi |
| `pengendali`, `kontak_pdp` | — | Ditampilkan di naskah persetujuan |

Environment variable lain: `RTSP_URL`, `FACEDB_PASSPHRASE` (kunci berbasis kata sandi), dan
`FACEDB_KEY_FILE` (lokasi file kunci).

---

## 7. Penggunaan CLI (`main.py`)

| Perintah | Opsi penting |
|---|---|
| `register --nama "X"` | `--id`, `--jumlah 5-10`, `--kamera 0`, `--jarak dekat\|sedang\|jauh`, `--petugas`, `--bukti-persetujuan "F-001"`, `--lewati-liveness` |
| `run` | `--kamera 0`, `--rtsp URL`, `--jarak`, `--headless` |
| `export` | `--dari YYYY-MM-DD`, `--sampai YYYY-MM-DD`, `--format csv\|xlsx` |
| `list` | Daftar orang + status persetujuan |
| `delete --nama "X"` | Hapus wajah dan tandai persetujuan |
| `download-models` | Unduh YuNet + SFace |
| `privasi <aksi>` | Lihat [§10](#10-privasi--uu-pdp) |

**Profil jarak (`--jarak`):**

| Profil | Kamera | Jarak ± | Catatan |
|---|---|---|---|
| `dekat` | 640×480 | 1 m | Paling ringan |
| `sedang` (bawaan) | 1280×720 | 3 m | Seimbang |
| `jauh` | 1280×720 | 4 m | Untuk CCTV; butuh CPU lebih kuat |

**Aturan absensi**

- Scan pertama hari itu sampai `jam_terlambat` → **MASUK, hadir**.
- Scan pertama setelah `jam_terlambat` → **MASUK, terlambat**.
- Scan setelah `jam_pulang` oleh orang yang sudah masuk, dan sudah lewat `min_durasi_menit` →
  **PULANG** (sekali per hari).
- Status direset otomatis setiap hari tanpa perlu restart.

**Contoh CCTV:**

```bash
python main.py run --rtsp "rtsp://user:pass@192.168.1.64:554/Streaming/Channels/102" --jarak jauh --headless
```

Password di URL disamarkan saat dicetak ke log.

---

## 8. Dashboard web (`dashboard.py`)

```bash
python dashboard.py                                   # lokal, http://127.0.0.1:8000
python dashboard.py --host 0.0.0.0 --pin 482915       # bisa dibuka dari HP/PC lain di jaringan
python dashboard.py --rtsp "rtsp://..." --jarak jauh  # CCTV selalu aktif di server
```

| Halaman | Isi |
|---|---|
| Ringkasan | Persentase hadir, terlambat, belum hadir, tren 7 hari, unduh CSV |
| Kamera absen | Kamera server / CCTV / kamera browser; kotak hijau = dikenali, kuning = memeriksa (+ instruksi liveness), merah = tidak dikenal / palsu |
| Kelola wajah | Tambah (wajib centang persetujuan + nama petugas), tambah sampel, ubah, hapus, catat persetujuan |
| Rekap | Persentase kehadiran dan jumlah terlambat per orang |
| `/privasi` | Naskah persetujuan (publik) |

**Keamanan dashboard:**

- Tanpa `--pin`, fitur kamera dan kelola wajah hanya aktif di localhost.
- PIN salah 5 kali dari satu IP dalam 60 detik → diblokir 120 detik dan dicatat di rekam jejak.
- Permintaan lintas situs ditolak, dan header `X-Frame-Options`, `X-Content-Type-Options`,
  `Referrer-Policy` dikirim.
- PIN dikirim lewat HTTP biasa, jadi di jaringan yang tidak tepercaya pakai reverse proxy HTTPS.
- Kamera browser hanya berfungsi di `localhost` atau HTTPS (aturan browser).

---

## 9. Liveness / anti-spoofing

Tanpa liveness, foto cetak atau foto di layar HP bisa dipakai untuk "titip absen". Absen **hanya
dicatat setelah wajah lolos uji keaslian**.

| `liveness_mode` | Cara kerja | Cocok untuk |
|---|---|---|
| `pasif` (bawaan) | Model CNN MiniFASNetV2 (crop 2,7×) + V1SE (crop 4,0×), keputusan dari rata-rata beberapa frame | CCTV / headless |
| `aktif` | Tantangan acak: "Tengok ke kiri lalu ke kanan" atau "Dongak lalu tunduk" | Kios dengan layar |
| `ganda` | Keduanya harus lolos | **Disarankan untuk kios** |
| `off` | Tanpa liveness | Hanya untuk pengujian |

```mermaid
stateDiagram-v2
    [*] --> Memeriksa: track wajah baru
    state "Memeriksa keaslian" as Memeriksa
    state "Pasif lolos" as PasifOK
    state "Wajah palsu" as Spoof
    state "Tantangan aktif" as Tantangan
    state "Lolos (absen boleh dicatat)" as Lolos

    Memeriksa --> PasifOK: >= 3 frame dan rata-rata >= ambang
    Memeriksa --> Spoof: 8 frame dan rata-rata < ambang
    Spoof --> PasifOK: rata-rata jendela kembali >= ambang
    PasifOK --> Lolos: mode pasif (atau ganda + tantangan selesai)
    Memeriksa --> Tantangan: mode aktif / ganda (berjalan bersamaan dengan cek pasif)

    state Tantangan {
        [*] --> Acak
        Acak --> Geleng: "Tengok kiri lalu kanan"
        Acak --> Angguk: "Dongak lalu tunduk"
        Geleng --> Selesai: rentang u dua arah >= ambang
        Angguk --> Selesai: rentang v dua arah >= 0,8 x ambang
        Geleng --> Acak: waktu habis (8 dtk)
        Angguk --> Acak: waktu habis (8 dtk)
    }
    Tantangan --> Lolos: selesai (mode ganda: pasif juga harus lolos)
    Lolos --> [*]: track hilang (status direset)
    Spoof --> [*]: track hilang
```

<sub>Gambar untuk dokumen: [PNG](docs/img/03_liveness_state.png) · [SVG](docs/img/03_liveness_state.svg)</sub>

**Tantangan aktif: koordinat afin hidung**

```
hidung = c + u·a + v·b      c = pusat mata, a = mata_kiri − mata_kanan, b = tengah_mulut − c
```

(u, v) tidak berubah oleh transformasi afin (geser, putar, skala, miring). Akibatnya, **foto datar yang
digerakkan memberi (u, v) yang tetap**, sedangkan **kepala 3D yang menoleh/dongak mengubahnya**. Gerakan
harus ke dua arah dari posisi awal.

![Tantangan afin](docs/img/07_tantangan_afin.png)

**Mematikan atau melonggarkan liveness**

| Kebutuhan | Pengaturan |
|---|---|
| Matikan total | `"liveness_mode": "off"` di `config.json`, atau env `LIVENESS_MODE=off` |
| Hanya cek otomatis (tanpa tantangan gerak) | `"liveness_mode": "pasif"` |
| Lewati saat registrasi satu orang | `python main.py register --nama "X" --lewati-liveness` (dicatat di rekam jejak) |
| Wajah asli sering ditolak | Turunkan `liveness_thresh` (mis. `0.6`) dan perbaiki pencahayaan |

```bash
# sementara, satu sesi terminal
set LIVENESS_MODE=off              # Windows CMD
$env:LIVENESS_MODE="off"           # PowerShell
LIVENESS_MODE=off python main.py run   # Linux/macOS
```

Percobaan palsu dicatat di `spoof_log.csv` dan `audit_log.jsonl`. Detail lengkap:
[docs/LIVENESS.md](docs/LIVENESS.md).

---

## 10. Privasi & UU PDP

Data wajah termasuk **data biometrik**, yaitu data pribadi yang bersifat spesifik menurut **Pasal 4
ayat (2) huruf b UU No. 27 Tahun 2022**. Indonesia tidak memiliki UU biometrik tersendiri.

```mermaid
flowchart LR
    classDef st fill:#ffffff,stroke:#7c3aed,color:#1f1d1a
    classDef aud fill:#dcf3e3,stroke:#15803d,color:#1f1d1a

    K["<b>1. Pengumpulan</b><br/>naskah + persetujuan eksplisit<br/><i>Pasal 20, 21, 22</i>"]:::st
    SM["<b>2. Penyimpanan</b><br/>AES-256-GCM<br/>foto tidak disimpan<br/><i>Pasal 35, 36</i>"]:::st
    PK["<b>3. Pemakaian</b><br/>dekripsi di RAM, hanya<br/>persetujuan aktif dicocokkan<br/><i>Pasal 20</i>"]:::st
    AK["<b>4. Hak subjek</b><br/>akses / ubah / tarik<br/><i>Pasal 5-10</i>"]:::st
    HP["<b>5. Penghapusan</b><br/>tarik persetujuan, retensi,<br/>timpa nol file lama<br/><i>Pasal 8, 9, 43, 44</i>"]:::st
    K --> SM --> PK --> AK --> HP

    AUD["<b>audit_log.jsonl</b> - setiap langkah dicatat, rantai hash SHA-256 (Pasal 31)"]:::aud
    K -.-> AUD
    SM -.-> AUD
    PK -.-> AUD
    AK -.-> AUD
    HP -.-> AUD
```

<sub>Gambar untuk dokumen: [PNG](docs/img/05_siklus_data.png) · [SVG](docs/img/05_siklus_data.svg)</sub>

### Registrasi dengan persetujuan

```mermaid
sequenceDiagram
    autonumber
    actor P as Petugas
    actor S as Subjek data
    participant A as main.py / dashboard
    participant PV as privacy.py
    participant L as Liveness pasif
    participant DB as faces_db.enc

    P->>A: Daftarkan "Budi" (nama, ID)
    A->>S: Tampilkan naskah persetujuan (Pasal 21)
    alt Subjek menolak
        S-->>A: tidak SETUJU
        A->>PV: audit("persetujuan_ditolak")
        A-->>P: Batal, tidak ada data wajah diproses
    else Subjek setuju
        S-->>A: SETUJU / centang + nama petugas
        A->>PV: consents.give(versi + SHA-256 naskah)
        PV->>PV: audit("persetujuan_diberikan")
        loop sampai 5-10 sampel valid
            A->>A: deteksi 1 wajah, lebar >= 90 px
            A->>A: gerbang kualitas (kecerahan, ketajaman)
            A->>L: skor keaslian (rata-rata 3 frame)
            alt buram / gelap / foto atau layar
                A-->>P: "Ditolak: ..." (sampel dibuang)
            else layak
                A->>A: embedding SFace 128-d (foto TIDAK disimpan)
            end
        end
        A->>PV: save_db (AES-256-GCM, tulis atomik + kunci)
        PV->>DB: tulis terenkripsi
        A->>PV: audit("wajah_didaftarkan")
        A-->>P: Tersimpan (terenkripsi)
    end
```

<sub>Gambar untuk dokumen: [PNG](docs/img/04_registrasi.png) · [SVG](docs/img/04_registrasi.svg)</sub>

### Pemetaan pasal → implementasi

| Pasal | Kewajiban | Implementasi |
|---|---|---|
| 20 (2) a, 22 | Persetujuan eksplisit, tertulis/terekam | Subjek mengetik `SETUJU` (CLI) / centang + petugas (dashboard) / `--bukti-persetujuan`; tanpa persetujuan aktif = tidak dikenali |
| 21 (1) | Informasi sebelum persetujuan | `privasi naskah`, halaman `/privasi`; versi + SHA-256 naskah dicatat |
| 5-10 | Hak subjek | `privasi akses` (salinan JSON), Ubah di dashboard, `privasi tarik` (hapus + tarik) |
| 31 | Perekaman kegiatan pemrosesan | `audit_log.jsonl` berantai hash SHA-256, diverifikasi dengan `privasi audit` |
| 35, 36 | Keamanan & kerahasiaan | AES-256-GCM, kunci izin 600 / passphrase scrypt, rotasi kunci, PIN + anti brute-force. Foto tidak disimpan |
| 43, 44 | Penghapusan / retensi | `privasi retensi`, `cloud_sync.py retensi`, file lama ditimpa nol |
| 34, 46, 53 | DPIA, notifikasi kebocoran, pejabat PDP | **Organisasional** (kerangka di [docs/PRIVASI.md](docs/PRIVASI.md)) |

### Perintah privasi

```bash
python main.py privasi naskah                     # tampilkan naskah persetujuan
python main.py privasi status                     # enkripsi, persetujuan, keutuhan rekam jejak
python main.py privasi catat --nama "Budi"        # catat persetujuan orang yang sudah terdaftar
python main.py privasi tarik --nama "Budi"        # tarik persetujuan + hapus data wajah
python main.py privasi akses --nama "Budi"        # salinan data subjek (Pasal 7)
python main.py privasi retensi --hari 365 --ya    # hapus absensi lama (tanpa --ya = simulasi)
python main.py privasi audit                      # verifikasi rantai hash
python main.py privasi rotasi-kunci [--passphrase]
python main.py privasi hapus-tanpa-persetujuan --ya
```

Naskah bisa diganti dengan file `naskah_persetujuan.txt` (placeholder `{pengendali}`, `{kontak}`,
`{retensi}`, `{versi}`).

---

## 11. Evaluasi kuantitatif (`evaluate.py`)

Pengukuran memakai **pipeline yang sama persis** dengan sistem. Hasilnya disimpan di
`hasil_evaluasi/<waktu>_<jenis>/`: `laporan.md` (tabel siap salin), `metrik.json`, CSV skor, dan
grafik PNG.

```mermaid
flowchart LR
    classDef d fill:#efe8fd,stroke:#7c3aed,color:#1f1d1a
    classDef c fill:#ffffff,stroke:#5c574f,color:#1f1d1a
    classDef o fill:#dcf3e3,stroke:#15803d,color:#1f1d1a

    KAM["Kamera"]:::d -- "evaluate.py kumpulkan" --> DS["dataset/&lt;nama&gt;/*.jpg"]:::d
    KAM -- "kumpulkan --liveness<br/>--tipe cetak|layar, --video" --> DL["dataset_liveness/<br/>asli/, serangan/&lt;jenis&gt;/"]:::d
    LAP["Uji di lokasi"]:::d --> CSV["uji_lapangan.csv"]:::d

    DS --> P1["pengenalan<br/>verifikasi 1:1 + identifikasi 1:N open-set"]:::c
    DL --> P2["liveness<br/>tingkat frame + tingkat video"]:::c
    DS --> P3["kinerja<br/>latensi, FPS, RAM, skala DB"]:::c
    CSV --> P4["lapangan<br/>black-box"]:::c

    P1 --> O1["EER, AUC, TAR@FAR,<br/>Rank-1, TPIR, FPIR, F1,<br/>ambang disarankan"]:::o
    P2 --> O2["APCER per jenis, BPCER,<br/>ACER, EER, lolos tantangan"]:::o
    P3 --> O3["ms per tahap (rerata/P95),<br/>FPS, MB"]:::o
    P4 --> O4["TAR, FAR, BPCER,<br/>APCER, waktu tercatat"]:::o
    O1 & O2 & O3 & O4 --> OUT["hasil_evaluasi/&lt;waktu&gt;/<br/>laporan.md, metrik.json, *.csv, *.png"]:::o
```

<sub>Gambar untuk dokumen: [PNG](docs/img/06_evaluasi.png) · [SVG](docs/img/06_evaluasi.svg)</sub>

```bash
python evaluate.py kumpulkan --label Budi --jumlah 30                  # ambil foto uji
python evaluate.py kumpulkan --label Budi --liveness --tipe cetak      # foto serangan
python evaluate.py kumpulkan --label Budi --liveness --video 6         # video untuk tantangan aktif

python evaluate.py pengenalan --data dataset/ --daftar 3 --ulang 10 --tak-dikenal 0.2
python evaluate.py liveness   --data dataset_liveness/
python evaluate.py kinerja    --data dataset/
python evaluate.py templat-lapangan && python evaluate.py lapangan --csv uji_lapangan.csv
```

```
dataset/<nama_orang>/*.jpg                         ≥ 2 foto/orang (disarankan 10)
dataset_liveness/asli/<nama>/*.jpg|*.mp4
dataset_liveness/serangan/<cetak|layar|video|...>/<nama>/*.jpg|*.mp4
```

| Evaluasi | Metrik |
|---|---|
| Verifikasi 1:1 | FAR, FRR, **EER**, AUC ROC, TAR@FAR 1% & 0,1%, d′ |
| Identifikasi 1:N open-set | Rank-1/5, TPIR, FNIR, salah identitas, **FPIR**, akurasi, presisi, recall, F1, **ambang disarankan** |
| Liveness (ISO/IEC 30107-3) | **APCER per jenis serangan**, BPCER, ACER, EER, BPCER@APCER 5%/10%, tingkat lolos tantangan |
| Kinerja | Latensi per tahap (rerata/median/P95), FPS, RAM, waktu cocok untuk 10-5.000 orang |
| Lapangan | TAR, FAR, BPCER, APCER, waktu sampai tercatat |

Contoh latensi pada CPU 2 core (OpenCV 5.0): deteksi YuNet ~7 ms, embedding SFace ~10 ms, liveness
pasif ~4 ms, dan pencocokan untuk 1.000 orang ~0,2 ms.

Protokol pengujian yang disarankan untuk skripsi ada di [docs/EVALUASI.md](docs/EVALUASI.md).

> Foto uji adalah data biometrik. Minta persetujuan peserta dan hapus setelah penelitian selesai.

---

## 12. Sinkronisasi cloud (opsional)

1. Buat service account di Google Cloud dan simpan JSON-nya sebagai `serviceAccount.json`.
2. Untuk Google Sheets, bagikan spreadsheet ke email service account tersebut.
3. Buat `cloud_config.json`:

```json
{
  "credentials": "serviceAccount.json",
  "firestore_enabled": true,
  "firestore_collection": "absensi",
  "firestore_daily": "rekap_harian",
  "gsheet_id": "ID_SPREADSHEET",
  "gsheet_tab": "Absensi"
}
```

Sinkronisasi berjalan otomatis saat `main.py run` atau `dashboard.py` aktif: absen masuk antrean,
dikirim ulang bila offline, dan tidak menghasilkan data ganda.

```bash
python cloud_sync.py test            # cek koneksi
python cloud_sync.py backfill        # paksa kirim semua yang belum terkirim
python cloud_sync.py retensi --hari 365 --ya
```

Yang dikirim hanya catatan absensi (nama, ID, jam, skor, status). **Embedding wajah tidak pernah
dikirim.**

---

## 13. Struktur proyek & file data

```
├── main.py                CLI + inti pengenalan
├── liveness.py            anti-spoofing
├── privacy.py             enkripsi, persetujuan, rekam jejak
├── attendance.py          aturan absensi
├── dashboard.py           dashboard web
├── evaluate.py            evaluasi kuantitatif
├── cloud_sync.py          Firestore + Google Sheets
├── config.py              konfigurasi terpusat
├── ipclock.py             kunci antar-proses
├── config.contoh.json     contoh konfigurasi
├── requirements.txt
├── models/                MiniFASNet ONNX (+ YuNet/SFace diunduh otomatis)
├── tools/                 konversi model liveness tfjs → ONNX
├── tests/                 uji unit
└── docs/                  dokumentasi lanjutan + diagram (Mermaid, PNG, SVG)
```

File yang dibuat saat berjalan (semuanya ada di `.gitignore`):

| File | Isi |
|---|---|
| `faces_db.enc` | Embedding wajah, **terenkripsi AES-256-GCM** |
| `kunci/facedb.key` | Kunci enkripsi (**cadangkan, jangan di-commit**) |
| `persetujuan.json` | Catatan persetujuan subjek |
| `audit_log.jsonl` | Rekam jejak berantai hash |
| `absensi.csv` | `tanggal, jam, nama, id, skor, tipe, status` |
| `spoof_log.csv` | Percobaan wajah palsu |
| `export/`, `hasil_evaluasi/` | Rekap dan laporan evaluasi |

> **Jangan pernah mengunggah data pribadi ke GitHub.** Pastikan `.gitignore` ikut ter-commit sebelum
> `git push`.

---

## 14. Pengujian

```bash
python -m unittest tests/uji_unit.py -v
```

Ada 15 uji unit yang berjalan tanpa kamera dan tanpa model: aturan masuk/pulang, retensi, enkripsi
(file diubah, kunci salah, passphrase), migrasi, persetujuan, rantai hash, filter persetujuan,
invariansi afin, tantangan dua arah, dan EER.

---

## 15. Pemecahan masalah

| Gejala | Penyebab / solusi |
|---|---|
| `Kamera index 0 tidak terdeteksi` | Coba `--kamera 1`, tutup aplikasi lain yang memakai kamera, cek izin kamera di Windows |
| Model gagal diunduh | Unduh manual dari opencv_zoo ke `models/` |
| `Butuh paket 'cryptography'` | `pip install cryptography` |
| `File kunci tidak ditemukan` | Kembalikan `kunci/facedb.key` dari cadangan atau set `FACEDB_KEY_FILE` |
| `Kunci tidak cocok` / `rusak atau telah diubah` | Kunci/passphrase salah atau file database dimodifikasi; pulihkan dari cadangan |
| Wajah terdaftar tetap "Unknown" | Belum ada persetujuan aktif: `privasi catat --nama "X"` atau tombol *Catat persetujuan* |
| Selalu "Memeriksa keaslian" / "PALSU?" padahal asli | Perbaiki cahaya (hindari cahaya dari belakang), dekatkan wajah, turunkan `liveness_thresh`, atau pakai `pasif` |
| Tantangan aktif tidak pernah lolos | Gerakkan kepala lebih jelas ke dua arah, wajah ≥ 100 px, turunkan `tantangan_ambang` (mis. 0.22) |
| Registrasi: "tidak lolos uji keaslian" terus | Perbaiki cahaya, atau `--lewati-liveness` (diawasi petugas) |
| Dashboard: "Terlalu banyak PIN salah" | Tunggu 120 detik, lalu isi PIN yang benar |
| Kamera browser tidak muncul | Buka lewat `localhost` atau HTTPS |
| `Konfigurasi '...' tidak valid` | Periksa format `config.json` (jam `HH:MM`, angka, `liveness_mode` berupa teks) |

---

## 16. Batasan

- Akurasi liveness pasif dan ambang tantangan aktif **harus diukur pada kamera dan kondisi cahaya
  sendiri** dengan `evaluate.py liveness`.
- Tantangan aktif bisa ditembus oleh video replay yang berisi gerakan yang sama, dan tidak ada lapisan
  yang dirancang untuk topeng 3D. Mode `ganda` menekan risiko ini tetapi tidak menghilangkannya.
- Enkripsi melindungi file di disk. Saat program berjalan, embedding berada di RAM dalam bentuk
  terdekripsi. Simpan kunci terpisah dari folder data.
- Kewajiban organisasi UU PDP (DPIA, prosedur notifikasi kebocoran, pejabat PDP) tidak bisa dipenuhi
  oleh kode saja.

---

## 17. Atribusi & lisensi model

| Model | Sumber | Lisensi |
|---|---|---|
| YuNet (deteksi wajah) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | MIT |
| SFace (pengenalan wajah) | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) | Apache-2.0 |
| MiniFASNetV2 / V1SE (liveness) | [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing), dikonversi ke ONNX dengan `tools/convert_minifasnet_tfjs.py` | Apache-2.0 |

Lihat [`models/LISENSI_MODEL.txt`](models/LISENSI_MODEL.txt).

**Dokumentasi lanjutan:** [Arsitektur](docs/ARSITEKTUR.md) · [Liveness](docs/LIVENESS.md) ·
[Privasi](docs/PRIVASI.md) · [Evaluasi](docs/EVALUASI.md) · [Perubahan & temuan teknis](docs/PERUBAHAN.md)

Rujukan hukum: [UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi](https://pasal.id/peraturan/uu/uu-no-27-tahun-2022)
