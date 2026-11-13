# Auto Fast Face Recognition: Sistem Absensi Wajah

Sistem absensi berbasis pengenalan wajah yang berjalan **sepenuhnya di CPU** (tanpa GPU), hemat RAM, dan bisa memakai webcam, kamera browser, atau CCTV RTSP. Setiap absen otomatis disimpan ke file CSV lokal dan **disinkronkan otomatis ke Firebase Firestore dan Google Sheets**. Dilengkapi dashboard admin web untuk memantau kehadiran dan mengelola wajah terdaftar.

## Fitur

- **Absen otomatis**: berdiri di depan kamera, wajah dikenali, kehadiran langsung tercatat.
- **Masuk, terlambat, dan pulang**: aturan jam bisa diatur, status direset otomatis tiap hari tanpa restart.
- **Hemat resource**: deteksi wajah tiap N frame, frame lain dilacak dengan optical flow. Berjalan di CPU.
- **Tiga sumber kamera**: webcam komputer, kamera browser (HP atau PC lain), dan CCTV RTSP always-on yang menyambung ulang otomatis bila putus.
- **Dashboard admin**: ringkasan kehadiran, kamera absen langsung di web, CRUD wajah, rekap per orang, tema terang, gelap, atau otomatis.
- **Auto sync ke cloud**: Firestore dan Google Sheets, tahan offline, dengan retry otomatis dan tanpa data ganda.
- **Ekspor rekap** ke CSV atau Excel (`.xlsx`).
- **Privasi**: foto tidak disimpan, hanya embedding wajah (vektor 128 dimensi).

## Arsitektur

```mermaid
flowchart LR
  CAM["Webcam / CCTV RTSP / Kamera browser"] --> DET["YuNet: deteksi wajah"]
  DET --> TRK["Pelacak: optical flow + IoU"]
  TRK --> EMB["SFace: embedding 128-d"]
  EMB --> MATCH["Pencocokan ke faces_db.npz"]
  MATCH --> LOG["AttendanceLogger"]
  LOG --> CSV[("absensi.csv")]
  LOG --> Q["Antrean cloud_sync"]
  Q --> FS[("Firestore")]
  Q --> GS[("Google Sheets")]
  CSV -. "catch-up otomatis" .-> Q
  DASH["dashboard.py"] --> CSV
  DASH --> LOG
```

## Tech stack

| Lapisan | Teknologi |
|---|---|
| Bahasa | Python 3 (dijalankan di Python 3.13, Windows) |
| Deteksi wajah | **YuNet** (`face_detection_yunet_2023mar.onnx`) lewat `cv2.FaceDetectorYN` |
| Pengenalan wajah | **SFace** (`face_recognition_sface_2021dec.onnx`) lewat `cv2.FaceRecognizerSF`, embedding 128 dimensi |
| Sumber model | [OpenCV Zoo](https://github.com/opencv/opencv_zoo), diunduh otomatis ke folder `models/` |
| Computer vision | **OpenCV** (`opencv-python` >= 4.8): akuisisi kamera, DirectShow dan V4L2, FFmpeg untuk RTSP, optical flow Lucas-Kanade |
| Numerik | **NumPy**: perkalian matriks untuk pencocokan, penyimpanan `.npz` terkompresi |
| Streaming CCTV | **RTSP** over TCP (opsi FFmpeg lewat OpenCV), reconnect dengan exponential backoff |
| Server dashboard | Python standard library: `http.server.ThreadingHTTPServer`, `threading`, `argparse`, `csv`, `hmac` |
| Frontend dashboard | **HTML, CSS, dan JavaScript murni** dalam satu file (tanpa framework dan tanpa build), CSS variables untuk tema, SVG ring chart |
| Kamera di browser | **WebRTC** `getUserMedia`, Canvas API, Fullscreen API |
| Font | Google Fonts: Lora dan Plus Jakarta Sans |
| Database cloud | **Firebase Firestore** lewat **Firebase Admin SDK** (`firebase-admin`) |
| Spreadsheet | **Google Sheets API** lewat **gspread** |
| Autentikasi cloud | **Google Service Account** (OAuth2 JWT) |
| Penyimpanan lokal | `absensi.csv` (log), `faces_db.npz` (embedding), JSONL (outbox), JSON (state sinkronisasi) |
| Ekspor Excel | **openpyxl** (opsional) |
| Monitoring RAM | **psutil** (opsional) |

### Model pengenalan wajah

- **YuNet** mendeteksi wajah beserta 5 landmark. Ringan dan cocok untuk CPU.
- **SFace** mengubah wajah yang sudah diluruskan (align) menjadi vektor 128 dimensi.
- **Pencocokan**: cosine similarity. Skor per orang adalah rata-rata 2 sampel paling mirip. Dianggap dikenali bila skor >= `MATCH_THRESH` (0.40), selisih dengan orang kedua >= `MATCH_MARGIN` (0.05), dan cocok `CONFIRM_MATCHES` (2) kali beruntun. Wajah diverifikasi ulang tiap `RECHECK_DETIK` (60 detik).

## Struktur proyek

```
.
├── main.py                    # CLI + mesin pengenalan + pencatat absensi
├── dashboard.py               # Dashboard admin web (server + UI satu file)
├── cloud_sync.py              # Sinkronisasi otomatis ke Firestore + Google Sheets
├── cloud_config.example.json  # Contoh konfigurasi cloud
├── requirements.txt
├── .gitignore
│
├── models/                    # (otomatis) model YuNet + SFace
├── faces_db.npz               # (dibuat otomatis) embedding wajah terdaftar
├── absensi.csv                # (dibuat otomatis) catatan absensi
├── export/                    # (dibuat otomatis) hasil ekspor rekap
├── serviceAccount.json        # (Anda buat) kredensial Google, JANGAN di-commit
└── cloud_config.json          # (Anda buat) konfigurasi cloud, JANGAN di-commit
```

## Instalasi

```bash
git clone <url-repo-anda>
cd <nama-folder>
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
python main.py download-models
```

Model juga diunduh otomatis saat pertama kali dibutuhkan.

## Penggunaan

### 1. Daftarkan wajah

```bash
python main.py register --nama "Budi" --id 2023001
```

Program mengambil 8 sampel otomatis (opsi `--jumlah 5` sampai `10`) dengan arahan pose. Wajah juga bisa ditambahkan lewat dashboard, tab **Kelola wajah**.

### 2. Jalankan absensi

```bash
python main.py run                                  # webcam, tekan Q / ESC untuk keluar
python main.py run --jarak jauh                     # profil jarak
python main.py run --rtsp "rtsp://user:pass@192.168.1.64:554/stream" --headless   # CCTV
```

URL RTSP juga bisa lewat environment variable `RTSP_URL`.

| Profil `--jarak` | Resolusi kamera | Perkiraan jarak | Catatan |
|---|---|---|---|
| `dekat` | 640x480 | sekitar 1 m | paling ringan |
| `sedang` (bawaan) | 1280x720 | sekitar 3 m | seimbang |
| `jauh` | 1280x720 | sekitar 4 m | akurasi menurun, butuh CPU lebih kuat |

### 3. Dashboard admin

```bash
python dashboard.py
python dashboard.py --rtsp "rtsp://..." --jarak jauh                  # CCTV always-on
python dashboard.py --host 0.0.0.0 --pin 1234                         # akses dari jaringan
```

Buka `http://127.0.0.1:8000`. Halaman: **Ringkasan**, **Kamera absen**, **Kelola wajah**, **Rekap kehadiran**.

| Opsi | Fungsi |
|---|---|
| `--port 8000` | port server |
| `--host 0.0.0.0` | bisa dibuka dari perangkat lain di jaringan (wajib `--pin`) |
| `--pin 1234` | PIN admin untuk kamera, pengelolaan wajah, dan data absensi |
| `--rtsp URL` | CCTV selalu aktif, absen jalan walau browser tertutup |
| `--jarak` | profil jarak yang dipakai `main.py` |
| `--no-browser` | jangan buka browser otomatis |

> Kamera browser hanya bisa dipakai di `localhost` atau HTTPS. PIN dikirim lewat HTTP biasa, jadi gunakan HTTPS di jaringan yang tidak tepercaya.

### 4. Perintah lain

```bash
python main.py list                          # daftar orang terdaftar
python main.py delete --nama "Budi"          # hapus data satu orang
python main.py export --format xlsx          # ekspor rekap (csv atau xlsx)
python main.py export --dari 2026-11-01 --sampai 2026-11-30
```

## Aturan absensi

Atur di bagian `KONFIGURASI` pada `main.py`:

| Konstanta | Bawaan | Arti |
|---|---|---|
| `JAM_TERLAMBAT` | `09:00:00` | scan pertama setelah jam ini tercatat **terlambat** |
| `JAM_PULANG` | `11:00:00` | scan setelah jam ini dari orang yang sudah masuk tercatat **pulang** |
| `MIN_DURASI_MENIT` | `0` | jeda minimum masuk ke pulang (naikkan bila perlu) |

- Scan pertama hari itu sampai `JAM_TERLAMBAT` tercatat **masuk (hadir)**.
- Pulang hanya sekali per hari. Semua status direset otomatis saat tanggal berganti.

Format `absensi.csv`: `tanggal, jam, nama, id, skor, tipe, status`.

## Sinkronisasi cloud (Firestore + Google Sheets)

Sinkronisasi **otomatis**: tidak ada perintah tambahan. Cukup jalankan `main.py run` atau `dashboard.py`.

### Setup

1. Buat project di [Firebase Console](https://console.firebase.google.com/) dan buat database Firestore (**Build, Firestore Database, Create database**).
2. **Project settings, Service accounts, Generate new private key**. Simpan sebagai `serviceAccount.json` di folder proyek.
3. Aktifkan **Google Sheets API** di Google Cloud Console untuk project yang sama.
4. Buat Google Sheet kosong, lalu **Share** sebagai Editor ke `client_email` yang ada di `serviceAccount.json`.
5. Salin `cloud_config.example.json` menjadi `cloud_config.json` dan isi `gsheet_id` (teks di URL sheet antara `/d/` dan `/edit`).
6. Cek koneksi:
   ```bash
   python cloud_sync.py test
   ```

### Konfigurasi

File `cloud_config.json` atau environment variable (nilai env diutamakan):

| Kunci | Env | Bawaan |
|---|---|---|
| `credentials` | `GOOGLE_CREDENTIALS` | `serviceAccount.json` |
| `gsheet_id` | `GSHEET_ID` | kosong (Sheets nonaktif) |
| `gsheet_tab` | `GSHEET_TAB` | `Absensi` |
| `firestore_enabled` | `FIRESTORE_ENABLED` | `true` |
| `firestore_collection` | `FIRESTORE_COLLECTION` | `absensi` |
| `firestore_daily` | `FIRESTORE_DAILY` | `rekap_harian` |

Contoh PowerShell:
```powershell
$env:GOOGLE_CREDENTIALS = "C:\path\serviceAccount.json"
$env:GSHEET_ID = "ID_SPREADSHEET"
python main.py run
```

### Cara kerja

1. Absen baru masuk antrean lalu dikirim oleh thread latar (tidak memperlambat pengenalan).
2. **Catch-up otomatis**: saat program mulai dan tiap 5 menit, `absensi.csv` dicocokkan dengan `cloud_state.json`. Baris yang belum ada di cloud dikirim sendiri, termasuk data lama.
3. Gagal kirim (internet putus dan sebagainya) dicoba lagi otomatis dengan jeda bertahap, juga setelah restart. Data tidak hilang karena `absensi.csv` tetap menjadi catatan lokal utama.
4. **Tanpa data ganda**: Firestore memakai ID dokumen deterministik, Sheets dicek lewat kolom `key`.

### Struktur data

**Firestore**

| Koleksi | ID dokumen | Isi |
|---|---|---|
| `absensi` | `{tanggal}_{nama}_{tipe}` | tanggal, jam, nama, id, skor, tipe, status, waktu (timestamp), sumber |
| `rekap_harian` | `{tanggal}_{nama}` | tanggal, nama, id, jam_masuk, status, jam_pulang |

**Google Sheets** (tab `Absensi`): `tanggal | jam | nama | id | skor | tipe | status | key`

## Keamanan dan privasi

- **Jangan commit** `serviceAccount.json`, `cloud_config.json`, `faces_db.npz`, dan `absensi.csv`. Semuanya sudah ada di `.gitignore`. Kalau `serviceAccount.json` pernah bocor, buat key baru dan hapus yang lama di Google Cloud Console.
- Atur **Firestore Security Rules** ke `allow read, write: if false;` selama hanya server Python yang menulis. Admin SDK tetap bisa menulis.
- Dashboard yang dibuka dari jaringan wajib memakai `--pin`. Tanpa PIN, fitur kamera dan pengelolaan wajah dinonaktifkan. Permintaan lintas situs ditolak, dan PIN dibandingkan dengan `hmac.compare_digest`.
- Input nama dan ID menolak awalan `=`, `+`, `-`, `@` untuk mencegah formula injection saat dibuka di Excel dan Sheets.
- Data wajah adalah **data biometrik**. Pastikan semua orang yang didaftarkan memberi persetujuan dan patuhi regulasi perlindungan data pribadi yang berlaku di tempat Anda.

## Pemecahan masalah

| Gejala | Penyebab dan solusi |
|---|---|
| `invalid_grant: Invalid JWT ...` | Jam komputer tidak akurat. Aktifkan **Set time automatically** lalu **Sync now** |
| `ModuleNotFoundError: firebase_admin` | `pip install -r requirements.txt` |
| Firestore `PermissionDenied` atau `NotFound` | Database Firestore belum dibuat, atau Firestore API belum aktif |
| Sheets `403` atau `SpreadsheetNotFound` | Sheet belum di-share ke `client_email`, atau Google Sheets API belum aktif |
| `Kamera index 0 tidak terdeteksi` | Periksa izin kamera, atau coba `--kamera 1`. Kamera tidak bisa dipakai bersamaan oleh dua program |
| `DB kosong` | Belum ada wajah terdaftar. Gunakan `register` atau dashboard, tab Kelola wajah |
| Wajah tidak dikenali dari jauh | Gunakan `--jarak jauh`, tambah cahaya, atau tambah sampel wajah |
| Tidak ada baris `[CLOUD]` | Pastikan `serviceAccount.json` ada dan `cloud_sync.py` satu folder dengan `main.py` |

## Catatan

- `main.py run` dan `dashboard.py` dengan kamera server tidak bisa memakai webcam yang sama bersamaan.
- Dashboard memakai OpenCV dan membaca kamera lokal, jadi dijalankan di komputer atau server sendiri (self-hosted), bukan di platform serverless.

## Lisensi

Tentukan lisensi proyek Anda di sini (misalnya MIT) dan tambahkan file `LICENSE`.

## Kredit

- [OpenCV](https://opencv.org/) dan [OpenCV Zoo](https://github.com/opencv/opencv_zoo) untuk model YuNet dan SFace
- [Firebase Admin SDK](https://firebase.google.com/docs/admin/setup) dan [gspread](https://github.com/burnash/gspread)
