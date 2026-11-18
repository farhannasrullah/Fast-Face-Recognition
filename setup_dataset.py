#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Melengkapi Dataset Uji hingga Total 100 Sampel Probe:
- terdaftar/Farhan           : 48 gambar total (8 galeri + 40 probe uji)
- tidak_terdaftar/Impostor   : 35 gambar probe orang asing acak
- spoof/Layar_HP_Farhan      : 25 gambar probe serangan layar/foto
Total probe evaluasi         : 40 + 35 + 25 = 100 sampel
"""

import time
from pathlib import Path
import cv2
import numpy as np

import config
from main import FaceEngine, ensure_models, small_size

DATASET_DIR = Path(config.BASE_DIR) / "dataset_uji"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

ARAHAN_VARIASI = [
    "Hadap lurus, cahaya normal",
    "Tengok sedikit ke kiri",
    "Tengok sedikit ke kanan",
    "Dongak sedikit ke atas",
    "Tunduk agak tajam (uji batas)",
    "Geser agak jauh dari kamera",
    "Geser agak dekat ke kamera",
    "Ekspresi senyum / bicara",
    "Miringkan kepala ringan",
    "Cahaya agak redup / membelakangi lampu",
]


def hitung_gambar(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def lengkapi_sesi(
    cap,
    engine: FaceEngine,
    folder_tujuan: Path,
    target_total: int,
    judul_sesi: str,
    instruksi: str,
    jeda: float = 0.45,
    mode_variasi: bool = False,
):
    folder_tujuan.mkdir(parents=True, exist_ok=True)
    existing = hitung_gambar(folder_tujuan)
    jumlah_awal = len(existing)

    if jumlah_awal >= target_total:
        print(f"[SKIP] {judul_sesi}: sudah memiliki {jumlah_awal}/{target_total} gambar.")
        return

    kekurangan = target_total - jumlah_awal
    print(f"\n=== {judul_sesi} ===")
    print(f"Saat ini: {jumlah_awal} gambar | Perlu tambah: {kekurangan} gambar (Target: {target_total})")
    print("Tekan [SPASI] untuk mulai merekam, [S] untuk lewati, atau [Q] untuk keluar.")

    mulai = False
    tersimpan = jumlah_awal
    waktu_terakhir = 0.0

    while tersimpan < target_total:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        show = cv2.flip(frame, 1)
        fw = frame.shape[1]
        sw, sh, scale = small_size(frame)
        faces = engine.detect(cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA))

        wajah_valid = None
        if faces:
            f = max(faces, key=lambda x: float(x[2] * x[3]))
            if f[2] * scale[0] >= 60:
                wajah_valid = f
                x, y, bw, bh = (int(v) for v in f[:4] * np.array([*scale, *scale], np.float32))
                mx = fw - x - bw
                warna = (0, 220, 0) if mulai else (0, 200, 255)
                cv2.rectangle(show, (mx, y), (mx + bw, y + bh), warna, 2)

        if not mulai:
            cv2.putText(show, f"{judul_sesi} ({tersimpan}/{target_total})", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(show, instruksi, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
            cv2.putText(show, "[SPASI] Mulai  |  [S] Lewati  |  [Q] Keluar", (15, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (200, 200, 200), 1)
        else:
            panduan = ARAHAN_VARIASI[tersimpan % len(ARAHAN_VARIASI)] if mode_variasi else instruksi
            cv2.putText(show, f"{judul_sesi}: {tersimpan}/{target_total}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            cv2.putText(show, panduan, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)

            if wajah_valid is None:
                cv2.putText(show, "Wajah belum terdeteksi!", (15, show.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            elif time.time() - waktu_terakhir >= jeda:
                waktu_terakhir = time.time()
                tersimpan += 1
                nama_file = folder_tujuan / f"sampel_{tersimpan:03d}.jpg"
                while nama_file.exists():
                    tersimpan += 1
                    nama_file = folder_tujuan / f"sampel_{tersimpan:03d}.jpg"
                cv2.imwrite(str(nama_file), frame)
                print(f"  [{tersimpan}/{target_total}] Tersimpan: {nama_file.name}")

        cv2.imshow("Lengkapi Dataset Uji (Target 100 Probe)", show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            mulai = True
        elif key == ord("s"):
            print(f"  Sesi '{judul_sesi}' dilewati.")
            return
        elif key in (ord("q"), 27):
            raise KeyboardInterrupt("Dibatalkan oleh pengguna.")


def main():
    nama = input("Masukkan nama subjek terdaftar (default: Farhan): ").strip() or "Farhan"
    engine = FaceEngine(*ensure_models())

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak dapat dibuka.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        # 1. Lengkapi hingga 48 foto (8 galeri + 40 sampel uji)
        lengkapi_sesi(
            cap, engine,
            DATASET_DIR / "terdaftar" / nama,
            target_total=48,
            judul_sesi="Sesi 1: Tambah Wajah Asli",
            instruksi="Variasikan jarak, sudut kepala, dan pencahayaan",
            jeda=0.45,
            mode_variasi=True,
        )

        # 2. Lengkapi hingga 35 foto orang asing acak
        lengkapi_sesi(
            cap, engine,
            DATASET_DIR / "tidak_terdaftar" / "Random_Impostor",
            target_total=35,
            judul_sesi="Sesi 2: Tambah Orang Asing Acak",
            instruksi="Geser foto/video wajah orang berbeda di HP ke webcam",
            jeda=0.7,
        )

        # 3. Lengkapi hingga 25 foto serangan spoof
        lengkapi_sesi(
            cap, engine,
            DATASET_DIR / "spoof" / f"Layar_HP_{nama}",
            target_total=25,
            judul_sesi="Sesi 3: Tambah Uji Spoof (Layar HP/Foto)",
            instruksi="Arahkan foto selfie/video di layar HP dari berbagai jarak",
            jeda=0.45,
        )

        print("\n" + "=" * 65)
        print("DATASET 100 SAMPEL PROBE SELESAI DILENGKAPI!")
        print("Rincian: 40 Uji Terdaftar + 35 Uji Orang Asing + 25 Uji Spoof = 100 Sampel")
        print("Jalankan pengujian sekarang dengan perintah:")
        print("   python evaluate.py run --galeri 8")
        print("=" * 65)

    except KeyboardInterrupt as e:
        print(f"\n{e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()