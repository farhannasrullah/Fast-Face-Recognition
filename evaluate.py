#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrip Pengujian Kuantitatif Model Presensi Wajah Otomatis (YuNet + SFace + Liveness)
Mengukur: Precision, Recall, F1-Score, Akurasi Identifikasi (Open-Set),
          FRR, FAR, Penolakan Wajah Palsu (Anti-Spoofing), dan Latensi per Siklus di CPU.

Cara Penggunaan:
1. Kumpulkan sampel uji dari kamera:
   python evaluate.py collect --kategori terdaftar --nama "Budi" --jumlah 5
   python evaluate.py collect --kategori tidak_terdaftar --nama "Tamu_1" --jumlah 5
   python evaluate.py collect --kategori spoof --nama "Foto_Cetak" --jumlah 10

2. Jalankan pengujian penuh & cetak tabel laporan:
   python evaluate.py run --galeri 8
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

import config
from liveness import LivenessEngine
from main import (
    CV_THREADS,
    Database,
    FaceEngine,
    MATCH_MARGIN,
    MATCH_THRESH,
    ensure_models,
    recog_quality,
    small_size,
)

DATASET_DIR = Path(config.BASE_DIR) / "dataset_uji"
REPORT_JSON = Path(config.BASE_DIR) / "hasil_evaluasi_presensi.json"
REPORT_CSV = Path(config.BASE_DIR) / "detail_uji_presensi.csv"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# 1. Pengambilan Sampel Uji dari Kamera Presensi
# ---------------------------------------------------------------------------
def collect_samples(kategori: str, nama: str, jumlah: int, cam_index: int = 0, jeda: float = 0.6):
    """Ambil gambar uji langsung dari kamera presensi ke struktur folder dataset_uji/."""
    out_dir = DATASET_DIR / kategori / nama
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len([p for p in out_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera index {cam_index} gagal dibuka.")

    print(f"[COLLECT] Kategori: {kategori} | Subjek: {nama} | Target: {jumlah} gambar baru")
    print("Tekan Q / ESC untuk berhenti lebih awal.")
    saved, last_t = 0, 0.0

    try:
        while saved < jumlah:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            now = time.time()
            show = cv2.flip(frame, 1)
            cv2.putText(
                show,
                f"{kategori}/{nama}: {saved}/{jumlah}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 0),
                2,
            )
            cv2.imshow("Pengambilan Dataset Uji", show)

            if now - last_t >= jeda:
                last_t = now
                fname = out_dir / f"img_{existing + saved + 1:03d}.jpg"
                cv2.imwrite(str(fname), frame)
                saved += 1
                print(f"  Tersimpan: {fname.name} ({saved}/{jumlah})")

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Selesai. Total gambar tersimpan di {out_dir}: {existing + saved}")


# ---------------------------------------------------------------------------
# 2. Ekstraksi Fitur & Siklus Inferensi Penuh pada CPU
# ---------------------------------------------------------------------------
def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def run_single_cycle(
    img_bgr: np.ndarray,
    engine: FaceEngine,
    db: Database,
    live: LivenessEngine | None,
    cek_kualitas: bool = True,
) -> dict:
    """
    Menjalankan satu siklus inferensi lengkap pada CPU:
    Deteksi YuNet -> Uji Keaslian Liveness -> Gerbang Kualitas -> Ekstraksi SFace -> Pencocokan Kosinus.
    """
    t0 = time.perf_counter()

    sw, sh, scale = small_size(img_bgr)
    small = cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    faces = engine.detect(small)

    if not faces:
        elapsed = time.perf_counter() - t0
        return {
            "status": "no_face",
            "pred_nama": None,
            "skor": 0.0,
            "live_skor": None,
            "latensi_detik": elapsed,
        }

    # Ambil wajah terbesar di frame
    face = max(faces, key=lambda f: float(f[2] * f[3]))
    k = np.array([*scale, *scale], np.float32)
    box_full = face[:4] * k

    # 1. Uji keaslian wajah (Anti-Spoofing pasif)
    live_skor = None
    if live is not None and live.enabled and live.passive is not None:
        live_skor = live.passive.score(img_bgr, box_full)
        if live_skor is not None and live_skor < live.thresh:
            elapsed = time.perf_counter() - t0
            return {
                "status": "spoof_rejected",
                "pred_nama": None,
                "skor": 0.0,
                "live_skor": float(live_skor),
                "latensi_detik": elapsed,
            }

    # 2. Pelurusan wajah & gerbang kualitas
    aligned = engine.align(img_bgr, face, scale)
    if cek_kualitas:
        bad = recog_quality(aligned, float(face[14]))
        if bad:
            elapsed = time.perf_counter() - t0
            return {
                "status": f"low_quality:{bad}",
                "pred_nama": None,
                "skor": 0.0,
                "live_skor": float(live_skor) if live_skor is not None else None,
                "latensi_detik": elapsed,
            }

    # 3. Ekstraksi embedding 128-d SFace & pencocokan kemiripan kosinus
    emb = engine.feature(aligned)
    pred_nama, _, skor = db.match(emb)
    dikenali = pred_nama is not None and skor >= MATCH_THRESH

    elapsed = time.perf_counter() - t0
    return {
        "status": "matched" if dikenali else "unknown",
        "pred_nama": pred_nama if dikenali else None,
        "skor": float(skor),
        "live_skor": float(live_skor) if live_skor is not None else None,
        "latensi_detik": elapsed,
    }


# ---------------------------------------------------------------------------
# 3. Evaluasi Open-Set Identification, FAR/FRR, dan Anti-Spoofing
# ---------------------------------------------------------------------------
def evaluate_dataset(n_galeri: int = 8, cek_kualitas: bool = False):
    """
    Membangun galeri in-memory dari n_galeri gambar pertama tiap subjek terdaftar,
    lalu menguji sisa gambar terdaftar (genuine), gambar tidak terdaftar (impostor),
    dan gambar serangan pemalsuan (spoof).
    """
    cv2.setNumThreads(CV_THREADS)
    engine = FaceEngine(*ensure_models())
    live = LivenessEngine(yunet_path=engine.yunet_path)

    reg_dir = DATASET_DIR / "terdaftar"
    unk_dir = DATASET_DIR / "tidak_terdaftar"
    spoof_dir = DATASET_DIR / "spoof"

    if not reg_dir.exists():
        raise FileNotFoundError(
            f"Folder {reg_dir} tidak ditemukan. Jalankan perintah 'collect' terlebih dahulu."
        )

    # A. Bangun basis data galeri terdaftar in-memory (tanpa mengubah faces_db.enc produksi)
    gal_names, gal_ids, gal_embs = [], [], []
    probe_terdaftar = []  # (nama_asli, path_gambar)

    subjek_terdaftar = sorted([d for d in reg_dir.iterdir() if d.is_dir()])
    for idx, subj_dir in enumerate(subjek_terdaftar, start=1):
        imgs = _list_images(subj_dir)
        if len(imgs) <= n_galeri:
            print(f"PERINGATAN: {subj_dir.name} hanya punya {len(imgs)} gambar (butuh > {n_galeri}), dilewati.")
            continue

        pid = f"SISWA_{idx:03d}"
        sampel_galeri = 0
        for p in imgs[:n_galeri]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            sw, sh, scale = small_size(img)
            faces = engine.detect(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA))
            if not faces:
                continue
            face = max(faces, key=lambda f: float(f[2] * f[3]))
            gal_names.append(subj_dir.name)
            gal_ids.append(pid)
            gal_embs.append(engine.embed(img, face, scale))
            sampel_galeri += 1

        if sampel_galeri > 0:
            for p in imgs[n_galeri:]:
                probe_terdaftar.append((subj_dir.name, p))

    if not gal_embs:
        raise RuntimeError("Tidak ada sampel galeri valid yang berhasil diekstrak.")

    db = Database.from_arrays(gal_names, gal_ids, np.vstack(gal_embs))

    # Kumpulkan daftar probe tidak terdaftar (impostor) & serangan palsu (spoof)
    probe_tidak_terdaftar = []
    if unk_dir.exists():
        for subj_dir in sorted([d for d in unk_dir.iterdir() if d.is_dir()]):
            for p in _list_images(subj_dir):
                probe_tidak_terdaftar.append((subj_dir.name, p))

    probe_spoof = []
    if spoof_dir.exists():
        for subj_dir in sorted([d for d in spoof_dir.iterdir() if d.is_dir()]):
            for p in _list_images(subj_dir):
                probe_spoof.append((subj_dir.name, p))

    # B. Jalankan Pengujian & Catat Detail
    detail_rows = []
    latencies = []

    # 1) Pengujian Kelompok Terdaftar (Genuine)
    tp = 0  # Dikenali dengan nama yang BENAR
    fn = 0  # Ditolak sebagai Unknown / salah orang (False Rejection)
    fp_mis_id = 0  # Dikenali sebagai siswa lain yang SALAH

    for nama_asli, path_img in probe_terdaftar:
        img = cv2.imread(str(path_img))
        res = run_single_cycle(img, engine, db, live, cek_kualitas=cek_kualitas)
        latencies.append(res["latensi_detik"])

        if res["pred_nama"] == nama_asli:
            tp += 1
            hasil_label = "TP (Benar)"
        elif res["pred_nama"] is None:
            fn += 1
            hasil_label = "FN (Salah Tolak)"
        else:
            fn += 1
            fp_mis_id += 1
            hasil_label = f"FN+FP (Salah Kenal -> {res['pred_nama']})"

        detail_rows.append([
            "terdaftar", nama_asli, path_img.name, res["pred_nama"] or "-",
            f"{res['skor']:.4f}", f"{res['live_skor']:.4f}" if res["live_skor"] is not None else "-",
            f"{res['latensi_detik']:.4f}", hasil_label
        ])

    # 2) Pengujian Kelompok Tidak Terdaftar (Impostor / Open-Set)
    tn = 0  # Ditolak dengan benar sebagai Unknown
    fp_impostor = 0  # Orang asing keliru diterima sebagai siswa (False Acceptance)

    for nama_asing, path_img in probe_tidak_terdaftar:
        img = cv2.imread(str(path_img))
        res = run_single_cycle(img, engine, db, live, cek_kualitas=cek_kualitas)
        latencies.append(res["latensi_detik"])

        if res["pred_nama"] is None:
            tn += 1
            hasil_label = "TN (Ditolak Benar)"
        else:
            fp_impostor += 1
            hasil_label = f"FP (Salah Terima -> {res['pred_nama']})"

        detail_rows.append([
            "tidak_terdaftar", nama_asing, path_img.name, res["pred_nama"] or "-",
            f"{res['skor']:.4f}", f"{res['live_skor']:.4f}" if res["live_skor"] is not None else "-",
            f"{res['latensi_detik']:.4f}", hasil_label
        ])

    # 3) Pengujian Serangan Pemalsuan (Anti-Spoofing)
    spoof_total = len(probe_spoof)
    spoof_rejected = 0

    for jenis_spoof, path_img in probe_spoof:
        img = cv2.imread(str(path_img))
        res = run_single_cycle(img, engine, db, live, cek_kualitas=cek_kualitas)
        latencies.append(res["latensi_detik"])

        # Ditolak baik oleh liveness maupun oleh ambang kemiripan
        if res["status"] == "spoof_rejected" or res["pred_nama"] is None:
            spoof_rejected += 1
            hasil_label = "Ditolak (Aman)"
        else:
            hasil_label = f"Lolos sebagai {res['pred_nama']} (Gagal)"

        detail_rows.append([
            "spoof", jenis_spoof, path_img.name, res["pred_nama"] or "-",
            f"{res['skor']:.4f}", f"{res['live_skor']:.4f}" if res["live_skor"] is not None else "-",
            f"{res['latensi_detik']:.4f}", hasil_label
        ])

    # C. Kalkulasi Metrik Evaluasi
    total_fp = fp_impostor + fp_mis_id
    total_genuine = tp + fn
    total_impostor = tn + fp_impostor
    total_ident = total_genuine + total_impostor

    precision = (tp / (tp + total_fp) * 100.0) if (tp + total_fp) > 0 else 0.0
    recall = (tp / total_genuine * 100.0) if total_genuine > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    akurasi_ident = ((tp + tn) / total_ident * 100.0) if total_ident > 0 else 0.0
    frr = 100.0 - recall
    far = (fp_impostor / total_impostor * 100.0) if total_impostor > 0 else 0.0
    spoof_reject_rate = (spoof_rejected / spoof_total * 100.0) if spoof_total > 0 else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    ringkasan = {
        "konfigurasi": {
            "match_thresh": MATCH_THRESH,
            "match_margin": MATCH_MARGIN,
            "sampel_galeri_per_orang": n_galeri,
            "liveness_aktif": bool(live and live.enabled),
        },
        "komposisi_data": {
            "individu_terdaftar": len(set(gal_names)),
            "sampel_uji_terdaftar": total_genuine,
            "individu_tidak_terdaftar": len(set(x[0] for x in probe_tidak_terdaftar)),
            "sampel_uji_tidak_terdaftar": total_impostor,
            "sampel_uji_spoof": spoof_total,
        },
        "confusion_counts": {
            "TP": tp,
            "TN": tn,
            "FP_impostor": fp_impostor,
            "FP_mis_id": fp_mis_id,
            "FN": fn,
            "Spoof_Rejected": spoof_rejected,
        },
        "metrik": {
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1-Score (%)": round(f1_score, 2),
            "Akurasi Identifikasi (%)": round(akurasi_ident, 2),
            "Kecepatan Inferensi per Siklus (detik)": round(avg_latency, 4),
            "FRR (%)": round(frr, 2),
            "FAR (%)": round(far, 2),
            "Penolakan Wajah Palsu (%)": round(spoof_reject_rate, 2),
        },
    }

    # Simpan CSV detail & JSON ringkasan
    with open(REPORT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kategori", "subjek_asli", "file", "prediksi", "skor_cos", "skor_live", "latensi_s", "hasil"])
        w.writerows(detail_rows)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(ringkasan, f, ensure_ascii=False, indent=2)

    # Cetak Tabel Laporan ke Terminal
    print("\n" + "=" * 72)
    print("RINGKASAN HASIL PENGUJIAN MODEL PRESENSI WAJAH (YUNET + SFACE)")
    print("=" * 72)
    print(f"Subjek terdaftar : {ringkasan['komposisi_data']['individu_terdaftar']} orang "
          f"({total_genuine} sampel probe)")
    print(f"Subjek asing     : {ringkasan['komposisi_data']['individu_tidak_terdaftar']} orang "
          f"({total_impostor} sampel impostor)")
    print(f"Sampel spoofing  : {spoof_total} sampel (ditolak: {spoof_rejected})")
    print(f"Rincian TP/TN/FP/FN : TP={tp}, TN={tn}, FP={total_fp}, FN={fn}")
    print("-" * 72)
    print(f"{'Parameter':<40} {'Target':<14} {'Hasil':<10} {'Status'}")
    print("-" * 72)

    rows_print = [
        ("Precision", ">= 90%", f"{precision:.2f}%", precision >= 90.0),
        ("Recall", ">= 85%", f"{recall:.2f}%", recall >= 85.0),
        ("F1-Score", ">= 85%", f"{f1_score:.2f}%", f1_score >= 85.0),
        ("Akurasi identifikasi", ">= 80%", f"{akurasi_ident:.2f}%", akurasi_ident >= 80.0),
        ("Kecepatan inferensi per siklus", "< 2 detik", f"{avg_latency:.2f} dtk", avg_latency < 2.0),
        ("FRR (= 100% - Recall)", "<= 15%", f"{frr:.2f}%", frr <= 15.0),
        ("FAR", "<= 5%", f"{far:.2f}%", far <= 5.0),
        ("Penolakan wajah palsu (foto/layar)", ">= 80%", f"{spoof_reject_rate:.2f}%", spoof_reject_rate >= 80.0),
    ]
    for param, target, hasil, ok in rows_print:
        print(f"{param:<40} {target:<14} {hasil:<10} {'Memenuhi' if ok else 'Tidak Memenuhi'}")
    print("=" * 72)
    print(f"Detail per sampel disimpan di : {REPORT_CSV.name}")
    print(f"Ringkasan JSON disimpan di    : {REPORT_JSON.name}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluasi Kuantitatif Model Presensi Wajah (YuNet + SFace)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_col = sub.add_parser("collect", help="ambil sampel gambar uji dari kamera")
    p_col.add_argument("--kategori", required=True, choices=["terdaftar", "tidak_terdaftar", "spoof"])
    p_col.add_argument("--nama", required=True, help="nama subjek atau jenis serangan (mis. Foto_Cetak)")
    p_col.add_argument("--jumlah", type=int, default=12, help="jumlah gambar yang diambil")
    p_col.add_argument("--kamera", type=int, default=0, help="index kamera")

    p_run = sub.add_parser("run", help="jalankan pengujian dan hitung seluruh metrik")
    p_run.add_argument("--galeri", type=int, default=8, help="jumlah sampel pertama per orang untuk galeri")
    p_run.add_argument("--cek-kualitas", action="store_true", help="aktifkan gerbang kualitas saat evaluasi")

    args = ap.parse_args()
    if args.cmd == "collect":
        collect_samples(args.kategori, args.nama.strip(), args.jumlah, args.kamera)
    elif args.cmd == "run":
        evaluate_dataset(n_galeri=args.galeri, cek_kualitas=args.cek_kualitas)


if __name__ == "__main__":
    main()