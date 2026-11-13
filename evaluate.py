#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - EVALUASI KUANTITATIF sistem absensi wajah (bahan Bab 4 skripsi).

Semua angka dihitung dengan pipeline yang SAMA dengan sistem (YuNet -> SFace -> aturan pencocokan
Database.match: rata-rata 2 sampel terbaik + ambang + margin; liveness MiniFASNet + tantangan afin).
Keluaran: folder hasil_evaluasi/<waktu>_<jenis>/ berisi laporan.md, metrik.json, skor *.csv, grafik *.png.

Perintah
  python evaluate.py pengenalan --data dataset/            # verifikasi 1:1 + identifikasi 1:N (open-set)
  python evaluate.py liveness   --data dataset_liveness/   # APCER / BPCER / ACER (ISO/IEC 30107-3)
  python evaluate.py kinerja    --data dataset/            # latensi per tahap, FPS, RAM, skala database
  python evaluate.py lapangan   --csv uji_lapangan.csv     # rekap uji langsung (black-box) di lokasi
  python evaluate.py templat-lapangan                      # buat contoh uji_lapangan.csv
  python evaluate.py kumpulkan  --label Budi --jumlah 30   # ambil foto uji dari kamera
  python evaluate.py kumpulkan  --label Budi --tipe cetak --liveness   # foto serangan (foto cetak)
  python evaluate.py kumpulkan  --label Budi --video 6 --liveness      # video 6 dtk (uji tantangan aktif)

Struktur data
  dataset/<nama_orang>/*.jpg                     minimal 2 foto per orang (disarankan 5-20). Format LFW cocok.
  dataset_liveness/asli/**/*.jpg|mp4             wajah asli (bona fide)
  dataset_liveness/serangan/<jenis>/**/*.jpg|mp4  serangan per jenis, mis. cetak/, layar/, video/, topeng/

Metrik
  Verifikasi (semua pasangan foto): FAR, FRR, EER, AUC ROC, TAR@FAR=1% & 0,1%, d-prime.
  Identifikasi (protokol sistem, diulang beberapa kali): sebagian orang DIDAFTARKAN dengan k foto,
  sisanya menjadi "orang tak dikenal". Rank-1, TPIR, FNIR, misidentifikasi, FPIR, akurasi, presisi,
  recall, F1, dan ambang yang disarankan pada target FPIR.
  Liveness: APCER per jenis serangan (maks = APCER sistem), BPCER, ACER, EER, BPCER@APCER 5%/10%.
  Untuk video: keputusan per video (rata-rata jendela frame) dan uji tantangan aktif (rentang u/v).

CATATAN ETIKA: foto uji adalah data biometrik. Minta persetujuan peserta uji, simpan terpisah,
dan hapus setelah penelitian selesai (UU No. 27/2022).
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import numpy as np

import config
import main
from liveness import CH_SCALE, LandmarkRefiner, PassiveLiveness, affine_coords, challenge_passed

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DETECT_W = 640
VIDEO_MAX_FRAMES = 40

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ============================== utilitas ==============================
def out_dir(base, kind):
    d = os.path.join(base, f"{datetime.now():%Y%m%d_%H%M%S}_{kind}")
    os.makedirs(d, exist_ok=True)
    return d


def pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}%"


def num(x, d=4):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def list_media(root):
    """[(path, relatif)] semua gambar/video di bawah root, urut."""
    out = []
    for d, _, files in os.walk(root):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT or ext in VID_EXT:
                p = os.path.join(d, f)
                out.append((p, os.path.relpath(p, root)))
    return sorted(out, key=lambda x: x[1])


def read_frames(path, max_frames=VIDEO_MAX_FRAMES):
    """Gambar -> [img]; video -> hingga max_frames frame berurutan (diambil merata)."""
    if os.path.splitext(path)[1].lower() in IMG_EXT:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)   # aman untuk path unicode
        return [img] if img is not None else []
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, n // max_frames) if n > 0 else 1
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


class Extractor:
    """Deteksi wajah terbesar + embedding + skor liveness, persis seperti pipeline sistem."""

    def __init__(self, need_live=False, need_refine=False):
        main.cv2.setNumThreads(main.CV_THREADS)
        self.engine = main.FaceEngine(*main.ensure_models())
        self.live = PassiveLiveness() if need_live else None
        self.refiner = LandmarkRefiner(self.engine.yunet_path) if need_refine else None

    def detect(self, img):
        h, w = img.shape[:2]
        sw = min(w, DETECT_W)
        sh = max(1, round(h * sw / w))
        small = img if sw == w else cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        faces = self.engine.detect(small)
        if not faces:
            return None, None
        f = max(faces, key=lambda r: r[2] * r[3])
        return f, (w / sw, h / sh)

    def embedding(self, img):
        f, scale = self.detect(img)
        return None if f is None else self.engine.embed(img, f, scale)

    def box_lm(self, img):
        f, scale = self.detect(img)
        if f is None:
            return None, None
        k = np.array([*scale, *scale], np.float32)
        return f[:4] * k, f[4:14].reshape(5, 2) * np.asarray(scale, np.float32)


def roc_points(gen, imp):
    """(thr, FAR, FRR) untuk semua ambang unik (tepat, tanpa grid)."""
    s = np.concatenate([gen, imp])
    y = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]          # ambang = setiap skor unik (>= thr diterima)
    thr = s[idx]
    far = fp[idx] / max(len(imp), 1)
    frr = 1 - tp[idx] / max(len(gen), 1)
    thr = np.r_[np.inf, thr]
    far = np.r_[0.0, far]
    frr = np.r_[1.0, frr]
    return thr, far, frr


def eer(thr, far, frr):
    i = int(np.argmin(np.abs(far - frr)))
    # interpolasi linear antara dua titik di sekitar persilangan
    j = i + 1 if i + 1 < len(far) and (far[i] - frr[i]) * (far[i + 1] - frr[i + 1]) <= 0 else max(i - 1, 0)
    a, b = (far[i] - frr[i]), (far[j] - frr[j])
    if a == b:
        return float((far[i] + frr[i]) / 2), float(thr[i])
    w = a / (a - b)
    e = far[i] + w * (far[j] - far[i])
    t = thr[i] if not np.isfinite(thr[j]) else thr[i] + w * (thr[j] - thr[i])
    return float(e), float(t)


def auc(far, tar):
    o = np.argsort(far, kind="mergesort")
    return float(np.trapezoid(tar[o], far[o]) if hasattr(np, "trapezoid") else np.trapz(tar[o], far[o]))


def tar_at_far(thr, far, frr, target):
    ok = np.where(far <= target)[0]
    if not len(ok):
        return None, None
    i = ok[np.argmin(frr[ok])]
    return float(1 - frr[i]), float(thr[i])


def rates_at(gen, imp, t):
    return float(np.mean(imp >= t)) if len(imp) else None, float(np.mean(gen < t)) if len(gen) else None


# ============================== 1. PENGENALAN ==============================
def eval_recognition(a):
    od = out_dir(a.keluar, "pengenalan")
    people = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    ex = Extractor()
    embs, labels, paths, fta = [], [], [], []
    t0 = time.time()
    for p in people:
        for path, rel in list_media(os.path.join(a.data, p)):
            if os.path.splitext(path)[1].lower() not in IMG_EXT:
                continue
            imgs = read_frames(path)
            e = ex.embedding(imgs[0]) if imgs else None
            if e is None:
                fta.append(os.path.join(p, rel))
                continue
            embs.append(e)
            labels.append(p)
            paths.append(os.path.join(p, rel))
    print(f"  ekstraksi: {len(embs)} embedding, {len(fta)} gagal deteksi ({time.time() - t0:.1f} dtk)")
    if len(set(labels)) < 2:
        raise SystemExit("Butuh minimal 2 orang dengan wajah terdeteksi.")
    E = np.stack(embs)
    L = np.array(labels)
    n_total = len(embs) + len(fta)

    # ---- verifikasi 1:1 (semua pasangan) ----
    S = E @ E.T
    iu = np.triu_indices(len(E), 1)
    same = (L[:, None] == L[None, :])[iu]
    sc = S[iu]
    gen, imp = sc[same], sc[~same]
    if not len(gen):
        raise SystemExit("Tidak ada pasangan genuine: setiap orang butuh >= 2 foto.")
    thr, far, frr = roc_points(gen, imp)
    e_rate, e_thr = eer(thr, far, frr)
    far_s, frr_s = rates_at(gen, imp, main.MATCH_THRESH)
    t1, th1 = tar_at_far(thr, far, frr, 0.01)
    t01, th01 = tar_at_far(thr, far, frr, 0.001)
    dprime = abs(gen.mean() - imp.mean()) / np.sqrt((gen.var() + imp.var()) / 2)
    ver = {"pasangan_genuine": int(len(gen)), "pasangan_impostor": int(len(imp)),
           "rerata_genuine": float(gen.mean()), "rerata_impostor": float(imp.mean()),
           "eer": e_rate, "ambang_eer": e_thr, "auc": auc(far, 1 - frr), "d_prime": float(dprime),
           "tar_far_1pct": t1, "ambang_far_1pct": th1, "tar_far_0_1pct": t01, "ambang_far_0_1pct": th01,
           "ambang_sistem": main.MATCH_THRESH, "far_ambang_sistem": far_s, "frr_ambang_sistem": frr_s}
    with open(os.path.join(od, "skor_verifikasi.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jenis", "skor"])
        w.writerows([("genuine", f"{x:.5f}") for x in gen])
        w.writerows([("impostor", f"{x:.5f}") for x in (imp if len(imp) <= 200000 else
                                                         np.random.RandomState(0).choice(imp, 200000, False))])

    # ---- identifikasi 1:N open-set, protokol sistem ----
    by = defaultdict(list)
    for i, l in enumerate(labels):
        by[l].append(i)
    elig = sorted(p for p, ix in by.items() if len(ix) >= a.daftar + 1)
    if len(elig) < 2:
        raise SystemExit(f"Butuh >= 2 orang dengan >= {a.daftar + 1} foto (atur --daftar).")
    grid = np.round(np.arange(0.0, 1.0001, 0.005), 3)
    rows_probe, runs = [], []
    agg_fnir = np.zeros(len(grid))
    agg_fpir = np.zeros(len(grid))
    n_known_tot = n_unk_tot = 0
    for r in range(a.ulang):
        rng = random.Random(a.seed + r)
        ppl = elig[:]
        rng.shuffle(ppl)
        n_unk = int(round(len(ppl) * a.tak_dikenal)) if len(ppl) >= 3 else 0
        unknown, known = set(ppl[:n_unk]), ppl[n_unk:]
        gal, probes = [], []
        for p in known:
            ix = by[p][:]
            rng.shuffle(ix)
            gal += ix[:a.daftar]
            probes += [(i, p) for i in ix[a.daftar:]]
        for p in unknown:
            probes += [(i, None) for i in by[p]]
        db = main.Database.from_arrays([labels[i] for i in gal], [""] * len(gal), E[gal])
        res = Counter()
        best_sc, best_ok, is_known = [], [], []
        for i, truth in probes:
            sc_all = db.person_scores(E[i])
            order = np.argsort(-sc_all)
            top, s1 = db.uniq[order[0]], float(sc_all[order[0]])
            margin = s1 - float(sc_all[order[1]]) if len(order) > 1 else 1.0
            accept = s1 >= main.MATCH_THRESH and (margin >= main.MATCH_MARGIN)
            pred = top if accept else None
            if truth is not None:
                res["rank1_benar"] += top == truth
                res["rank5_benar"] += truth in [db.uniq[j] for j in order[:5]]
                res["benar"] += pred == truth
                res["salah_identitas"] += pred is not None and pred != truth
                res["ditolak"] += pred is None
            else:
                res["fp"] += pred is not None
                res["tn"] += pred is None
            best_sc.append(s1 if margin >= main.MATCH_MARGIN else -1.0)   # aturan margin tetap dihitung
            best_ok.append(top == truth)
            is_known.append(truth is not None)
            rows_probe.append([r, paths[i], truth or "(tak dikenal)", top, f"{s1:.4f}", f"{margin:.4f}",
                               pred or "unknown", "benar" if (pred == truth) else "salah"])
        bs, bo, kn = np.array(best_sc), np.array(best_ok), np.array(is_known)
        nk, nu = int(kn.sum()), int((~kn).sum())
        for gi, t in enumerate(grid):
            acc_ = bs >= t
            agg_fnir[gi] += np.sum(kn & ~(acc_ & bo))
            agg_fpir[gi] += np.sum(~kn & acc_)
        n_known_tot += nk
        n_unk_tot += nu
        runs.append({"orang_terdaftar": len(known), "orang_tak_dikenal": len(unknown),
                     "probe_dikenal": nk, "probe_tak_dikenal": nu, **dict(res)})

    tot = Counter()
    for x in runs:
        tot.update({k: v for k, v in x.items() if isinstance(v, int)})
    nk, nu = tot["probe_dikenal"], tot["probe_tak_dikenal"]
    acc_all = (tot["benar"] + tot["tn"]) / max(nk + nu, 1)
    accepts = tot["benar"] + tot["salah_identitas"] + tot["fp"]
    prec = tot["benar"] / accepts if accepts else None
    rec = tot["benar"] / nk if nk else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    fnir_c = agg_fnir / max(n_known_tot, 1)
    fpir_c = agg_fpir / max(n_unk_tot, 1) if n_unk_tot else np.full(len(grid), np.nan)
    rec_thr = None
    if n_unk_tot:
        ok = np.where(fpir_c <= a.target_fpir)[0]
        if len(ok):
            gi = ok[np.argmin(fnir_c[ok])]
            rec_thr = {"ambang": float(grid[gi]), "fpir": float(fpir_c[gi]), "fnir": float(fnir_c[gi])}
    ident = {"ulang": a.ulang, "foto_daftar_per_orang": a.daftar, "porsi_tak_dikenal": a.tak_dikenal,
             "probe_dikenal": nk, "probe_tak_dikenal": nu,
             "rank1": tot["rank1_benar"] / nk if nk else None, "rank5": tot["rank5_benar"] / nk if nk else None,
             "tpir": tot["benar"] / nk if nk else None, "fnir": (tot["ditolak"] + tot["salah_identitas"]) / nk if nk else None,
             "salah_identitas": tot["salah_identitas"] / nk if nk else None, "ditolak": tot["ditolak"] / nk if nk else None,
             "fpir": tot["fp"] / nu if nu else None, "akurasi": acc_all, "presisi": prec, "recall": rec, "f1": f1,
             "ambang_disarankan": rec_thr, "target_fpir": a.target_fpir, "per_ulangan": runs}
    with open(os.path.join(od, "hasil_identifikasi.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ulangan", "foto", "kebenaran", "kandidat_teratas", "skor", "margin", "keputusan", "hasil"])
        w.writerows(rows_probe)

    # ---- grafik ----
    figs = []
    if plt:
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(-0.2, 1, 61)
        ax.hist(imp, bins, alpha=.6, density=True, label=f"impostor (n={len(imp)})", color="#d62728")
        ax.hist(gen, bins, alpha=.6, density=True, label=f"genuine (n={len(gen)})", color="#2ca02c")
        ax.axvline(main.MATCH_THRESH, color="k", ls="--", label=f"ambang sistem {main.MATCH_THRESH:.2f}")
        ax.axvline(e_thr, color="#1f77b4", ls=":", label=f"ambang EER {e_thr:.3f}")
        ax.set_xlabel("skor kemiripan kosinus SFace")
        ax.set_ylabel("kepadatan")
        ax.set_title("Distribusi skor genuine vs impostor")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(od, "distribusi_skor.png"), dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(np.clip(far, 1e-5, 1), 1 - frr, color="#1f77b4")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_xlabel("FAR (log)")
        ax.set_ylabel("TAR = 1 - FRR")
        ax.set_title(f"ROC verifikasi (AUC {ver['auc']:.4f}, EER {100 * e_rate:.2f}%)")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(od, "roc.png"), dpi=150)
        plt.close(fig)
        figs += ["distribusi_skor.png", "roc.png"]
        if n_unk_tot:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(grid, fpir_c, label="FPIR (tak dikenal diterima)", color="#d62728")
            ax.plot(grid, fnir_c, label="FNIR (terdaftar gagal dikenali)", color="#1f77b4")
            ax.axvline(main.MATCH_THRESH, color="k", ls="--", label="ambang sistem")
            if rec_thr:
                ax.axvline(rec_thr["ambang"], color="#2ca02c", ls=":", label=f"disarankan {rec_thr['ambang']:.3f}")
            ax.set_xlabel("ambang skor orang")
            ax.set_ylabel("laju galat")
            ax.set_title("Identifikasi open-set vs ambang")
            ax.legend()
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(os.path.join(od, "openset_ambang.png"), dpi=150)
            plt.close(fig)
            figs.append("openset_ambang.png")

    m = {"dataset": os.path.abspath(a.data), "orang": len(set(labels)), "foto": n_total,
         "gagal_deteksi": len(fta), "fta": len(fta) / max(n_total, 1), "verifikasi": ver, "identifikasi": ident}
    with open(os.path.join(od, "metrik.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

    rep = [f"# Evaluasi Pengenalan Wajah\n\nDataset: `{a.data}` - {m['orang']} orang, {n_total} foto, "
           f"gagal deteksi (FTA) {len(fta)} ({pct(m['fta'])}). Model: YuNet + SFace (CPU). "
           f"Dibuat {datetime.now():%Y-%m-%d %H:%M}.\n",
           "## 1. Verifikasi 1:1 (semua pasangan foto)\n",
           md_table(["Metrik", "Nilai"], [
               ["Pasangan genuine / impostor", f"{len(gen)} / {len(imp)}"],
               ["Rerata skor genuine / impostor", f"{gen.mean():.4f} / {imp.mean():.4f}"],
               ["d-prime (keterpisahan)", num(dprime, 3)],
               ["EER (ambang)", f"{pct(e_rate)} ({e_thr:.4f})"],
               ["AUC ROC", num(ver['auc'])],
               ["TAR @ FAR=1% (ambang)", f"{pct(t1)} ({num(th1)})"],
               ["TAR @ FAR=0,1% (ambang)", f"{pct(t01)} ({num(th01)})"],
               [f"FAR / FRR pada ambang sistem {main.MATCH_THRESH:.2f}", f"{pct(far_s)} / {pct(frr_s)}"],
           ]), "",
           "Catatan: verifikasi memakai 1 foto vs 1 foto; sistem memakai rata-rata 2 sampel terbaik per orang "
           "(bagian 2), sehingga kinerja sistem biasanya lebih baik daripada angka 1:1.\n",
           f"## 2. Identifikasi 1:N open-set (protokol sistem, {a.ulang}x ulangan acak)\n",
           f"Setiap ulangan: {int(100 * a.tak_dikenal)}% orang TIDAK didaftarkan (penyusup); orang lain didaftarkan "
           f"dengan {a.daftar} foto acak, sisanya menjadi probe. Keputusan memakai aturan sistem: skor >= "
           f"{main.MATCH_THRESH:.2f} dan margin >= {main.MATCH_MARGIN:.2f}.\n",
           md_table(["Metrik", "Nilai", "Keterangan"], [
               ["Rank-1 (closed-set)", pct(ident["rank1"]), "kandidat teratas benar, tanpa ambang"],
               ["Rank-5", pct(ident["rank5"]), ""],
               ["TPIR", pct(ident["tpir"]), "terdaftar dikenali dengan benar"],
               ["FNIR", pct(ident["fnir"]), "terdaftar ditolak atau salah identitas"],
               ["- salah identitas", pct(ident["salah_identitas"]), "dicatat sebagai orang lain (paling berbahaya)"],
               ["- ditolak", pct(ident["ditolak"]), "dianggap unknown"],
               ["FPIR", pct(ident["fpir"]), "penyusup diterima sebagai orang terdaftar"],
               ["Akurasi", pct(acc_all), "(benar + penyusup ditolak) / semua probe"],
               ["Presisi / Recall / F1", f"{pct(prec)} / {pct(rec)} / {pct(f1)}", "kelas positif = identitas benar"],
               ["Probe dikenal / tak dikenal", f"{nk} / {nu}", ""],
           ]), ""]
    if rec_thr:
        rep.append(f"**Ambang disarankan** (FPIR <= {pct(a.target_fpir)}): `match_thresh = {rec_thr['ambang']:.3f}` "
                   f"-> FPIR {pct(rec_thr['fpir'])}, FNIR {pct(rec_thr['fnir'])}. Ubah lewat config.json bila sesuai.\n")
    rep += [f"![{f}]({f})\n" for f in figs]
    rep.append("File pendukung: `metrik.json`, `skor_verifikasi.csv`, `hasil_identifikasi.csv`.")
    with open(os.path.join(od, "laporan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print(f"EER {pct(e_rate)} | Rank-1 {pct(ident['rank1'])} | TPIR {pct(ident['tpir'])} | FPIR {pct(ident['fpir'])}"
          f" | F1 {pct(f1)}\nLaporan: {os.path.join(od, 'laporan.md')}")
    return m, od


# ============================== 2. LIVENESS ==============================
def _liveness_label(rel):
    parts = rel.replace("\\", "/").split("/")
    top = parts[0].lower()
    if top in ("asli", "live", "real", "bonafide", "bona_fide"):
        return "asli", None
    if top in ("serangan", "spoof", "attack", "palsu") and len(parts) > 2:
        return "serangan", parts[1]
    return "serangan", top if top not in ("serangan", "spoof", "attack", "palsu") else "serangan"


def eval_liveness(a):
    od = out_dir(a.keluar, "liveness")
    ex = Extractor(need_live=True, need_refine=True)
    th = a.ambang if a.ambang is not None else config.LIVENESS_THRESH
    items = list_media(a.data)
    if not items:
        raise SystemExit(f"Tidak ada gambar/video di {a.data}")
    frame_rows, media_rows, fta = [], [], 0
    t0 = time.time()
    for path, rel in items:
        cls, kind = _liveness_label(rel)
        is_vid = os.path.splitext(path)[1].lower() in VID_EXT
        scores, us, vs = [], [], []
        for img in read_frames(path):
            box, lm = ex.box_lm(img)
            if box is None:
                fta += 1
                continue
            p = ex.live.probs(img, box)
            if p is None:
                fta += 1
                continue
            scores.append(float(p[1]))
            frame_rows.append([rel, cls, kind or "", f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
            if is_vid:
                rl = ex.refiner.refine(img, box)
                uv = affine_coords(rl) if rl is not None else None
                if uv:
                    us.append(uv[0])
                    vs.append(uv[1])
        if is_vid and scores:
            win = scores[:config.LIVENESS_WINDOW]
            T = config.TANTANGAN_AMBANG
            media_rows.append({"file": rel, "kelas": cls, "jenis": kind or "", "frame": len(scores),
                               "skor_jendela": float(np.mean(win)), "rentang_u": _two_sided(us),
                               "rentang_v": _two_sided(vs),
                               "lolos_aktif": bool(challenge_passed(us, T * CH_SCALE["geleng"])[0]
                                                   or challenge_passed(vs, T * CH_SCALE["angguk"])[0])})
    print(f"  {len(items)} file, {len(frame_rows)} frame berwajah ({time.time() - t0:.1f} dtk)")
    if not frame_rows:
        raise SystemExit("Tidak ada wajah terdeteksi.")
    with open(os.path.join(od, "skor_liveness.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "kelas", "jenis_serangan", "p_kertas", "p_asli", "p_layar"])
        w.writerows(frame_rows)

    cls = np.array([r[1] for r in frame_rows])
    kinds = np.array([r[2] for r in frame_rows])
    s = np.array([float(r[4]) for r in frame_rows])
    bona, atk = s[cls == "asli"], s[cls == "serangan"]

    def pad_metrics(bona, atk_by_kind, t):
        bpcer = float(np.mean(bona < t)) if len(bona) else None
        ap = {k: float(np.mean(v >= t)) for k, v in atk_by_kind.items() if len(v)}
        apcer = max(ap.values()) if ap else None
        acer = (apcer + bpcer) / 2 if apcer is not None and bpcer is not None else None
        return bpcer, ap, apcer, acer

    atk_by = {k: s[(cls == "serangan") & (kinds == k)] for k in sorted(set(kinds[cls == "serangan"]))}
    bpcer, ap, apcer, acer = pad_metrics(bona, atk_by, th)
    res = {"ambang": th, "frame_asli": int(len(bona)), "frame_serangan": int(len(atk)), "gagal_deteksi": fta,
           "bpcer": bpcer, "apcer_per_jenis": ap, "apcer": apcer, "acer": acer}
    curve = None
    if len(bona) and len(atk):
        thr, far, frr = roc_points(bona, atk)      # "far" = APCER (serangan lolos), "frr" = BPCER
        e_rate, e_thr = eer(thr, far, frr)
        b20, t20 = tar_at_far(thr, far, frr, 0.05)
        b10, t10 = tar_at_far(thr, far, frr, 0.10)
        res.update({"eer": e_rate, "ambang_eer": e_thr, "auc": auc(far, 1 - frr),
                    "bpcer_at_apcer_5pct": None if b20 is None else 1 - b20, "ambang_apcer_5pct": t20,
                    "bpcer_at_apcer_10pct": None if b10 is None else 1 - b10, "ambang_apcer_10pct": t10})
        curve = (far, frr)

    vid = None
    if media_rows:
        vb = np.array([m["skor_jendela"] for m in media_rows if m["kelas"] == "asli"])
        va = {k: np.array([m["skor_jendela"] for m in media_rows if m["kelas"] == "serangan" and m["jenis"] == k])
              for k in sorted({m["jenis"] for m in media_rows if m["kelas"] == "serangan"})}
        vbp, vap, vapc, vac = pad_metrics(vb, va, th)
        ct = config.TANTANGAN_AMBANG
        act_b = [m["lolos_aktif"] for m in media_rows if m["kelas"] == "asli"]
        act_a = {k: [m["lolos_aktif"] for m in media_rows
                     if m["kelas"] == "serangan" and m["jenis"] == k] for k in va}
        vid = {"video_asli": int(len(vb)), "video_serangan": int(sum(len(v) for v in va.values())),
               "pasif": {"bpcer": vbp, "apcer_per_jenis": vap, "apcer": vapc, "acer": vac},
               "aktif": {"ambang": ct, "lolos_asli": float(np.mean(act_b)) if act_b else None,
                         "lolos_serangan_per_jenis": {k: float(np.mean(v)) for k, v in act_a.items() if v}},
               "per_video": media_rows}
        res["video"] = vid
        with open(os.path.join(od, "per_video.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "kelas", "jenis", "frame", "skor_jendela", "rentang_u", "rentang_v", "lolos_aktif"])
            for m in media_rows:
                w.writerow([m["file"], m["kelas"], m["jenis"], m["frame"], f"{m['skor_jendela']:.4f}",
                            f"{m['rentang_u']:.4f}", f"{m['rentang_v']:.4f}", int(m["lolos_aktif"])])

    figs = []
    if plt:
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, 1, 41)
        if len(bona):
            ax.hist(bona, bins, alpha=.6, label=f"asli (n={len(bona)})", color="#2ca02c")
        for k, v in atk_by.items():
            ax.hist(v, bins, alpha=.5, label=f"serangan: {k} (n={len(v)})")
        ax.axvline(th, color="k", ls="--", label=f"ambang {th:.2f}")
        ax.set_xlabel("probabilitas wajah asli (rata-rata MiniFASNetV2 + V1SE)")
        ax.set_ylabel("jumlah frame")
        ax.legend()
        ax.set_title("Distribusi skor liveness")
        fig.tight_layout()
        fig.savefig(os.path.join(od, "distribusi_liveness.png"), dpi=150)
        plt.close(fig)
        figs.append("distribusi_liveness.png")
        if curve:
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.plot(curve[0], curve[1], color="#1f77b4")
            ax.set_xlabel("APCER (serangan diterima)")
            ax.set_ylabel("BPCER (wajah asli ditolak)")
            ax.set_title(f"Kurva DET liveness (EER {100 * res['eer']:.2f}%)")
            ax.grid(alpha=.3)
            fig.tight_layout()
            fig.savefig(os.path.join(od, "det_liveness.png"), dpi=150)
            plt.close(fig)
            figs.append("det_liveness.png")

    with open(os.path.join(od, "metrik.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    rows = [["Frame asli / serangan", f"{len(bona)} / {len(atk)}"], ["Gagal deteksi wajah", fta],
            [f"BPCER (ambang {th:.2f})", pct(bpcer)]]
    rows += [[f"APCER - {k} (n={len(atk_by[k])})", pct(v)] for k, v in ap.items()]
    rows += [["APCER sistem (maks per jenis)", pct(apcer)], ["ACER = (APCER + BPCER) / 2", pct(acer)]]
    if "eer" in res:
        rows += [["EER (ambang)", f"{pct(res['eer'])} ({res['ambang_eer']:.4f})"], ["AUC", num(res["auc"])],
                 ["BPCER @ APCER=5% (ambang)", f"{pct(res['bpcer_at_apcer_5pct'])} ({num(res['ambang_apcer_5pct'])})"],
                 ["BPCER @ APCER=10% (ambang)", f"{pct(res['bpcer_at_apcer_10pct'])} ({num(res['ambang_apcer_10pct'])})"]]
    rep = [f"# Evaluasi Liveness / Anti-Spoofing\n\nDataset: `{a.data}` - {len(items)} file. Metrik mengikuti "
           f"ISO/IEC 30107-3. Dibuat {datetime.now():%Y-%m-%d %H:%M}.\n",
           "## 1. Tingkat frame (model pasif)\n", md_table(["Metrik", "Nilai"], rows), ""]
    if vid:
        vr = [["Video asli / serangan", f"{vid['video_asli']} / {vid['video_serangan']}"],
              [f"Pasif - BPCER (rata-rata {config.LIVENESS_WINDOW} frame pertama)", pct(vid["pasif"]["bpcer"])]]
        vr += [[f"Pasif - APCER {k}", pct(v)] for k, v in vid["pasif"]["apcer_per_jenis"].items()]
        vr += [["Pasif - ACER", pct(vid["pasif"]["acer"])],
               [f"Aktif - video asli lolos tantangan (ambang {config.TANTANGAN_AMBANG})",
                pct(vid["aktif"]["lolos_asli"])]]
        vr += [[f"Aktif - serangan {k} lolos tantangan", pct(v)] for k, v in vid["aktif"]["lolos_serangan_per_jenis"].items()]
        rep += ["## 2. Tingkat video (keputusan sistem)\n", md_table(["Metrik", "Nilai"], vr), "",
                "Uji aktif: video asli harus berisi gerakan geleng/angguk sesuai protokol; serangan foto datar "
                "seharusnya tidak mengubah koordinat afin hidung (u, v).\n"]
    rep += [f"![{f}]({f})\n" for f in figs]
    rep.append("File pendukung: `metrik.json`, `skor_liveness.csv`" + (", `per_video.csv`." if vid else "."))
    with open(os.path.join(od, "laporan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    print(f"BPCER {pct(bpcer)} | APCER {pct(apcer)} | ACER {pct(acer)}" +
          (f" | EER {pct(res['eer'])}" if "eer" in res else "") + f"\nLaporan: {os.path.join(od, 'laporan.md')}")
    return res, od


# ============================== 3. KINERJA ==============================
def _two_sided(seq):
    """Rentang dua arah (maks-awal)+(awal-min) setelah median 3, sama seperti challenge_passed."""
    if len(seq) < 4:
        return 0.0
    sm = np.array([np.median(seq[max(0, i - 1):i + 2]) for i in range(len(seq))])
    base = float(np.median(sm[:3]))
    return float(sm.max() - base) + float(base - sm.min())


def stats(ms):
    ms = np.asarray(ms, np.float64)
    return {"rerata": float(ms.mean()), "median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
            "n": int(len(ms))}


def eval_performance(a):
    od = out_dir(a.keluar, "kinerja")
    main.apply_profile(a.jarak)
    imgs = []
    for path, _ in list_media(a.data):
        if os.path.splitext(path)[1].lower() in IMG_EXT:
            imgs += read_frames(path)
        if len(imgs) >= a.n:
            break
    if not imgs:
        raise SystemExit(f"Tidak ada gambar di {a.data}")
    ex = Extractor(need_live=True, need_refine=True)
    eng = ex.engine
    T = defaultdict(list)
    for i in range(a.n):
        img = imgs[i % len(imgs)]
        if img.shape[1] < main.CAM_WIDTH:          # samakan dengan resolusi kamera profil
            k = main.CAM_WIDTH / img.shape[1]
            img = cv2.resize(img, None, fx=k, fy=k)
        t0 = time.perf_counter()
        sw, sh, scale = main.small_size(img)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()
        faces = eng.detect(small)
        t2 = time.perf_counter()
        T["praproses (resize)"].append((t1 - t0) * 1000)
        T[f"deteksi YuNet ({sw}x{sh})"].append((t2 - t1) * 1000)
        if not faces:
            continue
        f = max(faces, key=lambda r: r[2] * r[3])
        t3 = time.perf_counter()
        e = eng.embed(img, f, scale)
        t4 = time.perf_counter()
        box = f[:4] * np.array([*scale, *scale], np.float32)
        ex.live.score(img, box)
        t5 = time.perf_counter()
        ex.refiner.refine(img, box)
        t6 = time.perf_counter()
        T["embedding SFace (align+feature)"].append((t4 - t3) * 1000)
        T["liveness pasif (2 model)"].append((t5 - t4) * 1000)
        T["landmark tantangan aktif"].append((t6 - t5) * 1000)
        del e
    scale_rows = []
    rs = np.random.RandomState(0)
    for n_people in (10, 100, 1000, 5000):
        emb = rs.randn(n_people * 8, 128).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        db = main.Database.from_arrays([f"p{i // 8}" for i in range(len(emb))], [""] * len(emb), emb)
        q = emb[0]
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            db.match(q)
            ts.append((time.perf_counter() - t0) * 1000)
        scale_rows.append([n_people, n_people * 8, f"{np.median(ts):.3f}", f"{len(emb) * 128 * 4 / 2 ** 20:.2f}"])
    res = {k: stats(v) for k, v in T.items()}
    det = next(v["rerata"] for k, v in res.items() if k.startswith("deteksi"))
    per_frame_pipe = res["praproses (resize)"]["rerata"] + det / main.DETECT_EVERY
    rss = None
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        pass
    summ = {"profil": a.jarak, "detect_every": main.DETECT_EVERY, "cv_threads": main.CV_THREADS,
            "fps_teoretis_tanpa_wajah_baru": 1000 / max(per_frame_pipe, 1e-6), "ram_mb": rss,
            "cpu": _cpu_name(), "opencv": cv2.__version__, "tahap": res, "skala_database": scale_rows}
    with open(os.path.join(od, "metrik.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    rows = [[k, f"{v['rerata']:.2f}", f"{v['median']:.2f}", f"{v['p95']:.2f}", v["n"]] for k, v in res.items()]
    rep = [f"# Evaluasi Kinerja (CPU)\n\nCPU: {summ['cpu']} | OpenCV {cv2.__version__} | profil `{a.jarak}` "
           f"(kamera {main.CAM_WIDTH}x{main.CAM_HEIGHT}, deteksi lebar {main.PROC_WIDTH}) | thread OpenCV "
           f"{main.CV_THREADS} | RAM proses {num(rss, 0)} MB\n",
           "## Latensi per tahap (ms)\n", md_table(["Tahap", "Rerata", "Median", "P95", "n"], rows), "",
           f"Deteksi hanya dijalankan tiap {main.DETECT_EVERY} frame (frame lain dilacak optical flow), dan "
           f"embedding/liveness hanya untuk wajah yang perlu diverifikasi. Perkiraan FPS saat tidak ada wajah baru: "
           f"**{summ['fps_teoretis_tanpa_wajah_baru']:.0f} FPS** (dibatasi FPS kamera).\n",
           "## Skala database (pencocokan 1 wajah, 8 sampel/orang)\n",
           md_table(["Orang", "Sampel", "Waktu cocok (ms)", "RAM embedding (MB)"], scale_rows), ""]
    with open(os.path.join(od, "laporan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    for r in rows:
        print(f"  {r[0]:<34} rerata {r[1]:>7} ms  p95 {r[3]:>7} ms")
    print(f"Laporan: {os.path.join(od, 'laporan.md')}")
    return summ, od


def _cpu_name():
    try:
        import platform
        name = platform.processor()
        if not name and os.path.exists("/proc/cpuinfo"):
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return name or platform.machine()
    except Exception:
        return "?"


# ============================== 4. UJI LAPANGAN ==============================
LAP_HEADER = ["percobaan", "subjek", "terdaftar", "jenis", "hasil", "waktu_detik", "catatan"]
LAP_EXAMPLE = [
    [1, "Budi", "ya", "asli", "Budi", 1.8, "cahaya normal"],
    [2, "Budi", "ya", "asli", "Budi", 2.4, "pakai kacamata"],
    [3, "Ani", "ya", "asli", "ditolak", "", "liveness gagal, cahaya redup"],
    [4, "Tamu1", "tidak", "asli", "unknown", "", "orang tidak terdaftar"],
    [5, "Budi", "ya", "cetak", "ditolak", "", "foto cetak A4"],
    [6, "Budi", "ya", "layar", "ditolak", "", "foto di layar HP"],
    [7, "Budi", "ya", "video", "Budi", 3.1, "video replay berhasil menembus (catat!)"],
]


def eval_field(a):
    od = out_dir(a.keluar, "lapangan")
    with open(a.csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)]
    if not rows:
        raise SystemExit("CSV kosong.")
    norm = lambda s: str(s or "").strip()
    gen = [r for r in rows if norm(r["jenis"]).lower() == "asli" and norm(r["terdaftar"]).lower() in ("ya", "y", "1", "true")]
    imp = [r for r in rows if norm(r["jenis"]).lower() == "asli" and r not in gen]
    atk = [r for r in rows if norm(r["jenis"]).lower() != "asli"]
    rejected = ("ditolak", "unknown", "tidak_terdeteksi", "-", "")
    tar = sum(norm(r["hasil"]) == norm(r["subjek"]) for r in gen) / len(gen) if gen else None
    mis = sum(norm(r["hasil"]) not in rejected and norm(r["hasil"]) != norm(r["subjek"]) for r in gen) / len(gen) if gen else None
    bpcer = sum(norm(r["hasil"]) == "ditolak" for r in gen) / len(gen) if gen else None
    far = sum(norm(r["hasil"]) not in rejected for r in imp) / len(imp) if imp else None
    kinds = sorted({norm(r["jenis"]).lower() for r in atk})
    apcer = {k: sum(norm(r["hasil"]) not in rejected for r in atk if norm(r["jenis"]).lower() == k) /
             max(1, sum(norm(r["jenis"]).lower() == k for r in atk)) for k in kinds}
    times = [float(r["waktu_detik"]) for r in gen if norm(r["hasil"]) == norm(r["subjek"]) and norm(r["waktu_detik"])]
    res = {"percobaan": len(rows), "genuine": len(gen), "penyusup": len(imp), "serangan": len(atk),
           "tar": tar, "frr": None if tar is None else 1 - tar, "salah_identitas": mis, "bpcer_liveness": bpcer,
           "far_penyusup": far, "apcer_per_jenis": apcer, "apcer": max(apcer.values()) if apcer else None,
           "waktu": stats([t * 1000 for t in times]) if times else None}
    with open(os.path.join(od, "metrik.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    tr = [["Percobaan (genuine / penyusup / serangan)", f"{len(gen)} / {len(imp)} / {len(atk)}"],
          ["TAR - terdaftar dicatat dengan benar", pct(tar)], ["FRR", pct(res["frr"])],
          ["Salah identitas", pct(mis)], ["BPCER - wajah asli ditolak liveness", pct(bpcer)],
          ["FAR - orang tak terdaftar diterima", pct(far)]]
    tr += [[f"APCER - serangan {k} berhasil", pct(v)] for k, v in apcer.items()]
    if times:
        tr.append(["Waktu sampai tercatat (rerata / p95)", f"{np.mean(times):.2f} / {np.percentile(times, 95):.2f} dtk"])
    with open(os.path.join(od, "laporan.md"), "w", encoding="utf-8") as f:
        f.write(f"# Uji Lapangan (black-box)\n\nSumber: `{a.csv}`, {len(rows)} percobaan.\n\n"
                + md_table(["Metrik", "Nilai"], tr) + "\n")
    for r in tr:
        print(f"  {r[0]:<46} {r[1]}")
    print(f"Laporan: {os.path.join(od, 'laporan.md')}")
    return res, od


# ============================== 5. PENGUMPULAN DATA ==============================
def collect(a):
    root = a.keluar_data or ("dataset_liveness" if a.liveness else "dataset")
    if a.liveness:
        sub = os.path.join(root, "asli" if a.tipe == "asli" else os.path.join("serangan", a.tipe), a.label)
    else:
        sub = os.path.join(root, a.label)
    os.makedirs(sub, exist_ok=True)
    print("INGAT: foto/video uji adalah data biometrik. Pastikan peserta menyetujui dan hapus setelah penelitian.")
    main.apply_profile(a.jarak)
    eng = main.FaceEngine(*main.ensure_models())
    cam = main.Camera(a.kamera)
    win = "Kumpulkan data (Q = selesai)"
    saved, last, info = 0, 0.0, ""
    writer, t_end = None, None
    try:
        if a.video:
            fr = main.next_frame(cam)
            while fr is None:
                fr = main.next_frame(cam)
            path = os.path.join(sub, f"{a.label}_{datetime.now():%Y%m%d_%H%M%S}.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (fr.shape[1], fr.shape[0]))
            t_end = time.time() + a.video
            print(f"Merekam {a.video} dtk -> {path}. " + ("Tengok kiri-kanan, lalu dongak-tunduk." if a.tipe == "asli" else ""))
        while True:
            fr = main.next_frame(cam)
            if fr is None:
                continue
            if writer is not None:
                if time.time() - last >= 1 / 15:
                    writer.write(fr)
                    last = time.time()
                if time.time() >= t_end:
                    saved = 1
                    break
            else:
                sw, sh, scale = main.small_size(fr)
                faces = eng.detect(cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA))
                info = "" if len(faces) == 1 else ("wajah tidak terlihat" if not faces else "harus satu wajah")
                if len(faces) == 1 and time.time() - last >= a.jeda:
                    cv2.imencode(".jpg", fr)[1].tofile(os.path.join(sub, f"{a.label}_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"))
                    saved += 1
                    last = time.time()
                if saved >= a.jumlah:
                    break
            show = cv2.flip(fr, 1)
            main.put_text(show, f"{a.label} [{a.tipe}] {'REKAM' if writer else f'{saved}/{a.jumlah}'} {info}",
                          (10, 24), (255, 255, 255), 0.7)
            cv2.imshow(win, show)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
        cam.close()
        cv2.destroyAllWindows()
    print(f"Tersimpan {saved} file di {sub}")


# ============================== CLI ==============================
def main_cli():
    ap = argparse.ArgumentParser(description="Evaluasi kuantitatif absensi wajah",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pengenalan", help="verifikasi 1:1 + identifikasi 1:N open-set")
    p.add_argument("--data", default="dataset")
    p.add_argument("--daftar", type=int, default=3, help="foto per orang yang didaftarkan (galeri)")
    p.add_argument("--ulang", type=int, default=5, help="jumlah ulangan pembagian acak")
    p.add_argument("--tak-dikenal", type=float, default=0.2, help="porsi orang yang tidak didaftarkan")
    p.add_argument("--target-fpir", type=float, default=0.01, help="target FPIR untuk ambang disarankan")
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("liveness", help="APCER/BPCER/ACER anti-spoofing")
    p.add_argument("--data", default="dataset_liveness")
    p.add_argument("--ambang", type=float, default=None, help="ambang skor asli (bawaan config)")
    p = sub.add_parser("kinerja", help="latensi, FPS, RAM, skala database")
    p.add_argument("--data", default="dataset")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--jarak", choices=list(main.PROFIL), default=main.PROFIL_AKTIF)
    p = sub.add_parser("lapangan", help="rekap uji lapangan dari CSV")
    p.add_argument("--csv", default="uji_lapangan.csv")
    sub.add_parser("templat-lapangan", help="buat contoh uji_lapangan.csv")
    p = sub.add_parser("kumpulkan", help="ambil foto/video uji dari kamera")
    p.add_argument("--label", required=True)
    p.add_argument("--jumlah", type=int, default=20)
    p.add_argument("--jeda", type=float, default=0.6)
    p.add_argument("--tipe", default="asli", help="asli | cetak | layar | video | topeng | ...")
    p.add_argument("--liveness", action="store_true", help="simpan ke struktur dataset_liveness/")
    p.add_argument("--video", type=float, default=0, help="rekam video N detik (uji tantangan aktif)")
    p.add_argument("--kamera", type=int, default=0)
    p.add_argument("--keluar-data", default="")
    p.add_argument("--jarak", choices=list(main.PROFIL), default="dekat")
    for sp in sub.choices.values():
        sp.add_argument("--keluar", default="hasil_evaluasi", help="folder hasil")
    a = ap.parse_args()
    if a.cmd == "pengenalan":
        eval_recognition(a)
    elif a.cmd == "liveness":
        eval_liveness(a)
    elif a.cmd == "kinerja":
        eval_performance(a)
    elif a.cmd == "lapangan":
        eval_field(a)
    elif a.cmd == "templat-lapangan":
        with open("uji_lapangan.csv", "x", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(LAP_HEADER)
            w.writerows(LAP_EXAMPLE)
        print("Dibuat uji_lapangan.csv (isi sesuai percobaan; hasil = nama yang tercatat sistem, 'unknown', "
              "'ditolak' (liveness), atau 'tidak_terdeteksi').")
    elif a.cmd == "kumpulkan":
        collect(a)


if __name__ == "__main__":
    try:
        main_cli()
    except KeyboardInterrupt:
        sys.exit("\nDihentikan.")
    except (RuntimeError, OSError, ValueError, cv2.error) as e:
        sys.exit(f"Error: {e}")
