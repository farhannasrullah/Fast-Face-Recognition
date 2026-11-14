# -*- coding: utf-8 -*-
"""
liveness.py - deteksi keaslian wajah (anti-spoofing / Presentation Attack Detection), CPU saja.

Dua lapis yang saling melengkapi (dipilih lewat config `liveness_mode`):

1. PASIF (tanpa interaksi) - MiniFASNetV2 (skala crop 2.7) + MiniFASNetV1SE (skala 4.0) dari
   Silent-Face-Anti-Spoofing (Minivision, Apache-2.0), dikonversi ke ONNX (tools/). Keluaran 3 kelas
   [kertas/print, ASLI, layar/replay]; skor = rata-rata probabilitas "asli" kedua model. Keputusan
   diambil dari beberapa frame (rata-rata jendela), bukan satu frame, sehingga lebih stabil.
   Cocok untuk CCTV/headless. Biaya ~2-4 ms per wajah.

2. AKTIF (tantangan-respons) - pengguna diminta MENOLEH kiri-kanan atau DONGAK-TUNDUK (dipilih acak). Gerakan diukur dengan
   koordinat AFIN hidung terhadap segitiga (mata kanan, mata kiri, tengah mulut):
        hidung = pusat_mata + u * (mata_kiri - mata_kanan) + v * (tengah_mulut - pusat_mata)
   Secara teori (u, v) TIDAK berubah oleh transformasi afin (geser, putar, skala, miring) - jadi FOTO DATAR
   yang digerak-gerakkan memberi (u, v) ~ konstan, sedangkan kepala 3D yang menoleh mengubah u dan
   dongak/tunduk mengubah v (hidung menonjol keluar bidang mata-mulut). Gerakan harus ke DUA arah dari
   posisi awal. Landmark diambil dari deteksi ulang YuNet pada crop resolusi penuh + median 3 sampel.
   Uji kami: foto yang diputar/digeser memberi rentang u ~0,15 dan foto yang dimiringkan +-35 derajat
   ~0,22 (landmark detektor tidak sepenuhnya ekuivarian), sedangkan menoleh +-20 derajat secara teori
   memberi ~0,35. Karena itu ambang bawaan 0,28 - KALIBRASI dengan video nyata (evaluate.py liveness).

Batasan (wajib ditulis di skripsi): model pasif dilatih pada data Minivision (kamera ponsel); akurasi
pada kamera/cahaya lain harus diukur sendiri (`python evaluate.py liveness`). Tantangan aktif bisa
dikalahkan video replay yang kebetulan berisi gerakan sama atau topeng 3D; acak tantangan + model
pasif menekan risiko itu, bukan menghilangkannya.
"""
import os
import random
import time
from collections import deque

import cv2
import numpy as np

import config

PASSIVE_MODELS = [("2.7_80x80_MiniFASNetV2.onnx", 2.7), ("4_0_0_80x80_MiniFASNetV1SE.onnx", 4.0)]
LABELS = ("kertas", "asli", "layar")
CHALLENGES = {"geleng": "Tengok ke kiri lalu ke kanan", "angguk": "Dongak lalu tunduk"}
REFINE_SIZE = 160          # ukuran crop untuk deteksi ulang landmark (resolusi penuh)


def spoof_crop(frame, box, scale, size=80):
    """Crop ala Silent-Face (CropImage): kotak wajah diperbesar `scale` kali, digeser agar tetap di dalam
    gambar, lalu diubah ke size x size. box = (x, y, w, h) pada koordinat frame."""
    H, W = frame.shape[:2]
    x, y, w, h = (float(v) for v in box[:4])
    if w <= 1 or h <= 1:
        return None
    s = min((H - 1) / h, (W - 1) / w, scale)
    nw, nh = w * s, h * s
    cx, cy = x + w / 2, y + h / 2
    l, t, r, b = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if l < 0:
        r, l = r - l, 0
    if t < 0:
        b, t = b - t, 0
    if r > W - 1:
        l, r = l - (r - W + 1), W - 1
    if b > H - 1:
        t, b = t - (b - H + 1), H - 1
    x0, y0 = int(np.clip(round(l), 0, W - 1)), int(np.clip(round(t), 0, H - 1))
    x1, y1 = int(np.clip(round(r), x0, W - 1)), int(np.clip(round(b), y0, H - 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return cv2.resize(frame[y0:y1 + 1, x0:x1 + 1], (size, size), interpolation=cv2.INTER_LINEAR)


class PassiveLiveness:
    """Skor keaslian pasif dari 1 frame. Input BGR uint8; model menerima BGR 0-255 tanpa normalisasi."""

    def __init__(self, model_dir=None):
        model_dir = model_dir or config.MODEL_DIR
        self.nets = []
        missing = [f for f, _ in PASSIVE_MODELS if not os.path.exists(os.path.join(model_dir, f))]
        if missing:
            raise RuntimeError("Model liveness tidak ditemukan di " + model_dir + ": " + ", ".join(missing) +
                               ". Salin folder models/ dari paket proyek, atau set liveness_mode=off/aktif.")
        for f, sc in PASSIVE_MODELS:
            net = cv2.dnn.readNetFromONNX(os.path.join(model_dir, f))
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.nets.append((net, sc))

    def probs(self, frame, box):
        """Probabilitas rata-rata [kertas, asli, layar] atau None bila crop tidak bisa dibuat."""
        acc, n = np.zeros(3, np.float64), 0
        for net, sc in self.nets:
            crop = spoof_crop(frame, box, sc)
            if crop is None:
                continue
            net.setInput(crop.transpose(2, 0, 1)[None].astype(np.float32))
            z = net.forward().reshape(-1).astype(np.float64)
            e = np.exp(z - z.max())
            acc += e / e.sum()
            n += 1
        return acc / n if n else None

    def score(self, frame, box):
        p = self.probs(frame, box)
        return None if p is None else float(p[1])


def affine_coords(lm):
    """(u, v) hidung pada basis afin wajah. lm = 5x2 [mata_kanan, mata_kiri, hidung, mulut_kanan, mulut_kiri]."""
    lm = np.asarray(lm, np.float64).reshape(5, 2)
    c = (lm[0] + lm[1]) / 2
    m = (lm[3] + lm[4]) / 2
    A = np.stack([lm[1] - lm[0], m - c], axis=1)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    u, v = np.linalg.solve(A, lm[2] - c)
    return float(u), float(v)


class LandmarkRefiner:
    """Deteksi ulang YuNet pada crop resolusi penuh agar landmark lebih presisi untuk tantangan aktif."""

    def __init__(self, yunet_path):
        self.det = cv2.FaceDetectorYN.create(yunet_path, "", (REFINE_SIZE, REFINE_SIZE), 0.6, 0.3, 5,
                                             cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)

    def refine(self, frame, box):
        H, W = frame.shape[:2]
        x, y, w, h = (float(v) for v in box[:4])
        side = 1.6 * max(w, h)
        x0, y0 = int(max(0, x + w / 2 - side / 2)), int(max(0, y + h / 2 - side / 2))
        x1, y1 = int(min(W, x + w / 2 + side / 2)), int(min(H, y + h / 2 + side / 2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = cv2.resize(frame[y0:y1, x0:x1], (REFINE_SIZE, REFINE_SIZE), interpolation=cv2.INTER_LINEAR)
        _, faces = self.det.detect(crop)
        if faces is None or not len(faces):
            return None
        sx, sy = (x1 - x0) / REFINE_SIZE, (y1 - y0) / REFINE_SIZE
        ctr = np.array([REFINE_SIZE / 2, REFINE_SIZE / 2])
        f = min(faces, key=lambda r: np.linalg.norm(r[0:2] + r[2:4] / 2 - ctr))   # wajah paling tengah
        lm = f[4:14].reshape(5, 2).astype(np.float64)
        return lm * [sx, sy] + [x0, y0]


class LivenessState:
    """Status liveness satu wajah yang dilacak (satu Track). Reset otomatis saat track hilang."""

    def __init__(self, eng):
        self.eng = eng
        self.scores = deque(maxlen=eng.window)
        self.passive_ok = eng.passive is None
        self.active_ok = not eng.active
        self.spoof = False
        self.spoof_reported = False
        self.last = None             # skor pasif terakhir
        self.mean = None
        self.ch = None               # tantangan aktif: {"jenis", "mulai", "u": [...], "v": [...]}
        self.ch_fail = 0
        self.too_small = False
        self.hint = ""
        self.uncertain = False       # zona abu-abu: belum cukup yakin asli, belum cukup yakin palsu
        self.gate = ""               # alasan wajah belum dinilai (terlalu kecil / terpotong tepi layar)
        self.last_probs = None       # [kertas, asli, layar] frame terakhir (diagnostik)
        self.last_w = 0.0            # lebar wajah (px) frame terakhir yang dinilai

    @property
    def passed(self):
        return self.passive_ok and self.active_ok and not self.spoof

    def prompt(self):
        if self.spoof:
            return "Wajah palsu terdeteksi"
        if not self.active_ok and self.eng.active:
            if self.too_small:
                return "Mendekat ke kamera"
            if self.ch:
                return CHALLENGES[self.ch["jenis"]] + (f" ({self.hint})" if self.hint else "")
        if not self.passive_ok:
            if self.gate:
                return self.gate
            if self.uncertain:
                return "Belum yakin: dekatkan wajah, hadap lurus, cahaya dari depan"
            return "Memeriksa keaslian wajah"
        return ""

    # ---- pasif ----
    def add_score(self, sc, probs=None):
        """Perbarui keputusan pasif. Tiga keadaan (bukan dua):
          LOLOS   rata-rata >= thresh (>= min_frames frame)               -> boleh dicatat
          PALSU   jendela penuh dan rata-rata < spoof_thresh (bukti kuat)  -> dicap palsu, dilaporkan
          RAGU    jendela penuh dan spoof_thresh <= rata-rata < thresh     -> TIDAK dicatat, TIDAK dicap
                  palsu; terus dinilai dan pengguna diberi petunjuk. Syarat lolos tidak dilonggarkan,
                  jadi keamanan sama, tetapi wajah asli tidak langsung dicap \"PALSU\".
        Mengembalikan \"spoof\" sekali saat pertama kali dicap palsu."""
        if sc is None:
            return None
        self.last, self.last_probs = sc, probs
        self.scores.append(sc)
        self.mean = float(np.mean(self.scores))
        n, ev = len(self.scores), None
        if n >= self.eng.min_frames and self.mean >= self.eng.thresh:
            self.passive_ok, self.spoof, self.uncertain = True, False, False
        elif n >= self.eng.window and self.mean < self.eng.spoof_thresh:
            self.spoof, self.uncertain = True, False
            if not self.spoof_reported:
                self.spoof_reported = True
                ev = "spoof"
        elif n >= self.eng.window:
            self.spoof, self.uncertain = False, True
        return ev

    # ---- aktif ----
    def add_landmarks(self, lm, now):
        uv = affine_coords(lm) if lm is not None else None
        if uv is None:
            return
        if self.ch is None or now - self.ch["mulai"] > self.eng.ch_timeout:
            if self.ch is not None:
                self.ch_fail += 1
                self.hint = "waktu habis, ulangi"
            self.ch = {"jenis": random.choice(list(CHALLENGES)), "mulai": now, "u": [], "v": []}
        self.ch["u"].append(uv[0])
        self.ch["v"].append(uv[1])
        seq = self.ch["u" if self.ch["jenis"] == "geleng" else "v"]
        ok, prog = challenge_passed(seq, self.eng.ch_thresh * CH_SCALE[self.ch["jenis"]])
        self.hint = "" if ok else f"{prog}%"
        if ok:
            self.active_ok = True


# Rentang yang dibutuhkan relatif terhadap tantangan_ambang. Angguk sedikit lebih ringan karena rentang
# vertikal hidung (relatif jarak mata-mulut) lebih kecil daripada rentang horizontal (relatif jarak mata).
CH_SCALE = {"geleng": 1.0, "angguk": 0.8}


def challenge_passed(seq, thresh):
    """Lolos bila deret koordinat (dihaluskan median 3) bergerak ke DUA arah dari posisi awal:
    (maks - awal) + (awal - min) >= thresh dan sisi terkecil >= 25% thresh.
    Mengembalikan (lolos, progres_persen)."""
    if len(seq) < 4:
        return False, 0
    sm = np.array([np.median(seq[max(0, i - 1):i + 2]) for i in range(len(seq))])
    base = float(np.median(sm[:3]))
    pos, neg = float(sm.max() - base), float(base - sm.min())
    ok = pos + neg >= thresh and min(pos, neg) >= 0.25 * thresh
    prog = min(99, round(100 * min(pos + neg, thresh) / thresh * (1.0 if min(pos, neg) >= 0.25 * thresh else 0.8)))
    return ok, (100 if ok else prog)


class LivenessEngine:
    """Satu objek per proses. Model OpenCV tidak thread-safe: panggil observe() di bawah kunci engine."""

    def __init__(self, mode=None, yunet_path=None, model_dir=None):
        self.mode = (mode or config.LIVENESS_MODE).lower()
        self.thresh = config.LIVENESS_THRESH
        self.min_frames = config.LIVENESS_MIN_FRAMES
        self.window = config.LIVENESS_WINDOW
        self.spoof_thresh = config.LIVENESS_SPOOF_THRESH
        self.min_px = config.LIVENESS_MIN_PX
        self.ch_thresh = config.TANTANGAN_AMBANG
        self.ch_timeout = config.TANTANGAN_TIMEOUT
        self.ch_min_px = config.TANTANGAN_MIN_PX
        self.passive = PassiveLiveness(model_dir) if self.mode in ("pasif", "ganda") else None
        self.active = self.mode in ("aktif", "ganda")
        self.refiner = LandmarkRefiner(yunet_path) if self.active and yunet_path else None
        self.timing = deque(maxlen=200)    # ms per observe (untuk evaluasi kinerja)

    @property
    def enabled(self):
        return self.mode != "off"

    def describe(self):
        if not self.enabled:
            return "liveness NONAKTIF (rentan foto/video!)"
        parts = []
        if self.passive:
            parts.append(f"pasif MiniFASNet (lolos >= {self.thresh:.2f}, palsu < {self.spoof_thresh:.2f}, "
                         f"{self.min_frames}-{self.window} frame, wajah >= {self.min_px}px)")
        if self.active:
            parts.append(f"aktif tengok kiri-kanan / dongak-tunduk (ambang {self.ch_thresh:.2f}, {self.ch_timeout} dtk)")
        return "liveness " + " + ".join(parts)

    def new_state(self):
        return LivenessState(self) if self.enabled else None

    def gate(self, frame, box):
        """\"\" bila wajah layak dinilai model pasif; selain itu alasan singkat untuk pengguna.
        Wajah kecil (tekstur hilang saat di-resize ke 80x80) atau terpotong tepi layar menghasilkan skor
        sembarang; menilainya hanya melahirkan penolakan palsu pada orang asli."""
        H, W = frame.shape[:2]
        x, y, w, h = (float(v) for v in box[:4])
        if self.min_px and w < self.min_px:
            return "Mendekat ke kamera"
        if x < 2 or y < 2 or x + w > W - 2 or y + h > H - 2:
            return "Posisikan seluruh wajah di dalam layar"
        return ""

    def observe(self, st, frame, box_full, lm_full, now=None):
        """Perbarui status liveness dari satu deteksi segar. box_full / lm_full pada koordinat frame penuh.
        Mengembalikan "spoof" sekali saat percobaan spoofing pertama kali diputuskan, selain itu None."""
        if st is None or st.passed:
            return None
        now = time.time() if now is None else now
        t0 = time.perf_counter()
        ev = None
        if self.passive is not None and (not st.passive_ok or st.spoof):
            st.gate = self.gate(frame, box_full)
            if not st.gate:
                p = self.passive.probs(frame, box_full)
                st.last_w = float(box_full[2])
                ev = st.add_score(None if p is None else float(p[1]), p)
        if self.active and not st.active_ok and not st.spoof:
            st.too_small = box_full[2] < self.ch_min_px
            if not st.too_small:
                # landmark dari deteksi ulang resolusi penuh; bila gagal, sampel DILEWATI (mencampur
                # landmark dari dua sumber menimbulkan lompatan palsu pada koordinat u/v)
                lm = self.refiner.refine(frame, box_full) if self.refiner else lm_full
                if lm is not None:
                    st.add_landmarks(lm, now)
        self.timing.append((time.perf_counter() - t0) * 1000)
        return ev