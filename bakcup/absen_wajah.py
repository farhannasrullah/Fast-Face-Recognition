"""
Auto Fast Face Recognition - Absensi Sederhana
Ringan: CPU only (0 VRAM), RAM ~100-200 MB.
Model: YuNet (deteksi, ~230 KB) + SFace (pengenalan, ~37 MB), via OpenCV.

Instalasi : pip install opencv-python numpy
Daftar    : python absen_wajah.py enroll "Nama Lengkap"
Absen     : python absen_wajah.py run
Daftar    : python absen_wajah.py list
Keluar    : tekan Q pada jendela kamera
"""
import csv
import os
import sys
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

# ---------------- Konfigurasi ----------------
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
PROC_WIDTH = 320          # lebar frame untuk diproses (makin kecil makin cepat)
DETECT_EVERY = 2          # proses 1 dari N frame
SCORE_THRESH = 0.9        # confidence deteksi wajah
MATCH_THRESH = 0.363      # cosine similarity SFace (rekomendasi resmi)
ENROLL_SAMPLES = 8        # jumlah sampel wajah saat pendaftaran
MIN_FACE = 60             # ukuran wajah minimum (px, pada frame proses)
DB_FILE = "faces_db.npz"
LOG_FILE = "absensi.csv"
MODEL_DIR = "models"
MODELS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
# ----------------------------------------------


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if not os.path.exists(path):
            print(f"Mengunduh {name} ...")
            urllib.request.urlretrieve(url, path)
    return (os.path.join(MODEL_DIR, "yunet.onnx"),
            os.path.join(MODEL_DIR, "sface.onnx"))


def load_models():
    yunet, sface = ensure_models()
    detector = cv2.FaceDetectorYN.create(
        yunet, "", (PROC_WIDTH, PROC_WIDTH), SCORE_THRESH, 0.3, 500)
    recognizer = cv2.FaceRecognizerSF.create(sface, "")
    return detector, recognizer


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit("Kamera tidak bisa dibuka.")
    return cap


def shrink(frame):
    h, w = frame.shape[:2]
    scale = PROC_WIDTH / w
    small = cv2.resize(frame, (PROC_WIDTH, int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def embed(recognizer, small, face_row):
    aligned = recognizer.alignCrop(small, face_row)
    feat = recognizer.feature(aligned).flatten().astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-9)


def load_db():
    if not os.path.exists(DB_FILE):
        return np.array([], dtype=str), np.zeros((0, 128), np.float32)
    d = np.load(DB_FILE, allow_pickle=False)
    return d["names"], d["embs"]


def save_db(names, embs):
    np.savez_compressed(DB_FILE, names=names, embs=embs)


# ---------------- Mode: daftar ----------------
def enroll(person):
    detector, recognizer = load_models()
    names, embs = load_db()
    cap = open_camera()
    samples, last = [], 0.0
    print(f"Mendaftarkan '{person}'. Hadap kamera, ubah sedikit posisi kepala.")
    while len(samples) < ENROLL_SAMPLES:
        ok, frame = cap.read()
        if not ok:
            break
        small, _ = shrink(frame)
        detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = detector.detect(small)
        if faces is not None and len(faces) == 1 and faces[0][2] >= MIN_FACE:
            if time.time() - last > 0.6:
                samples.append(embed(recognizer, small, faces[0]))
                last = time.time()
        cv2.putText(frame, f"Sampel {len(samples)}/{ENROLL_SAMPLES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Daftar Wajah", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(samples) < 3:
        sys.exit("Sampel kurang, pendaftaran dibatalkan.")
    new_names = np.array([person] * len(samples))
    save_db(np.concatenate([names, new_names]),
            np.vstack([embs, np.array(samples, np.float32)]))
    print(f"Selesai: {person} ({len(samples)} sampel).")


# ---------------- Mode: absen ----------------
def load_today_attended():
    today = datetime.now().strftime("%Y-%m-%d")
    done = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0] == today:
                    done.add(row[2])
    return done


def log_attendance(name):
    now = datetime.now()
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tanggal", "jam", "nama"])
        w.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), name])


def run():
    names, embs = load_db()
    if len(names) == 0:
        sys.exit("Database kosong. Jalankan: python absen_wajah.py enroll \"Nama\"")
    detector, recognizer = load_models()
    attended = load_today_attended()
    cap = open_camera()
    results, frame_no, fps, t0 = [], 0, 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        if frame_no % DETECT_EVERY == 0:
            small, scale = shrink(frame)
            detector.setInputSize((small.shape[1], small.shape[0]))
            _, faces = detector.detect(small)
            results = []
            if faces is not None:
                for f in faces:
                    if f[2] < MIN_FACE:
                        continue
                    sims = embs @ embed(recognizer, small, f)
                    i = int(np.argmax(sims))
                    name = str(names[i]) if sims[i] >= MATCH_THRESH else None
                    if name and name not in attended:
                        attended.add(name)
                        log_attendance(name)
                        print(f"[ABSEN] {name} - {datetime.now():%H:%M:%S}")
                    box = (f[:4] / scale).astype(int)
                    results.append((box, name, float(sims[i])))

        for (x, y, w, h), name, s in results:
            color = (0, 200, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{name} ({s:.2f})" if name else "Tidak dikenal"
            if name in attended:
                label += " [hadir]"
            cv2.putText(frame, label, (x, max(y - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if frame_no % 15 == 0:
            now = time.time()
            fps = 15 / (now - t0)
            t0 = now
        cv2.putText(frame, f"FPS {fps:.0f} | Hadir: {len(attended)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Absensi Wajah (Q = keluar)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1] == "enroll" and len(sys.argv) >= 3:
        enroll(" ".join(sys.argv[2:]))
    elif sys.argv[1] == "run":
        run()
    elif sys.argv[1] == "list":
        names, _ = load_db()
        for n in sorted(set(map(str, names))):
            print("-", n)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
