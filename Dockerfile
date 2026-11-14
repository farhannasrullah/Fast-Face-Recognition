FROM python:3.11-slim

# libgl/glib dibutuhkan beberapa build OpenCV; versi headless biasanya cukup, ini jaga-jaga
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Hanya paket WAJIB (tanpa matplotlib/psutil/openpyxl) supaya image kecil
COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

COPY . .

# Unduh model YuNet + SFace saat build (model liveness MiniFASNet harus sudah ada di folder models/ repo)
RUN python main.py download-models || echo "Unduh model gagal saat build; akan dicoba lagi saat jalan"

# Railway mengisi $PORT. ADMIN_PIN = PIN dashboard (WAJIB agar kamera & pengelolaan wajah aktif).
CMD ["sh", "-c", "python dashboard.py --host 0.0.0.0 --port ${PORT:-8000} --no-browser --pin \"${ADMIN_PIN}\""]
