#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_stage1.py - terapkan Tahap 1 ke main.py, dashboard.py, cloud_sync.py (satu kali, aman diulang).

Letakkan bersama config.py, ipclock.py, attendance.py di folder proyek, lalu:  python apply_stage1.py
  * Semua perubahan dihitung dulu di memori; bila ada satu pola yang tidak cocok, TIDAK ADA file
    yang diubah (pesan error menyebut bagian yang gagal).
  * File asli disalin ke *.bak sebelum ditimpa.
"""
import os
import re
import shutil
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def need(ok, msg):
    if not ok:
        sys.exit(f"GAGAL (tidak ada file yang diubah): {msg}")


def patch_main(s):
    if "from attendance import AttendanceLogger" in s:
        return s
    # 1) blok import cloud_sync -> import dari modul baru
    s, n = re.subn(r"try:[^\n]*\n    import cloud_sync\nexcept ImportError:\n    cloud_sync = None\n",
                   "from attendance import AttendanceLogger\n"
                   "from config import JAM_PULANG, JAM_TERLAMBAT, MIN_DURASI_MENIT\n", s, count=1)
    need(n == 1, "main.py: blok 'try: import cloud_sync' tidak ditemukan")
    # 2) tiga konstanta jam -> satu komentar (nilai kini dari config.py)
    seen = []
    def drop(m):
        seen.append(1)
        return ("# Jam terlambat / jam pulang / durasi minimum diatur terpusat di config.py "
                "(atau config.json / environment variable)\n") if len(seen) == 1 else ""
    s = re.sub(r"^(JAM_TERLAMBAT|JAM_PULANG|MIN_DURASI_MENIT)[ \t]*=.*\n", drop, s, flags=re.M)
    need(len(seen) == 3, f"main.py: ditemukan {len(seen)} dari 3 konstanta JAM_TERLAMBAT/JAM_PULANG/MIN_DURASI_MENIT")
    # 3) hapus class AttendanceLogger (kini di attendance.py)
    a = re.search(r"# -{5,} Pencatat absensi -{5,}\n", s)
    b = re.search(r"# -{5,} Pelacak wajah -{5,}\n", s)
    need(a and b and a.start() < b.start(), "main.py: penanda bagian 'Pencatat absensi' / 'Pelacak wajah' tidak ditemukan")
    s = s[:a.start()] + s[b.start():]
    need("class AttendanceLogger" not in s, "main.py: class AttendanceLogger masih ada")
    return s


def patch_dashboard(s):
    if "from config import JAM_TERLAMBAT" in s:
        return s
    s, n = re.subn(r"^JAM_TERLAMBAT[ \t]*=.*$", "from config import JAM_TERLAMBAT   # satu sumber konfigurasi: config.py",
                   s, count=1, flags=re.M)
    need(n == 1, "dashboard.py: baris JAM_TERLAMBAT tidak ditemukan")
    return s


def patch_cloud(s):
    if "from config import JAM_TERLAMBAT" in s:
        return s
    s, n = re.subn(r"^JAM_TERLAMBAT[ \t]*=[ \t]*cfg\(.*$", "from config import JAM_TERLAMBAT   # satu sumber konfigurasi: config.py",
                   s, count=1, flags=re.M)
    need(n == 1, "cloud_sync.py: baris JAM_TERLAMBAT = cfg(...) tidak ditemukan")
    s = re.sub(r"^  jam_terlambat .*$", "  (jam_terlambat kini diatur di config.py / config.json / env JAM_TERLAMBAT)", s, flags=re.M)
    return s


def main():
    for req in ("config.py", "ipclock.py", "attendance.py"):
        need(os.path.exists(os.path.join(BASE, req)), f"{req} belum ada di folder ini")
    jobs = {"main.py": patch_main, "dashboard.py": patch_dashboard, "cloud_sync.py": patch_cloud}
    out = {}
    for name, fn in jobs.items():
        path = os.path.join(BASE, name)
        need(os.path.exists(path), f"{name} tidak ditemukan di {BASE}")
        with open(path, encoding="utf-8") as f:
            old = f.read()
        new = fn(old)
        try:
            compile(new, name, "exec")
        except SyntaxError as e:
            sys.exit(f"GAGAL (tidak ada file yang diubah): hasil patch {name} tidak valid: {e}")
        out[name] = (old, new)
    for name, (old, new) in out.items():
        if old == new:
            print(f"- {name}: sudah diterapkan, dilewati")
            continue
        path = os.path.join(BASE, name)
        if not os.path.exists(path + ".bak"):
            shutil.copy2(path, path + ".bak")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        print(f"- {name}: diperbarui (cadangan: {name}.bak)")
    print("Selesai. Jalankan:  python -m unittest -v")


if __name__ == "__main__":
    main()