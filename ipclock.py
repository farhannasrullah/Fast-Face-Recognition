# -*- coding: utf-8 -*-
"""
ipclock.py - kunci antar-PROSES berbasis file (Windows: msvcrt, Linux/macOS: fcntl).

threading.Lock hanya melindungi satu proses. Bila `main.py run` dan `dashboard.py` berjalan
bersamaan, keduanya perlu kunci yang sama agar urutan "baca -> putuskan -> tulis" tidak
tumpang-tindih. Kunci ini juga serial antar-thread dalam proses yang sama.

Pemakaian:
    lock = InterProcessLock("absensi.csv.lock", timeout=5)
    with lock:
        ...   # hanya satu proses/thread di blok ini pada satu waktu
Melempar TimeoutError bila kunci tidak didapat dalam `timeout` detik.
Kunci dilepas otomatis oleh OS bila proses mati mendadak (tidak ada kunci yatim).
"""
import threading
import time

try:
    import msvcrt          # Windows
    _WIN = True
except ImportError:
    import fcntl           # Linux / macOS
    _WIN = False


def _try_lock(fh):
    if _WIN:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)       # OSError bila sedang dikunci proses lain
    else:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fh):
    try:
        if _WIN:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class InterProcessLock:
    def __init__(self, path, timeout=5.0, poll=0.02):
        self.path, self.timeout, self.poll = path, timeout, poll
        self._tlock = threading.Lock()      # serial antar-thread dalam satu proses
        self._fh = None

    def acquire(self):
        if not self._tlock.acquire(timeout=self.timeout):
            raise TimeoutError(f"kunci {self.path} tidak didapat dalam {self.timeout:.0f} detik")
        try:
            fh = open(self.path, "a+b")
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    _try_lock(fh)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        fh.close()
                        raise TimeoutError(f"kunci {self.path} tidak didapat dalam {self.timeout:.0f} detik")
                    time.sleep(self.poll)
            self._fh = fh
        except BaseException:
            self._tlock.release()
            raise

    def release(self):
        fh, self._fh = self._fh, None
        if fh is not None:
            _unlock(fh)
            fh.close()
        self._tlock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()