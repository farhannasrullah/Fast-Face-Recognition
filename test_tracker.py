import numpy as np
import main


def row(x, y, w):
    h = w * 1.2
    lm = [[x + .3 * w, y + .35 * w], [x + .7 * w, y + .35 * w], [x + .5 * w, y + .6 * w],
          [x + .35 * w, y + .85 * w], [x + .65 * w, y + .85 * w]]
    return np.array([x, y, w, h, *np.ravel(lm), .9], np.float32)


class FakeLive:
    passed = True
    resets = 0
    def reset(self):
        self.resets += 1
        self.passed = False


class FakeLogger:
    def __init__(self):
        self.calls = []
    def try_log(self, *a):
        self.calls.append(a)
        return ("masuk", "hadir")


def test_stale_track_dropped():
    tr = main.Tracker()
    tr.update([row(100, 100, 100)], 0.0)
    tr.update([row(100, 100, 100)], 10.0)          # sesi diam 10 dtk, orang lain di tempat yang sama
    assert list(tr.tracks) == [2]


def test_ambiguous_pair_forces_verify():
    tr = main.Tracker()
    tr.update([row(100, 100, 100)], 0.0)
    tr.update([row(100, 100, 100), row(110, 100, 100)], 0.1)
    a = tr.tracks[1]
    assert a.verify and a.next_embed == 0.0


def test_flow_jump_rejected(monkeypatch):
    tr = main.Tracker()
    tr.update([row(100, 100, 100)], 0.0)
    t = tr.tracks[1]
    box0, calls = t.box.copy(), []

    def fake(prev, gray, p0, nxt, **kw):
        calls.append(1)
        sh = np.array([120, 0], np.float32) * (1 if len(calls) == 1 else -1)   # maju lalu balik konsisten
        return p0 + sh, np.ones((len(p0), 1), np.uint8), None
    monkeypatch.setattr(main.cv2, "calcOpticalFlowPyrLK", fake)
    tr.advance(None, None)
    assert t.verify and np.allclose(t.box, box0)


def test_identity_change_resets_liveness_and_does_not_log():
    t = main.Track(1, row(100, 100, 100))
    t.name, t.cand, t.live = "Farhan", "Farhan", FakeLive()
    lg = FakeLogger()
    ev = main.apply_match(t, "Budi", "2", main.MATCH_THRESH + .2, 10.0, lg, 60)
    assert ev is None and t.name is None and t.live.resets == 1 and not lg.calls


def test_verify_skips_weak_tolerance():
    t = main.Track(1, row(100, 100, 100))
    t.name, t.live = "Farhan", FakeLive()
    main.apply_match(t, None, "", 0.0, 1.0, FakeLogger(), 60)       # tanpa verify: ditoleransi
    assert t.name == "Farhan"
    t.verify = True
    main.apply_match(t, None, "", 0.0, 2.0, FakeLogger(), 60)       # verify: langsung lepas
    assert t.name is None and t.live.resets == 1