import threading
import time

import numpy as np
import pytest

from fixpic.engine import Job, JobScheduler, JobState, WarmupState
from fixpic.errors import (CancelledError, EmptyMaskError, InferenceError, QueueFullError, SchedulerClosedError)
from fixpic.models import ModelConfig, ModelInfo
from fixpic.pipeline import Pipeline, PipelineConfig
from test_backends import tiny  # noqa: F401  (fixture)

T = 5.0   # תקרת המתנה בבדיקות; אף בדיקה לא אמורה להגיע אליה


class GateModel:
    """ממלא את המסכה בצבע קבוע. כל run נרשם ונחסם עד שמשחררים אותו (gate), כדי לשלוט במצב התור."""
    info = ModelInfo(name="gate")

    def __init__(self, gated=True):
        self.gated = gated
        self.gate = threading.Semaphore(0)
        self.entered = threading.Semaphore(0)
        self.log: list = []
        self.active = self.max_active = 0
        self._l = threading.Lock()

    def load(self, cfg): ...
    def unload(self): ...

    def run(self, image, mask, cancel=None):
        with self._l:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.log.append(("run", int(image[0, 0, 0])))
        self.entered.release()
        if self.gated:
            assert self.gate.acquire(timeout=T)
        out = image.copy()
        out[mask > 0] = (9, 9, 9)
        with self._l:
            self.active -= 1
        return out


def img(tag=0, h=120, w=120):
    a = np.zeros((h, w, 3), np.uint8)
    a[...] = tag                      # תג לזיהוי העבודה ביומן המודל
    return a


def msk(h=120, w=120):
    m = np.zeros((h, w), np.uint8)
    m[40:60, 40:60] = 255
    return m


def sched(model, **kw):
    return JobScheduler(Pipeline(model, PipelineConfig(mask_dilate_px=0)), **kw)


def test_run_and_result_outside_identical():
    with sched(GateModel(gated=False)) as s:
        a = img(50)
        job = s.submit(a, msk(), 256)
        res = job.result(T)
        assert job.state is JobState.DONE and job.finished
        eff = res.effective_mask.astype(bool)
        assert np.array_equal(res.image[~eff], a[~eff])
        assert (res.image[45:55, 45:55] == 9).all()
        assert job._image is None and job._mask is None    # הזיכרון של הקלט שוחרר


def test_fifo_order_single_worker():
    m = GateModel()
    with sched(m) as s:
        jobs = [s.submit(img(1), msk(), 256)]
        assert m.entered.acquire(timeout=T)                  # הראשונה רצה; 3 הבאות ממתינות
        jobs += [s.submit(img(i + 2), msk(), 256) for i in range(3)]
        m.gate.release()
        for _ in jobs[1:]:
            assert m.entered.acquire(timeout=T)
            m.gate.release()
        for j in jobs:
            j.result(T)
        assert [t for _, t in m.log] == [1, 2, 3, 4]
        assert m.max_active == 1


def test_queue_limit_three_pending_and_slot_frees():
    m = GateModel()
    with sched(m) as s:
        first = s.submit(img(1), msk(), 256)
        assert m.entered.acquire(timeout=T)                  # הראשונה רצה: התור ריק
        queued = [s.submit(img(i), msk(), 256) for i in (2, 3, 4)]
        assert s.pending == 3 and s.current is first
        with pytest.raises(QueueFullError):
            s.submit(img(5), msk(), 256)
        m.gate.release()
        first.result(T)
        assert m.entered.acquire(timeout=T)                  # השנייה עלתה: התפנה מקום
        s.submit(img(5), msk(), 256)
        assert s.pending == 3
        s.cancel_all()
        m.gate.release()                                     # queued[0] רצה; השאר הוסרו מהתור
        assert queued[0].wait(T) and queued[0].state is JobState.CANCELLED
        assert all(j.state is JobState.CANCELLED for j in queued[1:])


def test_cancel_queued_removed_immediately_and_never_runs():
    m = GateModel()
    done = []
    with sched(m) as s:
        a = s.submit(img(1), msk(), 256)
        assert m.entered.acquire(timeout=T)
        b = s.submit(img(2), msk(), 256, on_done=done.append)
        c = s.submit(img(3), msk(), 256)
        assert b.cancel() is True
        assert b.state is JobState.CANCELLED and s.pending == 1 and done == [b]
        with pytest.raises(CancelledError):
            b.result(0)
        assert b.cancel() is False                           # כבר סופית
        m.gate.release(); m.gate.release()
        a.result(T); c.result(T)
        assert [t for _, t in m.log] == [1, 3]


def test_cancel_running_never_returns_result():
    m = GateModel()
    with sched(m) as s:
        j = s.submit(img(1), msk(), 256)
        assert m.entered.acquire(timeout=T)
        assert j.cancel() is True and j.state is JobState.RUNNING
        m.gate.release()                                     # ההרצה מסתיימת בפועל, אך התוצאה נזרקת
        assert j.wait(T) and j.state is JobState.CANCELLED
        with pytest.raises(CancelledError):
            j.result(0)


def test_cancel_all_undo_race_then_scheduler_still_works():
    m = GateModel()
    with sched(m) as s:
        run = s.submit(img(1), msk(), 256)
        assert m.entered.acquire(timeout=T)
        q = [s.submit(img(i), msk(), 256) for i in (2, 3, 4)]
        assert s.cancel_all() == 4
        assert s.pending == 0 and all(j.state is JobState.CANCELLED for j in q)
        m.gate.release()
        assert run.wait(T) and run.state is JobState.CANCELLED
        m.gated = False
        nxt = s.submit(img(7), msk(), 256)
        assert nxt.result(T) is not None and nxt.state is JobState.DONE


def test_fixpic_error_marks_failed_and_worker_survives():
    with sched(GateModel(gated=False)) as s:
        bad = s.submit(img(1), np.zeros((120, 120), np.uint8), 256)
        assert bad.wait(T) and bad.state is JobState.FAILED
        with pytest.raises(EmptyMaskError):
            bad.result(0)
        assert s.submit(img(2), msk(), 256).result(T) is not None


def test_unexpected_exception_converted_and_worker_survives():
    class Boom(GateModel):
        def run(self, image, mask, cancel=None):
            if int(image[0, 0, 0]) == 1:
                raise RuntimeError("בום")
            return super().run(image, mask, cancel)

    with sched(Boom(gated=False)) as s:
        bad = s.submit(img(1), msk(), 256)
        assert bad.wait(T) and bad.state is JobState.FAILED
        assert isinstance(bad.error, InferenceError) and isinstance(bad.error.__cause__, RuntimeError)
        assert s.submit(img(2), msk(), 256).result(T) is not None


def test_callbacks_on_worker_thread_and_exceptions_isolated():
    seen, threads = [], set()

    def on_progress(job, p):
        threads.add(threading.current_thread().name)
        seen.append((job.id, p.stage))
        raise RuntimeError("callback שבור")

    def on_done(job):
        threads.add(threading.current_thread().name)
        raise RuntimeError("callback שבור")

    with sched(GateModel(gated=False)) as s:
        j = s.submit(img(1), msk(), 256, on_progress=on_progress, on_done=on_done)
        j.result(T)
        assert j.state is JobState.DONE
    assert threads == {"fixpic-worker"} and seen[-1] == (j.id, "done")


def test_submit_requires_running_scheduler():
    s = sched(GateModel(gated=False))
    with pytest.raises(SchedulerClosedError):
        s.submit(img(1), msk(), 256)                         # טרם הופעל
    s.start(); s.shutdown()
    with pytest.raises(SchedulerClosedError):
        s.submit(img(1), msk(), 256)                         # נסגר


def test_shutdown_cancels_everything_and_joins():
    m = GateModel()
    s = sched(m).start()
    run = s.submit(img(1), msk(), 256)
    assert m.entered.acquire(timeout=T)
    q = s.submit(img(2), msk(), 256)
    threading.Timer(0.2, m.gate.release).start()             # ההרצה הפעילה מסתיימת, shutdown מחכה לה
    s.shutdown(timeout=T)
    assert run.state is JobState.CANCELLED and q.state is JobState.CANCELLED
    assert not s._thread.is_alive()


# ------------------------------------------------------------------ Warm-up
class WarmModel(GateModel):
    def __init__(self, hold_first=False, fail=False):
        super().__init__(gated=False)
        self.warm_gate = threading.Event()
        self.warm_entered = threading.Event()
        self.hold_first, self.fail = hold_first, fail
        self.warmed: list = []

    def warmup(self, sizes, cancel=None):
        for s in sizes:
            if cancel:
                cancel.raise_if_cancelled()
            if self.hold_first and not self.warmed:
                self.warm_entered.set()
                assert self.warm_gate.wait(T)
            if self.fail:
                raise InferenceError("compile נכשל")
            self.warmed.append(s)
            self.log.append(("warm", s))


def test_warmup_background_default_bucket_first_all_buckets():
    m = WarmModel()
    with sched(m) as s:
        assert s.wait_warmup(T) and s.warmup_state is WarmupState.DONE
        assert m.warmed == [384, 256, 512]


def test_job_runs_between_warmup_buckets():
    m = WarmModel(hold_first=True)
    with sched(m) as s:
        assert m.warm_entered.wait(T) and s.warmup_state is WarmupState.PENDING
        j = s.submit(img(1), msk(), 256)                     # מגיעה באמצע ה-compile של 384
        m.warm_gate.set()
        j.result(T)
        assert s.wait_warmup(T)
        kinds = [(k, v) for k, v in m.log]
        assert kinds[0] == ("warm", 384) and kinds[1][0] == "run"   # העבודה לפני ה-bucket הבא
        assert [v for k, v in kinds if k == "warm"] == [384, 256, 512]


def test_warmup_failure_does_not_block_jobs():
    m = WarmModel(fail=True)
    with sched(m) as s:
        assert s.wait_warmup(T)
        assert s.warmup_state is WarmupState.FAILED and isinstance(s.warmup_error, InferenceError)
        assert s.submit(img(1), msk(), 256).result(T) is not None


def test_shutdown_during_warmup_marks_cancelled():
    m = WarmModel(hold_first=True)
    s = sched(m).start()
    threading.Timer(0.2, m.warm_gate.set).start()
    s.shutdown(timeout=T)
    assert s.warmup_state is WarmupState.CANCELLED and m.warmed in ([], [384])


def test_no_warmup_when_disabled():
    m = WarmModel()
    with sched(m) as s:
        pass
    m2 = WarmModel()
    s2 = sched(m2).start(warmup=False)
    assert s2.warmup_state is WarmupState.IDLE
    s2.submit(img(1), msk(), 256).result(T)
    s2.shutdown()
    assert m2.warmed == []


# -------------------------------------------------------- backend אמיתי (סינתטי)
def test_with_real_backends_warmup_compiles_buckets(tiny):  # noqa: F811
    from fixpic.models import AutoModel
    am = AutoModel()
    am.load(ModelConfig(tiny, threads=2))
    with JobScheduler(Pipeline(am)) as s:
        assert s.wait_warmup(60) and s.warmup_state is WarmupState.DONE
        a = np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8)
        m = np.zeros((300, 400), np.uint8)
        m[100:140, 150:230] = 255
        res = s.submit(a, m, 256).result(30)
        eff = res.effective_mask.astype(bool)
        assert np.array_equal(res.image[~eff], a[~eff])
