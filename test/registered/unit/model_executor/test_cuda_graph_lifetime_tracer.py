# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model-free regressions for the optional CUDA-graph lifetime tracer.

Real captures exercise ownership, replay and inspection failures. Patches
are scoped per test; importing this extra-tier suite changes no graph methods.
"""

import contextlib
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.cuda_graph_lifetime import harness, tracer
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="extra-a", runner_config="1-gpu-small")

_TORCH_GRAPH_METHODS = ("__init__", "capture_begin", "capture_end", "replay", "reset")


@contextlib.contextmanager
def _traced_torch():
    """Restore methods and bookkeeping after a test or partial installation.

    An already-instrumented process is left alone.
    """
    cls = torch.cuda.CUDAGraph
    if "torch" in tracer._patched:
        yield
        return
    saved = {name: cls.__dict__.get(name) for name in _TORCH_GRAPH_METHODS}
    try:
        tracer.patch_torch()
        yield
    finally:
        for name, original in saved.items():
            if original is None:
                # Inherited before: a wrapper, if one was installed, shadows
                # it on this class; one never installed needs nothing.
                if name in cls.__dict__:
                    delattr(cls, name)
            else:
                setattr(cls, name, original)
        tracer._patched.discard("torch")


class TestPatchScope(unittest.TestCase):
    """Partial installation must restore state and preserve its exception."""

    def test_partial_installation_restores_everything(self):
        cls = torch.cuda.CUDAGraph
        self.assertNotIn("torch", tracer._patched)
        before = {name: cls.__dict__.get(name) for name in _TORCH_GRAPH_METHODS}
        calls = []
        real_wrap = tracer._wrap

        def failing_wrap(klass, name, make):
            calls.append(name)
            if len(calls) == 3:
                raise RuntimeError("injected: wrapper installation failed")
            real_wrap(klass, name, make)

        with patch.object(tracer, "_wrap", failing_wrap):
            with self.assertRaises(RuntimeError) as raised:
                with _traced_torch():
                    self.fail("the scope must not be entered")
        self.assertIn("injected", str(raised.exception))
        self.assertEqual(calls, ["__init__", "capture_begin", "capture_end"])
        self.assertNotIn("torch", tracer._patched)
        for name in _TORCH_GRAPH_METHODS:
            self.assertIs(cls.__dict__.get(name), before[name], name)
            self.assertFalse(
                getattr(getattr(cls, name), "_lifetime_tracer_wrapped", False), name
            )
        # a complete installation restores the same way
        with _traced_torch():
            self.assertTrue(getattr(cls.replay, "_lifetime_tracer_wrapped", False))
        self.assertNotIn("torch", tracer._patched)
        for name in _TORCH_GRAPH_METHODS:
            self.assertIs(cls.__dict__.get(name), before[name], name)


class TestLifetimeTracer(CustomTestCase):
    def setUp(self):
        scope = _traced_torch()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        self.t = tracer.reset_for_tests()
        self.t.check_every = 1
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.stream = torch.cuda.Stream()

    def tearDown(self):
        tracer.reset_for_tests()

    def _capture(self, fn, label="g", pool=None):
        with self.t.labelled(label):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream, pool=pool):
                out = fn()
        return graph, out

    def test_second_capture_dropping_the_first_graphs_buffer_fails_before_replay(self):
        static = torch.ones(1024, device=self.device)
        table = {"meta": torch.full((1024,), 2.0, device=self.device)}
        first, _ = self._capture(lambda: static + table["meta"], "variant-a")
        # The second variant rebuilds the per-key buffer and drops the first.
        table["meta"] = torch.full((1024,), 3.0, device=self.device)
        self._capture(lambda: static + table["meta"], "variant-b")

        with self.assertRaises(RuntimeError) as raised:
            first.replay()
        message = str(raised.exception)
        self.assertIn("variant-a", message)
        self.assertIn("allocated before capture", message)
        self.assertEqual(self.t.status, "dangling")

    def test_static_inputs_and_shared_pool_temporaries_pass(self):
        static = torch.ones(1024, device=self.device)
        pool = torch.cuda.graph_pool_handle()
        first, out = self._capture(lambda: (static * 2.0) + 1.0, "first", pool)
        second, _ = self._capture(lambda: out + static, "second", pool)
        first.replay()
        second.replay()
        torch.cuda.synchronize()
        self.assertEqual(self.t.status, "clean")
        self.assertEqual(self.t.replays, 2)
        self.assertTrue(torch.equal(out, torch.full((1024,), 3.0, device=self.device)))

    def test_persistent_buffer_from_an_earlier_forward_must_stay_alive(self):
        static = torch.ones(1024, device=self.device)
        _, produced = self._capture(lambda: static * 2.0, "producer")
        holder = [produced]
        consumer, _ = self._capture(lambda: holder[0] + 1.0, "consumer")
        consumer.replay()  # clean while the producer's output lives
        self.assertEqual(self.t.status, "clean")
        holder.clear()
        del produced
        with self.assertRaises(RuntimeError) as raised:
            consumer.replay()
        self.assertIn("consumer", str(raised.exception))

    def test_eager_bridge_from_default_and_custom_pools(self):
        for use_custom_pool in (False, True):
            with self.subTest(custom_pool=use_custom_pool):
                self.t = tracer.reset_for_tests()
                self.t.check_every = 1
                static = torch.ones(1024, device=self.device)
                mem_pool = torch.cuda.MemPool() if use_custom_pool else None
                with self.t.forward():
                    _, first = self._capture(lambda: static * 2.0, "seg0")
                    if mem_pool is not None:
                        with torch.cuda.use_mem_pool(mem_pool):
                            bridges = [first + 1.0]
                    else:
                        bridges = [first + 1.0]
                    seg1, _ = self._capture(lambda: bridges[0] * 2.0, "seg1")
                seg1.replay()
                self.assertEqual(self.t.status, "clean")
                bridges.clear()
                with self.assertRaises(RuntimeError) as raised:
                    seg1.replay()
                self.assertIn(
                    "eager bridge allocated during the captured forward",
                    str(raised.exception),
                )

    def test_replay_while_capturing_defers_the_check(self):
        # cuGraphLaunch is refused during stream capture on this driver, so the
        # capturing state is simulated: a replay seen while a capture is open
        # must not synchronize or snapshot; the check runs at the next safe point.
        static = torch.ones(1024, device=self.device)
        graph, _ = self._capture(lambda: static * 2.0, "inner")
        checks = self.t.checks
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.t.before_replay(graph=graph)
        self.assertEqual(self.t.deferred_checks, 1)
        self.assertEqual(self.t.checks, checks)
        self.assertTrue(self.t.deferred)
        outer, _ = self._capture(
            lambda: static * 3.0, "outer"
        )  # safe point: capture end
        self.assertGreater(self.t.checks, checks)
        self.assertFalse(self.t.deferred)
        graph.replay()
        outer.replay()
        torch.cuda.synchronize()
        self.assertEqual(self.t.status, "clean")

    def test_failed_capture_reset_destruction_and_recapture(self):
        static = torch.ones(1024, device=self.device)
        with self.assertRaises(ValueError):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                raise ValueError("capture failed")
        # torch ends the capture on the way out; the discarded graph's record
        # goes with the object, and nothing stays pending.
        self.assertEqual(self.t._pending_begin, {})
        del graph
        import gc

        gc.collect()
        self.assertEqual(len(self.t.records), 0)

        graph, _ = self._capture(lambda: static * 2.0, "kept")
        self.assertEqual(len(self.t.records), 1)
        graph.reset()
        self.assertEqual(len(self.t.records), 0)
        graph2, _ = self._capture(lambda: static * 3.0, "recaptured")
        self.assertEqual(len(self.t.records), 1)
        del graph2
        gc.collect()
        self.assertEqual(len(self.t.records), 0)
        self.assertEqual(self.t.retired, 3)

    def test_dedup_direct_launch_is_checked(self):
        from sglang.srt.model_executor.runner_backend import (
            cuda_graph_dedup_mixin as dedup,
        )

        if dedup.cuda_drv is None or dedup.cuda_rt is None:
            self.skipTest("cuda-python not available")
        name = "sglang.srt.model_executor.runner_backend.cuda_graph_dedup_mixin"
        if name not in tracer._patched:
            original_replay = dedup.DedupedCudaGraph.__dict__.get("replay")

            def restore():
                if original_replay is None:
                    delattr(dedup.DedupedCudaGraph, "replay")
                else:
                    dedup.DedupedCudaGraph.replay = original_replay
                tracer._patched.discard(name)

            self.addCleanup(restore)
        tracer.patch_sglang_module(name, dedup)
        static = torch.ones(1024, device=self.device)
        table = {"meta": torch.full((1024,), 2.0, device=self.device)}
        registry = dedup.DedupedCudaGraphRegistry()
        first, _ = self._capture(lambda: static + table["meta"], "dedup-a")
        deduped = registry.register(first)
        table["meta"] = torch.full((1024,), 3.0, device=self.device)
        second, _ = self._capture(lambda: static + table["meta"], "dedup-b")
        registry.register(second)  # same signature: shares the executable via update
        self.assertEqual(registry.stats(), (2, 1))
        with self.assertRaises(RuntimeError) as raised:
            deduped.replay(self.stream.cuda_stream)
        self.assertIn("dedup-a", str(raised.exception))
        registry.close()

    def test_incomplete_inspection_fails(self):
        static = torch.ones(1024, device=self.device)
        graph, _ = self._capture(lambda: (static * 2.0) + 1.0, "no-pool-info")
        real_snapshot = torch.cuda.memory._snapshot

        def without_pool_ids(device=None):
            snapshot = real_snapshot(device)
            for entry in snapshot["device_traces"][self.device.index]:
                entry.pop("pool_id", None)
            return snapshot

        with patch.object(torch.cuda.memory, "_snapshot", side_effect=without_pool_ids):
            with self.assertRaises(RuntimeError) as raised:
                graph.replay()
        self.assertIn("incomplete", str(raised.exception))
        self.assertIn("no allocator pool information", str(raised.exception))

    def test_live_report_matches_harness(self):
        static = torch.ones(1024, device=self.device)
        graph, _ = self._capture(lambda: static * 2.0, "reported")
        graph.replay()
        self.t.flush("SIGUSR1")
        report = self.t.last_report
        self.assertEqual(report["status"], "clean")
        self.assertEqual(report["label_captures"], {"reported": 1})
        self.assertEqual(report["counts"]["replays"], 1)

        self.assertEqual(report["tracer_version"], tracer.TRACER_VERSION)
        self.assertEqual(report["tracer_version"], harness.TRACER_VERSION)
        self.assertTrue(report["device_uuid"])
        self.assertTrue(report["source"]["sglang_file"].endswith("sglang/__init__.py"))
        self.assertTrue(
            report["parallel"] is None or isinstance(report["parallel"], dict)
        )
        self.assertEqual(harness.lifetime_verdict([report], 1)["verdict"], "PASS")
        # label replays are counted by the SGLang backend wrappers, not by a
        # raw CUDAGraph.replay(): the wrapper's count reaches the report
        self.assertEqual(report["label_replays"], {})
        self.t.count_label_replay("reported")
        self.t.flush("SIGUSR1")
        self.assertEqual(self.t.last_report["label_replays"], {"reported": 1})

    def test_flush_request_file_is_honoured_at_the_next_replay(self):
        """The harness's file trigger: an idle worker whose signal handler
        cannot run flushes at the next (probe-driven) replay."""
        with tempfile.TemporaryDirectory() as report_dir:
            self.t.report_dir = report_dir
            self.t.run_id = "run-x"
            self.t.check_every = 1000
            static = torch.ones(1024, device=self.device)
            graph, _ = self._capture(lambda: static * 2.0, "flushed")
            graph.replay()  # first replay of a new graph: inspected
            self.assertEqual(self.t.last_report["reason"], "before replay")
            graph.replay()  # not due: no new report
            self.assertEqual(self.t.last_report["reason"], "before replay")
            with open(self.t.flush_file(), "w") as f:
                f.write("run-x-nonce")
            graph.replay()
            self.assertEqual(self.t.last_report["reason"], "flush-request")
            self.assertEqual(self.t.last_report["flush_nonce"], "run-x-nonce")
            self.assertEqual(self.t.last_report["run_id"], "run-x")
            self.assertFalse(os.path.exists(self.t.flush_file()))
            self.assertEqual(self.t.last_report["counts"]["replays"], 3)
            self.assertEqual(harness._read_reports(report_dir, "worker-", "run-y"), [])
            reports = harness._read_reports(report_dir, "worker-", "run-x")
            self.assertEqual([r["reason"] for r in reports], ["flush-request"])
            self.assertEqual(
                harness.lifetime_verdict(reports, 1, "run-x", "run-x-nonce")["verdict"],
                "PASS",
            )
            # the acknowledgement of another nonce is not this run's flush
            self.assertEqual(
                harness.lifetime_verdict(reports, 1, "run-x", "run-x-later")["verdict"],
                "FAIL",
            )

    def test_second_capture_device_is_rejected(self):
        static = torch.ones(1024, device=self.device)
        self._capture(lambda: static * 2.0, "first-device")
        # Simulated: a real second device is not always available, and torch
        # itself would reject a patched current device before the tracer runs.
        with patch.object(
            tracer, "_current_device_index", return_value=self.device.index + 1
        ):
            with self.assertRaises(RuntimeError) as raised:
                self._capture(lambda: static * 3.0, "second-device")
        self.assertIn("one capture device per process", str(raised.exception))
        self.assertEqual(self.t.effective_status(), "failed")
        self.assertEqual(self.t.last_report["reason"], "device conflict")
        self.assertEqual(self.t._pending_begin, {})

    def test_failure_history_survives_retirement_without_fail_fast(self):
        self.t.fail_fast = False
        static = torch.ones(1024, device=self.device)
        table = {"meta": torch.full((1024,), 2.0, device=self.device)}
        first, _ = self._capture(lambda: static + table["meta"], "variant-a")
        table["meta"] = torch.full((1024,), 3.0, device=self.device)
        second, _ = self._capture(lambda: static + table["meta"], "variant-b")
        second.replay()  # the check finds variant-a dangling; no raise
        self.assertEqual(self.t.status, "dangling")
        del first
        import gc

        gc.collect()  # the dangling graph is retired
        clean, _ = self._capture(lambda: static * 2.0, "clean")
        clean.replay()
        self.assertEqual(self.t.status, "clean")  # the latest check
        report = self.t.report_dict("check")
        self.assertEqual(report["status"], "dangling")  # never erased
        self.assertEqual(report["current_status"], "clean")
        self.assertEqual(report["status_history"][0]["status"], "dangling")
        self.assertEqual(
            harness.lifetime_verdict([self.t.report_dict("flush-request")], 1)[
                "verdict"
            ],
            "FAIL",
        )


if __name__ == "__main__":
    unittest.main()
