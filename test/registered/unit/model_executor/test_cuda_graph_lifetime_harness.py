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
"""Verdict logic of the CUDA-graph validation harness on synthetic evidence:
malformed or shortened responses, non-finite scores, requests without
comparable decode positions, crashes that a batch-sensitivity control must
not excuse, and the per-request control itself. No server, no GPU."""

import hashlib
import json
import math
import os
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cuda_graph_lifetime import harness

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NEW = 9  # generated positions per request: 8 decode positions each
ALT = 1000  # the runner-up token at every generated position


def _response(
    prompt_tokens, generated, prompt_lp=-1.0, gen_lp=-0.5, alt_lp=-0.9, runner_up=ALT
):
    """A /generate response: the greedy token first in every top list, the
    runner-up within the bound so an argmax flip is comparable."""
    tops = []
    for token in generated:
        second = runner_up if runner_up != token else ALT + 1
        entries = [[gen_lp, token, None], [alt_lp, second, None]]
        entries += [[-9.0 - i, 2000 + i, None] for i in range(harness.TOP_LOGPROBS - 2)]
        tops.append(entries)
    return {
        "meta_info": {
            "input_token_logprobs": [[None, prompt_tokens[0], None]]
            + [[prompt_lp, t, None] for t in prompt_tokens[1:]],
            "output_token_logprobs": [[gen_lp, t, None] for t in generated],
            "output_top_logprobs": tops,
        }
    }


def _outputs(**overrides):
    base = {
        "bs2-none-r0-0": _response([1, 2, 3], [11] * NEW),
        "bs2-none-r0-1": _response([4, 5], [12] * NEW),
    }
    base.update(overrides)
    return base


class TestCompare(unittest.TestCase):
    def test_identical_outputs_pass(self):
        result = harness.compare(_outputs(), _outputs(), "cand", NEW)
        self.assertEqual((result["verdict"], result["kind"]), ("PASS", "drift"))
        self.assertEqual(result["compared_decode_positions"], 16)
        self.assertEqual(result["problems"], [])

    def test_non_finite_scores_are_malformed(self):
        nan_prompt = _response([1, 2, 3], [11] * NEW)
        nan_prompt["meta_info"]["input_token_logprobs"][1][0] = math.nan
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": nan_prompt}), "cand", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "malformed"))
        self.assertTrue(any("not finite" in p for p in result["problems"]))
        inf_top = _response([1, 2, 3], [11] * NEW)
        inf_top["meta_info"]["output_top_logprobs"][3][1][0] = math.inf
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": inf_top}), "cand", NEW
        )
        self.assertEqual(result["kind"], "malformed")
        self.assertTrue(any("malformed top-logprob" in p for p in result["problems"]))

    def test_shortened_or_inconsistent_arrays_are_malformed(self):
        short = _response(
            [1, 2, 3], [11] * (NEW - 1)
        )  # stopped early despite ignore_eos
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": short}), "c", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "malformed"))
        self.assertTrue(any("expected 9 (ignore_eos)" in p for p in result["problems"]))
        fewer_tops = _response([1, 2, 3], [11] * NEW)
        fewer_tops["meta_info"]["output_top_logprobs"].pop()
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": fewer_tops}), "c", NEW
        )
        self.assertEqual(result["kind"], "malformed")
        thin_top = _response([1, 2, 3], [11] * NEW)
        thin_top["meta_info"]["output_top_logprobs"][0] = thin_top["meta_info"][
            "output_top_logprobs"
        ][0][:3]
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": thin_top}), "c", NEW
        )
        self.assertTrue(any("expected 5" in p for p in result["problems"]))
        wrong_greedy = _response([1, 2, 3], [11] * NEW)
        wrong_greedy["meta_info"]["output_token_logprobs"][2][1] = 4242
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": wrong_greedy}), "c", NEW
        )
        self.assertTrue(any("not in its own top-5" in p for p in result["problems"]))

    def test_inconsistent_scores_and_repeated_top_ids_are_malformed(self):
        # the greedy token's own score changed while its top-list entry did not
        off = _response([1, 2, 3], [11] * NEW)
        off["meta_info"]["output_token_logprobs"][4][0] = -99.0
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": off}), "c", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "malformed"))
        self.assertTrue(any("scores -99.000 but" in p for p in result["problems"]))
        within = _response([1, 2, 3], [11] * NEW)
        within["meta_info"]["output_token_logprobs"][4][0] = -0.5 + 5e-4
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": within}), "c", NEW
        )
        self.assertEqual(result["verdict"], "PASS")
        # five copies of the greedy token are not a top-5
        copies = _response([1, 2, 3], [11] * NEW)
        copies["meta_info"]["output_top_logprobs"][0] = [[-0.5, 11, None]] * 5
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": copies}), "c", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "malformed"))
        self.assertTrue(any("repeated token ids" in p for p in result["problems"]))

    def test_prompt_token_mismatch_and_missing_requests_are_malformed(self):
        other_prompt = _response([1, 2, 9], [11] * NEW)
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": other_prompt}), "c", NEW
        )
        self.assertEqual(result["kind"], "malformed")
        self.assertTrue(any("prompt tokens differ" in p for p in result["problems"]))
        candidate = _outputs()
        candidate["bs9-none-r0-0"] = candidate.pop("bs2-none-r0-1")
        result = harness.compare(_outputs(), candidate, "c", NEW)
        self.assertEqual(result["kind"], "malformed")
        self.assertIn("bs2-none-r0-1: missing in c", result["problems"])
        self.assertIn("bs9-none-r0-0: unexpected in c", result["problems"])

    def test_request_without_decode_positions_is_insufficient_evidence(self):
        # an argmax flip at position 0: the reference's token is the runner-up
        flipped = _response([1, 2, 3], [ALT] + [11] * (NEW - 1), runner_up=11)
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": flipped}), "c", NEW
        )
        self.assertEqual(
            (result["verdict"], result["kind"]),
            ("INCONCLUSIVE", "insufficient-evidence"),
        )
        self.assertEqual(result["requests_without_decode_positions"], ["bs2-none-r0-0"])
        self.assertEqual(result["drift_problems"], [])
        self.assertEqual(
            result["request_drift"]["bs2-none-r0-0"]["decode_positions"], 0
        )
        # the other request alone does not satisfy the global minimum of 8 either
        self.assertEqual(result["compared_decode_positions"], 8)

    def test_drift_beyond_the_bound_fails(self):
        drifted = _response([1, 2, 3], [11] * NEW, prompt_lp=-2.6)
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": drifted}), "c", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "drift"))
        self.assertEqual(
            result["failing_requests"], {"bs2-none-r0-0": {"prompt": True}}
        )
        self.assertAlmostEqual(result["request_drift"]["bs2-none-r0-0"]["prompt"], 1.6)
        decode = _response([1, 2, 3], [11] * NEW, gen_lp=-1.7)
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": decode}), "c", NEW
        )
        self.assertEqual(
            result["failing_requests"], {"bs2-none-r0-0": {"decode": True}}
        )

    def test_token_absent_from_the_other_top_list_is_bounded_not_measured(self):
        # the candidate's first token (3000) is far outside the reference's
        # top-5: the difference is at least 10.5 nats, a drift failure
        far = _response([1, 2, 3], [3000] + [11] * (NEW - 1))
        result = harness.compare(
            _outputs(), _outputs(**{"bs2-none-r0-0": far}), "c", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "drift"))
        self.assertTrue(any("at least 10.500" in p for p in result["problems"]))
        drift = result["request_drift"]["bs2-none-r0-0"]
        self.assertEqual((drift["decode_unbounded"], drift["decode_positions"]), (2, 0))
        # the same request moving as much in the control does not explain a
        # bound: nothing was measured
        result = harness.numerical_verdict(
            {
                "reference": _outputs(),
                "ordinary": _outputs(**{"bs2-none-r0-0": far}),
                "reference2": _outputs(**{"bs2-none-r0-0": far}),
            },
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "drift"))
        self.assertTrue(
            any("not a measured difference" in p for p in result["control_unexplained"])
        )
        # a near tie: the absent token's bound stays under the tolerance, so the
        # position is not measurable (insufficient evidence, never a pass)
        reference = _outputs()
        reference["bs2-none-r0-0"]["meta_info"]["output_top_logprobs"][0] = [
            [-0.5, 11, None],
            [-0.6, 1000, None],
            [-0.7, 2000, None],
            [-0.8, 2001, None],
            [-0.9, 2002, None],
        ]
        near = _response([1, 2, 3], [3000] + [11] * (NEW - 1))
        near["meta_info"]["output_top_logprobs"][0] = [
            [-0.5, 3000, None],
            [-0.6, 11, None],
            [-0.7, 1000, None],
            [-0.8, 2000, None],
            [-0.9, 2001, None],
        ]
        result = harness.compare(
            reference, _outputs(**{"bs2-none-r0-0": near}), "c", NEW
        )
        self.assertEqual(
            (result["verdict"], result["kind"]),
            ("INCONCLUSIVE", "insufficient-evidence"),
        )
        self.assertTrue(any("not measurable" in p for p in result["problems"]))
        self.assertEqual(result["drift_problems"], [])


class TestNumericalVerdict(unittest.TestCase):
    def test_crash_and_missing_outputs_keep_their_status(self):
        outputs = {
            "reference": _outputs(),
            "reference2": _outputs(
                **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, prompt_lp=-5.0)}
            ),
        }
        phases = {"ordinary": {"error": "RuntimeError('server exited')"}}
        result = harness.numerical_verdict(outputs, phases, "ordinary", NEW)
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "crash"))
        self.assertNotIn("batch_sensitivity_control", result)
        # a crashed candidate is a crash even when the reference left no
        # outputs: errors of both sides come before either side's outputs
        result = harness.numerical_verdict(
            {"reference2": _outputs()}, phases, "ordinary", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "crash"))
        result = harness.numerical_verdict(
            {"ordinary": _outputs()}, {"reference": {"error": "x"}}, "ordinary", NEW
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "crash"))
        result = harness.numerical_verdict(outputs, {}, "ordinary", NEW)
        self.assertEqual(result["verdict"], "NOT-RUN")
        self.assertEqual(
            harness.numerical_verdict({}, {}, "ordinary", NEW, skip_reference=True)[
                "verdict"
            ],
            "NOT-RUN",
        )

    def test_malformed_candidate_is_not_reclassified(self):
        broken = _response([1, 2, 3], [11] * (NEW - 2))
        outputs = {
            "reference": _outputs(),
            "ordinary": _outputs(**{"bs2-none-r0-0": broken}),
            "reference2": _outputs(
                **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, prompt_lp=-5.0)}
            ),
        }
        result = harness.numerical_verdict(outputs, {}, "ordinary", NEW)
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "malformed"))

    def test_control_explains_only_the_same_request_and_kind(self):
        drifted = _response(
            [1, 2, 3], [11] * NEW, prompt_lp=-2.5
        )  # 1.5 nats on request 0
        reference, ordinary = _outputs(), _outputs(**{"bs2-none-r0-0": drifted})
        same = _outputs(
            **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, prompt_lp=-2.4)}
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": ordinary, "reference2": same},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual(
            (result["verdict"], result["kind"]), ("INCONCLUSIVE", "batch-sensitivity")
        )
        self.assertEqual(result["control_unexplained"], [])
        self.assertIn("request_drift", result["batch_sensitivity_control"])
        # the control moves on the other request only: the failure stands
        other = _outputs(
            **{"bs2-none-r0-1": _response([4, 5], [12] * NEW, prompt_lp=-4.0)}
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": ordinary, "reference2": other},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "drift"))
        self.assertEqual(len(result["control_unexplained"]), 1)
        # the control moves on decode, the candidate drifts on the prompt: not explained
        decode_only = _outputs(
            **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, gen_lp=-1.9)}
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": ordinary, "reference2": decode_only},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual(result["verdict"], "FAIL")
        # the control moves less than the bound, or less than half the candidate: not explained
        small = _outputs(
            **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, prompt_lp=-1.7)}
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": ordinary, "reference2": small},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual(result["verdict"], "FAIL")
        far = _outputs(
            **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, prompt_lp=-6.0)}
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": far, "reference2": same},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(any("does not explain" in p for p in result["problems"]))

    def test_control_must_measure_the_same_decode_evidence(self):
        reference = _outputs()
        # 1.3 nats measured on every decode position of request 0
        measured = _outputs(
            **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, gen_lp=-1.8)}
        )
        # a control whose evidence for that request is a lower bound only: its
        # first token is far outside the reference's top-5 (two absent-token
        # cases, a 10.5-nat bound, no comparable decode position)
        bound_only = _outputs(
            **{"bs2-none-r0-0": _response([1, 2, 3], [3000] + [11] * (NEW - 1))}
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": measured, "reference2": bound_only},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "drift"))
        self.assertTrue(
            any("control's decode evidence" in p for p in result["control_unexplained"])
        )
        # a control with no comparable decode position (an argmax flip at
        # position 0 within the bound) explains nothing either
        flipped = _outputs(
            **{
                "bs2-none-r0-0": _response(
                    [1, 2, 3], [ALT] + [11] * (NEW - 1), runner_up=11
                )
            }
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": measured, "reference2": flipped},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "drift"))
        # measured control evidence of the same size still explains it
        same = _outputs(
            **{"bs2-none-r0-0": _response([1, 2, 3], [11] * NEW, gen_lp=-1.7)}
        )
        result = harness.numerical_verdict(
            {"reference": reference, "ordinary": measured, "reference2": same},
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual(
            (result["verdict"], result["kind"]), ("INCONCLUSIVE", "batch-sensitivity")
        )

    def test_malformed_control_leaves_the_failure(self):
        drifted = _response([1, 2, 3], [11] * NEW, prompt_lp=-2.5)
        broken = _outputs(**{"bs2-none-r0-0": _response([1, 2, 3], [11] * 3)})
        result = harness.numerical_verdict(
            {
                "reference": _outputs(),
                "ordinary": _outputs(**{"bs2-none-r0-0": drifted}),
                "reference2": broken,
            },
            {},
            "ordinary",
            NEW,
        )
        self.assertEqual((result["verdict"], result["kind"]), ("FAIL", "drift"))
        self.assertTrue(any("control is malformed" in p for p in result["problems"]))


def _parallel(tp_rank=0, pp_rank=0, dp_rank=None, tp_size=1, pp_size=1, dp_size=1):
    return {
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "dp_rank": dp_rank,
        "attn_dp_rank": 0,
        "tp_size": tp_size,
        "pp_size": pp_size,
        "dp_size": dp_size,
        "attn_dp_size": 1,
        "nnodes": 1,
        "node_rank": 0,
    }


def _report(**overrides):
    report = {
        "tracer_version": harness.TRACER_VERSION,
        "run_id": "run-1",
        "pid": 100,
        "hostname": "node-a",
        "rank": 0,
        "parallel": _parallel(),
        "device": "cuda:0",
        "device_uuid": "GPU-0000",
        "source": {
            "sglang_file": "/tree/python/sglang/__init__.py",
            "git": ["c1", "t1"],
        },
        "status": "clean",
        "reason": "flush-request",
        "flush_nonce": "run-1-7",
        "counts": {"captures": 6, "replays": 40, "checks": 5, "retired": 0},
        "findings": {"status": "clean", "graphs": 6, "pointers": 900, "dangling": []},
        "label_captures": {"DecodeCudaGraphRunner size=1 variant_label=lora": 1},
        "label_replays": {"DecodeCudaGraphRunner size=1 variant_label=lora": 9},
        "errors": [],
    }
    report.update(overrides)
    return report


class TestLifetimeVerdict(unittest.TestCase):
    def test_evidence_based_pass(self):
        verdict = harness.lifetime_verdict([_report()], 1, "run-1", "run-1-7")
        self.assertEqual(verdict["verdict"], "PASS", verdict["problems"])
        self.assertEqual(verdict["workers"][0]["rank"], 0)

    def test_old_run_zero_checks_and_stale_flush_do_not_pass(self):
        cases = {
            "report of another run": _report(run_id="run-0"),
            "zero inspections": _report(
                counts={"captures": 6, "replays": 40, "checks": 0}
            ),
            "no acknowledged final flush": _report(flush_nonce="run-1-6"),
            "no clean inspection findings": _report(findings={}),
            "no replay observed": _report(
                counts={"captures": 6, "replays": 0, "checks": 5}
            ),
            "no captured graph": _report(
                counts={"captures": 0, "replays": 3, "checks": 1}
            ),
            "unexpected tracer version": _report(tracer_version=1),
        }
        for needle, report in cases.items():
            with self.subTest(needle):
                verdict = harness.lifetime_verdict([report], 1, "run-1", "run-1-7")
                self.assertEqual(verdict["verdict"], "FAIL")
                self.assertTrue(
                    any(needle in p for p in verdict["problems"]), verdict["problems"]
                )
        self.assertEqual(
            harness.lifetime_verdict([], 1, "run-1", "run-1-7")["verdict"],
            "NO-WORKERS",
        )
        verdict = harness.lifetime_verdict(
            [_report(), _report(pid=101)], 2, "run-1", "run-1-7"
        )
        self.assertTrue(any("same device" in p for p in verdict["problems"]))
        verdict = harness.lifetime_verdict([_report()], 2, "run-1", "run-1-7")
        self.assertTrue(any("expected 2" in p for p in verdict["problems"]))

    def test_dangling_report_fails_with_its_finding(self):
        report = _report(
            status="dangling",
            findings={
                "status": "dangling",
                "graphs": 6,
                "dangling_total": 1,
                "dangling": [
                    {
                        "graph": "g",
                        "pointer": "0x1",
                        "allocated_at": "a",
                        "freed_at": "f",
                    }
                ],
            },
        )
        verdict = harness.lifetime_verdict([report], 1, "run-1", "run-1-7")
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertTrue(any("freed while alive" in p for p in verdict["problems"]))

    def test_reports_of_other_runs_are_not_read(self):
        with tempfile.TemporaryDirectory() as d:
            for name, run in (("worker-1.json", "run-0"), ("worker-2.json", "run-1")):
                with open(os.path.join(d, name), "w") as f:
                    json.dump(_report(run_id=run, pid=int(name[7])), f)
            with open(os.path.join(d, "worker-3.json"), "w") as f:
                f.write("{not json")
            reports = harness._read_reports(d, "worker-", "run-1")
            self.assertEqual([r["pid"] for r in reports], [2])
            self.assertEqual(len(harness._read_reports(d, "worker-")), 2)


def _resolved_topology(**sizes):
    resolved = {
        "tp_size": "1",
        "pp_size": "1",
        "dp_size": "1",
        "attn_dp_size": "1",
        "nnodes": "1",
    }
    resolved.update({k: str(v) for k, v in sizes.items()})
    return resolved


class TestTopology(unittest.TestCase):
    def test_workers_follow_the_resolved_parallelism(self):
        topo = harness.topology(_resolved_topology(tp_size=2))
        self.assertEqual(topo["expected_workers"], 2)
        self.assertEqual(topo["identities"], [[0, 0, 0], [0, 0, 1]])
        self.assertIsNone(topo["unsupported"])
        # attention data parallelism partitions the tensor ranks: no extra worker
        topo = harness.topology(_resolved_topology(tp_size=2, attn_dp_size=2))
        self.assertEqual(topo["expected_workers"], 2)
        # replicas multiply them, with the replica in the identity
        topo = harness.topology(_resolved_topology(tp_size=2, dp_size=2))
        self.assertEqual(topo["expected_workers"], 4)
        self.assertIn([1, 0, 1], topo["identities"])
        for name, sizes in {
            "multi-node": {"nnodes": 2, "tp_size": 4},
            "replicas with attention": {"dp_size": 2, "attn_dp_size": 2, "tp_size": 2},
            "unreadable": {"tp_size": "x"},
        }.items():
            with self.subTest(name):
                topo = harness.topology(_resolved_topology(**sizes))
                self.assertIsNone(topo["expected_workers"])
                self.assertTrue(topo["unsupported"])
        self.assertEqual(harness.topology({})["source"], "unavailable")

    def test_command_line_aliases_only_estimate(self):
        for command, workers in {
            "python -m sglang.launch_server --model-path m --tp 2": 2,
            "python -m sglang.launch_server --model-path m --tensor-parallel-size 2 --pp-size 2": 4,
            "python -m sglang.launch_server --model-path m --tp-size=2 --dp 2": 4,
            "python -m sglang.launch_server --model-path m": 1,
        }.items():
            with self.subTest(command):
                cmd = harness.ServingCommand.parse(command)
                self.assertEqual(cmd.expected_workers, workers)

    def test_every_expected_worker_must_report_itself(self):
        topo = harness.topology(_resolved_topology(tp_size=2))
        rank0 = _report(pid=1, parallel=_parallel(0, tp_size=2), device_uuid="GPU-a")
        rank1 = _report(
            pid=2,
            parallel=_parallel(1, tp_size=2),
            device="cuda:1",
            device_uuid="GPU-b",
        )
        verdict = harness.lifetime_verdict([rank0, rank1], 2, "run-1", "run-1-7", topo)
        self.assertEqual(verdict["verdict"], "PASS", verdict["problems"])
        self.assertEqual(verdict["identity"], "verified")
        # the same logical name on distinct physical devices is two workers
        same_name = _report(
            pid=2,
            parallel=_parallel(1, tp_size=2),
            device="cuda:0",
            device_uuid="GPU-b",
        )
        verdict = harness.lifetime_verdict(
            [rank0, same_name], 2, "run-1", "run-1-7", topo
        )
        self.assertEqual(verdict["verdict"], "PASS", verdict["problems"])
        cases = {
            "no report from worker dp=0 pp=0 tp=1": [rank0],
            "reported by 2 processes": [
                rank0,
                _report(pid=3, parallel=_parallel(0, tp_size=2), device_uuid="GPU-c"),
            ],
            "cannot stand in for a worker": [
                rank0,
                _report(pid=4, parallel=None, device_uuid="GPU-d"),
            ],
            "same physical device": [
                rank0,
                _report(pid=2, parallel=_parallel(1, tp_size=2), device_uuid="GPU-a"),
            ],
            "physical device identity unavailable": [
                _report(pid=1, parallel=_parallel(0, tp_size=2), device_uuid=None),
                _report(pid=2, parallel=_parallel(1, tp_size=2), device_uuid=None),
            ],
            "reports tp_size=1, resolved 2": [
                _report(pid=1, parallel=_parallel(0), device_uuid="GPU-a"),
                rank1,
            ],
            "outside the resolved topology": [
                rank0,
                _report(pid=2, parallel=_parallel(5, tp_size=2), device_uuid="GPU-b"),
            ],
        }
        for needle, reports in cases.items():
            with self.subTest(needle):
                verdict = harness.lifetime_verdict(reports, 2, "run-1", "run-1-7", topo)
                self.assertEqual(verdict["verdict"], "FAIL")
                self.assertTrue(
                    any(needle in p for p in verdict["problems"]), verdict["problems"]
                )
                coverage = harness.coverage_verdict(
                    reports, 2, ["DecodeCudaGraphRunner"], [], None, topo
                )
                self.assertEqual(coverage["verdict"], "INCOMPLETE")
                self.assertTrue(any(needle in p for p in coverage["problems"]))
        # an unsupported topology fails explicitly
        unsupported = harness.topology(_resolved_topology(nnodes=2, tp_size=2))
        verdict = harness.lifetime_verdict(
            [rank0, rank1], 2, "run-1", "run-1-7", unsupported
        )
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertTrue(any("unsupported topology" in p for p in verdict["problems"]))
        # without a topology the identities are declared unverified
        verdict = harness.lifetime_verdict([rank0, rank1], 2, "run-1", "run-1-7")
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertTrue(verdict["identity"].startswith("unverified"))

    def test_source_identity_is_verified_only_against_worker_reports(self):
        cwd = ["c1", "t1"]
        verified = harness.source_identity(cwd, [_report()], cwd_dirty=False)
        self.assertTrue(
            verified["source_identity"].startswith("verified by git identity")
        )
        self.assertIn("not a byte check", verified["source_identity"])
        valid = _report()
        no_git = _report(
            pid=7,
            source={"sglang_file": "/site-packages/sglang/__init__.py", "git": None},
        )
        cases = {
            "no worker report": [],
            "reported no git identity": [no_git],
            "pid(s) [7]": [valid, no_git],  # one valid source does not cover the other
            "pid(s) [8]": [valid, _report(pid=8, source=None)],
            "workers imported": [
                _report(
                    source={
                        "sglang_file": "/other/sglang/__init__.py",
                        "git": ["c2", "t2"],
                    }
                )
            ],
            "uncommitted changes in the checkout of worker": [
                _report(
                    source={
                        "sglang_file": "/tree/python/sglang/__init__.py",
                        "git": ["c1", "t1"],
                        "dirty": True,
                    }
                )
            ],
        }
        for needle, reports in cases.items():
            with self.subTest(needle):
                status = harness.source_identity(cwd, reports)["source_identity"]
                self.assertTrue(status.startswith("unverified"), status)
                self.assertIn(needle, status)
        status = harness.source_identity(cwd, [valid], cwd_dirty=True)[
            "source_identity"
        ]
        self.assertIn("uncommitted changes in the harness working directory", status)

    def test_recompute_keeps_not_run_for_a_launch_that_never_ran(self):
        # an instrumented-only run (--skip-reference --skip-ordinary) archived
        # without its outputs file: the numerical verdict stays NOT-RUN (no
        # reference ever ran) and coverage has no agreement to lose
        with tempfile.TemporaryDirectory() as d:
            ledger = {
                "run_id": "run-1",
                "max_new_tokens": NEW,
                "expected_workers": 1,
                "expect_labels": ["DecodeCudaGraphRunner"],
                "verdicts": {
                    "numerical": {"verdict": "NOT-RUN"},
                    "coverage": {"verdict": "PASS"},
                },
                "evidence": {
                    "outputs": {
                        "instrumented": {
                            "path": "instrumented.outputs.json",
                            "sha256": "0" * 64,
                            "requests": 2,
                        }
                    }
                },
                "phases": {
                    "instrumented": {
                        "requests": 2,
                        "worker_reports": [_report()],
                        "flush_nonce": "run-1-7",
                        "resolved_args": _resolved_topology(),
                    }
                },
            }
            verdicts = _recompute(d, ledger)["verdicts"]
            self.assertEqual(verdicts["numerical"]["verdict"], "NOT-RUN")
            self.assertEqual(verdicts["lifetime"]["verdict"], "PASS")
            self.assertEqual(
                verdicts["coverage"]["verdict"],
                "PASS",
                verdicts["coverage"]["problems"],
            )
            self.assertNotIn("instrumented_vs_ordinary", verdicts["coverage"])

    def test_recompute_keeps_a_recorded_instrumented_failure(self):
        # clean reports and an acknowledged nonce survive a worker that died
        # afterwards; the recorded launch error must still fail the verdict
        with tempfile.TemporaryDirectory() as d:
            ledger = {
                "run_id": "run-1",
                "max_new_tokens": NEW,
                "expected_workers": 1,
                "expect_labels": ["DecodeCudaGraphRunner"],
                "verdicts": {"lifetime": {"verdict": "FAIL"}},
                "phases": {
                    "instrumented": {
                        "error": "RuntimeError('server exited with -9')",
                        "worker_reports": [_report()],
                        "flush_nonce": "run-1-7",
                        "resolved_args": _resolved_topology(),
                    }
                },
            }
            lifetime = _recompute(d, ledger)["verdicts"]["lifetime"]
            self.assertEqual(lifetime["verdict"], "FAIL")
            self.assertTrue(any("recorded an error" in p for p in lifetime["problems"]))
            self.assertEqual(lifetime["identity"], "verified")  # the reports were fine
        # the live path applies the same rule
        self.assertEqual(
            harness._instrumented_failure({"error": "x"}, {"verdict": "PASS"})[
                "verdict"
            ],
            "FAIL",
        )
        self.assertEqual(
            harness._instrumented_failure({}, {"verdict": "PASS"})["verdict"], "PASS"
        )

    def test_recompute_does_not_rejudge_reports_without_identity(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = {
                "run_id": "run-1",
                "max_new_tokens": NEW,
                "expected_workers": 1,
                "expect_labels": [],
                "verdicts": {"lifetime": {"verdict": "PASS"}},
                "phases": {
                    "instrumented": {
                        "worker_reports": [_report(tracer_version=2, parallel=None)],
                        "flush_nonce": "run-1-7",
                        "resolved_args": _resolved_topology(),
                    }
                },
            }
            result = _recompute(d, ledger)
            lifetime = result["verdicts"]["lifetime"]
            self.assertEqual(lifetime["verdict"], "NOT-RECOMPUTED")
            self.assertTrue(any("worker identity" in p for p in lifetime["problems"]))
            # coverage is still judged on the replays, with the identity of
            # the reporting process declared unverifiable
            coverage = result["verdicts"]["coverage"]
            self.assertEqual(coverage["verdict"], "PASS", coverage["problems"])
            self.assertTrue(coverage["identity"].startswith("unverified-legacy"))
        # legacy reports of a run whose ledger never recorded the resolved
        # arguments (the first harness) are judged on their replays too, with
        # the unresolved topology noted in the identity
        legacy = [_report(tracer_version=2, parallel=None)]
        coverage = harness.coverage_verdict(
            legacy, 1, ["DecodeCudaGraphRunner"], [], None, harness.topology({})
        )
        self.assertEqual(coverage["verdict"], "PASS", coverage["problems"])
        self.assertIn("unverified-legacy", coverage["identity"])
        self.assertIn("resolved server arguments unavailable", coverage["identity"])
        # a current report under an unresolved topology still fails
        coverage = harness.coverage_verdict(
            [_report()], 1, ["DecodeCudaGraphRunner"], [], None, harness.topology({})
        )
        self.assertEqual(coverage["verdict"], "INCOMPLETE")
        self.assertTrue(any("unsupported topology" in p for p in coverage["problems"]))
        # a legacy report next to an identified one is not excused
        topo = harness.topology(_resolved_topology(tp_size=2))
        verdict = harness.lifetime_verdict(
            [
                _report(pid=1, parallel=_parallel(0, tp_size=2), device_uuid="GPU-a"),
                _report(pid=2, tracer_version=2, parallel=None, device_uuid="GPU-b"),
            ],
            2,
            "run-1",
            "run-1-7",
            topo,
        )
        self.assertEqual(verdict["verdict"], "FAIL")
        self.assertTrue(any("cannot stand in" in p for p in verdict["problems"]))


class TestCoverage(unittest.TestCase):
    def test_normalize_label_keeps_variants(self):
        self.assertEqual(
            harness.normalize_label("DecodeCudaGraphRunner size=4 variant_label=lora"),
            "DecodeCudaGraphRunner variant_label=lora",
        )
        self.assertEqual(
            harness.normalize_label("PrefillCudaGraphRunner size=2048"),
            "PrefillCudaGraphRunner",
        )
        self.assertEqual(
            harness.normalize_label("RowBucketGraphRunner rows=64 attention_variant=x"),
            "RowBucketGraphRunner attention_variant=x",
        )

    def test_requirements_follow_the_resolved_configuration(self):
        resolved = {
            "cuda_graph_backend_prefill": "breakable",
            "cuda_graph_backend_decode": "None",
            "disable_cuda_graph": "False",
            "enable_lora": "True",
            "lora_backend": "triton_v2",
        }
        required, notes = harness.required_coverage(
            resolved, {"SGLANG_ENABLE_LORA_DUAL_GRAPH": "1"}, ["Extra"]
        )
        self.assertEqual(
            required,
            [
                "Extra",
                "DecodeCudaGraphRunner",
                "PrefillCudaGraphRunner",
                "variant_label=lora",
                "variant_label=nolora",
            ],
        )
        self.assertEqual(notes, [])
        required, _ = harness.required_coverage(
            {**resolved, "cuda_graph_backend_prefill": "disabled"}, {}, []
        )
        self.assertEqual(required, ["DecodeCudaGraphRunner"])
        # the knob alone promises nothing: production records the LoRA decode
        # variants only with decode graphs on, LoRA enabled and triton_v2
        knob = {"SGLANG_ENABLE_LORA_DUAL_GRAPH": "1"}
        for name, args in {
            "graphs disabled": {
                **resolved,
                "disable_cuda_graph": "True",
                "cuda_graph_backend_prefill": "None",
            },
            "decode graphs disabled": {
                **resolved,
                "cuda_graph_backend_decode": "disabled",
            },
            "lora disabled": {**resolved, "enable_lora": "False"},
            "other lora backend": {**resolved, "lora_backend": "triton"},
        }.items():
            with self.subTest(name):
                required, notes = harness.required_coverage(args, knob, [])
                self.assertNotIn("variant_label=lora", required)
                self.assertNotIn("variant_label=nolora", required)
                self.assertTrue(any("none required" in n for n in notes), notes)
        # the pre-rename knob name still counts, and says so
        required, notes = harness.required_coverage(
            resolved, {"SGLANG_ENABLE_LORA_NOLORA_GRAPH": "1"}, []
        )
        self.assertIn("variant_label=nolora", required)
        self.assertTrue(any("legacy name" in n for n in notes))
        required, notes = harness.required_coverage({}, {}, [])
        self.assertEqual(required, [])
        self.assertTrue(any("--expect-labels" in n for n in notes))

    def test_one_variant_replaying_does_not_cover_the_other(self):
        report = _report(
            label_captures={
                "DecodeCudaGraphRunner size=1 variant_label=lora": 1,
                "DecodeCudaGraphRunner size=1 variant_label=nolora": 1,
                "PrefillCudaGraphRunner size=512": 1,
            },
            label_replays={"DecodeCudaGraphRunner size=1 variant_label=lora": 9},
        )
        required = [
            "DecodeCudaGraphRunner",
            "variant_label=lora",
            "variant_label=nolora",
        ]
        coverage = harness.coverage_verdict([report], 1, required, [])
        self.assertEqual(coverage["verdict"], "INCOMPLETE")
        self.assertEqual(coverage["per_worker"][0]["missing"], ["variant_label=nolora"])
        self.assertEqual(
            coverage["captured_not_replayed"],
            [
                "DecodeCudaGraphRunner size=1 variant_label=nolora",
                "PrefillCudaGraphRunner size=512",
            ],
        )
        # the unreplayed prefill bucket is recorded, not failed, when not required
        self.assertFalse(any("Prefill" in p for p in coverage["problems"]))
        both = _report(
            label_replays={
                "DecodeCudaGraphRunner size=1 variant_label=lora": 9,
                "DecodeCudaGraphRunner size=2 variant_label=nolora": 3,
            }
        )
        self.assertEqual(
            harness.coverage_verdict([both], 1, required, [])["verdict"], "PASS"
        )

    def test_unreplayed_buckets_are_inventoried_by_raw_label(self):
        report = _report(
            label_captures={
                "PrefillCudaGraphRunner size=512": 1,
                "PrefillCudaGraphRunner size=4096": 1,
            },
            label_replays={"PrefillCudaGraphRunner size=512": 3},
        )
        coverage = harness.coverage_verdict([report], 1, ["PrefillCudaGraphRunner"], [])
        self.assertEqual(coverage["verdict"], "PASS", coverage["problems"])
        self.assertEqual(
            coverage["captured_not_replayed"], ["PrefillCudaGraphRunner size=4096"]
        )
        self.assertEqual(
            coverage["per_worker"][0]["captured_not_replayed"],
            ["PrefillCudaGraphRunner size=4096"],
        )

    def test_every_worker_must_replay_the_required_paths(self):
        prefill = {"PrefillCudaGraphRunner size=1024": 2}
        first = _report(
            pid=1,
            device="cuda:0",
            label_replays={**_report()["label_replays"], **prefill},
        )
        second = _report(pid=2, device="cuda:1")  # decode only
        coverage = harness.coverage_verdict(
            [first, second], 2, ["DecodeCudaGraphRunner", "PrefillCudaGraphRunner"], []
        )
        self.assertEqual(coverage["verdict"], "INCOMPLETE")
        self.assertEqual(
            [w["missing"] for w in coverage["per_worker"]],
            [[], ["PrefillCudaGraphRunner"]],
        )
        self.assertTrue(any("pid 2" in p for p in coverage["problems"]))
        coverage = harness.coverage_verdict([first], 2, ["DecodeCudaGraphRunner"], [])
        self.assertTrue(any("expected 2" in p for p in coverage["problems"]))
        coverage = harness.coverage_verdict(
            [first], 1, [], ["resolved server arguments unavailable"]
        )
        self.assertEqual(coverage["verdict"], "INCOMPLETE")
        self.assertTrue(any("--expect-labels" in p for p in coverage["problems"]))


def _write_outputs(d, run_id, phase, outputs):
    """An outputs file as the harness writes it, with its evidence record."""
    path = os.path.join(d, f"{phase}.outputs.json")
    harness._write_outputs(path, run_id, phase, outputs)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {
        "path": f"{phase}.outputs.json",
        "run_id": run_id,
        "phase": phase,
        "sha256": digest,
        "requests": len(outputs),
    }


def _recompute(directory, ledger):
    with open(os.path.join(directory, "ledger.json"), "w") as f:
        json.dump(ledger, f)
    return harness.recompute(directory)


class TestEvidence(unittest.TestCase):
    def test_recompute_rejudges_an_archived_run_without_touching_it(self):
        with tempfile.TemporaryDirectory() as d:
            report = _report(
                tracer_version=1,
                label_replays={
                    "DecodeCudaGraphRunner size=1 variant_label=lora": 9,
                    "DecodeCudaGraphRunner size=1 variant_label=nolora": 4,
                    "PrefillCudaGraphRunner size=512": 2,
                },
            )
            report.pop("run_id")
            ledger = {
                "max_new_tokens": NEW,
                "expected_workers": 1,
                "expect_labels": [],
                "verdicts": {"numerical": {"verdict": "FAIL"}},
                "phases": {
                    "reference": {},
                    "ordinary": {
                        "extra_env": {"SGLANG_ENABLE_LORA_NOLORA_GRAPH": "1"},
                        "resolved_args": {
                            "cuda_graph_backend_prefill": "breakable",
                            "enable_lora": "True",
                            "lora_backend": "triton_v2",
                        },
                    },
                    "instrumented": {"worker_reports": [report]},
                },
            }
            with open(os.path.join(d, "ledger.json"), "w") as f:
                json.dump(ledger, f)
            for phase in ("reference", "ordinary"):
                with open(os.path.join(d, f"{phase}.outputs.json"), "w") as f:
                    json.dump(_outputs(), f)
            before = open(os.path.join(d, "ledger.json")).read()
            result = harness.recompute(d)
            self.assertEqual(open(os.path.join(d, "ledger.json")).read(), before)
            self.assertTrue(os.path.exists(os.path.join(d, "ledger.recomputed.json")))
            self.assertEqual(result["original_verdicts"]["numerical"], "FAIL")
            self.assertEqual(result["verdicts"]["numerical"]["verdict"], "PASS")
            self.assertEqual(
                result["verdicts"]["lifetime"]["verdict"], "NOT-RECOMPUTED"
            )
            coverage = result["verdicts"]["coverage"]
            self.assertEqual(
                coverage["required"],
                [
                    "DecodeCudaGraphRunner",
                    "PrefillCudaGraphRunner",
                    "variant_label=lora",
                    "variant_label=nolora",
                ],
            )
            self.assertEqual(coverage["verdict"], "PASS", coverage["problems"])
            self.assertEqual(coverage["requirement_provenance"], "unverified-legacy")
            self.assertTrue(any("legacy name" in n for n in coverage["notes"]))
            # outputs without recorded digests are diagnostic, never verified
            self.assertEqual(result["outputs"]["ordinary"], "unverified-legacy")
            self.assertEqual(
                result["verdicts"]["numerical"]["evidence_provenance"],
                "unverified-legacy",
            )
            self.assertEqual(harness.main(["--recompute", d]), 0)
            # a second recomputation keeps the first under its tool digest
            tag = result["tool"]["sha256"][:12]
            self.assertTrue(
                os.path.exists(os.path.join(d, f"ledger.recomputed.{tag}.json"))
            )

    def test_recompute_never_weakens_the_coverage_requirements(self):
        four = [
            "DecodeCudaGraphRunner",
            "PrefillCudaGraphRunner",
            "variant_label=lora",
            "variant_label=nolora",
        ]
        # the launch-wide knob is absent from every phase's extra_env (it was
        # passed with --env); only the declaration, or the original verdict's
        # requirement list, carries it
        resolved = {
            "cuda_graph_backend_prefill": "breakable",
            "enable_lora": "True",
            "lora_backend": "triton_v2",
        }
        cases = {
            "declared": (
                {
                    "launch_env": {"SGLANG_ENABLE_LORA_DUAL_GRAPH": "1"},
                    "required": four,
                },
                {},
            ),
            "unverified-legacy": (None, {"coverage": {"required": four}}),
        }
        for provenance, (declaration, verdicts) in cases.items():
            with self.subTest(provenance):
                with tempfile.TemporaryDirectory() as d:
                    ledger = {
                        "run_id": "run-1",
                        "max_new_tokens": NEW,
                        "expected_workers": 1,
                        "expect_labels": [],
                        "verdicts": verdicts,
                        "phases": {
                            "ordinary": {"extra_env": {}, "resolved_args": resolved},
                            "instrumented": {
                                "extra_env": {"PYTHONPATH": "x"},
                                "worker_reports": [_report()],  # lora decode only
                            },
                        },
                    }
                    if declaration is not None:
                        ledger["coverage_declaration"] = declaration
                    coverage = _recompute(d, ledger)["verdicts"]["coverage"]
                    self.assertEqual(coverage["required"], four)
                    self.assertEqual(coverage["verdict"], "INCOMPLETE")
                    self.assertEqual(
                        coverage["per_worker"][0]["missing"],
                        ["PrefillCudaGraphRunner", "variant_label=nolora"],
                    )
                    self.assertEqual(coverage["requirement_provenance"], provenance)

    def test_recompute_carries_an_unrecomputable_agreement_forward(self):
        # both launches produced outputs at the time, but the archive holds
        # none: the original instrumented-vs-ordinary judgment stands, and a
        # missing or failed one cannot turn into a pass
        for original, verdict, needle in (
            (
                {"verdict": "FAIL", "kind": "drift"},
                "INCOMPLETE",
                "historical judgment FAIL",
            ),
            (None, "INCOMPLETE", "historical judgment None"),
            ({"verdict": "PASS", "kind": "drift"}, "PASS", None),
        ):
            with self.subTest(str(original)):
                with tempfile.TemporaryDirectory() as d:
                    coverage = {"required": ["DecodeCudaGraphRunner"]}
                    if original is not None:
                        coverage["instrumented_vs_ordinary"] = original
                    ledger = {
                        "run_id": "run-1",
                        "max_new_tokens": NEW,
                        "expected_workers": 1,
                        "expect_labels": ["DecodeCudaGraphRunner"],
                        "verdicts": {"coverage": coverage},
                        "phases": {
                            "reference": {"requests": 2},
                            "ordinary": {"requests": 2},
                            "instrumented": {
                                "requests": 2,
                                "worker_reports": [
                                    _report(tracer_version=2, parallel=None)
                                ],
                            },
                        },
                    }
                    result = _recompute(d, ledger)["verdicts"]["coverage"]
                    self.assertEqual(result["verdict"], verdict, result["problems"])
                    agreement = result["instrumented_vs_ordinary"]
                    self.assertEqual(agreement["kind"], "not-recomputed")
                    self.assertTrue(agreement["historical"])
                    if needle:
                        self.assertTrue(any(needle in p for p in result["problems"]))
                    else:
                        self.assertEqual(agreement["original_verdict"], "PASS")
                        self.assertTrue(any("historical" in n for n in result["notes"]))
        # recorded outputs that are rejected (here: missing) inherit nothing,
        # even from an original PASS
        with tempfile.TemporaryDirectory() as d:
            ledger = {
                "run_id": "run-1",
                "max_new_tokens": NEW,
                "expected_workers": 1,
                "expect_labels": ["DecodeCudaGraphRunner"],
                "verdicts": {
                    "coverage": {
                        "required": ["DecodeCudaGraphRunner"],
                        "instrumented_vs_ordinary": {
                            "verdict": "PASS",
                            "kind": "drift",
                        },
                    }
                },
                "evidence": {
                    "outputs": {
                        phase: {
                            "path": f"{phase}.outputs.json",
                            "sha256": "0" * 64,
                            "requests": 2,
                        }
                        for phase in ("ordinary", "instrumented")
                    }
                },
                "phases": {
                    "reference": {"requests": 2},
                    "ordinary": {"requests": 2},
                    "instrumented": {
                        "requests": 2,
                        "worker_reports": [_report(tracer_version=2, parallel=None)],
                    },
                },
            }
            verdicts = _recompute(d, ledger)["verdicts"]
            coverage = verdicts["coverage"]
            self.assertEqual(coverage["verdict"], "INCOMPLETE")
            self.assertEqual(
                coverage["instrumented_vs_ordinary"]["kind"], "evidence-rejected"
            )
            self.assertFalse(coverage["instrumented_vs_ordinary"]["historical"])
            self.assertTrue(any("not inherited" in p for p in coverage["problems"]))
            self.assertEqual(
                (verdicts["numerical"]["verdict"], verdicts["numerical"]["kind"]),
                ("INCONCLUSIVE", "evidence-missing"),
            )

    def test_recompute_never_consumes_outputs_of_a_failed_phase(self):
        for recorded in (False, True):
            with self.subTest(recorded_evidence=recorded):
                with tempfile.TemporaryDirectory() as d:
                    reference = _write_outputs(d, "run-1", "reference", _outputs())
                    # a stale file from an earlier launch in the same directory
                    with open(os.path.join(d, "ordinary.outputs.json"), "w") as f:
                        json.dump(_outputs(), f)
                    ledger = {
                        "run_id": "run-1",
                        "max_new_tokens": NEW,
                        "expected_workers": 1,
                        "expect_labels": [],
                        "verdicts": {"numerical": {"verdict": "FAIL"}},
                        "phases": {
                            "reference": {},
                            "ordinary": {"error": "RuntimeError('server exited')"},
                            "instrumented": {"worker_reports": []},
                        },
                    }
                    if recorded:
                        ledger["evidence"] = {"outputs": {"reference": reference}}
                    result = _recompute(d, ledger)
                    numerical = result["verdicts"]["numerical"]
                    self.assertEqual(
                        (numerical["verdict"], numerical["kind"]), ("FAIL", "crash")
                    )
                    self.assertEqual(result["outputs"]["ordinary"], "failed-phase")

    def test_recompute_consumes_only_recorded_and_matching_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            reference = _write_outputs(d, "run-1", "reference", _outputs())
            ordinary = _write_outputs(d, "run-1", "ordinary", _outputs())
            # present but never recorded: not evidence of this run
            _write_outputs(d, "run-1", "instrumented", _outputs())
            ledger = {
                "run_id": "run-1",
                "max_new_tokens": NEW,
                "expected_workers": 1,
                "expect_labels": ["x"],
                "verdicts": {},
                "evidence": {"outputs": {"reference": reference, "ordinary": ordinary}},
                "phases": {
                    "reference": {"requests": 2},
                    "ordinary": {"requests": 2},
                    "instrumented": {"worker_reports": []},
                },
            }
            result = _recompute(d, ledger)
            self.assertEqual(result["recomputed_from"], os.path.abspath(d))
            numerical = result["verdicts"]["numerical"]
            self.assertEqual(numerical["verdict"], "PASS")
            self.assertEqual(numerical["evidence_provenance"], "verified")
            self.assertEqual(result["outputs"]["instrumented"], "never-archived")
            self.assertNotIn("instrumented_vs_ordinary", result["verdicts"]["coverage"])
            # the recorded file replaced by another run's outputs
            _write_outputs(d, "run-0", "ordinary", _outputs())
            numerical = harness.recompute(d)["verdicts"]["numerical"]
            self.assertEqual(
                (numerical["verdict"], numerical["kind"]),
                ("INCONCLUSIVE", "evidence-mismatch"),
            )
            self.assertTrue(any("digest" in p for p in numerical["problems"]))
            # the digest matches but the file names another run
            ledger["evidence"]["outputs"]["ordinary"] = _write_outputs(
                d, "run-0", "ordinary", _outputs()
            )
            numerical = _recompute(d, ledger)["verdicts"]["numerical"]
            self.assertEqual(numerical["kind"], "evidence-mismatch")
            self.assertTrue(
                any("belongs to run 'run-0'" in p for p in numerical["problems"])
            )
            # the recorded file is gone: rejected evidence, not a run that
            # never produced outputs
            os.remove(os.path.join(d, "ordinary.outputs.json"))
            numerical = harness.recompute(d)["verdicts"]["numerical"]
            self.assertEqual(
                (numerical["verdict"], numerical["kind"]),
                ("INCONCLUSIVE", "evidence-missing"),
            )
        # the optional control is judged on its own: verified reference and
        # ordinary outputs 1.5 nats apart stay FAIL when the recorded
        # reference2 file is missing or does not match its digest
        for break_control in ("missing", "mismatch"):
            with self.subTest(break_control):
                with tempfile.TemporaryDirectory() as d2:
                    reference = _write_outputs(d2, "run-1", "reference", _outputs())
                    drifted = _outputs(
                        **{
                            "bs2-none-r0-0": _response(
                                [1, 2, 3], [11] * NEW, prompt_lp=-2.5
                            )
                        }
                    )
                    ordinary = _write_outputs(d2, "run-1", "ordinary", drifted)
                    control = _write_outputs(d2, "run-1", "reference2", drifted)
                    if break_control == "missing":
                        os.remove(os.path.join(d2, "reference2.outputs.json"))
                    else:
                        _write_outputs(d2, "run-0", "reference2", drifted)
                    ledger = {
                        "run_id": "run-1",
                        "max_new_tokens": NEW,
                        "expected_workers": 1,
                        "expect_labels": ["x"],
                        "verdicts": {},
                        "evidence": {
                            "outputs": {
                                "reference": reference,
                                "ordinary": ordinary,
                                "reference2": control,
                            }
                        },
                        "phases": {
                            "reference": {"requests": 2},
                            "reference2": {"requests": 2},
                            "ordinary": {"requests": 2},
                            "instrumented": {"worker_reports": []},
                        },
                    }
                    numerical = _recompute(d2, ledger)["verdicts"]["numerical"]
                    self.assertEqual(
                        (numerical["verdict"], numerical["kind"]), ("FAIL", "drift")
                    )
                    self.assertEqual(
                        numerical["batch_sensitivity_control"]["verdict"], "UNUSABLE"
                    )
                    self.assertEqual(
                        numerical["batch_sensitivity_control"]["kind"],
                        f"evidence-{break_control}",
                    )
                    self.assertTrue(any("unusable" in p for p in numerical["problems"]))
                    self.assertEqual(numerical["evidence_provenance"], "verified")
        # an --out directory resolves to its single run directory, never to two
        with tempfile.TemporaryDirectory() as out:
            run = os.path.join(out, "20261027T000000-abcd1234")
            os.makedirs(run)
            with open(os.path.join(run, "ledger.json"), "w") as f:
                json.dump({"run_id": "r", "phases": {}, "verdicts": {}}, f)
            self.assertEqual(harness._locate_ledger(out), run)
            other = os.path.join(out, "20261027T000001-abcd1234")
            os.makedirs(other)
            with open(os.path.join(other, "ledger.json"), "w") as f:
                f.write("{}")
            with self.assertRaises(FileNotFoundError):
                harness._locate_ledger(out)


class TestServerLogEvidence(unittest.TestCase):
    """The scheduler log lines the harness reads (pure parsing, no GPU)."""

    def test_server_log_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "server.log")
            with open(path, "w") as f:
                f.write(
                    "[x] server_args=ServerArgs(model_path='m', attention_backend='flashinfer', "
                    "cuda_graph_backend_prefill=breakable, tp_size=2, lora_backend='triton_v2')\n"
                    "[x] Prefill batch, #new-seq: 1, #new-token: 700, cuda graph: True, x\n"
                    "[x] Prefill batch, #new-seq: 1, #new-token: 7, cuda graph: False, x\n"
                    "[x] Decode batch, #running-req: 2, cuda graph: True, x\n"
                    "[x] Decode batch, #running-req: 2, cuda graph: True, x\n"
                )
            evidence = harness.server_log_evidence(path)
        self.assertEqual(
            evidence["graph_batches"],
            {
                "prefill_graph": 1,
                "prefill_eager": 1,
                "decode_graph": 2,
                "decode_eager": 0,
            },
        )
        self.assertEqual(
            evidence["resolved_args"],
            {
                "attention_backend": "flashinfer",
                "cuda_graph_backend_prefill": "breakable",
                "lora_backend": "triton_v2",
                "tp_size": "2",
            },
        )


if __name__ == "__main__":
    unittest.main()
