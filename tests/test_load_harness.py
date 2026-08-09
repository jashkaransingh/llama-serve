"""The benchmark harness's own arithmetic.

A measurement tool that computes its statistics wrongly produces confident,
committed, wrong numbers — the exact failure this repo is trying not to have. So
the parts of `bench/load.py` that turn samples into reported values are unit
tested like any other code.
"""

from __future__ import annotations

import importlib.util
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_load", ROOT / "bench" / "load.py")
load = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(load)


# --- percentiles ------------------------------------------------------------
def test_percentile_indexes_the_real_samples_without_interpolating():
    """Index = round(q * (n - 1)), the same convention `llama_serve.metrics`
    uses, so a number from this harness and one scraped from /metrics mean the
    same thing."""
    v = [float(i) for i in range(1, 101)]  # 1..100
    assert load.pct(v, 0.50) == 51.0  # index 50
    assert load.pct(v, 0.90) == 90.0  # index 89
    assert load.pct(v, 0.99) == 99.0  # index 98
    assert load.pct(v, 0.0) == 1.0
    assert load.pct(v, 1.0) == 100.0


def test_percentile_matches_the_servers_own_convention():
    """Pinned against the server-side implementation, not restated."""
    from llama_serve.metrics import _pct

    v = [0.4, 0.1, 0.9, 0.2, 0.7, 0.3, 0.8, 0.5, 0.6, 1.0]
    for q in (0.5, 0.9, 0.99):
        assert load.pct(v, q) == _pct(v, q)


def test_percentile_of_nothing_is_none_not_zero():
    """Zero is a measurement; None is the absence of one, and the difference
    matters when the value ends up in a table."""
    assert load.pct([], 0.5) is None


def test_percentile_of_one_sample():
    assert load.pct([0.25], 0.99) == 0.25


# --- spread -----------------------------------------------------------------
def test_spread_reports_mean_and_sample_stdev():
    s = load.spread([1.0, 2.0, 3.0])
    assert s["n"] == 3
    assert s["mean"] == 2.0
    assert s["stdev"] == round(statistics.stdev([1.0, 2.0, 3.0]), 5)
    assert (s["min"], s["max"]) == (1.0, 3.0)


def test_spread_of_a_single_run_reports_zero_spread_not_an_error():
    s = load.spread([0.5])
    assert s == {"n": 1, "mean": 0.5, "stdev": 0.0, "min": 0.5, "max": 0.5}


def test_spread_drops_missing_values_rather_than_treating_them_as_zero():
    """A level where p99 was not reportable contributes no sample to the p99
    spread. Counting it as 0.0 would silently pull the mean down."""
    s = load.spread([1.0, None, 3.0])
    assert s["n"] == 2
    assert s["mean"] == 2.0


def test_spread_of_nothing_is_empty():
    assert load.spread([]) == {"n": 0}
    assert load.spread([None, None]) == {"n": 0}


# --- comparison rendering ---------------------------------------------------
def _fake_level(qps, achieved, p50, p99, saturated=False):
    return {
        "offered_qps": qps,
        "saturated": saturated,
        "achieved_qps": {"mean": achieved, "stdev": 0.1},
        "ttft_pooled": {"n": 200, "p50": p50, "p90": None, "p99": p99},
    }


def _fake_results(tmp_path):
    import json

    env = {"model": "m.gguf", "platform": "test"}
    harness = {"workload": "w", "max_tokens": 32, "runs_per_level": 3, "duration_s": 20.0}
    data = {
        "base": {
            "harness": harness,
            "environment": env,
            "levels": [_fake_level(1, 1.0, 0.4, 2.0), _fake_level(2, 1.2, 2.0, 8.0, True)],
        },
        "cand": {
            "harness": harness,
            "environment": env,
            "levels": [_fake_level(1, 1.0, 0.1, 0.5), _fake_level(2, 2.0, 0.2, 1.0)],
        },
    }
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(data))
    return p


def test_compare_reports_the_real_reduction(tmp_path):
    out = tmp_path / "cmp.md"
    assert load.compare("base,cand", _fake_results(tmp_path), out) == 0
    body = out.read_text()
    assert "75.0 %" in body, "0.4 -> 0.1 is a 75% reduction"
    assert "90.0 %" in body, "2.0 -> 0.5 p99 is a 90% reduction"


def test_compare_marks_saturated_levels(tmp_path):
    out = tmp_path / "cmp.md"
    load.compare("base,cand", _fake_results(tmp_path), out)
    assert "SAT base" in out.read_text()


def test_compare_picks_the_highest_unsaturated_level_for_the_headline(tmp_path):
    """The headline capacity number must not come from a level where the server
    was already falling behind."""
    out = tmp_path / "cmp.md"
    load.compare("base,cand", _fake_results(tmp_path), out)
    body = out.read_text()
    # base's only unsaturated level achieved 1.0; cand's best is 2.0.
    assert "Capacity ratio: **2.00x**" in body
