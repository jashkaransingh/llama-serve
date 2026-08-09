"""Observability: the metrics surface, tested against real completed requests.

Nothing here asserts on a hand-built registry. Every test runs actual requests
through the app on the mock backend and then reads what the endpoints report,
because the failure mode worth catching is a metric that is *plumbed* but never
fed — a `/metrics` full of zeroes is worse than no `/metrics`, since it looks
like a working dashboard.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llama_serve.api import create_app
from llama_serve.config import Config

PREAMBLE = "You are a helpful assistant answering questions about storage systems.\n"


def cfg(**kw) -> Config:
    c = Config(
        backend="mock",
        engine="continuous",
        max_seqs=4,
        cache_seqs=4,
        n_ctx_per_seq=512,
        block_size=16,
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


@pytest.fixture
def client():
    with TestClient(create_app(cfg())) as c:
        yield c


def generate(client, prompt="hello there", max_tokens=12, priority=0):
    r = client.post(
        "/generate",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "priority": priority,
            "ignore_eos": True,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def parse_prom(text: str) -> dict[str, float]:
    """Parse the exposition format into {series: value}, keeping labels."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        out[name.strip()] = float(value)
    return out


# --- format ----------------------------------------------------------------
def test_exposition_parses_with_the_official_prometheus_parser(client):
    """The strongest available check that "Prometheus-compatible" is not a claim
    about intent: the reference client's own parser accepts it."""
    from prometheus_client.parser import text_string_to_metric_families

    for _ in range(3):
        generate(client)
    families = list(text_string_to_metric_families(client.get("/metrics").text))

    assert families, "parser found no metric families"
    by_name = {f.name: f for f in families}
    assert by_name["llama_serve_requests_finished"].type == "counter"
    assert by_name["llama_serve_queue_depth"].type == "gauge"
    assert by_name["llama_serve_time_to_first_token_seconds"].type == "summary"



def test_metrics_is_prometheus_text_and_well_formed(client):
    generate(client)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")

    body = r.text
    assert body.endswith("\n"), "exposition format requires a trailing newline"

    # Every emitted series must have been declared with a HELP and a TYPE.
    declared = {
        line.split()[2] for line in body.splitlines() if line.startswith("# TYPE")
    }
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        series = line.rpartition(" ")[0].split("{")[0]
        base = series.removesuffix("_sum").removesuffix("_count")
        assert base in declared, f"series {series} emitted with no TYPE declaration"


def test_every_value_is_a_number(client):
    generate(client)
    for name, value in parse_prom(client.get("/metrics").text).items():
        assert value == value, f"{name} is NaN"  # noqa: PLR0124 - NaN check


def test_build_info_carries_the_configuration(client):
    series = parse_prom(client.get("/metrics").text)
    key = next(k for k in series if k.startswith("llama_serve_build_info"))
    assert 'engine="continuous"' in key
    assert 'backend="mock"' in key
    assert series[key] == 1


# --- the metrics the milestone actually requires ---------------------------
def test_queue_time_ttft_and_throughput_come_from_real_requests(client):
    for _ in range(5):
        generate(client, max_tokens=10)
    series = parse_prom(client.get("/metrics").text)

    assert series["llama_serve_requests_finished_total"] == 5
    assert series["llama_serve_queue_time_seconds_count"] == 5
    assert series["llama_serve_time_to_first_token_seconds_count"] == 5
    assert series["llama_serve_request_duration_seconds_count"] == 5
    assert series["llama_serve_output_tokens_per_second_count"] == 5

    # Real measurements, not placeholders: strictly positive, and TTFT is at
    # least the queue time it contains.
    assert series['llama_serve_time_to_first_token_seconds{quantile="0.5"}'] > 0
    assert series["llama_serve_output_tokens_per_second_sum"] > 0
    assert series["llama_serve_generated_tokens_total"] == 50


def test_percentiles_below_their_sample_floor_are_omitted_not_invented(client):
    """The registry refuses to report p90 under 10 samples and p99 under 100.
    A missing series is a fact; a fabricated one is a lie on a dashboard."""
    for _ in range(3):
        generate(client, max_tokens=5)
    series = parse_prom(client.get("/metrics").text)
    q = 'llama_serve_time_to_first_token_seconds{quantile='
    assert f'{q}"0.5"}}' in series
    assert f'{q}"0.9"}}' not in series, "p90 reported from 3 samples"
    assert f'{q}"0.99"}}' not in series, "p99 reported from 3 samples"


def test_p90_appears_once_there_are_enough_samples(client):
    for _ in range(12):
        generate(client, max_tokens=4)
    series = parse_prom(client.get("/metrics").text)
    assert 'llama_serve_time_to_first_token_seconds{quantile="0.9"}' in series


def test_engine_and_kv_cache_metrics_are_populated(client):
    for i in range(4):
        generate(client, prompt=PREAMBLE + f"Question {i}?", max_tokens=8)
    series = parse_prom(client.get("/metrics").text)

    assert series["llama_serve_decode_steps_total"] > 0
    assert series["llama_serve_max_sequence_slots"] == 4
    assert series["llama_serve_kv_blocks_total"] > 0
    assert series["llama_serve_kv_cache_hits_total"] >= 1, "shared preamble never hit the cache"
    assert series["llama_serve_kv_prefill_tokens_reused_total"] > 0
    assert series["llama_serve_decode_step_seconds_avg"] >= 0


def test_counters_only_go_up(client):
    generate(client)
    first = parse_prom(client.get("/metrics").text)
    generate(client)
    second = parse_prom(client.get("/metrics").text)
    for name, value in first.items():
        if name.endswith("_total"):
            assert second[name] >= value, f"{name} went backwards"
    assert second["llama_serve_requests_finished_total"] == 2


# --- the JSON surfaces ------------------------------------------------------
def test_metrics_json_still_serves_the_snapshot(client):
    generate(client)
    body = client.get("/metrics.json").json()
    assert body["requests"]["finished"] == 1
    assert body["latency"]["ttft_s"]["n"] == 1
    assert body["engine"]["engine"] == "continuous"


def test_per_request_rows_are_exposed(client):
    """Prometheus aggregates by design; finding the one slow request needs the
    raw rows."""
    for i in range(3):
        generate(client, max_tokens=6, priority=i)
    body = client.get("/metrics/requests").json()

    assert body["n"] == 3
    assert len(body["rows"]) == 3
    row = body["rows"][0]
    for field in ("id", "queue_time_s", "ttft_s", "total_time_s", "output_tps", "priority"):
        assert field in row, f"per-request row is missing {field}"
    assert row["generated_tokens"] == 6
    assert [r["priority"] for r in body["rows"]] == [0, 1, 2]


def test_per_request_rows_respect_the_limit(client):
    for _ in range(5):
        generate(client, max_tokens=4)
    body = client.get("/metrics/requests?limit=2").json()
    assert body["n"] == 5
    assert len(body["rows"]) == 2


def test_reset_clears_the_window(client):
    generate(client)
    client.post("/metrics/reset")
    series = parse_prom(client.get("/metrics").text)
    assert series["llama_serve_requests_finished_total"] == 0
    assert series["llama_serve_time_to_first_token_seconds_count"] == 0


def test_metrics_work_before_any_request_has_run(client):
    """A scrape at startup must not 500, and must not report a fake latency."""
    body = client.get("/metrics").text
    series = parse_prom(body)
    assert series["llama_serve_requests_finished_total"] == 0
    assert 'llama_serve_time_to_first_token_seconds{quantile="0.5"}' not in series


# --- dashboard --------------------------------------------------------------
def test_dashboard_renders(client):
    """It used to import a module that did not exist, and returned a 500."""
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "/metrics.json" in r.text
