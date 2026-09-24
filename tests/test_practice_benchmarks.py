"""Practice curricula, mastery gates and A/B benchmarks against headless Blender."""

from __future__ import annotations

import pytest

from lucius.app import Lucius
from tests.conftest import HAS_BPY
from tests.test_e2e_learning import record
from tests.fixtures.demos import sword_blockout_demo

pytestmark = [pytest.mark.bpy, pytest.mark.skipif(not HAS_BPY, reason="needs Blender")]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    instance = Lucius(data_dir=tmp_path_factory.mktemp("practice") / "data", background_processing=False)
    yield instance
    instance.close()


def test_practice_reaches_mastery_on_seeded_fundamentals(app):
    backend = app.headless_backend()
    report = app.practice.train("hard surface", stage=1, attempts=5, backend=backend, seed=3)
    assert report.stage_name == "Primitive manipulation"
    assert all(a.verdict == "success" for a in report.attempts), [a.model_dump() for a in report.attempts]
    assert len({str(a.params) for a in report.attempts}) == 5  # sampled instances, not one memorised task
    assert report.status == "mastered" and report.advanced_to == 2
    assert report.metrics.completion_rate == 1.0 and report.metrics.generalization == 1.0
    overview = app.practice.overview("hard_surface")
    assert overview["stages"][0]["status"] == "mastered" and overview["current_stage"] == 2

    extrude = app.practice.train("hard surface", stage=2, attempts=2, backend=backend, seed=4)
    assert all(a.verdict == "success" for a in extrude.attempts), [a.model_dump() for a in extrude.attempts]
    topology = app.practice.train("topology", stage=2, attempts=2, backend=backend, seed=5)
    assert all(a.verdict == "success" for a in topology.attempts), [a.model_dump() for a in topology.attempts]
    assert extrude.status == "practicing" and extrude.unmet  # two attempts are not enough for mastery


def test_practice_is_honest_about_missing_capabilities(app):
    backend = app.headless_backend()
    gui = app.practice.train("sculpting", stage=1, attempts=1, backend=backend)
    assert gui.status == "requires_gui" and not gui.attempts
    blade = app.practice.train("hard surface", stage=3, attempts=2, backend=backend, seed=1)
    assert blade.status == "needs_demonstration" and len(blade.attempts) == 1
    assert "demonstrate" in blade.message


def test_benchmark_arms_measure_the_value_of_memory(app):
    backend = app.headless_backend()
    record(app, sword_blockout_demo(blade_length=6.0, blade_width=0.3, variant="a", t0=1_700_000_000.0))
    record(app, sword_blockout_demo(blade_length=4.0, blade_width=0.24, variant="b", with_mistake=False,
                                    t0=1_700_100_000.0))
    result = app.benchmarks.experiment("memory vs none", arms=["baseline", "memory_enhanced", "raw_demonstrations",
                                                               "trained_policy"],
                                       benchmarks=["sized_box", "blade_blockout"], backend=backend)
    arms = result["arms"]
    assert arms["baseline"]["success_rate"] == 0.0  # no retrieval, nothing to execute
    assert arms["memory_enhanced"]["success_rate"] == 1.0
    assert arms["memory_enhanced"]["delta_vs_baseline"] == 1.0
    # Replaying a past demonstration verbatim cannot hit unseen proportions: generalisation matters.
    assert arms["raw_demonstrations"]["success_rate"] < arms["memory_enhanced"]["success_rate"]
    assert result["unavailable_arms"] == {"trained_policy": "no trained policy is registered (see lucius.training)"}
    assert arms["memory_enhanced"]["success_ci95"] is not None
