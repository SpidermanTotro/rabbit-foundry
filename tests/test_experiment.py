from rabbit_foundry.experiment import SamplingExperiment


def test_equal_compute_and_winner():
    exp = SamplingExperiment(100)
    exp.record_fixed(0.40)
    exp.observe_adaptive("code_repair", False)
    exp.record_adaptive(0.55)
    assert all(r.compute_steps == 100 for r in exp.results)
    assert exp.winner() == "adaptive"
