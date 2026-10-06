import random

from rabbit_foundry.sampler import AdaptiveSampler


def test_failures_raise_sampling_weight():
    sampler = AdaptiveSampler()
    sampler.add("python")
    before = sampler.skills["python"].weight
    sampler.record("python", passed=False)
    assert sampler.skills["python"].weight > before


def test_choose_registered_skill():
    sampler = AdaptiveSampler()
    sampler.add("python")
    sampler.add("rust")
    assert sampler.choose(random.Random(1)) in {"python", "rust"}
