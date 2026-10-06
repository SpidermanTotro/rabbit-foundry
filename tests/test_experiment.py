import pytest

from rabbit_foundry.experiment import ArmResult, choose_winner, run_experiment


def arm(name, loss, *, finite=True, compute=300):
    return ArmResult(name, compute, loss, finite, [1, 2, 3], [])


def test_equal_compute_and_lower_validation_loss_wins():
    results = [arm("fixed", 0.40), arm("adaptive", 0.35)]
    assert all(r.compute_steps == 300 for r in results)
    assert choose_winner(results) == "adaptive"


def test_nonfinite_arm_cannot_win():
    assert choose_winner([arm("fixed", 0.50), arm("adaptive", 0.10, finite=False)]) == "fixed"


def test_three_skill_experiment_rejects_incomplete_manifest(tmp_path):
    manifest = tmp_path / "episodes.json"
    manifest.write_text(
        '{"version":2,"episodes":['
        '{"episode_id":"train","skill":"code_prediction","source_repository":"r/x",'
        '"source_commit":"a","source_path":"x.py","prompt_hex":"0001020304050607",'
        '"target_hex":"0102030405060708"},'
        '{"episode_id":"valid","skill":"code_prediction","source_repository":"r/y",'
        '"source_commit":"b","source_path":"y.py","prompt_hex":"0001020304050607",'
        '"target_hex":"0102030405060708"}'
        ']}'
    )
    with pytest.raises(ValueError, match="missing training skills"):
        run_experiment(manifest, tmp_path / "result.json", steps=1, batch=1, seq=8, seeds=(1,), device_name="cpu")
