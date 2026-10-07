from rabbit_foundry.alpha_course_quality import enrich_alpha_capture, score_alpha_capture


def test_complete_debug_trajectory_is_high_value():
    row = {
        "environment": {"platform": {"system": "Linux"}},
        "messages": [
            {"role": "user", "content": "pytest failed with an error"},
            {"role": "assistant", "content": "The root cause is stale state. I will fix the state restore."},
            {"role": "tool", "content": "pytest: tests passed"},
        ],
    }
    enriched = enrich_alpha_capture(row)
    assert enriched["training_quality"]["high_value_repair"] is True
    assert enriched["training_quality"]["score"] == 1.0
    assert set(enriched["training_quality"]["signals"]) >= {
        "failure", "diagnosis", "repair", "test", "tool", "environment"
    }


def test_shallow_answer_scores_below_debug_trajectory():
    shallow = score_alpha_capture({
        "messages": [{"role": "assistant", "content": "Here is some code."}]
    })
    rich = score_alpha_capture({
        "environment": {"linux": {"id": "fedora"}},
        "messages": [
            {"role": "user", "content": "error"},
            {"role": "assistant", "content": "because config is wrong; fix it then verify"},
            {"role": "tool", "content": "tests passed"},
        ],
    })
    assert shallow.score < rich.score
    assert rich.score == 1.0
