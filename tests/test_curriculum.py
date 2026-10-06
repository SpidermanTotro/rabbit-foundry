from rabbit_foundry.curriculum import Curriculum, Outcome


def test_failure_increases_skill_attention():
    c = Curriculum()
    before = c.state()["code_repair"]["weight"]
    c.observe([Outcome("code_repair", False)])
    after = c.state()["code_repair"]["weight"]
    assert after > before
