from rabbit_foundry.episodes import next_byte_episodes


def test_episode_generation_is_deterministic():
    data = bytes(range(256)) * 2
    a = next_byte_episodes("o/r", "abcdef012345", "x.py", data, window=32)
    b = next_byte_episodes("o/r", "abcdef012345", "x.py", data, window=32)
    assert a
    assert [x.episode_id for x in a] == [x.episode_id for x in b]
    assert a[0].prompt[1:] == a[0].target[:-1]
