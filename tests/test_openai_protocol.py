from rabbit_foundry.openai_protocol import ServedModel, last_user_text, models_response


def test_models_endpoint_shape():
    body = models_response([ServedModel("rabbit-local", 4096, 1024)])
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "rabbit-local"


def test_extract_last_user_text():
    assert last_user_text([
        {"role": "system", "content": "code"},
        {"role": "user", "content": "fix this"},
    ]) == "fix this"
