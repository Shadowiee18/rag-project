from ragapp.generate import build_prompt


def test_build_prompt_numbers_chunks_in_order_and_includes_question():
    chunks = [
        {"title": "VPN Access", "text": "Install the VPN client."},
        {"title": "YouTrack", "text": "Log your tasks."},
    ]
    prompt = build_prompt("How do I get VPN?", chunks)

    assert "[1] (page: VPN Access)\nInstall the VPN client." in prompt
    assert "[2] (page: YouTrack)\nLog your tasks." in prompt
    assert prompt.index("[1]") < prompt.index("[2]")
    assert prompt.endswith("Question: How do I get VPN?")
