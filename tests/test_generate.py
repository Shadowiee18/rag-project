from ragapp import config, llm
from ragapp.generate import build_prompt, generate, system_prompt


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


def test_calendar_lines_come_after_the_student_line_and_before_the_question():
    chunks = [{"title": "VPN Access", "text": "Install the VPN client."}]
    prompt = build_prompt("When does week 2 start?", chunks, "Today is Tuesday.", ["Week 2 of Year 2 Block B: ..."])
    assert "Student: Today is Tuesday.\n\nCalendar:\n- Week 2 of Year 2 Block B: ...\n\nQuestion:" in prompt


def test_no_calendar_section_without_facts():
    assert "Calendar" not in build_prompt("q", [{"title": "t", "text": "x"}], "Today is Tuesday.")


def test_with_calendar_facts_the_prompt_counts_them_as_a_source(monkeypatch):
    monkeypatch.setattr(config, "STUDENT_LINE_AS_SOURCE", False)
    plain, with_calendar = system_prompt(), system_prompt(calendar=True)
    assert "Calendar" not in plain
    assert "Use ONLY the numbered context passages below and the Calendar lines" in with_calendar
    assert "If neither the passages nor the Calendar lines contain the answer" in with_calendar
    assert "need no citation" in with_calendar


def test_generate_sends_the_calendar_prompt_only_when_there_are_facts(monkeypatch):
    sent = []
    monkeypatch.setattr(llm, "chat", lambda messages: sent.append(messages) or "ok")
    chunks = [{"title": "t", "text": "x"}]
    generate("q", chunks, "Today is Tuesday.")
    generate("q", chunks, "Today is Tuesday.", ["Week 2 of Year 2 Block B: ..."])
    assert "Calendar" not in sent[0][0]["content"] and "Calendar" in sent[1][0]["content"]
    assert "Calendar:\n- Week 2" in sent[1][1]["content"]
