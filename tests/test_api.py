"""The HTTP API, with the RAG pipeline faked (no Ollama, no Qdrant) and profiles in a temporary database."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

import ragapp.api as api
from ragapp.profile_store import ProfileStore

TODAY = "2026-11-26"  # Year 2: Block B, week 3


@pytest.fixture
def store(tmp_path):
    store = ProfileStore(tmp_path / "profiles.sqlite3")
    api.app.dependency_overrides[api.get_store] = lambda: store
    yield store
    api.app.dependency_overrides.clear()


@pytest.fixture
def client(store):
    return TestClient(api.app)  # keeps cookies between requests, like a browser


@pytest.fixture
def seen(monkeypatch):
    """What the fake pipeline was called with; empty when it was not called."""
    calls = {}

    def fake_answer(question, k, student):
        calls.update(question=question, k=k, student=student)
        return {"question": question, "answer": "Friday [1]", "chunks": [{"text": "full chunk text"}], "sources": []}

    monkeypatch.setattr(api, "answer", fake_answer)
    return calls


def save_year(client, year="Year2", today=TODAY):
    return client.put("/profile", json={"year": year}, params={"today": today})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------- /profile


def test_a_new_visitor_is_asked_the_year_and_gets_no_cookie(client):
    response = client.get("/profile", params={"today": TODAY})
    assert response.json()["ask_year"] is True
    assert "set-cookie" not in response.headers  # an id is only handed out when there is something to save


def test_saving_the_year_sets_a_private_cookie_and_returns_where_the_student_is(client):
    response = save_year(client)
    assert response.status_code == 200
    assert response.json()["label"] == "Year 2 · Block B · week 3"
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie


def test_the_saved_year_is_remembered(client):
    save_year(client)
    profile = client.get("/profile", params={"today": TODAY}).json()
    assert (profile["ask_year"], profile["year"], profile["block"], profile["week"]) == (False, "Year2", "BlockB", 3)


def test_a_new_academic_year_asks_again(client):
    save_year(client)
    profile = client.get("/profile", params={"today": "2027-09-15"}).json()
    assert (profile["ask_year"], profile["academic_year"], profile["year"]) == (True, "2027-28", "Year2")


def test_answering_again_in_the_new_year_keeps_the_same_user(client, store):
    first = save_year(client).cookies[api.COOKIE]
    second = save_year(client, "Year3", "2027-09-15")
    assert api.COOKIE not in second.cookies or second.cookies[api.COOKIE] == first
    assert store.get(first).year == "Year3"


def test_two_browsers_have_their_own_profiles(store):
    one, two = TestClient(api.app), TestClient(api.app)
    save_year(one, "Year1")
    save_year(two, "Year3")
    assert one.get("/profile", params={"today": TODAY}).json()["year"] == "Year1"
    assert two.get("/profile", params={"today": TODAY}).json()["year"] == "Year3"


def test_a_malformed_cookie_counts_as_no_profile(client, store):
    store.save("x' OR '1'='1", "Year2", "2026-27")
    client.cookies.set(api.COOKIE, "x' OR '1'='1")
    assert client.get("/profile", params={"today": TODAY}).json()["ask_year"] is True


def test_saving_rejects_an_unknown_year(client):
    assert client.put("/profile", json={"year": "Year4"}).status_code == 422


# ---------------------------------------------------------------- /ask


def test_ask_uses_the_saved_year_and_the_block_of_today(client, seen):
    save_year(client)
    response = client.get("/ask", params={"q": "whats due", "today": TODAY})

    assert response.status_code == 200
    assert response.json()["answer"] == "Friday [1]"
    assert response.json()["student"]["label"] == "Year 2 · Block B · week 3"
    assert "chunks" not in response.json()  # full chunk texts are for the eval, not for clients
    assert (seen["student"].year, seen["student"].block, seen["student"].today) == ("Year2", "BlockB", date(2026, 11, 26))


def test_ask_without_a_profile_asks_for_the_year_first(client, seen):
    response = client.get("/ask", params={"q": "vpn", "today": TODAY})
    assert response.status_code == 409
    assert seen == {}  # the pipeline did not run


def test_ask_with_a_profile_from_last_academic_year_asks_again(client, seen):
    save_year(client)
    assert client.get("/ask", params={"q": "vpn", "today": "2027-09-15"}).status_code == 409
    assert seen == {}


def test_ask_in_the_summer_says_the_block_is_unknown(client, seen):
    save_year(client, today="2027-07-20")
    response = client.get("/ask", params={"q": "whats due", "today": "2027-07-20"})
    assert response.json()["student"]["label"] == "Year 2 · block unknown (summer holiday)"
    assert seen["student"].block == ""


def test_year_and_block_override_the_profile_for_testing(client, seen):
    response = client.get("/ask", params={"q": "whats due", "year": "Year1", "block": "BlockC", "today": TODAY})
    assert response.status_code == 200
    assert (seen["student"].year, seen["student"].block) == ("Year1", "BlockC")


def test_a_year_override_alone_takes_the_block_from_the_date(client, seen):
    client.get("/ask", params={"q": "whats due", "year": "Year3", "today": TODAY})
    assert seen["student"].block == "Semester1"


def test_ask_rejects_an_unknown_block(client):
    assert client.get("/ask", params={"q": "vpn", "year": "Year2", "block": "BlockZ"}).status_code == 422


# ---------------------------------------------------------------- the chat page


def test_the_chat_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/profile" in response.text and "/ask" in response.text
