import pytest
from fastapi.testclient import TestClient

from ex06_api import create_app

DOCS = [
    {"id": "cats", "text": "Cats are great pets. Cats sleep a lot.", "source": "pets.txt"},
    {"id": "dogs", "text": "Dogs are loyal pets.", "source": "pets.txt"},
    {"id": "stocks", "text": "The stock market fell today.", "source": "news.txt"},
]


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def filled_client(client):
    for doc in DOCS:
        assert client.post("/documents", json=doc).status_code == 201
    return client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_add_document_returns_201_and_id(client):
    response = client.post("/documents", json=DOCS[0])
    assert response.status_code == 201
    assert response.json() == {"id": "cats"}


def test_duplicate_id_returns_409(client):
    client.post("/documents", json=DOCS[0])
    assert client.post("/documents", json=DOCS[0]).status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"text": "no id", "source": "s"},
        {"id": "1", "source": "s"},
        {"id": "1", "text": "", "source": "s"},
    ],
)
def test_invalid_document_returns_422(client, body):
    assert client.post("/documents", json=body).status_code == 422


def test_get_document(filled_client):
    response = filled_client.get("/documents/dogs")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "dogs"
    assert body["text"] == DOCS[1]["text"]
    assert body["source"] == "pets.txt"


def test_get_missing_document_returns_404(client):
    assert client.get("/documents/nope").status_code == 404


def test_each_app_starts_empty(filled_client):
    fresh = TestClient(create_app())
    assert fresh.get("/documents/cats").status_code == 404


def test_search_ranks_best_match_first(filled_client):
    response = filled_client.get("/search", params={"q": "cats pets"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["id"] for r in results] == ["cats", "dogs"]
    assert all(isinstance(r["score"], float) for r in results)
    assert results[0]["score"] > results[1]["score"]


def test_search_respects_k(filled_client):
    results = filled_client.get("/search", params={"q": "pets", "k": 1}).json()["results"]
    assert len(results) == 1


def test_search_with_no_documents_is_empty(client):
    assert client.get("/search", params={"q": "cats"}).json() == {"results": []}


def test_search_with_no_matches_is_empty(filled_client):
    assert filled_client.get("/search", params={"q": "zebra"}).json() == {"results": []}


@pytest.mark.parametrize("params", [{"q": "cats", "k": 0}, {"q": "cats", "k": 21}, {"q": ""}, {}])
def test_search_invalid_params_return_422(filled_client, params):
    assert filled_client.get("/search", params=params).status_code == 422
