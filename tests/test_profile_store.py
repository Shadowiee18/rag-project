"""The saved profiles (SQLite in a temporary folder)."""

from concurrent.futures import ThreadPoolExecutor

from ragapp.profile_store import ProfileStore, SavedProfile


def test_unknown_user_has_no_profile(tmp_path):
    assert ProfileStore(tmp_path / "p.sqlite3").get("nobody") is None


def test_save_then_get(tmp_path):
    store = ProfileStore(tmp_path / "p.sqlite3")
    store.save("u1", "Year2", "2026-27")
    assert store.get("u1") == SavedProfile("Year2", "2026-27")


def test_saving_again_replaces_the_profile(tmp_path):
    store = ProfileStore(tmp_path / "p.sqlite3")
    store.save("u1", "Year2", "2026-27")
    store.save("u1", "Year3", "2027-28")
    assert store.get("u1") == SavedProfile("Year3", "2027-28")


def test_users_are_kept_apart(tmp_path):
    store = ProfileStore(tmp_path / "p.sqlite3")
    store.save("u1", "Year1", "2026-27")
    store.save("u2", "Year3", "2026-27")
    assert store.get("u1").year == "Year1"
    assert store.get("u2").year == "Year3"


def test_profiles_survive_a_restart_and_the_folder_is_created(tmp_path):
    path = tmp_path / "new folder" / "p.sqlite3"
    ProfileStore(path).save("u1", "Year2", "2026-27")
    assert ProfileStore(path).get("u1") == SavedProfile("Year2", "2026-27")


def test_many_requests_at_once(tmp_path):
    """The API serves requests on several threads; each save opens its own connection."""
    store = ProfileStore(tmp_path / "p.sqlite3")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.save(f"u{i}", "Year2", "2026-27"), range(50)))
    assert all(store.get(f"u{i}") for i in range(50))
