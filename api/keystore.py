"""Local key store: one secret per name in the OS credential store, read one name at a time.

Backend is the `keyring` package: Windows Credential Manager, macOS Keychain, or the
Linux Secret Service. Nothing is written to disk by this module except a small
sqlite counter of reads per name per day, used for the daily read limit.

    python api/keystore.py set GITHUB_REPORTS_TOKEN      # value typed hidden
    python api/keystore.py get GITHUB_REPORTS_TOKEN
    python api/keystore.py list
    python api/keystore.py delete GITHUB_REPORTS_TOKEN

Or paste NAME=value lines at http://localhost:8080/keys while `python api/main.py` runs
(that page is added only outside Cloud Run).

What this gives you: no plaintext .env for a process or agent to copy in one read,
values handed out singly, and a cap on how many reads a day each name allows.
What it cannot give you: the user that stored a key can always read it back,
because the OS unlocks the store for that user's processes. For keys that must
never be readable on a laptop, use Secret Manager on Cloud Run instead.
"""
import json
import os
import sqlite3
from datetime import date

SERVICE = "modelearth"
INDEX = "__names__"
DAILY_READ_LIMIT = int(os.environ.get("DAILY_KEY_READS", "50"))
COUNTER_DB = os.path.join(os.path.expanduser("~"), ".modelearth", "key-reads.sqlite")


class KeyReadLimit(Exception):
    pass


def _keyring():
    try:
        import keyring
        from keyring.errors import NoKeyringError  # noqa: F401
        keyring.get_keyring()
        return keyring
    except Exception:
        return None  # no credential store on this machine (for example a Cloud Run container)


def available():
    return _keyring() is not None


def names():
    kr = _keyring()
    if kr is None:
        return []
    raw = kr.get_password(SERVICE, INDEX)
    return json.loads(raw) if raw else []


def set_key(name, value):
    kr = _keyring()
    if kr is None:
        raise RuntimeError("No OS credential store available on this machine.")
    name, value = name.strip(), value.strip()
    if not name or not value:
        raise ValueError("Both a name and a value are required.")
    kr.set_password(SERVICE, name, value)
    kr.set_password(SERVICE, INDEX, json.dumps(sorted(set(names()) | {name})))


def delete_key(name):
    kr = _keyring()
    if kr is None:
        return
    try:
        kr.delete_password(SERVICE, name)
    except Exception:
        pass
    kr.set_password(SERVICE, INDEX, json.dumps(sorted(set(names()) - {name})))


def _count(name, caller, increment):
    os.makedirs(os.path.dirname(COUNTER_DB), exist_ok=True)
    with sqlite3.connect(COUNTER_DB) as db:
        db.execute("CREATE TABLE IF NOT EXISTS reads (day TEXT, name TEXT, caller TEXT, n INTEGER, PRIMARY KEY (day, name, caller))")
        today = date.today().isoformat()
        if increment:
            db.execute("INSERT INTO reads VALUES (?, ?, ?, 1) ON CONFLICT(day, name, caller) DO UPDATE SET n = n + 1",
                       (today, name, caller))
        row = db.execute("SELECT COALESCE(SUM(n), 0) FROM reads WHERE day = ? AND name = ?", (today, name)).fetchone()
        return row[0]


def get_key(name, caller="local"):
    """Return the stored value, or None when absent or no store exists. Raises KeyReadLimit past the daily cap."""
    kr = _keyring()
    if kr is None:
        return None
    value = kr.get_password(SERVICE, name)
    if value is None:
        return None
    if _count(name, caller, increment=False) >= DAILY_READ_LIMIT:
        raise KeyReadLimit(f"{name} has been read {DAILY_READ_LIMIT} times today; limit resets at midnight.")
    _count(name, caller, increment=True)
    return value


def reads_today():
    """{name: reads so far today} for every stored name."""
    out = {}
    for name in names():
        out[name] = _count(name, "", increment=False)
    return out


def main(argv):
    import getpass
    cmd = argv[1] if len(argv) > 1 else "list"
    if cmd == "list":
        counts = reads_today()
        for n in names():
            print(f"{n}  (reads today: {counts.get(n, 0)}/{DAILY_READ_LIMIT})")
    elif cmd == "set" and len(argv) == 3:
        set_key(argv[2], getpass.getpass(f"Value for {argv[2]}: "))
        print(f"stored {argv[2]}")
    elif cmd == "get" and len(argv) == 3:
        value = get_key(argv[2], caller="cli")
        print(value if value is not None else f"{argv[2]} not set")
    elif cmd == "delete" and len(argv) == 3:
        delete_key(argv[2])
        print(f"deleted {argv[2]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    import sys
    main(sys.argv)
