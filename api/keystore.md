[RealityStream](../) | [Deploy](../deploy/)
# Local key store

Keys such as `GITHUB_REPORTS_TOKEN` and `DATACOMMONS_API_KEY` can live in your operating system's credential store instead of a plaintext `.env` file. The store is Windows Credential Manager, macOS Keychain, or the Linux Secret Service, reached through the `keyring` package. Each key is one entry under the app name `modelearth`. Readers ask for one name at a time, and each name allows a limited number of reads per day.

## Steps

1. Install the package once, from the realitystream folder:

       pip install keyring

2. Store a key. The value is typed hidden and never echoed:

       python api/keystore.py set GITHUB_REPORTS_TOKEN

   Paste the token at the prompt. Repeat for other names, for example `DATACOMMONS_API_KEY` or `REALITYSTREAM_API_KEY`.

3. Check what is stored. Only names and today's read counts are shown, never values:

       python api/keystore.py list

4. Remove the same line from your `.env` file, so the credential store is the only copy.

5. Run anything as usual. `api/run_models.py`, `api/main.py` and the deploy scripts look a name up in this order: the environment, then the credential store, then the `.env` file named by `automation/paths.yaml`. Nothing changes in how you call them.

To remove a key: `python api/keystore.py delete GITHUB_REPORTS_TOKEN`. To read one back yourself: `python api/keystore.py get NAME` (this counts as a read).

## Daily read limit

Each name allows `DAILY_KEY_READS` reads per day (default 50), counted in a small SQLite file at `~/.modelearth/key-reads.sqlite`. Past the limit the reader gets a `KeyReadLimit` error until midnight. Raise it for a busy machine with `DAILY_KEY_READS=200` in the environment.

## What this protects, and what it does not

- It removes the plaintext `.env` that one file read could copy in full. A process or agent that asks for `GITHUB_REPORTS_TOKEN` gets that one value, nothing else, and the read is counted.
- It does not make a key unreadable on the computer. The user who stored a key can always read it back, because the operating system unlocks the store for that user's processes.
- Keys that must never be readable on a laptop belong in Secret Manager on Cloud Run. The Team Passphrase already lives there (`realitystream-api-key`), attached by `deploy/deploy-cloud-run.sh`.

## Browser keys are separate

The [keys page](https://model.earth/keys/) on model.earth keeps tokens in your browser's own storage and sends them with requests to the Cloud Run API (header `X-GitHub-Token`). That path never touches this credential store. Use the browser page for the Run Models form; use this store for scripts and local runs.
