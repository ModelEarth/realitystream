[RealityStream](../)
# Deploy RealityStream to Google Cloud Run

The RealityStream API ([api/main.py](../api/main.py) with [api/run_models.py](../api/run_models.py)) runs as the Cloud Run service "realitystream". The [Run Models](../models/) page calls it. Formerly documented in the cloud repo's run/README.md.

Deploy from the realitystream folder:

    ./deploy/deploy-cloud-run.sh        # CPU service "realitystream"
    ./deploy/deploy-cloud-run.sh gpu    # GPU service "realitystream-gpu" (NVIDIA L4)

The script reads GOOGLE_PROJECT_ID (and optional GOOGLE_REGION, GOOGLE_TEAM_EMAILS) from the environment or your shared .env, runs [add-team-members.sh](add-team-members.sh), downloads the Random Bits Forest binary, and attaches the Team Passphrase secret when it exists.

<div class="bucket" markdown="1"><a id="googleaccountautomation"></a>

## Google Account for Automation

Live service: [realitystream-kwr4qrkopq-uc.a.run.app](https://realitystream-kwr4qrkopq-uc.a.run.app) (project "realitystream", us-central1), also used by [Run Models](../models/)

**Running more models:** without the Team Passphrase, the service runs 1 model once per day for each visitor. To run more, either:

- **Add the REALITYSTREAM_API_KEY for this Google Cloud project.** Ask the project's admin for the Team Passphrase (stored in Secret Manager as `realitystream-api-key`; it's our own shared passphrase, not a Google key), then enter it in the Team Passphrase field on [Run Models](../models/). Your browser remembers it until you choose Clear.
- **Run on your own cloud account.** Follow the steps below with your own project ID and billing account, then deploy. You get your own service URL and your own daily limits.

Our RealityStream serverless APIs reside in [api/main.py](../api/main.py) (`/health`, `/parameters`, `/run`). Settings are read by name from your shared .env, or from Cloud Run env vars and secrets once deployed.
To let a Coding CLI create a "realitystream" Google Cloud project and deploy main.py, your Google account needs a credential that can manage projects.

**A Google "API key" can't do this.** API keys (APIs & Services > Credentials > Create API key) only identify calls to APIs such as Gemini or Maps. Creating projects, linking billing and deploying to Cloud Run require an OAuth login or a service account. Choose one:

**Option A: Sign in with your Google account (recommended)**  
One login covers every project your account can access. Credentials are stored by gcloud, not in .env.

    gcloud auth login
    gcloud auth application-default login

**Option B: Service account key (unattended automation)**  
Use for scripts that run without you. Creating new projects with a service account requires a Google Cloud Organization, so for a personal Gmail account, create the project with Option A first.

1. In [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts), create "realitystream-deployer".
2. Grant roles: Cloud Run Admin, Cloud Build Editor, Artifact Registry Administrator, Service Account User, Service Usage Admin, Storage Admin.
3. Under Keys, add a JSON key and save it outside the webroot (never commit it).
4. In your shared .env (set by `env_file` in automation/paths.yaml), add GOOGLE_APPLICATION_CREDENTIALS with the JSON key's path.

If key creation is blocked, your organization enforces `iam.disableServiceAccountKeyCreation`. Use Option A instead.

**Add these to your shared .env** (names only shown here, never paste values into the terminal):

    GOOGLE_PROJECT_ID=realitystream-yourname   # Project IDs are globally unique, so add a suffix
    GOOGLE_BILLING_ID=000000-000000-000000     # From: gcloud billing accounts list
    GOOGLE_REGION=us-central1                  # Optional, used by deploy/deploy-cloud-run.sh
    REALITYSTREAM_API_KEY=                     # Optional: any long random string. Without it in header X-API-Key, callers get 1 model once per day

**Commands the Coding CLI runs to add the "realitystream" project and deploy:**

    gcloud projects create $GOOGLE_PROJECT_ID --name="RealityStream"
    gcloud billing projects link $GOOGLE_PROJECT_ID --billing-account=$GOOGLE_BILLING_ID
    gcloud config set project $GOOGLE_PROJECT_ID
    gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
    cd realitystream
    ./deploy/deploy-cloud-run.sh cpu

No Dockerfile is needed. Google Cloud buildpacks read realitystream's `requirements.txt`, `Procfile` (starts `api/main.py` with gunicorn) and `.python-version`, and `.gcloudignore` limits the upload to `api/`, `parameters/` and those files.

To use a Team Passphrase on Cloud Run, store it in Secret Manager as `realitystream-api-key` and let the service read it. deploy/deploy-cloud-run.sh attaches the secret automatically when it exists:

    gcloud services enable secretmanager.googleapis.com
    printf %s "$REALITYSTREAM_API_KEY" | gcloud secrets create realitystream-api-key --data-file=-
    gcloud secrets add-iam-policy-binding realitystream-api-key \
      --member="serviceAccount:$(gcloud projects describe $GOOGLE_PROJECT_ID --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
      --role="roles/secretmanager.secretAccessor"

Runs with the Team Passphrase push their output folder to GitHub modelearth/reports when the service has GITHUB_REPORTS_TOKEN. With GITHUB_REPORTS_TOKEN in your shared .env, deploy/deploy-cloud-run.sh copies it to the secret `github-reports-token` (when the secret is missing or differs), lets the service read it, and attaches it.

</div>

<div class="bucket" markdown="1"><a id="localkeystore"></a>

## Local key store

`python api/keystore.py list|set|get|delete NAME` keeps keys in the OS credential store (Windows Credential Manager, macOS Keychain, Linux Secret Service) under the app name `modelearth`. `set` asks for the value without showing it.

To push Run Models output with your own GitHub token instead, save `GITHUB_REPORTS_TOKEN` on the webroot's keys page ([localhost:8887/keys/#need=GITHUB_REPORTS_TOKEN](http://localhost:8887/keys/#need=GITHUB_REPORTS_TOKEN)). It stays in the browser, and the Run Models page sends it with each run.

`api/run_models.py` checks the store before the env file, one name at a time. Reads through `api/keystore.py` are capped at `DAILY_KEY_READS` a day per name (default 50).

Limit to be clear about: the user who stored a key can always read it back, because the OS unlocks the store for that user's processes. This removes the plaintext `.env` that one file read could copy; it does not make a key unreadable on the laptop. Keys that must never be readable locally belong in Secret Manager on Cloud Run.

</div>
