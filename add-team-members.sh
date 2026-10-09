#!/usr/bin/env bash
# Give each team member in GOOGLE_TEAM_EMAILS (comma-separated, in the shared env file) access to GOOGLE_PROJECT_ID.
# Usage: ./add-team-members.sh        (also run by deploy-cloud-run.sh; safe to repeat)
# Default roles are what Google requires to deploy to Cloud Run from source. Override with GOOGLE_TEAM_ROLES.
# Removing an email from the list does not revoke access; use: gcloud projects remove-iam-policy-binding
set -euo pipefail
cd "$(dirname "$0")"
source ./read-setting.sh

PROJECT="$(read_setting GOOGLE_PROJECT_ID)"
EMAILS="$(read_setting GOOGLE_TEAM_EMAILS)"
ROLES="$(read_setting GOOGLE_TEAM_ROLES)"
ROLES="${ROLES:-roles/run.sourceDeveloper,roles/iam.serviceAccountUser,roles/serviceusage.serviceUsageConsumer,roles/logging.viewer}"
if [ -z "$PROJECT" ]; then echo "Set GOOGLE_PROJECT_ID in the shared env file"; exit 1; fi
if [ -z "$EMAILS" ]; then echo "No GOOGLE_TEAM_EMAILS set, no team members added"; exit 0; fi

for email in ${EMAILS//,/ }; do
  for role in ${ROLES//,/ }; do
    if gcloud projects add-iam-policy-binding "$PROJECT" --member="user:$email" --role="$role" \
         --condition=None --quiet > /dev/null; then
      echo "$email: $role"
    else
      echo "$email: could not add $role"
    fi
  done
done
