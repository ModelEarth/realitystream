# Sourced by deploy-cloud-run.sh and add-team-members.sh.
# read_setting NAME: print NAME from the environment, else from the env file named in
# automation/paths.yaml, without printing the file's location.
# automation/ sits beside the realitystream folder, two levels up from this file.
RS_AUTOMATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/automation"
read_setting() {
  if [ -n "${!1:-}" ]; then echo "${!1}"; return; fi
  python3 - "$1" "$RS_AUTOMATION_DIR" <<'PY'
import os, re, sys
automation = sys.argv[2]
try:
    with open(os.path.join(automation, "paths.yaml"), encoding="utf-8") as fh:
        value = re.search(r"^\s*env_file:\s*(.+)$", fh.read(), re.M).group(1)
    value = re.sub(r"\s+#.*$", "", value).strip().strip("\"'")
    with open(os.path.join(automation, value), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(sys.argv[1] + "="):
                print(line.split("=", 1)[1].strip().strip("\"'"))
                break
except Exception:
    pass
PY
}
