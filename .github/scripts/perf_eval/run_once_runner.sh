#!/usr/bin/env bash
# Launch a one-shot (--ephemeral) GitHub Actions self-hosted runner in /dev/shm.
#
# It registers against REPO with label "lmdeploy-build", executes AT MOST ONE
# queued/dispatched job, and immediately exits and unregisters itself. Nothing
# stays online in the background and nothing is written to / or /nvme.
#
# Environment knobs:
#   REPO          target GitHub repository      [zzwwrr/lmdeploy-working]
#   RUNNER_LABEL  runner label                  [lmdeploy-build]
#   RUNNER_ROOT   tmpfs scratch directory       [/dev/shm/$USER-actions-runner]
set -euo pipefail

REPO=${REPO:-zzwwrr/lmdeploy-working}
RUNNER_LABEL=${RUNNER_LABEL:-lmdeploy-build}
RUNNER_ROOT=${RUNNER_ROOT:-/dev/shm/${USER:-$(id -un)}-actions-runner}
MIN_FREE_GIB=${MIN_FREE_GIB:-20}

command -v gh >/dev/null || { echo "gh CLI is required" >&2; exit 2; }

mkdir -p "$RUNNER_ROOT/cache"
free_gib=$(df -BG --output=avail "$RUNNER_ROOT" | tail -1 | tr -dc '0-9')
if (( free_gib < MIN_FREE_GIB )); then
    echo "only ${free_gib} GiB free under $RUNNER_ROOT, need >= ${MIN_FREE_GIB}" >&2
    exit 2
fi

read -r tag url digest < <(
    gh api repos/actions/runner/releases/latest --jq '
        .tag_name as $t |
        (.assets[] | select(.name | test("^actions-runner-linux-x64-[0-9.]+\\.tar\\.gz$"))) |
        "\($t) \(.browser_download_url) \(.digest // "")"
    '
)
[[ -n ${url:-} ]] || { echo "failed to resolve actions/runner download URL" >&2; exit 2; }

tarball=$RUNNER_ROOT/cache/$(basename "$url")
if [[ ! -s $tarball ]]; then
    echo "downloading $(basename "$url") ($tag) into $RUNNER_ROOT/cache ..."
    curl -fL --retry 3 -o "$tarball.part" "$url"
    if [[ ${digest:-} == sha256:* ]]; then
        echo "${digest#sha256:}  $tarball.part" | sha256sum -c -
    fi
    mv "$tarball.part" "$tarball"
fi

INST=$RUNNER_ROOT/inst-$$
mkdir -p "$INST"

cleanup() {
    set +e
    if [[ -f $INST/.runner ]]; then
        rm_tok=$(gh api -X POST "repos/${REPO}/actions/runners/remove-token" --jq .token 2>/dev/null || true)
        if [[ -n $rm_tok ]]; then
            (cd "$INST" && ./config.sh remove --token "$rm_tok" >/dev/null 2>&1) || true
        fi
    fi
    rm -rf "$INST"
}
trap cleanup EXIT INT TERM

echo "unpacking runner into $INST ..."
tar -xzf "$tarball" -C "$INST"

reg_tok=$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)
runner_name="once-$(hostname -s)-$$"

(
    cd "$INST"
    ./config.sh --unattended \
        --url "https://github.com/${REPO}" \
        --token "$reg_tok" \
        --name "$runner_name" \
        --labels "$RUNNER_LABEL" \
        --work "_work" \
        --ephemeral \
        --replace
    echo "runner '$runner_name' ready (ephemeral: will run 1 job and exit)."
    ./run.sh
)
