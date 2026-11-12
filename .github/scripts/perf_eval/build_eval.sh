#!/usr/bin/env bash
# Build selected lmdeploy components on the local machine from a clean tree and
# record build-time / memory metrics. No image is built or pushed.
#
# Everything heavy (source copy, venvs, build trees, TMPDIR, uv cache) lives
# under BUILD_ROOT, which defaults to tmpfs, so the build never touches full disks.
#
# Components (COMPONENTS, comma/space separated, or "all"):
#   turbomind      lmdeploy wheel including the turbomind CUDA kernels   [default]
#   python-only    lmdeploy wheel with DISABLE_TURBOMIND=1 (pure Python, seconds)
#   deep_ep | deep_gemm | flash_mla | fast_hadamard
#                  third-party CUDA wheels; versions are read from
#                  docker/prepare_3rdparty_wheel.sh. They need torch, which is
#                  downloaded into a separate venv (several GB).
#   "all" = turbomind + the four third-party wheels. turbomind and python-only
#   produce the same wheel name, so they cannot be combined.
#
# Knobs (environment variables):
#   JOBS          ninja / MAX_JOBS parallelism                            [16]
#   CPU_LIST      taskset cpu list hard-capping the build, e.g. 48-63.   [auto]
#                 "auto": if the host exposes more CPUs than JOBS, pin to the
#                 last JOBS allowed CPUs; otherwise do not pin. "none": never pin.
#   NVCC_THREADS  -DNVCC_THREADS for turbomind                            [unset: CMakeLists default]
#   CUDA_ARCHS    CUDAARCHS for turbomind, e.g. "80-real;90a-real"        [unset: CMakeLists default]
#   CMAKE_DEFS    extra turbomind cmake defines WITHOUT the -D prefix,
#                 ';'-separated, e.g. "BUILD_FAST_MATH:BOOL=OFF"          [unset]
#   TORCH_INDEX   pip index for torch (third-party components)            [https://download.pytorch.org/whl/cu<ver>0]
#   CUDA_HOME     CUDA toolkit root                                       [/usr/local/cuda]
#   BUILD_ROOT    scratch root                                            [/dev/shm/$USER-lmdeploy-perf-eval]
#   RUN_TAG       sub-directory name for this run                         [timestamp]
#   OUT_DIR       where metrics/logs/wheels are written                   [$BUILD_ROOT/$RUN_TAG/out]
#   KEEP_WHEEL    1 = copy wheels into OUT_DIR                            [0]
#   KEEP_BUILD    1 = keep the build tree after the run                   [0]
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC_DIR=${SRC_DIR:-$(git -C "$HERE" rev-parse --show-toplevel)}
COMPONENTS=${COMPONENTS:-turbomind}
JOBS=${JOBS:-16}
CPU_LIST=${CPU_LIST:-auto}
NVCC_THREADS=${NVCC_THREADS:-}
CUDA_ARCHS=${CUDA_ARCHS:-}
CMAKE_DEFS=${CMAKE_DEFS:-}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
BUILD_ROOT=${BUILD_ROOT:-/dev/shm/${USER:-$(id -un)}-lmdeploy-perf-eval}
KEEP_WHEEL=${KEEP_WHEEL:-0}
KEEP_BUILD=${KEEP_BUILD:-0}
MIN_FREE_GIB=${MIN_FREE_GIB:-30}

WORK=$BUILD_ROOT/$RUN_TAG
OUT=${OUT_DIR:-$WORK/out}

# ---- validate inputs -------------------------------------------------------
THIRD_PARTY=(deep_ep deep_gemm flash_mla fast_hadamard)
comps=()
for c in $(echo "${COMPONENTS//,/ }"); do
    if [[ $c == all ]]; then comps+=(turbomind "${THIRD_PARTY[@]}"); else comps+=("$c"); fi
done
declare -A seen=()
uniq_comps=()
for c in "${comps[@]}"; do
    case $c in
        turbomind|python-only|deep_ep|deep_gemm|flash_mla|fast_hadamard) ;;
        *) echo "unknown component '$c'" >&2; exit 2 ;;
    esac
    [[ -n ${seen[$c]:-} ]] || { seen[$c]=1; uniq_comps+=("$c"); }
done
comps=("${uniq_comps[@]}")
(( ${#comps[@]} )) || { echo "no component selected" >&2; exit 2; }
[[ -n ${seen[turbomind]:-} && -n ${seen[python-only]:-} ]] && {
    echo "turbomind and python-only produce the same wheel; pick one" >&2; exit 2; }

[[ $JOBS =~ ^[0-9]+$ && $JOBS -ge 1 ]] || { echo "JOBS must be a positive integer: '$JOBS'" >&2; exit 2; }
[[ -z $NVCC_THREADS || $NVCC_THREADS =~ ^[0-9]+$ ]] || { echo "NVCC_THREADS must be an integer" >&2; exit 2; }
[[ -x $CUDA_HOME/bin/nvcc ]] || { echo "nvcc not found under CUDA_HOME=$CUDA_HOME" >&2; exit 2; }

mkdir -p "$WORK" "$OUT" "$WORK/tmp"
free_gib=$(df -BG --output=avail "$BUILD_ROOT" | tail -1 | tr -dc '0-9')
if (( free_gib < MIN_FREE_GIB )); then
    echo "only ${free_gib} GiB free under $BUILD_ROOT, need >= ${MIN_FREE_GIB}" >&2
    exit 2
fi

cleanup() {
    if [[ $KEEP_BUILD != 1 ]]; then
        rm -rf "$WORK/src" "$WORK/venv" "$WORK/venv-3p" "$WORK/tmp" "$WORK/wheels"
    fi
}
trap cleanup EXIT

export TMPDIR=$WORK/tmp
export UV_CACHE_DIR=$BUILD_ROOT/uv-cache
export PATH=$CUDA_HOME/bin:$PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}
export CMAKE_BUILD_PARALLEL_LEVEL=$JOBS
export MAX_JOBS=$JOBS   # honoured by torch cpp_extension (third-party wheels)

CUDA_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
if [[ $CUDA_MAJOR == 13 ]]; then TORCH_CU=cu130; else TORCH_CU=cu${CUDA_MAJOR}8; fi
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/$TORCH_CU}
PYV=${PYTHON_VERSION:-3.12}

echo "::group::sync source"
# Tracked files only: untracked scratch/worktrees must not leak into the build.
mkdir -p "$WORK/src" "$WORK/wheels"
git -C "$SRC_DIR" ls-files -z --cached | rsync -a -0 --files-from=- "$SRC_DIR"/ "$WORK/src"/
echo "::endgroup::"

has() { [[ -n ${seen[$1]:-} ]]; }
need_3p=0
for c in "${THIRD_PARTY[@]}"; do has "$c" && need_3p=1; done

echo "::group::venv + build requirements"
if ! command -v uv >/dev/null; then
    python3 -m pip install --user --break-system-packages -q uv
    export PATH=$HOME/.local/bin:$PATH
fi
uv venv -q -p "$PYV" "$WORK/venv"
uv pip install -q -p "$WORK/venv/bin/python" build wheel psutil \
    -r "$WORK/src/requirements/build.txt"
has turbomind && uv pip install -q -p "$WORK/venv/bin/python" "nvidia-nccl-cu${CUDA_MAJOR}>=2.30.4"
if (( need_3p )); then
    uv venv -q -p "$PYV" "$WORK/venv-3p"
    uv pip install -q -p "$WORK/venv-3p/bin/python" pip "cmake<4.0" wheel ninja setuptools packaging
    # Install torch strictly from TORCH_INDEX (mixing PyPI causes multi-GB resolver backtracking).
    uv pip install -p "$WORK/venv-3p/bin/python" --index-url "$TORCH_INDEX" "torch<=2.12.1"
    if has deep_ep; then
        if [[ $CUDA_MAJOR == 13 ]]; then
            uv pip install -q -p "$WORK/venv-3p/bin/python" \
                nvidia-nvshmem-cu13==3.4.5 mooncake-transfer-engine-cuda13==0.3.12.post1
        else
            uv pip install -q -p "$WORK/venv-3p/bin/python" \
                nvidia-nvshmem-cu12==3.4.5 mooncake-transfer-engine==0.3.12.post1
        fi
    fi
fi
echo "::endgroup::"

pin=()
if [[ $CPU_LIST == auto ]]; then
    allowed=()
    for part in $(taskset -pc $$ | sed 's/.*: //; s/,/ /g'); do
        if [[ $part == *-* ]]; then
            lo=${part%-*}; hi=${part#*-}
            for ((i = lo; i <= hi; i++)); do allowed+=("$i"); done
        else
            allowed+=("$part")
        fi
    done
    if (( ${#allowed[@]} > JOBS )); then
        tail=("${allowed[@]: -JOBS}")
        CPU_LIST=$(IFS=,; echo "${tail[*]}")
    else
        CPU_LIST=
    fi
elif [[ $CPU_LIST == none ]]; then
    CPU_LIST=
fi
if [[ -n $CPU_LIST ]]; then
    taskset -c "$CPU_LIST" true || { echo "CPU_LIST='$CPU_LIST' is not usable on this host (has $(nproc --all) CPUs)" >&2; exit 2; }
    pin=(taskset -c "$CPU_LIST")
fi

{
    echo "commit:        $(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "components:    ${comps[*]}"
    echo "nvcc:          $(nvcc --version | tail -2 | head -1)"
    echo "gcc:           $(gcc --version | head -1)"
    echo "cmake/ninja:   $("$WORK/venv/bin/cmake" --version | head -1) / ninja $("$WORK/venv/bin/ninja" --version)"
    echo "host cpus:     $(nproc --all)"
    echo "jobs:          $JOBS"
    echo "cpu_list:      ${CPU_LIST:-<unpinned>}"
    echo "nvcc_threads:  ${NVCC_THREADS:-<default>}"
    echo "cuda_archs:    ${CUDA_ARCHS:-<default>}"
    echo "cmake_defs:    ${CMAKE_DEFS:-<none>}"
    echo "loadavg start: $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "mem avail GiB: $(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo)"
} | tee "$OUT/env.txt"

# third_party_pin NAME -> version pinned in docker/prepare_3rdparty_wheel.sh
third_party_pin() {
    sed -n "s/^$1=\([^ #]*\).*/\1/p" "$WORK/src/docker/prepare_3rdparty_wheel.sh" | head -1
}

# build_component NAME: runs under the process-tree monitor, writes $OUT/NAME/*
build_component() {
    local name=$1 d=$OUT/$1 rc=0 vpy
    mkdir -p "$d"
    local cmd env_prefix=(env)
    case $name in
        turbomind|python-only)
            local cfg=() defs=()
            vpy=$WORK/venv
            if [[ $name == python-only ]]; then
                env_prefix+=(DISABLE_TURBOMIND=1)
            else
                [[ -n $CUDA_ARCHS ]] && env_prefix+=("CUDAARCHS=$CUDA_ARCHS")
                # typed: an untyped define is overridden by set(... CACHE) in CMakeLists.txt
                [[ -n $NVCC_THREADS ]] && defs+=("NVCC_THREADS:STRING=$NVCC_THREADS")
                [[ -n $CMAKE_DEFS ]] && defs+=("$CMAKE_DEFS")
                if (( ${#defs[@]} )); then
                    local joined
                    joined=$(IFS=';'; echo "${defs[*]}")
                    # cmake_build_extension reads `build_ext -D"A=B;C=D"` and prepends -D itself.
                    cfg=(-C--global-option=build_ext "-C--global-option=-D${joined}")
                fi
            fi
            cmd=(python -m build --wheel --no-isolation -o "$WORK/wheels/$name" "${cfg[@]}" .)
            ;;
        *)
            vpy=$WORK/venv-3p
            local url ver
            case $name in
                deep_ep) url=https://github.com/deepseek-ai/DeepEP.git; ver=$(third_party_pin DEEP_EP_VERSION)
                    [[ $CUDA_MAJOR == 13 ]] && env_prefix+=("CPLUS_INCLUDE_PATH=$CUDA_HOME/include/cccl:${CPLUS_INCLUDE_PATH:-}") ;;
                deep_gemm) url=https://github.com/deepseek-ai/DeepGEMM.git; ver=$(third_party_pin DEEP_GEMM_VERSION) ;;
                flash_mla) url=https://github.com/deepseek-ai/FlashMLA.git; ver=$(third_party_pin FLASH_MLA_VERSION)
                    env_prefix+=(FLASH_MLA_DISABLE_SM100=1) ;;
                fast_hadamard) url=https://github.com/Dao-AILab/fast-hadamard-transform.git
                    ver=$(third_party_pin FAST_HADAMARD_TRANSFORM_VERSION) ;;
            esac
            [[ -n $ver ]] || { echo "no pinned version for $name" >&2; return 2; }
            cmd=(pip wheel -v --no-build-isolation --no-deps -w "$WORK/wheels/$name" "git+${url}@${ver}")
            echo "$name @ $ver" >"$d/pin.txt"
            ;;
    esac

    echo "::group::build $name"
    set +e
    (cd "$WORK/src" && PATH=$vpy/bin:$PATH VIRTUAL_ENV=$vpy "${pin[@]}" "${env_prefix[@]}" \
        "$WORK/venv/bin/python" "$HERE/monitor.py" \
        --summary "$d/monitor.json" --series "$d/series.csv" -- "${cmd[@]}") 2>&1 | tee "$d/build.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "::endgroup::"
    echo "$rc" >"$d/rc"

    if [[ $name == turbomind ]]; then
        # FetchContent sub-builds leave tiny .ninja_log files too; the main one is the largest.
        local nl eff_threads eff_archs
        nl=$(find "$WORK/src" -name .ninja_log -printf '%s %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2- || true)
        if [[ -n $nl ]]; then
            cp "$nl" "$d/ninja_log"
            "$WORK/venv/bin/python" "$HERE/ninja_report.py" "$nl" --top 15 >"$d/ninja_report.txt" || true
        fi
        eff_threads=$(find "$WORK/src" -name CMakeCache.txt -print0 2>/dev/null | xargs -0 -r grep -h '^NVCC_THREADS' | head -1 | cut -d= -f2 || true)
        eff_archs=$(grep -m1 'Building with CUDA archs' "$d/build.log" | sed 's/.*archs: *//' || true)
        {
            echo "effective NVCC_THREADS: ${eff_threads:-?}"
            echo "effective CUDA archs:   ${eff_archs:-?}"
        } | tee "$d/effective.txt"
        if [[ -n $NVCC_THREADS && $eff_threads != "$NVCC_THREADS" && $rc == 0 ]]; then
            echo "requested NVCC_THREADS=$NVCC_THREADS but CMake used '${eff_threads:-?}'; results are invalid" >&2
            echo 3 >"$d/rc"
        fi
    fi
    if [[ $KEEP_WHEEL == 1 ]]; then
        cp "$WORK"/wheels/"$name"/*.whl "$OUT/" 2>/dev/null || true
    fi
}

for c in "${comps[@]}"; do
    build_component "$c" || echo 2 >"$OUT/$c/rc"
done

rc=0
"$WORK/venv/bin/python" - "$OUT" "$WORK/wheels" "${comps[@]}" <<'PY' || rc=$?
import glob, json, os, sys
out, wheels, comps = sys.argv[1], sys.argv[2], sys.argv[3:]
lines = ['### Build evaluation', '', '| component | result | wall | peak RSS | peak compile procs | wheel |',
         '|---|---|---|---|---|---|']
overall = 0
detail = []
for c in comps:
    d = f'{out}/{c}'
    rc = int(open(f'{d}/rc').read()) if os.path.exists(f'{d}/rc') else 2
    mon = json.load(open(f'{d}/monitor.json')) if os.path.exists(f'{d}/monitor.json') else {}
    whl = sorted(glob.glob(f'{wheels}/{c}/*.whl'))
    mon.update(build_exit_code=rc, wheels=[{'name': os.path.basename(w), 'mib': round(os.path.getsize(w) / 2**20, 1)} for w in whl])
    json.dump(mon, open(f'{d}/summary.json', 'w'), indent=2)
    overall = overall or rc
    wall = mon.get('wall_s')
    lines.append('| {} | {} | {} | {} | {} | {} |'.format(
        c, 'ok' if rc == 0 else f'**FAILED ({rc})**',
        f'{wall} s ({wall / 60:.1f} min)' if wall is not None else '?',
        f"{mon.get('peak_tree_rss_gib', '?')} GiB", mon.get('peak_concurrent_compile_procs', '?'),
        ', '.join(f"{w['name']} ({w['mib']} MiB)" for w in mon['wheels']) or 'none'))
    for fname, title in (('effective.txt', 'effective settings'), ('ninja_report.txt', 'ninja report')):
        p = f'{d}/{fname}'
        if os.path.exists(p):
            detail += ['', f'<details><summary>{c}: {title}</summary>', '', '```', open(p).read().rstrip(), '```', '</details>']
lines += ['', '<details><summary>env</summary>', '', '```', open(f'{out}/env.txt').read().rstrip(), '```', '</details>']
open(f'{out}/summary.md', 'w').write('\n'.join(lines + detail) + '\n')
json.dump({'components': comps, 'exit_code': overall}, open(f'{out}/summary.json', 'w'), indent=2)
sys.exit(overall)
PY
cat "$OUT/summary.md"
echo "metrics written to $OUT"
exit "$rc"
