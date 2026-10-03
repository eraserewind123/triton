#!/usr/bin/env bash
# Create a uv venv, build this Triton checkout, and run the Gluon MoE BMM1 fused-gather example.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

INTER="${INTER:-768}"
TOKENS="${TOKENS:-8192}"
HIDDEN="${HIDDEN:-3072}"
# EP = experts per token (top-k). TP = expert shards (local experts = E / TP).
EP="${EP:-${TOPK:-8}}"
TP="${TP:-${NUM_EXPERT_SHARDS:-1}}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
# BLOCK_N=512 (auto-selected for this token count) does not compile for N=768.
BLOCK_N="${BLOCK_N:-256}"
NUM_CTAS="${NUM_CTAS:-1}"

log() { echo "[moe-bmm1] $*"; }

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

# Triton CMake needs Python.h. Prefer uv-managed CPython so we do not depend on
# distro python*-dev packages.
uv python install "$PYTHON_VERSION" >/dev/null
MANAGED_PY="$(uv python list --only-installed | awk -v ver="$PYTHON_VERSION" '
  $0 ~ ("cpython-" ver) && $0 ~ /\/uv\/python\// { print $2; exit }
')"
if [[ -z "$MANAGED_PY" || ! -x "$MANAGED_PY" ]]; then
  echo "could not find a uv-managed CPython $PYTHON_VERSION interpreter" >&2
  exit 1
fi

has_python_h() {
  local py="$1"
  [[ -x "$py" ]] || return 1
  "$py" -c 'import os, sysconfig; raise SystemExit(0 if os.path.isfile(os.path.join(sysconfig.get_path("include"), "Python.h")) else 1)'
}

if [[ ! -x "$VENV/bin/python" ]] || ! has_python_h "$VENV/bin/python"; then
  log "creating uv venv at $VENV ($MANAGED_PY)"
  uv venv --clear --python "$MANAGED_PY" "$VENV"
else
  log "reusing venv $VENV"
fi

PY="$VENV/bin/python"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

# Prebuilt LLVM used by Triton requires the ZLIB CMake target. Distro zlib-dev
# is often missing, so build a user-local copy.
DEPS_PREFIX="${DEPS_PREFIX:-$ROOT/.deps}"
if [[ ! -f "$DEPS_PREFIX/include/zlib.h" || ! -e "$DEPS_PREFIX/lib/libz.so" && ! -e "$DEPS_PREFIX/lib/libz.a" ]]; then
  log "building zlib into $DEPS_PREFIX"
  ZLIB_VER="1.3.1"
  ZLIB_SRC="$(mktemp -d)"
  trap 'rm -rf "$ZLIB_SRC"' EXIT
  curl -fsSL "https://github.com/madler/zlib/releases/download/v${ZLIB_VER}/zlib-${ZLIB_VER}.tar.gz" \
    | tar -xz -C "$ZLIB_SRC" --strip-components=1
  (
    cd "$ZLIB_SRC"
    ./configure --prefix="$DEPS_PREFIX"
    make -j"$(nproc)"
    make install
  )
  trap - EXIT
  rm -rf "$ZLIB_SRC"
fi
export CMAKE_PREFIX_PATH="$DEPS_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export ZLIB_ROOT="$DEPS_PREFIX"
export LIBRARY_PATH="$DEPS_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$DEPS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LDFLAGS="${LDFLAGS:-} -L$DEPS_PREFIX/lib"

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  log "installing torch from $TORCH_INDEX_URL"
  uv pip install --python "$PY" torch --index-url "$TORCH_INDEX_URL"
fi

# Both PyTorch wheels and this checkout own the triton/ package; drop the wheel copy.
uv pip uninstall --python "$PY" triton pytorch-triton pytorch-triton-cu13 || true

log "installing Triton build dependencies"
uv pip install --python "$PY" -r python/requirements.txt -r python/test-requirements.txt

log "building Triton (editable, no build isolation)"
uv pip install --python "$PY" -e . --no-build-isolation

log "installing triton_kernels"
uv pip install --python "$PY" -e python/triton_kernels

export PYTHONPATH="$ROOT/python/triton_kernels${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DEPS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

log "python=$PY"
"$PY" - <<'PY'
import torch
import triton

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"triton={triton.__file__}")
PY

log "running 05-moe-bmm1-fused-gather.py hidden=$HIDDEN inter=$INTER tokens=$TOKENS EP=$EP TP=$TP"
exec "$PY" python/examples/gluon/05-moe-bmm1-fused-gather.py \
  --hidden "$HIDDEN" \
  --inter "$INTER" \
  --tokens "$TOKENS" \
  --ep "$EP" \
  --tp "$TP" \
  --num-experts "$NUM_EXPERTS" \
  --block-n "$BLOCK_N" \
  --num-ctas "$NUM_CTAS" \
  "$@"
