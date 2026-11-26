#!/usr/bin/env bash
# Universal Pascal (sm_60 / sm_61) installer for llama.cpp-p100 (p100-universal-2026).
# Works on any Linux x86_64 machine: single-CPU desktops or multi-socket servers,
# with 1 to 4+ NVIDIA Pascal GPUs (Tesla P100 sm_60, P40/P4/GTX 10-series/Titan Xp sm_61).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR=""
CUDA_ARCH=""
JOBS="$(nproc 2>/dev/null || echo 4)"
CHECK_ONLY=0
INSTALL_DEPS=0
BUILD_TESTS=0

usage() {
    cat <<'EOF'
Usage: ./install-pascal.sh [options]

Auto-detects your NVIDIA Pascal GPU(s) (sm_60 or sm_61), CUDA 12.x toolkit,
and CPU topology, then builds llama-server, llama-bench, and llama-completion
with single-architecture Pascal optimizations enabled.

Options:
  --check             Detect hardware & toolchain only (no build)
  --install-deps      Install missing build tools via apt/dnf/pacman (uses sudo)
  --arch 60|61        Override CUDA SM architecture (60=P100, 61=P40/P4/1080Ti)
  --build-dir DIR     Custom CMake build directory (default: build-p100 or build-sm61)
  --jobs N            Parallel compile jobs (default: nproc)
  --with-tests        Also build test-backend-ops for kernel verification
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)        CHECK_ONLY=1; shift ;;
        --install-deps) INSTALL_DEPS=1; shift ;;
        --arch)         CUDA_ARCH="${2:-}"; shift 2 ;;
        --build-dir)    BUILD_DIR="${2:-}"; shift 2 ;;
        --jobs)         JOBS="${2:-}"; shift 2 ;;
        --with-tests)   BUILD_TESTS=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *)              echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

echo "======================================================================"
echo "  llama.cpp-p100 (p100-universal-2026) Pascal Installer"
echo "======================================================================"

# 1. Detect CPU / RAM / NUMA topology (single-CPU or multi-socket)
SOCKETS="$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2}' || echo 1)"
[[ -z "$SOCKETS" ]] && SOCKETS=1
CORES="$(nproc 2>/dev/null || echo 4)"
RAM_GIB="$(awk '/MemTotal:/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo "?")"
NUMA_NODES="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l || echo 1)"
SIMD="SSE/Scalar"
if grep -qE '\bavx512f\b' /proc/cpuinfo 2>/dev/null; then SIMD="AVX-512"
elif grep -qE '\bavx2\b' /proc/cpuinfo 2>/dev/null; then SIMD="AVX2"
elif grep -qE '\bavx\b' /proc/cpuinfo 2>/dev/null; then SIMD="AVX1"
fi

echo "[CPU]  ${SOCKETS} socket(s), ${CORES} logical thread(s), ${NUMA_NODES} NUMA node(s), ${SIMD}, ${RAM_GIB} GiB RAM"

# 2. Detect NVIDIA Pascal GPUs
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERR]  nvidia-smi not found. Install NVIDIA driver 580.x (last branch supporting Pascal)." >&2
    exit 1
fi

GPU_LIST="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true)"
if [[ -z "$GPU_LIST" ]]; then
    echo "[ERR]  No NVIDIA GPUs detected by nvidia-smi." >&2
    exit 1
fi

GPU_COUNT=0
HAS_60=0
HAS_61=0
TOTAL_VRAM_MIB=0
echo "[GPUs]"
while IFS=',' read -r idx name cc mem; do
    idx="$(echo "$idx" | xargs)"
    name="$(echo "$name" | xargs)"
    cc="$(echo "$cc" | xargs)"
    mem_num="$(echo "$mem" | grep -oE '[0-9]+' | head -n1 || echo 0)"
    TOTAL_VRAM_MIB=$((TOTAL_VRAM_MIB + mem_num))
    GPU_COUNT=$((GPU_COUNT + 1))
    echo "       GPU ${idx}: ${name} (compute_cap ${cc}, ${mem_num} MiB)"
    [[ "$cc" == "6.0" ]] && HAS_60=1
    [[ "$cc" == "6.1" ]] && HAS_61=1
done <<< "$GPU_LIST"

echo "[VRAM] ${GPU_COUNT} GPU(s), $((TOTAL_VRAM_MIB / 1024)) GiB total VRAM"

if [[ -z "$CUDA_ARCH" ]]; then
    if [[ "$HAS_60" -eq 1 && "$HAS_61" -eq 0 ]]; then
        CUDA_ARCH="60"
    elif [[ "$HAS_61" -eq 1 && "$HAS_60" -eq 0 ]]; then
        CUDA_ARCH="61"
    elif [[ "$HAS_60" -eq 1 && "$HAS_61" -eq 1 ]]; then
        echo "[WARN] Mixed sm_60 + sm_61 GPUs detected. Defaulting to --arch 60 for P100 kernels."
        echo "       Pass --arch 61 if targeting your sm_61 card(s) specifically."
        CUDA_ARCH="60"
    else
        echo "[WARN] Non-Pascal GPU detected. Defaulting to --arch 60; pass --arch 60 or 61 explicitly."
        CUDA_ARCH="60"
    fi
fi

if [[ "$CUDA_ARCH" != "60" && "$CUDA_ARCH" != "61" ]]; then
    echo "[ERR]  --arch must be 60 (Tesla P100) or 61 (P40/P4/GTX 10-series/Titan Xp)." >&2
    exit 1
fi

if [[ -z "$BUILD_DIR" ]]; then
    if [[ "$CUDA_ARCH" == "60" ]]; then
        BUILD_DIR="${SCRIPT_DIR}/build-p100"
    else
        BUILD_DIR="${SCRIPT_DIR}/build-sm61"
    fi
fi

# 3. Locate CUDA 12.x nvcc (CUDA 13+ dropped native sm_60/sm_61 support)
NVCC_BIN=""
find_cuda12() {
    local candidates=()
    command -v nvcc >/dev/null 2>&1 && candidates+=("$(command -v nvcc)")
    for d in /usr/local/cuda-12*/bin/nvcc /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
        [[ -x "$d" ]] && candidates+=("$d")
    done
    for c in "${candidates[@]}"; do
        local ver
        ver="$("$c" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' || true)"
        if [[ "$ver" == 12.* ]]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

if NVCC_BIN="$(find_cuda12)"; then
    CUDA_VER="$("$NVCC_BIN" --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
    CUDA_ROOT="$(cd "$(dirname "$NVCC_BIN")/.." && pwd)"
    echo "[CUDA] Found CUDA ${CUDA_VER}: ${NVCC_BIN}"
else
    echo "[ERR]  CUDA 12.x (nvcc) not found!" >&2
    echo "       Pascal (sm_60/sm_61) requires CUDA Toolkit 12.x (e.g., 12.4–12.9) and NVIDIA driver 580.x." >&2
    echo "       CUDA 13.x removed native sm_60/sm_61 SASS compilation." >&2
    exit 1
fi

# 4. Optional build dependency installation
if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    echo "[DEPS] Installing build dependencies..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build git libcurl4-openssl-dev
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y gcc gcc-c++ cmake ninja-build git libcurl-devel
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --needed base-devel cmake ninja git curl
    else
        echo "[WARN] Unsupported package manager; please install cmake, g++, ninja (or make), git, and libcurl dev headers."
    fi
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "[ERR]  cmake not found. Re-run with --install-deps or install cmake >= 3.23." >&2
    exit 1
fi

GEN_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GEN_ARGS=(-G Ninja)
    echo "[GEN]  Ninja"
else
    echo "[GEN]  Unix Makefiles (install ninja-build for faster builds)"
fi

echo "[PLAN] Arch: sm_${CUDA_ARCH} (single-arch, unlocks Pascal fast paths)"
echo "       Build dir: ${BUILD_DIR}"
echo "       Jobs: ${JOBS}"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "[OK]   Hardware & toolchain check passed (--check mode; skipping build)."
    exit 0
fi

# 5. Configure & build (single-arch sm_60 or sm_61 ONLY!)
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"

TEST_FLAG="OFF"
TARGETS=(llama-server llama-bench llama-completion)
if [[ "$BUILD_TESTS" -eq 1 ]]; then
    TEST_FLAG="ON"
    TARGETS+=(test-backend-ops)
fi

cmake "${GEN_ARGS[@]}" \
    -S "${SCRIPT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${NVCC_BIN}" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DGGML_CUDA_P100_VMAD=OFF \
    -DLLAMA_BUILD_TESTS="${TEST_FLAG}" \
    -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build "${BUILD_DIR}" --target "${TARGETS[@]}" -j"${JOBS}"

echo ""
echo "======================================================================"
echo "  Build Complete!"
echo "======================================================================"
echo "  Binaries: ${BUILD_DIR}/bin/{llama-server,llama-bench,llama-completion}"
echo ""
echo "  Quick start (auto-tunes for your ${GPU_COUNT} Pascal GPU(s)):"
echo "    ./scripts/run-pascal.sh -m /path/to/model.gguf"
echo "    ./scripts/run-pascal.sh -m /path/to/model.gguf -d /path/to/draft.gguf"
echo "======================================================================"
