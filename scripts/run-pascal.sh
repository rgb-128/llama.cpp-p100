#!/usr/bin/env bash
# Hardware-adaptive launcher for llama.cpp-p100 (p100-universal-2026).
# Automatically selects optimal multi-GPU split mode, CUDA Graph policy,
# and speculative decoding defaults for 1, 2, 3, or 4+ Pascal GPUs on any PC.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=""
DRAFT=""
GPUS="auto"
SPLIT="auto"
SPEC_TYPE="auto"
DRAFT_MAX=""
DRAFT_P_MIN=""
CTX="16384"
PORT="8080"
HOST="127.0.0.1"
BUILD_DIR=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./scripts/run-pascal.sh -m /path/to/model.gguf [options] [-- extra llama-server args]

Options:
  -m, --model PATH       Main GGUF model path (required)
  -d, --draft PATH       Optional MTP / DFlash / DSpark drafter GGUF
  --gpus N|0,1,...|auto  Number of GPUs (1..4) or explicit CUDA indices (default: auto)
  --split auto|tensor|layer
                         Multi-GPU split mode (default: auto -> tensor on 2/4 GPUs, layer on 3)
  --spec-type auto|mtp|dflash|dflash2|dspark|eagle3
                         Drafter type (default: auto-detect from draft filename)
  --draft-max N          Speculative draft width (default: 4 for MTP, 3 for DFlash/DSpark)
  --draft-p-min F        Min draft probability (optional, e.g. 0.1 for MTP)
  -c, --ctx N            Context length (default: 16384)
  --port N               HTTP listen port (default: 8080)
  --host ADDR            HTTP bind address (default: 127.0.0.1)
  --build-dir DIR        Build directory (auto-detects build-p100 or build-sm61)
  --dry-run              Print the resolved command without running it
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)      MODEL="${2:-}"; shift 2 ;;
        -d|--draft)      DRAFT="${2:-}"; shift 2 ;;
        --gpus)          GPUS="${2:-}"; shift 2 ;;
        --split)         SPLIT="${2:-}"; shift 2 ;;
        --spec-type)     SPEC_TYPE="${2:-}"; shift 2 ;;
        --draft-max)     DRAFT_MAX="${2:-}"; shift 2 ;;
        --draft-p-min)   DRAFT_P_MIN="${2:-}"; shift 2 ;;
        -c|--ctx)        CTX="${2:-}"; shift 2 ;;
        --port)          PORT="${2:-}"; shift 2 ;;
        --host)          HOST="${2:-}"; shift 2 ;;
        --build-dir)     BUILD_DIR="${2:-}"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; EXTRA_ARGS+=("$@"); break ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model /path/to/model.gguf is required." >&2
    usage
    exit 1
fi

# Locate binary
if [[ -z "$BUILD_DIR" ]]; then
    if [[ -x "${SCRIPT_DIR}/build-p100/bin/llama-server" ]]; then
        BUILD_DIR="${SCRIPT_DIR}/build-p100"
    elif [[ -x "${SCRIPT_DIR}/build-sm61/bin/llama-server" ]]; then
        BUILD_DIR="${SCRIPT_DIR}/build-sm61"
    else
        echo "Error: llama-server not found. Run ./install-pascal.sh first." >&2
        exit 1
    fi
fi
SERVER_BIN="${BUILD_DIR}/bin/llama-server"

# Detect available GPUs
AVAIL_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)"
[[ "$AVAIL_GPUS" -lt 1 ]] && AVAIL_GPUS=1

CUDA_DEVS=""
NUM_GPUS=1
if [[ "$GPUS" == "auto" ]]; then
    NUM_GPUS="$AVAIL_GPUS"
    if [[ "$NUM_GPUS" -gt 4 ]]; then NUM_GPUS=4; fi
    CUDA_DEVS="$(seq -s, 0 $((NUM_GPUS - 1)))"
elif [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    NUM_GPUS="$GPUS"
    CUDA_DEVS="$(seq -s, 0 $((NUM_GPUS - 1)))"
else
    CUDA_DEVS="$GPUS"
    NUM_GPUS="$(echo "$CUDA_DEVS" | tr ',' '\n' | wc -l)"
fi

# Choose split mode safely for 1, 2, 3, or 4 GPUs
SPLIT_AUTO=0
SPLIT_ARGS=()
if [[ "$NUM_GPUS" -gt 1 ]]; then
    if [[ "$SPLIT" == "auto" ]]; then
        SPLIT_AUTO=1
        # Pascal zero-sync AllReduce accelerates 2-GPU and 4-GPU tensor parallelism.
        # 3-GPU rigs use layer split. If a model architecture does not implement tensor split,
        # auto mode falls back to layer split automatically below.
        if [[ "$NUM_GPUS" -eq 2 || "$NUM_GPUS" -eq 4 ]]; then
            SPLIT="tensor"
        else
            SPLIT="layer"
        fi
    fi
    SPLIT_ARGS=(-sm "$SPLIT")
fi

# On multi-socket hosts, if all selected GPUs share one NUMA node, bind CPU and memory
# to that node so host-mapped AllReduce traffic never crosses inter-socket links.
NUMA_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    NUMA_NODES="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l || echo 1)"
    if [[ "$NUMA_NODES" -gt 1 ]]; then
        GPU_NUMA=""
        SAME_NUMA=1
        IFS=',' read -ra DEV_ARR <<< "$CUDA_DEVS"
        for dev in "${DEV_ARR[@]}"; do
            bdf="$(nvidia-smi --id="$dev" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | tr '[:upper:]' '[:lower:]' | sed 's/^ *//; s/^00000000:/0000:/')"
            node="$(cat "/sys/bus/pci/devices/${bdf}/numa_node" 2>/dev/null || echo -1)"
            if [[ "$node" -lt 0 ]]; then
                SAME_NUMA=0
                break
            fi
            if [[ -z "$GPU_NUMA" ]]; then
                GPU_NUMA="$node"
            elif [[ "$GPU_NUMA" != "$node" ]]; then
                SAME_NUMA=0
                break
            fi
        done
        if [[ "$SAME_NUMA" -eq 1 && -n "$GPU_NUMA" ]]; then
            NUMA_PREFIX=(numactl "--cpunodebind=${GPU_NUMA}" "--membind=${GPU_NUMA}")
        fi
    fi
fi

# Choose CUDA Graph policy & speculative decoding args
# Mode 3 (single-token graphs) is the Pascal default in llama.cpp-p100 and works for dense, MoE, and MTP.
GRAPHS_VAL="3"
SPEC_ARGS=()
if [[ -n "$DRAFT" ]]; then
    if [[ "$SPEC_TYPE" == "auto" ]]; then
        if [[ "$DRAFT" =~ [Dd][Ss]park ]]; then
            SPEC_TYPE="dspark"
        elif [[ "$DRAFT" =~ [Dd][Ff]lash2 ]]; then
            SPEC_TYPE="dflash2"
        elif [[ "$DRAFT" =~ [Dd][Ff]lash ]]; then
            SPEC_TYPE="dflash"
        elif [[ "$DRAFT" =~ [Ee]agle3 ]]; then
            SPEC_TYPE="eagle3"
        else
            SPEC_TYPE="mtp"
        fi
    fi
    SPEC_ARGS+=(-md "$DRAFT" -ngld 99)
    case "$SPEC_TYPE" in
        dspark)
            SPEC_ARGS+=(--spec-type draft-dspark --spec-draft-n-max "${DRAFT_MAX:-3}")
            ;;
        dflash2)
            SPEC_ARGS+=(--spec-type draft-dflash2 --spec-draft-n-max "${DRAFT_MAX:-3}")
            ;;
        dflash)
            SPEC_ARGS+=(--spec-type draft-dflash --spec-draft-n-max "${DRAFT_MAX:-3}")
            ;;
        eagle3)
            SPEC_ARGS+=(--spec-type draft-eagle3 --spec-draft-n-max "${DRAFT_MAX:-4}")
            ;;
        *)
            SPEC_ARGS+=(--spec-type draft-mtp --spec-draft-n-max "${DRAFT_MAX:-4}")
            [[ -n "$DRAFT_P_MIN" ]] && SPEC_ARGS+=(--spec-draft-p-min "$DRAFT_P_MIN")
            ;;
    esac
fi

build_cmd() {
    local split_mode="$1"
    local s_args=()
    if [[ "$NUM_GPUS" -gt 1 ]]; then
        s_args=(-sm "$split_mode")
    fi
    CMD=(
        "${NUMA_PREFIX[@]}"
        env
        "CUDA_VISIBLE_DEVICES=${CUDA_DEVS}"
        "GGML_CUDA_GRAPHS_PRE_VOLTA=${GRAPHS_VAL}"
        "${SERVER_BIN}"
        -m "${MODEL}"
        -ngl 99
        -fa on
        -c "${CTX}"
        --host "${HOST}"
        --port "${PORT}"
        "${s_args[@]}"
        "${SPEC_ARGS[@]}"
        "${EXTRA_ARGS[@]}"
    )
}

build_cmd "$SPLIT"
NUMA_DESC="default"
[[ ${#NUMA_PREFIX[@]} -gt 0 ]] && NUMA_DESC="node${GPU_NUMA}"
echo "[pascal] GPUs=${CUDA_DEVS} (${NUM_GPUS}) | split=${SPLIT} | numa=${NUMA_DESC} | graphs=${GRAPHS_VAL} | ctx=${CTX} | port=${PORT}"
if [[ "$DRY_RUN" -eq 1 ]]; then
    printf ' %q' "${CMD[@]}"
    echo ""
    exit 0
fi

if [[ "$SPLIT_AUTO" -eq 1 && "$SPLIT" == "tensor" ]]; then
    ERR_LOG="$(mktemp)"
    trap 'rm -f "$ERR_LOG"' EXIT
    set +e
    "${CMD[@]}" 2> >(awk -v f="$ERR_LOG" 'NR<=60 { print >> f } NR==60 { close(f) } { print > "/dev/stderr"; fflush("/dev/stderr") }')
    RC=$?
    set -e
    if [[ "$RC" -ne 0 ]] && grep -qE "LLAMA_SPLIT_MODE_TENSOR (not implemented|is not supported)" "$ERR_LOG"; then
        rm -f "$ERR_LOG"
        trap - EXIT
        echo "[pascal] Architecture does not support tensor split; retrying with -sm layer..." >&2
        build_cmd "layer"
        exec "${CMD[@]}"
    fi
    exit "$RC"
fi

exec "${CMD[@]}"
