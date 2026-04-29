#!/usr/bin/env bash
# Shared functions for DeepSeek-V4-Flash UltraChat response regeneration.
# Source this file; do not execute directly.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts()      { date '+%Y-%m-%d %H:%M:%S'; }
info()     { echo "[$(_ts)] [INFO]  $*"; }
warn()     { echo "[$(_ts)] [WARN]  $*"; }
error()    { echo "[$(_ts)] [ERROR] $*" >&2; }

banner() {
    local title="$1" width=70
    local line
    line=$(printf '=%.0s' $(seq 1 $width))
    echo ""
    echo "[$(_ts)] ${line}"
    echo "[$(_ts)]   ${title}"
    echo "[$(_ts)] ${line}"
}

sub_banner() {
    local title="$1" width=70
    local line
    line=$(printf -- '-%.0s' $(seq 1 $width))
    echo ""
    echo "[$(_ts)] ${line}"
    echo "[$(_ts)]   ${title}"
    echo "[$(_ts)] ${line}"
}

elapsed() {
    local secs=$(( $(date +%s) - SCRIPT_START ))
    printf "%dh %02dm %02ds" $(( secs / 3600 )) $(( (secs % 3600) / 60 )) $(( secs % 60 ))
}

elapsed_since() {
    local start="$1"
    local secs=$(( $(date +%s) - start ))
    printf "%dh %02dm %02ds" $(( secs / 3600 )) $(( (secs % 3600) / 60 )) $(( secs % 60 ))
}

# ---------------------------------------------------------------------------
# Deadline
# ---------------------------------------------------------------------------
past_deadline() {
    [[ $(date +%s) -ge ${SCRIPT_DEADLINE} ]]
}

# ---------------------------------------------------------------------------
# GPU management
# ---------------------------------------------------------------------------
wait_for_gpus() {
    local needed="$1"
    info "Waiting for ${needed} AVAILABLE GPU(s) (polling every 5s)..."
    while ! past_deadline; do
        local available
        available=$(chg status 2>&1 \
            | sed 's/\x1b\[[0-9;]*[mGKHF]//g' \
            | { grep -E '^\s*[0-9]+\s' || true; } \
            | { grep 'AVAILABLE' || true; } \
            | wc -l)
        if [[ $available -ge $needed ]]; then
            info "GPU check passed: ${available} AVAILABLE GPU(s) found."
            return 0
        fi
        info "Only ${available}/${needed} GPU(s) AVAILABLE -- next check in 5s..."
        sleep 5
    done
    error "Deadline reached while waiting for GPUs."
    return 1
}

reserve_gpus() {
    local needed="$1"
    local duration="${2:-2h}"
    info "Reserving ${needed} GPU(s) for ${duration}..."
    GPU_IDS=$(chg reserve --gpus "${needed}" --duration "${duration}" --short --note "deepseekv4 ultrachat regen" 2>&1)
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        error "chg reserve failed (exit ${rc}): ${GPU_IDS}"
        GPU_IDS=""
        return 1
    fi
    RESERVATION_EXPIRY=$(( $(date +%s) + $(parse_duration "${duration}") ))
    info "GPUs reserved: ${GPU_IDS} (expires in ${duration}, ~$(date -d "@${RESERVATION_EXPIRY}" '+%H:%M:%S' 2>/dev/null || echo "${duration}"))"
    export GPU_IDS
    return 0
}

parse_duration() {
    local dur="$1"
    local num unit secs
    num=$(echo "${dur}" | sed 's/[^0-9.]//g')
    unit=$(echo "${dur}" | sed 's/[0-9.]//g')
    case "${unit}" in
        s)  secs=$(printf '%.0f' "${num}") ;;
        m)  secs=$(printf '%.0f' "$(echo "${num} * 60" | bc)") ;;
        h)  secs=$(printf '%.0f' "$(echo "${num} * 3600" | bc)") ;;
        d)  secs=$(printf '%.0f' "$(echo "${num} * 86400" | bc)") ;;
        *)  secs=$(printf '%.0f' "${num}") ;;
    esac
    echo "${secs}"
}

renew_reservation() {
    local needed="$1"
    local duration="${2:-2h}"
    local now
    now=$(date +%s)
    local remaining=$(( RESERVATION_EXPIRY - now ))

    if [[ $remaining -gt 1800 ]]; then
        return 0
    fi

    warn "Reservation expires in $(( remaining / 60 ))m -- renewing..."

    local saved_gpu_ids="${GPU_IDS}"
    chg release --gpu-ids "${saved_gpu_ids}" 2>/dev/null || true

    if chg reserve --gpu-ids "${saved_gpu_ids}" --duration "${duration}" --short --nonblock >/dev/null 2>&1; then
        RESERVATION_EXPIRY=$(( $(date +%s) + $(parse_duration "${duration}") ))
        info "Reservation renewed on GPUs ${saved_gpu_ids} for ${duration}."
        return 0
    fi

    warn "Could not re-reserve GPUs ${saved_gpu_ids} -- attempting fresh reservation..."
    GPU_IDS=""

    if ! wait_for_gpus "${needed}"; then
        return 1
    fi
    if ! reserve_gpus "${needed}" "${duration}"; then
        return 1
    fi
    info "Fresh reservation acquired: GPUs ${GPU_IDS}"
    return 0
}

release_gpus() {
    if [[ -n "${GPU_IDS:-}" ]]; then
        info "Releasing GPUs: ${GPU_IDS}..."
        chg release --gpu-ids "${GPU_IDS}" 2>/dev/null || true
        GPU_IDS=""
    fi
}

# ---------------------------------------------------------------------------
# Port utilities
# ---------------------------------------------------------------------------
port_in_use() {
    local p="$1"
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${p} "
    else
        lsof -i ":${p}" &>/dev/null
    fi
}

find_free_port() {
    local p="${1:-8000}"
    while port_in_use "${p}"; do
        p=$(( p + 1 ))
    done
    echo "${p}"
}

# ---------------------------------------------------------------------------
# Server management
# SERVE_MODE (set by run.sh): "native" or "docker"
# ---------------------------------------------------------------------------
SERVER_PID=""

wait_for_server() {
    local endpoint="$1"
    local timeout="${2:-3600}"
    local start_ts
    start_ts=$(date +%s)
    local per_start_deadline=$(( start_ts + timeout ))
    local effective_deadline=$(( per_start_deadline < SCRIPT_DEADLINE ? per_start_deadline : SCRIPT_DEADLINE ))

    info "Waiting for server at ${endpoint}/health (timeout: ${timeout}s)..."
    while ! curl -sf "${endpoint}/health" > /dev/null 2>&1; do
        if [[ $(date +%s) -ge $effective_deadline ]]; then
            if past_deadline; then
                error "Global deadline reached while waiting for server."
            else
                error "Server did not become ready within ${timeout}s."
            fi
            return 1
        fi
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            error "Server process (PID ${SERVER_PID}) died."
            return 1
        fi
        sleep 30
    done
    info "Server is ready! (startup took $(elapsed_since "${start_ts}"))"
    return 0
}

stop_server() {
    local port="${PORT:-8000}"

    if [[ "${SERVE_MODE:-native}" == "docker" ]]; then
        local name="vllm-deepseek-${port}"
        info "Stopping Docker server (container: ${name})..."

        # Kill the bash wrapper (serve_docker.sh) so the pipe exits cleanly
        if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill "${SERVER_PID}" 2>/dev/null || true
            wait "${SERVER_PID}" 2>/dev/null || true
        fi

        # docker stop: sends SIGTERM to PID 1 in the container, then SIGKILL after 10s
        if docker ps --format "{{.Names}}" 2>/dev/null | grep -q "^${name}$"; then
            info "Running docker stop ${name}..."
            docker stop "${name}" > /dev/null 2>&1 || true
        fi

        rm -f "/tmp/vllm_container_${port}.lock"
    else
        info "Stopping vLLM server (PID: ${SERVER_PID:-none})..."

        if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
            # Graceful: SIGTERM first
            kill -TERM "${SERVER_PID}" 2>/dev/null || true

            # Wait up to 30s for the process to exit on its own
            local deadline=$(( $(date +%s) + 30 ))
            while kill -0 "${SERVER_PID}" 2>/dev/null && [[ $(date +%s) -lt $deadline ]]; do
                sleep 1
            done

            # Force kill if still alive
            if kill -0 "${SERVER_PID}" 2>/dev/null; then
                warn "Still alive after 30s — sending SIGKILL..."
                kill -KILL "${SERVER_PID}" 2>/dev/null || true
                wait "${SERVER_PID}" 2>/dev/null || true
            fi
        fi
    fi

    SERVER_PID=""
    sleep 2
    info "Server stopped."
}

exit_after_cleanup() {
    stop_server 2>/dev/null || true
    release_gpus
    info "Exiting (total elapsed: $(elapsed))."
    exit 1
}
