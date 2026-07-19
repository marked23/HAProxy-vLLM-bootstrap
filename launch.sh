#!/usr/bin/env bash
################################################################################
# HAProxy-vLLM Configuration Override and Launcher
#
# This script provides a clean way to configure haproxllm.sh without modifying
# the main installation script. Simply uncomment and modify the variables you
# want to change, then run this script instead of the main one.
#
# Usage:
#   sudo bash launch.sh
#
# CURRENT WORKING CONFIG (tested on Ubuntu 25.10 with 6x RDNA 3 GPUs):
#   - Model: mistralai/Mistral-7B-Instruct-v0.2 (7B, ungated)
#   - GPUs: 0,1,2,3 (using 4 of 6 GPUs)
#   - Tensor Parallelism: TP=2 per backend (2 backends total)
#   - Container Runtime: Podman with podman-compose
#   - Endpoints: http://wide.local:8000/v1 (HAProxy), http://wide.local:8404/stats
################################################################################

# ============================================================================
# Stack Directory and Model Configuration
# ============================================================================

# Base directory for the entire LLM stack (models, configs, cache)
export STACK_DIR="/opt/llm-stack"

# Model ID from Hugging Face Hub (format: organization/model-name)
# Examples:
#   - mistralai/Mistral-7B-Instruct-v0.2 (7B, good for RDNA 3, ungated)
#   - meta-llama/Llama-3.1-8B-Instruct (8B, GATED - requires HF auth)
#   - eousphoros/persona_epsilon_20b_mxfp4 (20B, MXFP4 not compatible with RDNA 3)
#   - openai/gpt-oss-120b (120B, requires high VRAM)
#   - openai/gpt-oss-20b (20B, good for Radeon AI PRO R9700)
export MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"

# Local directory for model weights (auto-derived from MODEL_ID if not set)
# The script converts "/" to "_" in the MODEL_ID to create the directory name
# export MODEL_DIR="$STACK_DIR/models/${MODEL_ID//\//_}"

# ============================================================================
# GPU Configuration
# ============================================================================

# GPU type for automatic image selection
# Valid values: r9700, mi300, mi355, auto
#   - r9700: Radeon AI PRO R9700 (RDNA 4) → rocm/vllm:rocm7.13.0_gfx120X-all_*
#   - mi300: MI300X/MI325 (CDNA 3) → rocm/vllm:rocm7.13.0_gfx94X-dcgpu_*
#   - mi355: MI350X (CDNA 3) → rocm/vllm:rocm7.13.0_gfx950-dcgpu_*
#   - auto: Auto-detect GPU type (default); RDNA 3 → rocm7.13.0_gfx110X-all_*
export GPU_TYPE="auto"

# vLLM Docker image (leave empty for auto-detection based on GPU_TYPE)
# Manual override examples:
#   - rocm/vllm:latest
#   - rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210
#   - rocm/vllm:rocm7.13.0_gfx110X-all_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1
# export VLLM_IMAGE=""

# GPU isolation method: "env" or "devices"
#   - "env": Use HIP_VISIBLE_DEVICES environment variable (simpler)
#   - "devices": Mount only specific /dev/dri/renderD* devices (more reliable)
# Switch to "devices" if containers don't see GPUs correctly with "env"
export GPU_ISOLATION="devices"

# ============================================================================
# GPU Assignment per vLLM Backend
# ============================================================================

# GPU indices for first backend (comma-separated, no spaces)
# For 6 GPUs with Mistral-7B: Using 2 GPUs per backend (TP=2)
# Mistral has 32 attention heads which must divide evenly by TP
# Options for 6 GPUs:
#   - "0,1" / "2,3" : 2 backends × TP=2 (WORKING CONFIG - uses 4 GPUs)
#   - "0,1,2,3" : 1 backend × TP=4 (max context, uses 4 GPUs)
#   - "0" / "1" / "2" / "3" / "4" / "5" : 6 backends × TP=1 (max throughput, uses all 6)
# Note: TP=3 doesn't work with Mistral (32 heads not divisible by 3)
export VLLM1_GPUS="0,1"

# GPU indices for second backend
export VLLM2_GPUS="2,3"

# NUMA node binding for multi-socket systems (leave empty for auto)
# export VLLM1_NUMA_NODE=""
# export VLLM2_NUMA_NODE=""

# ============================================================================
# Network Ports
# ============================================================================

# HAProxy load balancer port (clients connect here)
export LB_PORT="8000"
# Public host name for accessing the load balancer (defaults to wide.local)
export LB_HOST="wide.local"

# vLLM backend ports (internal, load-balanced by HAProxy)
export VLLM1_PORT="8001"
export VLLM2_PORT="8002"

# ============================================================================
# vLLM Performance Tuning
# ============================================================================

# GPU memory utilization (0.0 to 1.0)
# 0.90 = 90% of available VRAM for KV cache and model weights
# Lower this if you encounter OOM errors
export GPU_MEM_UTIL="0.90"

# Maximum context length (in tokens)
# Examples:
#   - 8192: Safe default for most models
#   - 16384: 16K context (requires more VRAM)
#   - 32768: 32K context (MI300X+ with high-memory models)
# Must be ≤ model's maximum position embeddings
export MAX_MODEL_LEN="8192"

# Maximum number of sequences processed in parallel
# Higher = more throughput, but requires more VRAM
# Conservative default: 128
# High-VRAM systems (MI300X): Try 256-512
export MAX_NUM_SEQS="128"

# Maximum number of batched tokens (leave empty for auto)
# Affects latency/throughput trade-off
# export MAX_NUM_BATCHED_TOKENS=""

# ============================================================================
# vLLM Features
# ============================================================================

# Prefix caching: Cache prompt prefixes for 20-50% speedup on repeated prompts
# 1 = enabled, 0 = disabled
export ENABLE_PREFIX_CACHING="1"

# Chunked prefill: Better time-to-first-token for long prompts
# 1 = enabled, 0 = disabled
export ENABLE_CHUNKED_PREFILL="1"

# AITER (AMD Inference Tensor Engine for ROCm): Optimized attention kernels
# 1 = enabled (recommended for R9700/MI300X/MI350X), 0 = disabled
export ENABLE_AITER="1"

# Data type for model weights
# Valid values: auto, float16, bfloat16, float32
# "auto" detects best dtype; mxfp4 models typically require bfloat16
export DTYPE="auto"

# KV cache data type (MI300X supports fp8_e4m3 for memory savings)
# Valid values: auto, fp8_e4m3 (MI300X only), float16
export KV_CACHE_DTYPE="auto"

# ============================================================================
# Chat Template Configuration
# ============================================================================

# Path to Jinja2 chat template file
# Options:
#   - "chat_template.jinja": Look for template in model directory (default)
#   - "": Use model's built-in template from tokenizer_config.json
#   - "/path/to/custom_template.jinja": Use custom template (absolute path)
# Required for models with reasoning tokens (e.g., gpt-oss)
# Llama models have built-in templates, so using empty string
export CHAT_TEMPLATE=""

# ============================================================================
# TLS/HTTPS Configuration
# ============================================================================

# Enable TLS with automatic certificate generation
# 1 = enabled (HTTPS with HTTP/2), 0 = disabled (HTTP only)
export ENABLE_TLS="0"

# Certificate validity period (in days)
export TLS_CERT_DAYS="3650"

# Certificate Authority Common Name
export TLS_CA_CN="LLM-Stack-CA"

# Server certificate Common Name
export TLS_SERVER_CN="llm-server"

# Additional Subject Alternative Names (comma-separated)
# Examples: "DNS:api.example.com,IP:10.0.0.1"
# export TLS_SERVER_SAN=""

# ============================================================================
# Hugging Face Configuration
# ============================================================================

# Hugging Face API token (optional; some models require authentication)
# Get your token from: https://huggingface.co/settings/tokens
# Preserves existing HF_TOKEN from environment, or set explicitly: export HF_TOKEN="hf_..."
export HF_TOKEN="${HF_TOKEN:-}"

# ============================================================================
# ROCm and GPU Validation
# ============================================================================

# Minimum ROCm version (for compatibility checks)
export MIN_ROCM_VERSION="6.0"

# Expected number of GPUs (0 = skip validation)
# Set to your actual GPU count to enable validation warnings
# System has 6 RDNA 3 GPUs
export EXPECTED_GPU_COUNT="6"

# ROCm version to install (if not already installed)
export ROCM_VERSION="7.2.4"

# AMDGPU kernel driver repo version paired with ROCM_VERSION
# (since ROCm 7.x the driver is versioned separately; 30.30.4 pairs with 7.2.4)
export AMDGPU_DRIVER_VERSION="30.30.4"

# ============================================================================
# Launch the main installation script
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/haproxllm.sh"

if [[ ! -f "$MAIN_SCRIPT" ]]; then
  echo "ERROR: Main script not found: $MAIN_SCRIPT" >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: This script must be run as root (use: sudo bash $0)" >&2
  exit 1
fi

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║  Launching HAProxy-vLLM Installation with Custom Configuration               ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  STACK_DIR:    $STACK_DIR"
echo "  MODEL_ID:     $MODEL_ID"
echo "  GPU_TYPE:     $GPU_TYPE"
echo "  VLLM1_GPUS:   $VLLM1_GPUS (backend 1)"
echo "  VLLM2_GPUS:   $VLLM2_GPUS (backend 2)"
echo "  LB_PORT:      $LB_PORT"
echo "  ENABLE_TLS:   $ENABLE_TLS"
echo ""

# Execute the main script with all exported variables
exec bash "$MAIN_SCRIPT"
