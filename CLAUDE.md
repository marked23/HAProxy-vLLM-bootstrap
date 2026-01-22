# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **single-command bootstrap script** for deploying a production-ready, highly-available LLM inference stack on AMD ROCm GPUs. The system provides an OpenAI-compatible API endpoint with automatic failover, load balancing, and comprehensive system tuning.

**Key Architecture:**
- Multiple vLLM backend instances (each bound to a subset of GPUs via tensor parallelism)
- HAProxy load balancer in front providing failover, health checking, and sticky sessions
- Containerized deployment via Docker Compose with host networking
- Comprehensive system tuning (kernel, GPU, networking, CPU governor)

The design goal is **redundancy**: clients connect to a single HAProxy endpoint, and individual backend or GPU failures don't cause downtime.

## System Requirements and Compatibility

### Container Runtime: Docker vs Podman

**CRITICAL**: This script requires a **compose tool** to be installed. The script uses `docker compose` command, which requires either:

1. **Docker with Compose V2** (recommended by script author):
   - Real Docker Engine from `get.docker.com`
   - Built-in `docker compose` subcommand

2. **Podman with podman-compose** (common on modern Ubuntu):
   - Podman acts as Docker replacement
   - Ubuntu 25.10+ often has `podman-docker` package that creates `/usr/bin/docker` as a Podman wrapper
   - **REQUIRES** `podman-compose` to be installed separately: `sudo apt-get install podman-compose`
   - Script will fail with "looking up compose provider failed" error if compose tool is missing

**How to detect your setup:**
```bash
# Check if docker is really Podman
ls -la /usr/bin/docker  # Small file (~228 bytes) = Podman wrapper

# Check your container runtime
docker --version  # Shows "podman version" if using Podman

# Verify compose tool availability
docker compose version    # Docker Compose V2
podman-compose --version  # Podman Compose
```

**If script fails with compose errors:**
```bash
# For Podman users (Ubuntu 25.10+):
sudo apt-get install -y podman-compose

# For Docker users:
sudo apt-get install -y docker-compose-v2

# Or install real Docker (removes Podman):
sudo apt-get remove -y podman-docker
curl -fsSL https://get.docker.com | sudo sh
```

### OS Version Compatibility

- **Designed for**: Ubuntu 24.04 LTS
- **Tested on**: Ubuntu 25.10 (Questing Quokka) with Podman
- **Warning shown**: Script detects non-24.04 and shows warning but continues

### GPU Configuration Notes

The default configuration assumes **4 GPUs**:
- `EXPECTED_GPU_COUNT=4` in `launch.sh`
- `VLLM1_GPUS="0,1"` and `VLLM2_GPUS="2,3"` (2 backends × TP=2)

**If you have a different GPU count**, update `launch.sh`:
```bash
# For 6 GPUs (example):
export EXPECTED_GPU_COUNT="6"
export VLLM1_GPUS="0,1,2"    # Backend 1: GPUs 0,1,2 with TP=3
export VLLM2_GPUS="3,4,5"    # Backend 2: GPUs 3,4,5 with TP=3

# Or for 2 GPUs:
export EXPECTED_GPU_COUNT="2"
export VLLM1_GPUS="0"        # Backend 1: GPU 0 with TP=1
export VLLM2_GPUS="1"        # Backend 2: GPU 1 with TP=1
```

## Core Scripts

### haproxllm.sh (Main Installation Script)
The primary installation and deployment script (~1500 lines). This is a **complete system bootstrap** that:
- Installs ROCm 6.3 and AMDGPU drivers (if not present)
- Configures Docker for ROCm GPU access
- Downloads HuggingFace models
- Auto-detects GPU type and selects optimized vLLM container images
- Generates HAProxy configuration with health checks and sticky sessions
- Creates Docker Compose stack with two vLLM backends
- Applies production system tuning (sysctl, ulimits, GPU power profiles, CPU governor)
- Optionally generates TLS certificates for HTTPS with HTTP/2

**Important:** This script is designed to be **run once** to set up the stack. It is **idempotent** for most operations (skips installation if already present).

### launch.sh (Configuration Wrapper)
A user-friendly configuration wrapper that sets environment variables and then executes `haproxllm.sh`. Users should modify this file to customize their deployment instead of editing the main script.

## Running the Stack

### Initial Setup
```bash
# Configure your settings in launch.sh, then run:
sudo bash launch.sh

# Or run directly with environment overrides:
sudo MODEL_ID="openai/gpt-oss-20b" GPU_TYPE="r9700" bash haproxllm.sh
```

### Managing the Stack
```bash
# Note: Use 'podman-compose' if using Podman, or 'docker compose' for Docker

# View logs
docker compose -f /opt/llm-stack/compose/docker-compose.yml logs -f
# OR for Podman:
podman-compose -f /opt/llm-stack/compose/docker-compose.yml logs -f

# Restart services
docker compose -f /opt/llm-stack/compose/docker-compose.yml restart

# Stop services
docker compose -f /opt/llm-stack/compose/docker-compose.yml down

# Start services
docker compose -f /opt/llm-stack/compose/docker-compose.yml up -d

# Check GPU status
rocm-smi
rocm-smi --showmeminfo vram
watch -n 1 rocm-smi  # Continuous monitoring

# View HAProxy stats
curl http://wide.local:8404/stats

# Check individual backend health
curl http://wide.local:8001/health  # vllm1
curl http://wide.local:8002/health  # vllm2
```

### Testing the API
```bash
# Check available models
curl -s http://wide.local:8000/v1/models | jq .

# Health check
curl -s http://wide.local:8000/health

# Test inference
curl -s http://wide.local:8000/v1/completions \

  -H "Content-Type: application/json" \
  -d '{"model": "your-model-id", "prompt": "Hello", "max_tokens": 50}'
```

## Architecture Details

### GPU Isolation Modes
The script supports two methods for GPU isolation (`GPU_ISOLATION` variable):

1. **env mode** (default): Mount all `/dev/dri` devices and use `HIP_VISIBLE_DEVICES` environment variable to filter GPUs
   - Simpler configuration
   - May not work reliably on all GPU types

2. **devices mode**: Mount only specific `/dev/dri/renderD*` devices for each container
   - More deterministic and reliable
   - Recommended for RDNA 4 (R9700) and complex configurations
   - Uses `gpu_indices_to_render_devices()` helper to convert GPU indices to device paths

### GPU-Specific Optimizations

The script auto-detects GPU architecture (ISA) and selects optimized container images:

- **RDNA 4** (gfx1201 - R9700): `rocm/vllm-dev:open-r9700-08052025`
- **CDNA 3** (gfx942 - MI300X/MI325): `rocm/vllm-dev:open-mi300-08052025`
- **CDNA 3** (gfx950 - MI350X): `rocm/vllm-dev:open-mi355-08052025`
- **CDNA 2** (gfx90a - MI200 series): `rocm/vllm:latest`
- **Others**: `rocm/vllm:latest`

These GPU-specific images include AITER (AMD Inference Tensor Engine) optimized attention kernels for maximum performance.

### HAProxy Load Balancing Strategy

The HAProxy configuration uses:
- **leastconn** balancing: Routes to backend with fewest active connections (better for variable inference times)
- **Sticky sessions** via cookie: Maximizes prefix cache hit rates by routing repeat requests to same backend
- **Health checks**: `/health` endpoint, mark down after 3 failures, mark up after 2 successes
- **Failover**: 2 retries with redispatch, fails over to healthy backend if sticky backend is down
- **Slowstart**: Ramps up traffic over 180s when backend comes online
- **Long timeouts**: 1-hour client/server timeout for streaming inference

### Tensor Parallelism Configuration

Default setup assumes 4 GPUs split into 2 backends with TP=2 each:
- `VLLM1_GPUS="0,1"` → backend 1 uses GPUs 0,1 with TP=2
- `VLLM2_GPUS="2,3"` → backend 2 uses GPUs 2,3 with TP=2

Alternative configurations:
- **Max throughput**: 4 backends × TP=1 (`VLLM1_GPUS="0"`, `VLLM2_GPUS="1"`, add vllm3/vllm4)
- **Max context**: 1 backend × TP=4 (`VLLM1_GPUS="0,1,2,3"`, remove vllm2)
- **Multi-host**: Modify HAProxy backend config to point to remote hosts

## Key Configuration Variables

Environment variables that commonly need adjustment:

- `MODEL_ID`: HuggingFace model identifier (e.g., `openai/gpt-oss-20b`)
- `STACK_DIR`: Base directory for all stack components (default: `/opt/llm-stack`)
- `GPU_TYPE`: Force specific GPU type (`r9700`, `mi300`, `mi355`, `auto`)
- `VLLM_IMAGE`: Override container image selection
- `GPU_ISOLATION`: Isolation method (`env` or `devices`)
- `VLLM1_GPUS` / `VLLM2_GPUS`: GPU assignments (comma-separated indices)
- `GPU_MEM_UTIL`: GPU memory utilization (0.0-1.0, default: 0.90)
- `MAX_MODEL_LEN`: Context window length in tokens (default: 8192)
- `MAX_NUM_SEQS`: Max parallel sequences (default: 128)
- `ENABLE_PREFIX_CACHING`: Cache prompt prefixes for 20-50% speedup (default: 1)
- `ENABLE_CHUNKED_PREFILL`: Better TTFT for long prompts (default: 1)
- `ENABLE_TLS`: Generate certificates and enable HTTPS with HTTP/2 (default: 0)
- `HF_TOKEN`: HuggingFace API token for private models

## System Tuning Applied

The script configures production-grade system settings:

### Kernel/Network (sysctl):
- Large connection backlogs and ephemeral port range for high QPS
- TCP keepalive and TIME_WAIT reuse optimizations
- Large TCP buffers for chunky LLM responses
- BBR congestion control
- High `vm.max_map_count` for large model mappings
- Low `vm.swappiness` to avoid latency spikes

### GPU Configuration:
- Power profile set to COMPUTE
- Performance level set to high
- Persisted via systemd services

### CPU Configuration:
- Governor set to performance mode
- Persisted via systemd service

### System Limits:
- File handle limits raised to 1M
- THP (Transparent Huge Pages) disabled
- Docker service limits increased

### ROCm Environment:
- RCCL tuning for tensor parallelism
- HSA fine-grain PCIe enabled
- AITER attention kernels enabled

## File Structure

```
/opt/llm-stack/              # Default STACK_DIR
├── repos/                   # Git clones (gpt-oss, vllm)
├── models/                  # Model weights (MODEL_DIR)
│   └── [model_id]/         # HuggingFace model files
├── hf-cache/               # HuggingFace cache (HF_HOME)
├── config/                 # Configuration files
│   ├── .env               # Environment variables (HF_TOKEN, AITER flags)
│   └── haproxy/
│       ├── haproxy.cfg    # HAProxy configuration
│       ├── server.pem     # TLS certificate (if ENABLE_TLS=1)
│       └── tls/           # CA and server certs
├── compose/
│   └── docker-compose.yml # Docker Compose stack
└── venv/                  # Python venv for HF CLI
```

## Important Implementation Notes

### ROCm-Specific Constraints
- **DO NOT use `vllm/vllm-openai` image** - that's CUDA only
- Use `rocm/vllm` or `rocm/vllm-dev` images
- `--num-scheduler-steps` is NOT supported in ROCm vLLM builds
- `--compilation-config` with `full_cuda_graph` is NOT supported in ROCm builds
- Device mounts differ from CUDA: use `/dev/kfd` and `/dev/dri/*`, not nvidia-docker runtime

### Container Configuration
- Uses `network_mode: host` for minimum latency
- Uses `ipc: host` for shared memory
- Uses numeric GIDs (not group names) in `group_add` for portability
- Installs missing dependencies (`openai`, `colorama`) at container startup via bash wrapper

### Model Download
- Uses HuggingFace CLI in a dedicated Python venv
- Skips download if `config.json` exists in `MODEL_DIR`
- Excludes `original/*` directory to save space (vLLM uses HF-converted format)
- Supports `HF_TOKEN` for private models

### Chat Templates
- `CHAT_TEMPLATE` variable specifies Jinja2 template path
- Required for models with reasoning tokens (e.g., gpt-oss)
- Can be empty (use model's built-in), relative (in model dir), or absolute path

## Troubleshooting

### Compose Tool Not Found (Most Common Error)

**Error:**
```
Error: looking up compose provider failed
7 errors occurred:
	* exec: "docker-compose": executable file not found in $PATH
	* exec: "podman-compose": executable file not found in $PATH
```

**Cause:** No compose tool installed. The script requires either:
- Docker with built-in Compose V2, OR
- Podman with separately-installed `podman-compose`

**Solution:**
```bash
# Check your container runtime first
docker --version

# If using Podman (shows "podman version"):
sudo apt-get install -y podman-compose

# If using Docker (shows "Docker version"):
sudo apt-get install -y docker-compose-v2
```

**Verification:**
```bash
# For Podman:
podman-compose --version

# For Docker:
docker compose version
```

After installing compose tool, re-run the script or manually start the stack:
```bash
# For Podman:
podman-compose -f /opt/llm-stack/compose/docker-compose.yml up -d

# For Docker:
docker compose -f /opt/llm-stack/compose/docker-compose.yml up -d
```

### GPU Count Mismatch Warning

**Warning:**
```
WARNING: Expected 4 GPUs but found 6.
         You may need to adjust VLLM1_GPUS and VLLM2_GPUS.
```

**Cause:** The script's default configuration assumes 4 GPUs, but your system has a different count.

**Solution:** Edit `launch.sh` and adjust GPU configuration:
```bash
# Find the GPU configuration section in launch.sh
export EXPECTED_GPU_COUNT="6"     # Match your actual GPU count
export VLLM1_GPUS="0,1,2"        # First backend: GPUs 0,1,2 (TP=3)
export VLLM2_GPUS="3,4,5"        # Second backend: GPUs 3,4,5 (TP=3)
```

**Note:** This is a warning, not an error. The script will continue, but performance may be suboptimal if GPU assignments don't match your hardware.

### GPU Not Detected
- Check if ROCm is loaded: `rocm-smi`
- Verify kernel module: `lsmod | grep amdgpu`
- If ROCm was just installed: **reboot required** to load kernel modules
- Check detected GPU info: `rocminfo | grep "Marketing Name"`

### Container Can't See GPUs
- Switch `GPU_ISOLATION` from `env` to `devices` mode
- Verify device permissions: `ls -l /dev/dri/`
- Check container logs: `docker compose logs vllm1` or `podman-compose logs vllm1`
- Verify video/render group membership: `groups` (should include 'video' and 'render')

### RDNA 3 GPUs (gfx1100) Detected as "Not Optimal"

**Message:**
```
RDNA 3 GPU detected (gfx1100). ROCm support available but not optimal for LLM inference.
```

**Explanation:** RDNA 3 consumer/workstation GPUs have ROCm support but lack the specialized features of CDNA data center GPUs:
- Smaller VRAM (16-48GB vs 128-288GB for MI-series)
- Missing specialized matrix acceleration
- No GPU-specific optimized vLLM images available

**Performance expectations:**
- Will work with `rocm/vllm:latest` image
- Best suited for smaller models (7B-20B parameter range)
- May need `HSA_OVERRIDE_GFX_VERSION` environment variable for some models
- Consider using `GPU_ISOLATION="devices"` mode for better reliability

**RDNA 3 is supported, but consider:**
- Adjust `MAX_MODEL_LEN` to fit available VRAM
- Lower `MAX_NUM_SEQS` if encountering OOM errors
- Use quantized models (e.g., GPTQ, AWQ) for better memory efficiency

### Model Download Fails
- Verify `HF_TOKEN` is set for private models
- Check internet connectivity
- Verify disk space in `STACK_DIR`

### Performance Issues
- Check GPU utilization: `rocm-smi`
- Check VRAM usage: `rocm-smi --showmeminfo vram`
- Review HAProxy stats: `curl http://wide.local:8404/stats`
- Check if requests are hitting cache: HAProxy sticky cookies + prefix caching

### Image Pull Fails
- Verify image tag exists: `curl -s 'https://hub.docker.com/v2/repositories/rocm/vllm/tags?page_size=20' | jq -r '.results[].name'`
- Check for typos in `VLLM_IMAGE` variable
- Ensure using ROCm images, not CUDA images
