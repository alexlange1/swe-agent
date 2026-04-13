# ── Tau Mining Agent Dockerfile ──────────────────────────────────────────────
#
# Build on top of the tau pi-mono-base image which provides:
#   - Node.js 20
#   - tsgo (TypeScript Go compiler) at /opt/pi-mono-base/node_modules/.bin/
#   - shx and other dev tools
#
# USAGE: The validator builds this image from the miner's fork, then runs:
#   docker exec ... bash -lc "
#     export PATH=$TAU_AGENT_DIR/node_modules/.bin:/opt/pi-mono-base/node_modules/.bin:$PATH
#     cd $TAU_AGENT_DIR
#     tsgo -p packages/tui/tsconfig.build.json   &&
#     tsgo -p packages/ai/tsconfig.build.json    &&
#     tsgo -p packages/agent/tsconfig.build.json &&
#     tsgo -p packages/coding-agent/tsconfig.build.json &&
#     node packages/coding-agent/dist/cli.js --mode json ... -p PROMPT
#   "
#
# Our cli.ts shim (compiled to cli.js) calls:
#   python3 $TAU_AGENT_DIR/agent/main.py <same args>
# ─────────────────────────────────────────────────────────────────────────────

# Try tau's published base image first; fall back to node:20-slim for local dev
ARG BASE_IMAGE=ghcr.io/unarbos/pi-mono-base:latest
FROM ${BASE_IMAGE} AS tau-base

# ── Python setup ──────────────────────────────────────────────────────────────
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv keeps Python deps clean
RUN python3 -m venv /opt/tau-agent-venv
ENV PATH="/opt/tau-agent-venv/bin:$PATH"

# Install Python dependencies
COPY agent/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Agent source ──────────────────────────────────────────────────────────────
# Copy the full repo into the image; TAU_AGENT_DIR will point at the
# agent/ subdirectory (/work/agent-src at runtime).
COPY . /work/agent-src

# Ensure the Python agent is executable
RUN chmod +x /work/agent-src/agent/main.py

# ── Environment defaults ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
# TAU_AGENT_DIR points at the pi-mono agent workspace within the repo.
# The validator may override this at runtime; this default ensures the
# tsgo compile step (cd $TAU_AGENT_DIR && tsgo -p packages/...) works correctly.
ENV TAU_AGENT_DIR=/work/agent-src/agent

WORKDIR /work
