/**
 * cli.ts — Drop-in replacement for tau's packages/coding-agent/src/cli.ts
 *
 * Delegates all solving to the Python agent at the workspace root.
 *
 * Layout after placement in the miner's tau fork:
 *   agent/                        ← pi-mono workspace (TAU_AGENT_DIR)
 *     main.py                     ← our Python entry point
 *     solver.py, tools.py, ...    ← our Python agent
 *     packages/
 *       coding-agent/
 *         src/cli.ts              ← THIS FILE (replaces native cli.ts)
 *         dist/cli.js             ← compiled output (runtime)
 *
 * __dirname at runtime = agent/packages/coding-agent/dist/
 * Three levels up       = agent/   (workspace root, where main.py lives)
 */

import { spawnSync } from "child_process";
import * as path from "path";

// Resolve main.py relative to the compiled JS file — no env var dependency.
const workspaceRoot = path.join(__dirname, "..", "..", "..");
const agentScript = path.join(workspaceRoot, "main.py");
const args = process.argv.slice(2);

const result = spawnSync("python3", [agentScript, ...args], {
  stdio: "inherit",
  env: process.env,
  cwd: process.cwd(),
});

process.exit(result.status ?? 1);
