#!/usr/bin/env node
/**
 * Cross-platform task runner - a Node port of the Makefile targets so Windows
 * contributors do not need GNU make, bash, pkill, or lsof.
 *
 * Usage: node scripts/run.mjs <task>   (or: npm run <task>)
 */
import { spawn, spawnSync } from "node:child_process";
import { cp, rm } from "node:fs/promises";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const isWindows = process.platform === "win32";
const BACKEND_PORT = 8000;
const FRONTEND_PORT = 3000;

const log = (msg = "") => console.log(msg);

/** Run a command to completion; exit the process if it fails. */
function run(command, { cwd = ROOT, allowFailure = false } = {}) {
  const result = spawnSync(command, { cwd, stdio: "inherit", shell: true });
  if (!allowFailure && result.status !== 0) {
    process.exit(result.status ?? 1);
  }
  return result.status ?? 0;
}

/** Capture stdout of a command without echoing it. */
function capture(command) {
  const result = spawnSync(command, { cwd: ROOT, shell: true, encoding: "utf8" });
  return result.stdout ?? "";
}

/** Kill a process tree by pid (Windows needs taskkill to reach grandchildren). */
function killTree(pid) {
  if (!pid) return;
  if (isWindows) {
    spawnSync(`taskkill /pid ${pid} /T /F`, { shell: true, stdio: "ignore" });
    return;
  }
  try {
    process.kill(-pid, "SIGTERM");
  } catch {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
}

/** Free a TCP port left behind by a crashed dev server. */
function killPort(port) {
  if (isWindows) {
    const pids = new Set();
    for (const line of capture("netstat -ano -p tcp").split(/\r?\n/)) {
      if (!line.includes("LISTENING")) continue;
      const parts = line.trim().split(/\s+/);
      if ((parts[1] ?? "").endsWith(`:${port}`)) pids.add(parts[parts.length - 1]);
    }
    for (const pid of pids) {
      if (pid && pid !== "0") {
        spawnSync(`taskkill /pid ${pid} /T /F`, { shell: true, stdio: "ignore" });
      }
    }
    return;
  }
  spawnSync(`lsof -ti:${port} | xargs kill -9`, { shell: true, stdio: "ignore" });
}

/** Recursively delete __pycache__ directories and .pyc files. */
async function cleanPythonCache(dir = ROOT) {
  const skip = new Set(["node_modules", ".git", ".venv", "venv", ".next", "out"]);
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (skip.has(entry.name)) continue;
      if (entry.name === "__pycache__") {
        await rm(full, { recursive: true, force: true });
        continue;
      }
      await cleanPythonCache(full);
    } else if (entry.name.endsWith(".pyc")) {
      await rm(full, { force: true });
    }
  }
}

/**
 * Seed frontend/.env.local from .env.example when it is missing.
 *
 * Without NEXT_PUBLIC_API_URL the dev UI posts to its own origin (:3000)
 * instead of the API (:8000), so every request 404s from Next.js.
 */
async function ensureFrontendEnv() {
  const envLocal = path.join(ROOT, "frontend", ".env.local");
  const envExample = path.join(ROOT, "frontend", ".env.example");
  if (existsSync(envLocal) || !existsSync(envExample)) return;
  await cp(envExample, envLocal);
  log("Created frontend/.env.local from .env.example (NEXT_PUBLIC_API_URL).");
}

/** Start backend + frontend together and stop both on Ctrl+C. */
function runBoth() {
  log("Starting QuOptuna dev environment...");
  log("");
  log(`Backend:  http://localhost:${BACKEND_PORT} (API docs: /api/docs)`);
  log(`Frontend: http://localhost:${FRONTEND_PORT}`);
  log("");
  log("Press Ctrl+C to stop both services");
  log("");

  const spawnOpts = { stdio: "inherit", shell: true, detached: !isWindows };
  const backend = spawn(
    `uv run --no-sync uvicorn quoptuna.server.main:app --host 0.0.0.0 --port ${BACKEND_PORT} --reload`,
    { ...spawnOpts, cwd: ROOT },
  );
  const frontend = spawn("npm run dev", {
    ...spawnOpts,
    cwd: path.join(ROOT, "frontend"),
  });

  let shuttingDown = false;
  const shutdown = () => {
    if (shuttingDown) return;
    shuttingDown = true;
    log("");
    log("Stopping services...");
    killTree(backend.pid);
    killTree(frontend.pid);
    killPort(BACKEND_PORT);
    killPort(FRONTEND_PORT);
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
  backend.on("exit", shutdown);
  frontend.on("exit", shutdown);
}

const tasks = {
  help() {
    log("QuOptuna Development Commands (npm run <task>)");
    log("");
    log("Setup:");
    log("  install            - Install backend + frontend dependencies");
    log("  install:backend    - Install backend dependencies");
    log("  install:frontend   - Install frontend dependencies");
    log("  init               - Clean Python caches and sync the uv environment");
    log("");
    log("Running:");
    log("  dev                - Run backend + frontend together (Ctrl+C stops both)");
    log("  dev:backend        - Run FastAPI backend (port 8000)");
    log("  dev:frontend       - Run Next.js frontend (port 3000)");
    log("  dev:streamlit      - Run legacy Streamlit interface");
    log("");
    log("Code Quality:");
    log("  format             - Format the code (ruff)");
    log("  lint               - Run ruff + mypy");
    log("  lint:fix           - Run ruff with --fix");
    log("  test               - Run pytest");
    log("  coverage           - Run pytest with coverage");
    log("  precommit          - Run pre-commit on all files");
    log("");
    log("Build & Clean:");
    log("  build              - Build the Python package (wheel/sdist)");
    log("  build:package      - Build frontend + bundle it + build the package (for uvx)");
    log("  clean              - Free ports 8000/3000 and stop dev servers");
    log("  clean:cache        - Remove __pycache__ dirs and .pyc files");
    log("  clean:all          - clean:cache + clean");
  },

  async install() {
    tasks["install:backend"]();
    await tasks["install:frontend"]();
  },

  "install:backend"() {
    log("Installing backend dependencies...");
    run("uv pip install -e .", { cwd: path.join(ROOT, "backend") });
    log("Installing main quoptuna package into backend venv...");
    run("uv pip install -e ..", { cwd: path.join(ROOT, "backend") });
    log("Backend dependencies installed!");
  },

  async "install:frontend"() {
    log("Installing frontend dependencies...");
    run("npm install", { cwd: path.join(ROOT, "frontend") });
    await ensureFrontendEnv();
    log("Frontend dependencies installed!");
  },

  async init() {
    await cleanPythonCache();
    run("uv sync");
  },

  async dev() {
    await ensureFrontendEnv();
    runBoth();
  },

  "dev:backend"() {
    log(`Starting FastAPI backend on http://localhost:${BACKEND_PORT}...`);
    log(`API docs: http://localhost:${BACKEND_PORT}/api/docs`);
    run(
      `uv run --no-sync uvicorn quoptuna.server.main:app --host 0.0.0.0 --port ${BACKEND_PORT} --reload`,
    );
  },

  async "dev:frontend"() {
    await ensureFrontendEnv();
    log(`Starting Next.js frontend on http://localhost:${FRONTEND_PORT}...`);
    run("npm run dev", { cwd: path.join(ROOT, "frontend") });
  },

  "dev:streamlit"() {
    log("Starting legacy Streamlit interface...");
    run("uv run streamlit run src/quoptuna/frontend/app.py");
  },

  format: () => run("uv run ruff format ."),

  lint() {
    run("uv run ruff check .");
    run("uv run mypy .");
  },

  "lint:fix": () => run("uv run ruff check --fix ."),

  test: () => run("uv run pytest"),

  coverage() {
    run("uv run coverage run -m pytest");
    run("uv run coverage report");
  },

  precommit: () => run("uv run pre-commit run --all-files"),

  build: () => run("uv build"),

  async "build:package"() {
    log("Building frontend static export...");
    run("npm ci", { cwd: path.join(ROOT, "frontend") });
    run("npm run build", { cwd: path.join(ROOT, "frontend") });

    log("Bundling frontend into the package (src/quoptuna/web)...");
    const outDir = path.join(ROOT, "frontend", "out");
    if (!existsSync(outDir)) {
      console.error(`Expected the static export at ${outDir} but it was not produced.`);
      process.exit(1);
    }
    const webDir = path.join(ROOT, "src", "quoptuna", "web");
    await rm(webDir, { recursive: true, force: true });
    await cp(outDir, webDir, { recursive: true });

    log("Building Python package...");
    run("uv build");
  },

  clean() {
    log("Stopping all services and cleaning up ports...");
    killPort(BACKEND_PORT);
    killPort(FRONTEND_PORT);
    log(`Ports ${BACKEND_PORT} and ${FRONTEND_PORT} cleaned up`);
  },

  async "clean:cache"() {
    await cleanPythonCache();
    log("Python cache cleaned");
  },

  async "clean:all"() {
    await tasks["clean:cache"]();
    tasks.clean();
  },
};

const task = process.argv[2] ?? "help";
if (!tasks[task]) {
  console.error(`Unknown task: ${task}\n`);
  tasks.help();
  process.exit(1);
}
await tasks[task]();
