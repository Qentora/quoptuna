/**
 * GPU support for the dev task runner.
 *
 * JAX publishes CUDA wheels for Linux only, so on Windows the GPU backend runs
 * inside WSL2 (which passes the host NVIDIA driver through) while the Next.js
 * frontend stays on Windows. The frontend is pointed at the distro's own IP
 * rather than localhost -- see backendUrlForBrowser for why. macOS has no CUDA
 * backend for this stack at all.
 */
import { spawn, spawnSync } from "node:child_process";
import path from "node:path";

export const GPU_VENV = "$HOME/.quoptuna-gpu";
const UV = "$HOME/.local/bin/uv";

const isWindows = process.platform === "win32";
const isMac = process.platform === "darwin";
const log = (msg = "") => console.log(msg);

/** Decode `wsl.exe` output, which is UTF-16LE rather than UTF-8. */
function decodeWsl(buffer) {
  if (!buffer || !buffer.length) return "";
  // A UTF-8 payload from `bash` has no interleaved NULs; wsl.exe's own
  // listings do, so pick the decoding from the bytes rather than guessing.
  const hasNulls = buffer.includes(0);
  return buffer.toString(hasNulls ? "utf16le" : "utf8").replace(/\0/g, "");
}

/** Name of the WSL distro to use (QUOPTUNA_WSL_DISTRO, else the default). */
export function wslDistro() {
  if (process.env.QUOPTUNA_WSL_DISTRO) return process.env.QUOPTUNA_WSL_DISTRO;
  const out = decodeWsl(spawnSync("wsl.exe", ["-l", "-q"], { encoding: "buffer" }).stdout);
  const names = out
    .split(/\r?\n/)
    .map((n) => n.trim())
    .filter((n) => n && !n.startsWith("docker-"));
  return names[0] ?? "";
}

/** Run a shell snippet on the Linux side; returns {status, stdout}. */
export function linuxCapture(script, { distro = null } = {}) {
  const args = isWindows
    ? ["-d", distro ?? wslDistro(), "--", "bash", "-lc", script]
    : null;
  const result = args
    ? spawnSync("wsl.exe", args, { encoding: "buffer" })
    : spawnSync("bash", ["-lc", script], { encoding: "buffer" });
  return { status: result.status ?? 1, stdout: decodeWsl(result.stdout).trim() };
}

/** Spawn a long-running Linux command with inherited stdio. */
export function linuxSpawn(script, { distro = null } = {}) {
  return isWindows
    ? spawn("wsl.exe", ["-d", distro ?? wslDistro(), "--", "bash", "-lc", script], {
        stdio: "inherit",
      })
    : spawn("bash", ["-lc", script], { stdio: "inherit", detached: true });
}

/**
 * Address the Windows browser should use to reach the Linux-side backend.
 *
 * WSL2's localhost forwarding is the documented path, but its relay does not
 * always publish the port (a force-killed relay leaves nothing listening on
 * 127.0.0.1 while the server is healthy inside the VM). Addressing the distro
 * directly is deterministic, so resolve its IP and only fall back to localhost
 * when that is unavailable. http://localhost:3000 is already an allowed CORS
 * origin, so the browser can call this cross-origin.
 */
export function backendUrlForBrowser(port) {
  if (!isWindows) return `http://localhost:${port}`;
  const { status, stdout } = linuxCapture("hostname -I | awk '{print $1}'");
  const ip = status === 0 ? stdout.split(/\s+/)[0] : "";
  return /^\d+\.\d+\.\d+\.\d+$/.test(ip)
    ? `http://${ip}:${port}`
    : `http://localhost:${port}`;
}

/** Repo path as the Linux side sees it (/mnt/c/... under WSL). */
export function linuxRepoPath(root) {
  if (!isWindows) return root;
  const drive = root[0].toLowerCase();
  return `/mnt/${drive}${root.slice(2).split(path.sep).join("/")}`;
}

/**
 * Why the GPU path cannot run, or null when it can.
 * Ordered so the message names the first thing the user has to fix.
 */
export function gpuBlocker() {
  if (isMac) {
    return [
      "macOS has no CUDA backend for this stack: JAX ships CUDA wheels for",
      "Linux only, and Apple silicon has no supported GPU path for PennyLane",
      "here. Use `npm run dev` (CPU) on this machine.",
    ].join("\n");
  }
  if (isWindows) {
    if (spawnSync("wsl.exe", ["-l", "-q"], { encoding: "buffer" }).status !== 0) {
      return "WSL2 is not available. Install it with: wsl --install";
    }
    if (!wslDistro()) return "No WSL distro found. Install one with: wsl --install -d Ubuntu";
  }
  if (linuxCapture("command -v nvidia-smi >/dev/null && nvidia-smi -L").status !== 0) {
    return isWindows
      ? "No NVIDIA GPU visible inside WSL. Update the Windows NVIDIA driver (it\nprovides the WSL CUDA passthrough); no driver is installed inside WSL."
      : "No NVIDIA GPU visible (nvidia-smi failed).";
  }
  return null;
}

/** Whether the GPU venv exists and reports a CUDA backend. */
export function gpuVenvStatus() {
  const { status, stdout } = linuxCapture(
    `${GPU_VENV}/bin/python -c "import jax;print(jax.default_backend());print(jax.devices()[0].device_kind)" 2>/dev/null`,
  );
  if (status !== 0 || !stdout) return { ready: false, backend: null, device: null };
  const [backend, device] = stdout.split(/\r?\n/);
  return { ready: backend === "gpu", backend, device: device ?? null };
}

export function reportGpu(root) {
  const blocker = gpuBlocker();
  if (blocker) {
    log("GPU: unavailable");
    log("");
    log(blocker);
    return 1;
  }
  const distro = isWindows ? wslDistro() : "(native linux)";
  log(`GPU host:   ${distro}`);
  log(`Driver:     ${linuxCapture("nvidia-smi -L").stdout || "unknown"}`);
  const venv = gpuVenvStatus();
  if (!venv.ready) {
    log(`GPU venv:   not ready (${venv.backend ?? "missing"}) - run: npm run gpu:setup`);
    return 1;
  }
  log(`GPU venv:   ready - JAX backend '${venv.backend}' on ${venv.device}`);
  log(`Repo path:  ${linuxRepoPath(root)}`);
  return 0;
}

/** Create the Linux-side venv and install the project plus CUDA JAX into it. */
export function setupGpu(root, jaxVersion = "0.7.1") {
  const blocker = gpuBlocker();
  if (blocker) {
    log(blocker);
    return 1;
  }
  const repo = linuxRepoPath(root);
  log(`Setting up the GPU environment (${isWindows ? wslDistro() : "linux"})...`);
  log("This installs uv, a Python 3.12 venv and CUDA JAX under ~/.quoptuna-gpu.");
  log("");
  const script = [
    "set -e",
    'export PATH="$HOME/.local/bin:$PATH"',
    "if ! command -v uv >/dev/null; then",
    '  echo "Installing uv...";',
    "  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null;",
    "fi",
    `echo "Creating venv at ${GPU_VENV}..."`,
    `${UV} venv --python 3.12 ${GPU_VENV} >/dev/null 2>&1 || true`,
    `echo "Installing quoptuna (editable) from ${repo}..."`,
    `${UV} pip install --python ${GPU_VENV}/bin/python -q -e "${repo}"`,
    `echo "Installing jax[cuda12]==${jaxVersion}..."`,
    `${UV} pip install --python ${GPU_VENV}/bin/python -q "jax[cuda12]==${jaxVersion}"`,
    'echo "Verifying..."',
    `${GPU_VENV}/bin/python -c "import jax;print('JAX backend:',jax.default_backend());print('Device:',jax.devices()[0].device_kind)"`,
  ].join("\n");
  const result = isWindows
    ? spawnSync("wsl.exe", ["-d", wslDistro(), "--", "bash", "-lc", script], { stdio: "inherit" })
    : spawnSync("bash", ["-lc", script], { stdio: "inherit" });
  if ((result.status ?? 1) !== 0) {
    log("");
    log("GPU setup failed. See the output above.");
    return result.status ?? 1;
  }
  log("");
  log("GPU environment ready. Start it with: npm run dev:gpu");
  return 0;
}

/**
 * Command that starts the GPU backend on the Linux side.
 *
 * float32 is not optional here: GeForce cards run FP64 at a fraction of FP32
 * throughput, so leaving x64 on would give back most of the GPU's advantage.
 */
export function gpuBackendCommand(root, port) {
  const repo = linuxRepoPath(root);
  return [
    `cd "${repo}"`,
    "export QUOPTUNA_JAX_X64=0",
    `exec ${GPU_VENV}/bin/python -m uvicorn quoptuna.server.main:app ` +
      `--host 0.0.0.0 --port ${port} --reload`,
  ].join(" && ");
}
