import path from 'node:path';
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Pin the workspace root so Next does not pick up unrelated parent lockfiles.
  outputFileTracingRoot: path.join(__dirname),
  // Emit a fully static site (frontend/out) that FastAPI serves from the wheel.
  output: 'export',
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
