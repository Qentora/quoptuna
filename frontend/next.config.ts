import path from 'node:path';
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Pin the workspace root so Next does not pick up unrelated parent lockfiles.
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
