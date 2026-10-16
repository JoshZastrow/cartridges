import { defineConfig, devices } from '@playwright/test';

// One model turn may take ~60s for mock-echo and several minutes for a real
// Teacher model (cold start). Each test does 3 turns, so the per-test timeout
// is 3x the turn timeout plus slack for registration.
const turnTimeoutMs = Number(process.env.PW_TURN_TIMEOUT_MS ?? 120000);

export default defineConfig({
  testDir: './e2e',
  timeout: turnTimeoutMs * 3 + 60000,
  expect: { timeout: 15000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:3080',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  // Start the whole stack unless it is already up (reuseExistingServer). The
  // Modal GPU containers need no start step: the gateway's first request to a
  // model cold-starts them. The gateway's /v1/models answers 401 without a
  // key, which Playwright counts as "up".
  webServer: [
    {
      command: 'uv run cartridges gateway_serve --port 8008',
      url: 'http://127.0.0.1:8008/v1/models',
      reuseExistingServer: true,
      timeout: 120000,
    },
    {
      command: 'LIBRECHAT_FOREGROUND=1 ./librechat/start.sh',
      url: 'http://localhost:3080/health',
      reuseExistingServer: true,
      timeout: 600000, // first run: npm ci + frontend build
    },
  ],
});
