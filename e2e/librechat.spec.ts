import { expect, test, type Page } from '@playwright/test';
import tasks from './fixtures/tasks.json';

const MODEL = process.env.LIBRECHAT_MODEL ?? 'mock-echo';
const ENDPOINT = 'Journal';
const TURN_TIMEOUT_MS = Number(process.env.PW_TURN_TIMEOUT_MS ?? 120000);
const EMAIL = process.env.LIBRECHAT_EMAIL ?? 'e2e@journal.test';
const PASSWORD = process.env.LIBRECHAT_PASSWORD ?? 'Journal-e2e-pass1';

const questions = tasks.tasks.map((t) => t.text);

/** Returns the /api/auth/login status. */
async function login(page: Page) {
  const form = page.getByRole('form', { name: 'Login form' });
  await expect(form).toBeVisible();
  await form.getByRole('textbox', { name: 'Email' }).fill(EMAIL);
  await form.getByRole('textbox', { name: 'Password' }).fill(PASSWORD);
  const [resp] = await Promise.all([
    page.waitForResponse((r) => r.url().endsWith('/api/auth/login')),
    page.getByTestId('login-button').click(),
  ]);
  return resp.status();
}

async function register(page: Page) {
  await page.goto('/register');
  await page.getByTestId('name').fill('E2E Journal');
  await page.getByTestId('email').fill(EMAIL);
  await page.getByTestId('password').fill(PASSWORD);
  await page.getByTestId('confirm_password').fill(PASSWORD);
  const [resp] = await Promise.all([
    page.waitForResponse((r) => r.url().endsWith('/api/auth/register')),
    page.getByLabel('Submit registration').click(),
  ]);
  expect(resp.status(), `register ${EMAIL}: ${await resp.text()}`).toBe(200);
  console.log(`registered ${EMAIL} / ${PASSWORD}`);
}

/** Log in as the fixed e2e user; register it first if it does not exist yet. */
async function loginOrRegister(page: Page) {
  await page.goto('/login');
  const status = await login(page);
  if (status !== 200) {
    await register(page);
    // Registration bounces through /c/new back to /login (no token yet).
    await page.goto('/login');
    expect(await login(page), 'login after registration').toBe(200);
  }
  await expect(page.getByTestId('text-input')).toBeVisible({ timeout: 30000 });
}

async function selectModel(page: Page) {
  await page.getByTestId('model-selector-button').click();
  const dialog = page.getByRole('dialog', { name: 'Select a model' });
  await expect(dialog).toBeVisible();
  // Endpoint options read "Journal" / "Journal selected"; hovering opens its model list.
  const endpoint = dialog.getByRole('option', { name: new RegExp(`^${ENDPOINT}`) }).first();
  await endpoint.hover();
  await endpoint.click();
  const model = page.getByRole('option', { name: new RegExp(`^${MODEL}( selected)?$`) }).first();
  await expect(model).toBeVisible();
  await model.click();
  await expect(page.getByTestId('model-selector-button')).toContainText(MODEL);
}

async function sendTurn(page: Page, text: string, turn: number) {
  const input = page.getByTestId('text-input');
  await input.click();
  await input.fill(text);
  await page.getByTestId('send-button').click();
  const bodies = page.getByTestId('message-body');
  await expect(bodies).toHaveCount(turn * 2, { timeout: TURN_TIMEOUT_MS });
  await expect(page.getByTestId('stop-generation-button')).toBeHidden({ timeout: TURN_TIMEOUT_MS });
  const reply = (await bodies.nth(turn * 2 - 1).innerText()).trim();
  expect(reply, `assistant reply ${turn} is empty`).not.toBe('');
  console.log(`turn ${turn} reply (${reply.length} chars): ${reply.slice(0, 120)}`);
}

test(`Journal / ${MODEL}: 3-turn chat with Task Store questions`, async ({ page }) => {
  await loginOrRegister(page);
  await page.goto('/c/new');
  await expect(page.getByTestId('text-input')).toBeVisible();
  await selectModel(page);

  for (const [i, q] of questions.entries()) {
    await sendTurn(page, q, i + 1);
  }

  // Each turn is a group "Message N" headed "Prompt N: <user>" or "Response N: <model>".
  const groups = page.getByRole('group', { name: /^Message \d+$/ });
  await expect(groups).toHaveCount(6);
  await expect(page.getByRole('heading', { name: /^Prompt \d+:/ }), 'user messages').toHaveCount(3);
  const responses = groups.filter({
    has: page.getByRole('heading', { name: /^Response \d+:/ }),
  });
  await expect(responses, 'assistant messages').toHaveCount(3);
  for (const t of await responses.getByTestId('message-body').allInnerTexts()) {
    expect(t.trim(), 'assistant message body').not.toBe('');
  }

  await page.screenshot({ path: `test-results/librechat-${MODEL}.png`, fullPage: true });
});
