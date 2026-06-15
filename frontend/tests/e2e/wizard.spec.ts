import { expect, test } from '@playwright/test';

test('home page renders and links to the optimizer', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: /QuOptuna/i }).first()).toBeVisible();

  // Navigate to the optimizer wizard via the sidebar.
  await page.getByRole('link', { name: 'Optimizer' }).click();
  await expect(page).toHaveURL(/\/optimizer$/);
  await expect(page.getByRole('heading', { name: /QuOptuna Optimizer/i })).toBeVisible();

  // First wizard step should be the dataset selection.
  await expect(page.getByRole('heading', { name: /Dataset Selection/i })).toBeVisible();
});

test('settings page exposes API key inputs', async ({ page }) => {
  await page.goto('/settings');
  await expect(page.getByText('OpenAI API Key')).toBeVisible();
  await expect(page.getByText('Anthropic API Key')).toBeVisible();
  await expect(page.getByText('Google API Key')).toBeVisible();
});
