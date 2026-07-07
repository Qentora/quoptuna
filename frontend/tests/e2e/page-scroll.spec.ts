import { expect, test } from '@playwright/test';

// The migration's layout invariant: pages rely on page-level scrolling only.
// No element (besides the app's main scroll container) may be a vertical
// scroll container. Horizontal-only scroll (wide tables) is allowed, as are
// popover/dropdown viewports which only scroll when open.
const findVerticalScrollers = () =>
  Array.from(document.querySelectorAll<HTMLElement>('*'))
    .filter((el) => {
      const style = getComputedStyle(el);
      const scrollsY = style.overflowY === 'auto' || style.overflowY === 'scroll';
      return scrollsY && el.scrollHeight > el.clientHeight + 1;
    })
    .filter((el) => !el.dataset.appScrollContainer)
    .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 80)}`);

for (const path of ['/', '/optimizer', '/runs', '/settings']) {
  test(`only the page scrolls on ${path}`, async ({ page }) => {
    await page.goto(path);
    await page.waitForLoadState('networkidle');
    const scrollers = await page.evaluate(findVerticalScrollers);
    expect(scrollers).toEqual([]);
  });
}
