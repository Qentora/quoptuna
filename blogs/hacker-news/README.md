# Hacker News — how to launch (and not get flagged)

> **Status:** First Show HN (20 Jul 2026, account `edwinjosewmu`) was **`[flagged]`**. Cause = brand-new account + first-ever action was self-promotion, **not** the post content. This README is the recovery + do-it-right playbook. The post copy itself lives in [`show-hn.md`](show-hn.md).

---

## Why the first post got flagged

HN aggressively filters (and users flag) a specific pattern: **a days-old account with ~1 karma and no comment history whose very first submission is its own project.** That reads as a marketing drop, regardless of how good the project is. Contributing factors last time:

1. **New account, karma 1, zero history** ← the main cause.
2. **No prior participation** on HN (no comments, no upvotes given).
3. **A stray `?` on the repo URL** (`.../quoptuna?`) — a broken-looking link adds to the "low-effort" read.
4. Possibly **cross-platform velocity** — blasting Reddit/PH/X in the same window can draw extra scrutiny.

None of this is about the writing. Show HN posts from fresh accounts get flagged constantly.

---

## Recover the CURRENT flagged post

1. **Fix the URL typo** if you re-paste anywhere — the repo is `https://github.com/Qentora/quoptuna` (no trailing `?`).
2. **Email the mods** — this genuinely works; dang routinely restores legit Show HNs from new accounts. Send to **hn@ycombinator.com**:

   > **Subject:** Show HN flagged — new account, genuine first launch
   >
   > Hi, my Show HN "QuOptuna" (item id `XXXXX`) got flagged. It's a genuine open-source project — my PhD work at Western Michigan University, described in three IEEE papers — not spam. My account is just new with no history yet. Could you take a look and unflag it if it qualifies? Happy to answer anything. Thanks! — Edwin

3. **Do NOT immediately re-submit** a fresh copy. That reads as evasion and gets you rate-limited / shadow-penalized. Email first; only re-submit if the mods say to.

---

## Before the NEXT attempt — a pre-launch checklist

**Account warm-up (do this over several days — most important):**
- [ ] Leave **5–15 genuine, useful comments** on other threads (answer questions in your area — ML, Python, quantum). This builds karma + history so the auto-flag doesn't trip.
- [ ] Upvote things you actually like. Behave like a real reader, because you are one.
- [ ] Aim for **~20+ karma** and a few days of age before self-posting.

**Post readiness:**
- [ ] **Hosted live demo link** ready (the biggest anti-flag signal — a clickable "try it now" beats a repo). Deploy read-only on HF Spaces / Railway. *(Still a TODO — see root `../README.md`.)*
- [ ] **Demo GIF / 60-sec video** embedded in the README.
- [ ] Repo README is clean above the fold: one-line value prop + `uvx quoptuna` + demo link.
- [ ] `CONTRIBUTING.md` present (it is ✅).
- [ ] URL has no typos.

**Read once:**
- [ ] [Show HN rules](https://news.ycombinator.com/showhn.html)
- [ ] [HN guidelines](https://news.ycombinator.com/newsguidelines.html)

---

## How to post

1. **Timing:** Tuesday–Thursday, **~8–10am ET** (weekday morning US = peak traffic, best shot at the /newest → front-page path).
2. **URL field:** the **hosted demo** if you have one; otherwise the repo (`https://github.com/Qentora/quoptuna`).
3. **Title:** pick one from [`show-hn.md`](show-hn.md). Rules: starts with `Show HN:`, ≤80 chars, **no hype words** ("revolutionary", "amazing", "!!!"), no emoji.
4. **Text/first comment:** paste the body from [`show-hn.md`](show-hn.md) verbatim (it includes the CLI + a clean repo link). This becomes the top comment — it's your story, so it matters.
5. **Post it, then stay put.**

---

## Launch-day behavior (this is where posts live or die)

- **Reply to the first 2–3 comments within minutes.** Active-maintainer signal keeps the post alive and is what converts lurkers into stargazers. Pre-drafted replies are at the bottom of [`show-hn.md`](show-hn.md).
- **Answer every comment for the first few hours**, including skeptical ones — especially "does quantum actually beat classical?" (your honest answer: usually not yet on tabular data, and the tool says so — that candor plays well on HN).
- **Never ask for upvotes or stars** anywhere. Asking for feedback is fine; asking for votes violates the rules and gets you flagged.
- **Don't argue defensively.** Concede fair points, thank critics, note what you'll fix.

---

## After

- Check **hide/past** and your profile for whether it stuck.
- If it did well, the same-day spike can hit **GitHub Trending** — keep replies flowing.
- Log the result in the tracking table in the root `../README.md`.
- Reuse the same body on **Lobsters** (tags `ml`, `python`) only if you have an invite — same account-reputation rules apply there.

---

## The one-line rule

**HN rewards a real person who has been part of the community showing something genuine, and punishes a fresh account that shows up only to promote.** Warm up first; the copy is already good.
