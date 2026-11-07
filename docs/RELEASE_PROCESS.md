---
doc_id: fusion-trainer/release-process
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "fusion-trainer 发布流程", en: "fusion-trainer release process" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - docs/RELEASE_PROCESS.md
---

# Release Process — fusion-trainer

This document defines how a fusion-trainer release is cut, reviewed, tagged,
and — if necessary — rolled back. It applies to every release from 1.0 onward.
The public API surface is frozen under SemVer; see
[`API_STABILITY.md`](API_STABILITY.md) for what a breaking change is.

---

## 1. Branch strategy

- `master` is the always-green trunk. Never commit directly to `master`; every
  change lands via a reviewed branch + merge (or fast-forward PR).
- **Release branch:** `release/vX.Y.Z` (e.g. `release/v1.0.0`), cut from
  `master` at release time. Only release-specific fixes land on it
  (cherry-picked from `master` or authored directly if `master` is ahead).
- **Feature branches:** `feat/<topic>`, `fix/<topic>` off `master`, one per
  logical change, merged back to `master` and deleted.
- Release branches are long-lived only as long as that version receives
  patch fixes; once superseded, leave them (history) but stop committing.

---

## 2. Versioning

Single source of truth: `fusion_trainer/_version.py` (`__version__`).
`pyproject.toml` reads it dynamically
(`version = {attr = "fusion_trainer._version.__version__"}`). Bump ONE place.

- **Patch** `1.0.0 → 1.0.1`: bug fixes only, every frozen surface identical.
- **Minor** `1.0.0 → 1.1.0`: new frozen surface added (new CLI option, new
  `TrainerConfig` field with default, new `RunManager` method), backward-
  compatible.
- **Major** `1.0.0 → 2.0.0`: a frozen surface broke (removed/renamed CLI
  subcommand or option, renamed field/method, changed signature).

---

## 3. Cutting a release

1. **Ensure `master` is green.** `pytest tests/` clean, `ruff check .` clean.
   For a minor/major release, run the real-model integration suite if a
   fusion-mlx is available:
   `FUSION_TRAINER_REAL=1 FUSION_MLX_ADMIN_TOKEN=<key> pytest -m integration -v`.
2. **Cut the release branch** from `master`:
   `git checkout master && git pull && git checkout -b release/vX.Y.Z`.
3. **Bump the version** on the release branch:
   edit `fusion_trainer/_version.py`, set `__version__ = "X.Y.Z"`.
4. **Add a CHANGELOG section** under `## [X.Y.Z] — YYYY-MM-DD` (move the
   `[Unreleased]` entries into it, or write new ones). Keep an empty
   `## [Unreleased]` above it for the next cycle.
5. **Update docs if needed** (README badges point at the new version;
   `API_STABILITY.md` "Frozen at" line for a major bump).
6. **Verify again on the release branch:**
   `pytest tests/ && ruff check .` clean.
7. **Commit + open a PR** for the release branch → `master` with the full
   review checklist (§4) in the PR description.
8. **On PR merge**, tag the merge commit on `master`:
   `git tag vX.Y.Z && git push origin vX.Y.Z`.
9. **Announce** the release (release notes from the CHANGELOG section).

---

## 4. Review checklist (release PR)

The release PR reviewer confirms:

- [ ] Version in `_version.py` matches the CHANGELOG section header and the
      git tag that will be cut.
- [ ] CHANGELOG `[X.Y.Z]` section present, dated, with an empty
      `[Unreleased]` left above it.
- [ ] SemVer bump is correct for the changes listed: patch = fixes only,
      minor = additions, major = a frozen-surface break.
- [ ] If major: `docs/API_STABILITY.md` "Frozen at" updated + a migration
      note for every broken surface.
- [ ] `pytest tests/` clean; `ruff check .` clean.
- [ ] Real-model integration suite run if a server was available (note the
      fusion-mlx version in the PR).
- [ ] README version badge updated.
- [ ] No `runtime_cache/` artifacts, no debug logging, no `TODO`/`FIXME`
      introduced in the release diff.

---

## 5. Rollback

A release is a git tag + a merge to `master`. To roll back:

1. **If the bad release was NOT yet tagged/merged:** delete the release
   branch, fix on `master`, re-cut. No rollback needed.
2. **If merged to `master` but not tagged:** revert the merge commit
   (`git revert -m 1 <merge-sha>`), open a PR, re-merge. `master` is clean
   again; no tag exists to retract.
3. **If tagged:** the tag is public history — do NOT delete it. Cut a new
   patch release (`vX.Y.(Z+1)`) that reverts the offending change, and mark
   the bad version as yanked in the CHANGELOG:
   `**[X.Y.Z] — YANKED** (reason: …). Use X.Y.(Z+1).`

Never rewrite or delete published tags. Never force-push `master`.

---

## 6. Hotfix flow

A critical bug on a released version that can't wait for the next scheduled
release:

1. `git checkout release/vX.Y.Z` (or cut `release/vX.Y.(Z+1)` from the tag).
2. Fix + bump to `X.Y.(Z+1)` + CHANGELOG patch section.
3. PR the release branch → `master` (cherry-pick the fix to `master` too if
   it's not already there).
4. Tag `vX.Y.(Z+1)` on merge.

---

## 7. What lives where

- `fusion_trainer/_version.py` — the one version source.
- `CHANGELOG.md` — human-readable release notes, one section per version.
- `docs/API_STABILITY.md` — the frozen surface + SemVer rules.
- `docs/RELEASE_PROCESS.md` — this file.
- Git tags `vX.Y.Z` — the immutable release points.
