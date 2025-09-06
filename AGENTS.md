````md
# AGENTS.md

This document tells the Copilot coding agent **how to work in this repository**: environment setup, build & test, submodule handling, and PR workflow. It is aligned with `.github/copilot-instructions.md`.

---

## 0) Ground rules

- **Python**: Use **Python 3.10**.
- **Install (root)**: Prefer `pip install -e .`.
- **Do not install (IMM extras)**: **Do not** run `pip install -e tools/image-matching_models[all]` unless strictly necessary (PEP 517/build-isolation can cause issues).
- **Git LFS**: Assume **LFS is OFF** unless a task explicitly needs LFS assets.
- **Submodules (detached HEAD)**: Do **not** `git pull` while a submodule is in detached HEAD; either use `git submodule update --init --recursive` or check out a branch **inside** the submodule first.
- **System libs (only if needed)**: If OpenCV/Open3D complain about shared libraries, install `libgl1` and `libglib2.0-0`.

---

## 1) Repository layout

- Top repo: **colon_matching** (this repo).
- Submodule: **tools/image-matching_models** (IMM).
  - IMM contains **nested submodules**; two are forks we maintain (see §5).
- Tests live under **`tests/`** (unit + integration).
  - Pytest is configured to collect only from `tests/` (not from `third_party` trees).

---

## 2) Environment & build

1. Sync and initialize submodules (top → down):
   ```bash
   git submodule sync --recursive
   git submodule update --init --recursive
````

2. Install root package (editable):

   ```bash
   python -m pip install -U pip setuptools wheel
   pip install -e .
   ```
---

## 3) Tests

* **Unit tests**: `tests/unit/**`
* **Integration tests**: `tests/integration/**`

Run all tests (quiet):

```bash
pytest -q
```

Run suites separately when needed:

```bash
pytest tests/unit
pytest tests/integration
```

**Policy**: If any tests fail, **stop, fix, and re-run** before updating or opening a PR.

---

## 4) Code style, typing, docs

* **Style**: Black (line length 88) + isort. Prefer early returns and expressive `snake_case` names.
* **Typing**: Add Python type hints on public functions/classes.
* **Docstrings**: Use **Google-style** docstrings with `Args:` and `Returns:`, focused on **why** as well as what.

Example:

```python
def example(x: int) -> str:
    """Summarize purpose and result.

    Args:
        x: Meaningful description.

    Returns:
        Description of the returned value.
    """
```

---

## 5) Submodule workflow (critical)

**Our forks** (maintained by us):

* `tools/image-matching_models` — [https://github.com/ipastore/image-matching\_models](https://github.com/ipastore/image-matching_models)
* `tools/image-matching_models/third_party/LightGlue_mask` — [https://github.com/ipastore/LightGlue\_mask](https://github.com/ipastore/LightGlue_mask)
* `tools/image-matching_models/third_party/RoMa_mask` — [https://github.com/ipastore/RoMa\_mask](https://github.com/ipastore/RoMa_mask)

### 5.1 Updating submodules to the latest commit on their `main` branches

```bash
cd tools/image-matching_models
git switch main
git pull origin main

cd third_party/LightGlue_mask
git switch main
git pull origin main

cd ../RoMa_mask
git switch main
git pull origin main

cd ../../../..
git add tools/image-matching_models
git commit -m "Update image-matching_models submodule"
git push origin <your-branch>
```

> If a conflict arises, prefer **rebasing** your branch over merging.

### 5.2 Editing a leaf submodule safely

1. **If the leaf is not our fork and needs edits**: fork it first, update `.gitmodules` to point to the fork, then:

   ```bash
   git submodule sync --recursive
   git submodule update --init --recursive
   ```
2. **Implement inside the leaf**:

   * Enter the leaf submodule, check out a branch (avoid detached HEAD), make changes, commit, and **push to the leaf remote**.
3. **Update parent pointer(s)**:

   * From the immediate parent repo, stage the leaf path and commit the new pointer:

     ```bash
     git add <path-to-leaf>
     git commit -m "Update submodule pointer: <leaf> to <new SHA>"
     ```
   * Repeat upward (e.g., from IMM up to **colon\_matching**) until the top repo records the new SHAs.
4. **Re-run tests at the top**:

   ```bash
   pytest -q
   ```

Never leave parent and child pointers out of sync.

---

## 6) Branching, rebasing, PRs

* Work on a **feature branch**; **do not** push directly to `main`.
* **Rebase** your branch onto the latest `main` before merging:

  ```bash
  git fetch origin
  git rebase origin/main
  ```
* **Squash** trivial/iterative commits before merging.
* **PR checklist**:

  * Include a short **What / Why / Test** section.
  * Ensure CI checks (tests) are **green**.
  * For breaking changes, update README/docs.
  * For dependency changes, update packaging/requirements.

After merging, **delete the feature branch**.

---

## 7) Agent task flow (step-by-step)

1. **Plan**

   * Identify the components to change and whether they reside in a leaf submodule.
2. **Prepare**

   * Confirm Python 3.10.
   * Run submodule sync/update (top → down).
   * Install the root package (`pip install -e .`). Only install IMM extras if required by the task.
3. **Locate or scaffold tests**

   * Search `tests/` for related unit/integration tests.
   * If none exist for the changed surface area, **scaffold minimal tests** first.
4. **Implement**

   * If editing a leaf: create a branch **inside that leaf**, implement, commit, and push.
   * Update parent pointer(s) as described in §5.2.
5. **Validate**

   * Run `pytest tests/unit -q`, then `pytest tests/integration -q`, then `pytest -q` from repo root.
6. **Prepare PR**

   * Provide **What / Why / Test**, note decisions/assumptions, and keep scope tight (split large changes into smaller PRs).
7. **Safety rails**

   * Do not fetch LFS assets unless the task requires them.
   * Avoid modifying CI/secrets/branch protection unless explicitly requested.
   * Prefer prebuilt wheels over source builds for heavy extensions when possible.
