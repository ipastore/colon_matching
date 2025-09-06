<!-- .github/copilot-instructions.md -->

# Repository Code Standards & Workflow
- Always use **Python 3.10** for local examples and CI steps.
- When installing, prefer `pip install -e .` 
- If it is not necessary, do not install `pip install -e tools/image-matching_models[all]` because sometimes there is an incompatibility with PEP517 and isolation builds.
- Assume **Git LFS is OFF** unless explicitly enabled.


##  CI Steps
- Every PR must include passing tests:
  - Run `pytest -q`.
  - Code must build and import correctly.

##  Testing
- Add unit tests in `tests/unit/`.
- Name tests `test_*.py`.
- For key functionality, add integration tests under `tests/integration/`.

##  Docstrings & Typing
- Write **Google-style docstrings** for all modules, functions, and classes. Include `Args:` and `Returns:`.
- Use snake_case for functions/variables, CamelCase for classes.
- Annotate public functions with type hints whenever practical.
- Style: Use Black (line length 88) + isort. Prefer early returns and expressive `snake_case` names.

##  Submodule Workflow
- Image Matching Models (tools/image-matching_models) uses nested submodules; two are your forks.
- After making changes:
  1. Commit & push in the leaf.
  2. In the parent repo, run `git add <leaf>`, commit, and push.
- There are 3 submodules that are my forks:
  - `tools/image-matching_models` (https://github.com/ipastore/image-matching_models)
  - `tools/image-matching_models/third_party/LightGlue_mask` (https://github.com/ipastore/LightGlue_mask)
  - `tools/image-matching_models/third_party/RoMa_mask` (https://github.com/ipastore/RoMa_mask)
- To update submodules to latest remote commit:
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
  If there is a merge conflict, try to rebase instead of merging.

##  Naming & Comments
- Use underscores and descriptive variable names (`image_pairs`, not `ip`).
- Add inline comments sparingly to explain *why*, not *what*.

##  Type Hints
- Use type annotations:
  ```python
  def foo(bar: int) -> str: ...
  ```

## Pull Requests
- PRs must include a short “What/Why/Test” section; do not push to `main` directly.
- Ensure PRs pass CI checks before merging.
- Use descriptive commit messages; squash minor commits before merging.
- After merging, delete the feature branch.
- If a PR introduces breaking changes, update the README and relevant documentation.
- Use GitHub's code review tools to request reviews and address feedback.
- For any changes affecting dependencies, update `requirements.txt` or `setup.py` accordingly.
