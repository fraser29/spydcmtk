# Sphinx Documentation

gh-pages is set up to automatically build the documentation from the `main` branch and deploy it to the `gh-pages` branch.

## Updating Documentation from Main Branch

To update the documentation with changes from the main branch and rebuild:

1. Ensure you are on the `main` branch:
   ```bash
   git checkout main
   ```

2. [Optional] Merge changes from development branches:
   ```bash
   git merge my-dev
   ```

3. Rebuild the Sphinx documentation:
   ```bash
   cd sphinx
   make clean
   make html
   ```

4. Preview the documentation by opening `sphinx/_build/html/index.html` in your web browser.

5. If everything looks correct, commit the changes:
   ```bash
   git add .
   git commit -m "Update documentation from main branch"
   git push 
   ```

Note: Make sure you have all required Sphinx dependencies installed before rebuilding the documentation.
