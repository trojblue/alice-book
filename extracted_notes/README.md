# HTML Exports for Notes

Serve HTML exports from the `html/` directory via GitHub Pages.

GitHub Pages
------------
- In repo settings → Pages, choose "Deploy from a branch", branch `main`, folder `/ (root)`.
- The root `index.html` reads from `manifest.json` (static) and will also try the GitHub API if the repo is public.
- To add more exports, drop them in `html/` and run `npm run build:manifest` (requires Node) which regenerates `manifest.json`.
- If you have asset folders (e.g., `something.assets` from Markdown/Quarto exports), place them in `content/` and run `npm run build:pages` to copy them into `html/`.
- The published site will live at `https://<your-username>.github.io/<this-repo>/` with links to `/html/<file>.html`.
