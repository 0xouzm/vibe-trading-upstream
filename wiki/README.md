# Vibe-Trading Wiki

Static source for `https://vibetrading.wiki`.

## Local preview

```bash
cd wiki
python3 -m http.server 8088
```

Open `http://localhost:8088/home/` for the landing page and these wiki sections:

- `http://localhost:8088/docs/`
- `http://localhost:8088/tutorials/`
- `http://localhost:8088/alpha-library/`
- `http://localhost:8088/research-lab/`

Direct docs URLs such as `/docs/latest/getting-started/vibe-trading-overview` are handled by Cloudflare Pages via `_redirects`. The simple Python preview server does not apply those rewrite rules, so use `/docs/` as the local entry point.

## Docs content

The docs are one page shell in two languages. English pages live in
`docs/content.en.js` and are served at `/docs/latest/<page>`; Chinese pages live
in `docs/content.zh.js` and are served at `/docs/zh/<page>`. The header toggle
switches language and remembers the choice; a first visit from a Chinese browser
opens the Chinese pages. Both files must list the same pages and sections in the
same order — CI runs:

```bash
node wiki/scripts/check_docs_parity.mjs
```

`_headers` lets browsers cache `.js` and `.css` for an hour, so the docs load
their modules with a `?v=<date>` query (`docs/index.html` → `docs/main.js` →
`docs/content.js` → the two language files). Bump it in all five places when
any of those files change, or a returning visitor can combine a new module with
a cached old one and get a blank page.

Counts on the docs pages (data sources, engines, skills, presets, MCP tools,
brokers, alphas) are measured from the code of the version in
`docs/content.js`; re-measure them for a new release rather than editing one
number.

## Cloudflare Pages

- Project root: `wiki`
- Build command: leave empty
- Output directory: `.`
- Custom domain: `vibetrading.wiki`

The site is static and needs no build step. The only dynamic piece is a small
analytics layer in `functions/` (Cloudflare Pages Functions): `_middleware.js`
classifies each page request as AI-agent / bot / human by User-Agent and counts
it into a D1 database (`vibetrading-analytics`, binding `DB`), and
`api/stats.js` serves the footer's aggregate counts alongside public PyPI and
GitHub numbers. The counter is anonymous and first-party — no cookies, no
per-visitor identifier, no IP retention.
