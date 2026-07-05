# satyamchatrola.dev — portfolio

Personal portfolio for **Satyam Chatrola**, AI Systems Engineer specializing in LLM inference & serving.

Live: **https://nightshade14.github.io/satyamchatrola**

## Stack

- **[Astro](https://astro.build)** — static, zero-JS-by-default output (fast under any condition)
- Self-hosted fonts via `@fontsource-variable` (Space Grotesk · Geist · Geist Mono)
- Single scrolling page, dark theme with one electric-cyan accent
- One small client script drives the hero "decode" animation + scroll reveals; everything else is static HTML/CSS
- Deployed to **GitHub Pages via GitHub Actions**

## Develop

```bash
npm install     # once
npm run dev      # local dev server (http://localhost:4321/satyamchatrola)
npm run build    # production build → dist/
npm run preview  # preview the built site
```

## Editing content

All copy and data live in `src/pages/index.astro` (top of the file): the hero response,
`focus`, `highlights`, `creds`, `projects`, and `links` arrays. Design tokens (colors, type
scale, spacing) are in `src/styles/global.css`. Static assets (headshot, résumé, research PDF)
are in `public/`.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds with Astro and publishes
to GitHub Pages. **One-time setup:** in the repo's **Settings → Pages**, set **Source** to
**GitHub Actions**.

If the repo is ever renamed to `Nightshade14` (a user page at `nightshade14.github.io`), change
`base` in `astro.config.mjs` from `'/satyamchatrola'` to `'/'`.
