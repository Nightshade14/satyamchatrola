# satyamchatrola.dev — portfolio

Personal portfolio for **Satyam Chatrola**, AI Systems Engineer specializing in LLM inference & serving.

Live: **https://nightshade14.github.io/satyamchatrola**

## Stack

- **[Astro](https://astro.build)** — static, zero-JS-by-default output (fast under any condition)
- Self-hosted fonts via `@fontsource-variable` (Bricolage Grotesque · Geist · Geist Mono)
- Single scrolling page, "compute night" dark theme with one electric-cyan accent
- **Full-screen GPU "engine"** ([three.js](https://threejs.org), lazy-loaded): a fixed WebGL backdrop
  the page boots on — scrolling dollies the camera back and crossfades it to a faint always-running
  presence. Real `.glb` (Draco/Meshopt) lit by an HDRI studio env; degrades to a particle field
  if WebGL/the model is unavailable, and holds static under `prefers-reduced-motion`
- Client scripts drive the boot crossfade, the hero "decode" stream, and scroll reveals; the rest is static HTML/CSS
- Deployed to **GitHub Pages via GitHub Actions**

## Develop

```bash
npm install     # once
npm run dev      # local dev server (http://localhost:4321/satyamchatrola)
npm run build    # production build → dist/
npm run preview  # preview the built site
```

## Editing content

All copy and data live in `src/pages/index.astro` (top of the file): the `RESPONSES` decode pool,
`stats`, `focus`, `highlights`, `creds`, `projects`, and `links` arrays. Design tokens (colors, type
scale, spacing) are in `src/styles/global.css`. The 3D engine lives in `src/components/GpuScene.astro`
(model at `public/models/*.glb`, HDRI at `public/hdri/*.hdr`). Static assets (résumé, research PDF)
are in `public/`.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds with Astro and publishes
to GitHub Pages. **One-time setup:** in the repo's **Settings → Pages**, set **Source** to
**GitHub Actions**.

If the repo is ever renamed to `Nightshade14` (a user page at `nightshade14.github.io`), change
`base` in `astro.config.mjs` from `'/satyamchatrola'` to `'/'`.
