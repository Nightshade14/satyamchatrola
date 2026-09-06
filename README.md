# Satyam Chatrola — portfolio

Personal portfolio for **Satyam Chatrola**, AI Systems Engineer specializing in LLM inference & serving.

Live: **https://nightshade14.github.io/satyamchatrola/**

## Stack and rendering

- Astro static HTML on GitHub Pages, with a single scrolling page and the “compute night” dark theme.
- Self-hosted Latin variable fonts: Bricolage Grotesque, Geist, and Geist Mono.
- The name, introduction, statistics, and links are available without JavaScript. Scripts progressively add scroll reveals, response streaming, and pointer interactions.
- The decorative Three.js GPU scene loads while the hero is visible and the document is active. Reduced motion, data-saving mode, and a manual pause prevent initialization. These conditions are checked again after asynchronous imports.
- Scene rendering is capped at approximately 30 fps, stops outside the hero or in a hidden tab, and resumes on return. Touch/narrow viewports use one render pass and DPR at most 1; desktop DPR is capped at 1.5, with a two-million-pixel drawing-buffer budget.
- A **Pause animation** button stops the scene and CSS animations. Reduced motion uses a static CSS glow without downloading Three.js, the model, or the HDR environment. A missing model falls back to particles; unavailable WebGL leaves readable static content.
- Draco decoding uses one worker and files copied from the installed Three.js package to the same origin. No runtime decoder CDN is required. Scene resources are disposed on ordinary page exit; browser back/forward-cache suspension preserves them for resumption.
- Canonical/social URLs and the sitemap include the GitHub Pages project path. Social cards use a compressed JPEG rather than the original headshot PNG.

## Develop and verify

Use **Node.js 24** and the committed lockfile:

```bash
npm ci
npm run dev      # http://localhost:4321/satyamchatrola/
npm run check    # build static output, then run regression tests
npm run preview  # inspect dist/ in a browser
npm audit --audit-level=high
```

`npm run build` generates `dist/`. The `predev` and `prebuild` hooks run `scripts/prepare-assets.mjs` to copy the Draco decoder into ignored `public/draco/`; do not edit or commit that generated directory. `npm test` alone expects an existing production build.

Tests cover the generated HTML, local asset/anchor paths, canonical URLs, rendering frequency, stop/resume behavior, and eligibility changes during asynchronous scene loading. The lifecycle tests replace browser and Three.js boundaries; they complement manual browser checks rather than replace a real GPU/browser test.

Before publishing UI changes, check narrow phones (320px and 390px), tablet and desktop widths, keyboard focus, section links, resampling, and pause/resume. Confirm that the page remains readable if scripts or WebGL are unavailable. See [the production-readiness audit](docs/production-readiness.md) for findings and verification limits.

## Editing content

Copy and data live in `src/pages/index.astro`: the `RESPONSES`, `stats`, `focus`, `highlights`, `creds`, `projects`, `articles`, and `links` arrays. The latency claim is **71%**, without the previously inconsistent before/after timings. Hero inference numbers are explicitly labeled as a demo.

Design tokens and shared accessibility styles are in `src/styles/global.css`; font declarations are in `src/styles/fonts.css`. The 3D scene is in `src/components/GpuScene.astro`, with scheduling in `src/scripts/animation-loop.js`. Its model and HDR environment are under `public/models/` and `public/hdri/`. PDFs live under `public/pdf/`.

`public/images/satyam_headshot.png` is the original image. To regenerate the optimized social image after changing it:

```bash
node --input-type=module -e 'import sharp from "sharp"; await sharp("public/images/satyam_headshot.png").rotate().resize({width:1200,height:1200,fit:"inside",withoutEnlargement:true}).jpeg({quality:82,mozjpeg:true}).toFile("public/images/satyam-social.jpg");'
```

## Deployment

Pull requests run `.github/workflows/check.yml` with read-only repository permissions. Pushing to `main` runs `.github/workflows/deploy.yml`, which installs from the lockfile, builds, tests, audits dependencies, and publishes the static artifact. High/critical dependency advisories block publication. Build and deployment jobs each have a ten-minute timeout.

In **Settings → Pages**, set **Source** to **GitHub Actions**. The current project site uses `site: 'https://nightshade14.github.io'` and `base: '/satyamchatrola'` in `astro.config.mjs`.

A root user site requires a repository named **`Nightshade14.github.io`**, with `base: '/'`. If changing the domain or base, update the expected published URL in `tests/build.test.mjs` and verify generated links, social metadata, and `sitemap.xml` before deployment. This distinction follows the [Astro GitHub Pages guide](https://docs.astro.build/en/guides/deploy/github/).
