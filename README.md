# Satyam Chatrola — portfolio

Personal portfolio for **Satyam Chatrola**, AI Systems Engineer specializing in LLM inference, reliability, and evals.

Live: **https://nightshade14.github.io/satyamchatrola/**

## Stack and rendering

- Astro static HTML on GitHub Pages: one scrolling page in a light "blue engraving" style, plus a branded `404.html`.
- Self hosted Latin variable fonts: Geist and Geist Mono. Icons are Phosphor SVGs inlined at build time by `src/components/Icon.astro`. Nothing loads from a CDN.
- All illustrations are inline SVG engravings in `src/components/art/`. Their hatch patterns and the star symbol are defined once in `Defs.astro`.
- All text, statistics, and links are in the static HTML. Scripts only add scroll reveals, the mobile menu, the current section in the nav, the tagline word reveal, and the hero ticker tape.
- Reduced motion turns off reveals, the tagline animation, and the ticker tape. The ticker tape also pauses off screen and in hidden tabs.
- Canonical and social URLs and the sitemap include the GitHub Pages project path.

## Develop and verify

Use **Node.js 24.22.1 or newer within 24.x** (the first 24.x release with the October 2026 security fixes) and the committed lockfile. `.npmrc` sets `engine-strict=true`, so `npm ci` refuses an older Node.js, and CI installs the latest 24.x on every run:

```bash
npm ci
npm run dev      # http://localhost:4321/satyamchatrola/
npm run check    # build static output, then run regression tests
npm run preview  # inspect dist/ in a browser
npm audit --audit-level=high
```

`npm run build` generates `dist/`. `npm test` alone expects an existing production build.

Tests check the generated HTML: the headline and statistics without JavaScript, canonical URLs, local links and anchors, no placeholder text or empty links, no third party scripts or stylesheets, and the 404 page.

Before publishing UI changes, check narrow phones (320px and 390px), tablet and desktop widths, keyboard focus, the mobile menu, and section links. Confirm that the page stays readable with scripts off and with reduced motion. [The production readiness audit](docs/production-readiness.md) describes the earlier 3D design and is kept for history.

## Editing content

Copy and data live at the top of `src/pages/index.astro`: the `proof`, `stats`, `pillars`, `skills`, `highlights`, `featured`, `projects`, `articles`, and `socials` arrays. The latency claim is **71%**.

Design tokens (colors, spacing, easing) and named text styles are in `src/styles/global.css`; font declarations are in `src/styles/fonts.css`. PDFs live under `public/pdf/`. `public/images/satyam-avatar.jpg` is the 112px portrait shown beside the name in the hero.

`public/images/satyam-social-card.jpg` is the link preview image (`og:image`). It is a 1200 × 627 card with the hero headline, the three focus areas, and the engraved GPU drawing, with no photo. Update it when the headline changes.

## Deployment

Pull requests run `.github/workflows/check.yml` with read-only repository permissions. Pushing to `main` runs `.github/workflows/deploy.yml`, which installs from the lockfile, builds, tests, audits dependencies, and publishes the static artifact. High/critical dependency advisories block publication. Build and deployment jobs each have a ten-minute timeout.

In **Settings → Pages**, set **Source** to **GitHub Actions**. The current project site uses `site: 'https://nightshade14.github.io'` and `base: '/satyamchatrola'` in `astro.config.mjs`.

A root user site requires a repository named **`Nightshade14.github.io`**, with `base: '/'`. If changing the domain or base, update the expected published URL in `tests/build.test.mjs` and verify generated links, social metadata, and `sitemap.xml` before deployment. This distinction follows the [Astro GitHub Pages guide](https://docs.astro.build/en/guides/deploy/github/).
