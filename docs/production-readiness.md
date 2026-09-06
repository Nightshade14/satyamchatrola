# Portfolio production-readiness audit

Audited September 5–6, 2026 against the repository starting at `710caf8`.
The work preserves the Astro static site and existing GPU/compute visual design.
Changes are local; they have not been committed or deployed by this audit.

## Findings addressed

| Finding | Change |
| --- | --- |
| Hero identity/actions waited for the decorative scene, with a 3.4-second safety timeout. | Essential content renders immediately; WebGL initializes independently. |
| The static introduction was empty and statistics rendered as zero without scripts. | Real copy and final numbers are emitted in HTML. Reveal styling is enabled only after its observer is installed. |
| `semantic_cache_and_RAG` measured 260px inside a 233px text area at a 320px viewport. | Long titles wrap; grid columns can shrink, role text can wrap, and footer links can wrap. |
| Muted text had weak contrast and the animated background competed with hero copy. | Lightened the muted text token and strengthened the hero overlay. Added keyboard skip navigation, anchor clearance, and larger controls. |
| Streaming changed paragraph height as text arrived. | All response variants reserve their layout space; a separate accessible span announces complete responses. |
| Scene rendering continued throughout the page, including behind the nearly opaque content area. | Rendering stops outside the hero and in hidden documents. One resumable loop is capped at approximately 30 fps, independent of display refresh rate. |
| Reduced-motion visitors still downloaded and initialized Three.js, its model, lighting, and postprocessing. | Reduced motion and data-saving preferences prevent scene initialization. A CSS glow remains. A pause control also stops animations. |
| A preference/visibility change during asynchronous imports could still initiate heavy work. | Eligibility is checked after all imports, before renderer allocation or model/environment downloads. Twelve lifecycle tests cover those transitions and page exit. |
| Touch devices allocated bloom render targets and high-resolution full-screen buffers. | Touch/narrow viewports use one render pass with DPR at most 1; desktop is capped at 1.5 DPR. A two-million-pixel drawing-buffer budget limits large displays. |
| Decoder code came from a separate CDN; unused Meshopt code was loaded for a Draco-only model. | Decoder files come from the installed Three.js package and are served locally; one decoder worker is used and the unused loader is removed. |
| Environment/rendering resources were not fully disposed on exit. | Explicit disposal covers the renderer, materials, geometry, skeletons, textures, environment targets, postprocessing, and decoder workers. Back/forward-cache suspension preserves the scene. |
| The social preview referenced a 3,700,828-byte PNG. | A 72,361-byte JPEG is used for social previews, about 98% smaller. The original remains available as source material. |
| Canonical URL, `og:url`, and sitemap were missing. | They now point to the GitHub Pages project URL. The README corrects the repository name required for a root user site. |
| Six npm advisories affected the installed build toolchain. | Compatible updates advance Astro from 7.0.6 to 7.3.1 and update affected transitive dependencies. The final npm audit reported zero advisories. |
| Builds were deployed without repository regression checks. | Pull requests and deployments run a build, tests, and a dependency audit. High/critical advisories block deployment; jobs have bounded execution times. |
| The 71% latency statistic conflicted with its displayed timings. | Per the owner's decision, retain 71% and remove the conflicting timings from both the statistic and experience copy. Hero inference numbers are labeled as a demo. |

## Verification

- `npm run check`: production build and 19 passing tests.
- `npm audit --json`: zero vulnerabilities at final verification.
- `git diff --check`: no whitespace errors.
- Responsive checks in the Codex browser at 320, 390, 768, and 1280px: no page overflow or clipped project titles, hero role, focus headings, or footer rows.
- Real browser rendering: the pause control changes state; the WebGL draw counter remains unchanged while paused and after scrolling beyond the hero, then rendering resumes on return/resume.
- Generated temporary test pages with scripts removed, reduced motion simulated, data saving simulated, and WebGL unavailable: content remains readable. Reduced-motion/data-saving pages load no Three.js modules, model, HDR, or decoder and create no canvas.
- Resampling: paragraph height remained 204px before, during, and after a response change at the tested desktop width; accessible text updated to the full selected response.
- Keyboard: the first Tab reveals the skip link; activating it focuses the main content. Project navigation lands below the sticky header.
- Independent code review found an async initialization race and an incomplete print override; both were fixed and the reviewer verified the corrections.

Browser test fixtures were temporary and are excluded from the final rebuilt artifact. Automated tests do not require a browser or GPU. They verify the actual scene initialization script with mocked browser/module boundaries, plus the real scheduling helper and generated HTML.

## Remaining performance tradeoffs and limits

The animated desktop experience still needs Three.js, a roughly 0.89 MB GLB model, and a roughly 1.51 MB HDR environment before HTTP compression. The approximately 695 KB minified Three.js chunk still produces the build's >500 KB warning. It is dynamically loaded; artificially splitting it would not eliminate the bytes needed to display the scene. Replacing the interactive scene with a static poster would reduce normal-visitor downloads further, but would change the portfolio's main visual feature.

The image size reduction improves social-card fetches, not the normal page's largest contentful paint: that image is metadata, not a visible page image. No Lighthouse score, field Core Web Vitals result, physical-device battery benchmark, or full Safari/Firefox matrix is claimed. UI checks used the local production artifact; CI and public deployment must run after the changes are pushed.

The resource cleanup follows [Three.js's disposal guidance](https://threejs.org/manual/en/how-to-dispose-of-objects.html). Project-path deployment follows [Astro's GitHub Pages guidance](https://docs.astro.build/en/guides/deploy/github/).
