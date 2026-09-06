// @ts-check
import { defineConfig } from 'astro/config';

// Project page: served at https://nightshade14.github.io/satyamchatrola
// A user page requires a repo named `Nightshade14.github.io` and base: '/'.
export default defineConfig({
  site: 'https://nightshade14.github.io',
  base: '/satyamchatrola',
  trailingSlash: 'ignore',
  build: {
    inlineStylesheets: 'auto',
  },
});
