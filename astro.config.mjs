// @ts-check
import { defineConfig } from 'astro/config';

// Project page: served at https://nightshade14.github.io/satyamchatrola
// If you later rename the repo to `Nightshade14` (a user page), set base to '/'.
export default defineConfig({
  site: 'https://nightshade14.github.io',
  base: '/satyamchatrola',
  trailingSlash: 'ignore',
  build: {
    inlineStylesheets: 'auto',
  },
});
