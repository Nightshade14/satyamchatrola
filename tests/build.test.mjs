import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import test from 'node:test';

const html = await readFile(new URL('../dist/index.html', import.meta.url), 'utf8');
const tags = [...html.matchAll(/<[^!][^>]*>/g)].map(([tag]) => tag);
const attr = (tag, name) => tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1];

test('the introduction is readable before JavaScript runs', () => {
  const text = html.match(/<span\b[^>]*\bid="decode"[^>]*>([\s\S]*?)<\/span>/)?.[1];
  assert.ok(text?.includes('AI Systems Engineer.'), 'the server-rendered introduction must not be empty');
});

test('statistics contain their final values in the static page', () => {
  const counters = [...html.matchAll(/<span\b([^>]*\bdata-count="[^"]+"[^>]*)>([^<]*)<\/span>/g)];
  assert.equal(counters.length, 4);
  for (const [, attrs, text] of counters) {
    assert.equal(Number(text.replace(/[^\d.]/g, '')), Number(attr(attrs, 'data-count')));
  }
});

test('the canonical and social URLs use the GitHub Pages project path', () => {
  const canonical = tags.find(tag => attr(tag, 'rel') === 'canonical');
  const social = tags.find(tag => attr(tag, 'property') === 'og:url');
  assert.equal(attr(canonical ?? '', 'href'), 'https://nightshade14.github.io/satyamchatrola/');
  assert.equal(attr(social ?? '', 'content'), 'https://nightshade14.github.io/satyamchatrola/');
});

test('local links, scripts and styles resolve inside the published artifact', async () => {
  const ids = new Set(tags.map(tag => attr(tag, 'id')).filter(Boolean));
  const urls = new Set(tags.flatMap(tag => [attr(tag, 'href'), attr(tag, 'src')]).filter(Boolean));
  for (const url of urls) {
    if (url.startsWith('#')) {
      assert.ok(ids.has(url.slice(1)), `missing anchor ${url}`);
    } else if (url.startsWith('/')) {
      assert.ok(url.startsWith('/satyamchatrola/'), `missing project base in ${url}`);
      const relative = url.slice('/satyamchatrola/'.length) || 'index.html';
      assert.ok((await stat(new URL(`../dist/${relative}`, import.meta.url))).isFile(), url);
    }
  }
});
