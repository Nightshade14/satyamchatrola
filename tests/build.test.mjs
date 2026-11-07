import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import test from 'node:test';

const read = (file) => readFile(new URL(`../dist/${file}`, import.meta.url), 'utf8');
const html = await read('index.html');
const tags = [...html.matchAll(/<[^!][^>]*>/g)].map(([tag]) => tag);
const attr = (tag, name) => tag.match(new RegExp(`\\b${name}="([^"]*)"`))?.[1];

test('the headline is readable before JavaScript runs', () => {
  assert.match(html, /<h1\b[^>]*>I make LLMs fast,<br>reliable, and measured\.<\/h1>/);
});

test('statistics show their final values in the static page', () => {
  const values = [...html.matchAll(/<div class="num">([^<]*)<\/div>/g)].map(([, value]) => value);
  assert.deepEqual(values, ['71%', '355 MB', '5M+', '+34%']);
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

test('the page has no placeholder text, empty links, or third party scripts and styles', () => {
  assert.ok(!html.includes('{{'), 'unfilled placeholder');
  assert.ok(!html.includes('href="#"'), 'link with no target');
  assert.doesNotMatch(html, /<script\b[^>]*\bsrc="https?:/, 'external script');
  assert.doesNotMatch(html, /<link\b[^>]*\brel="stylesheet"[^>]*\bhref="https?:/, 'external stylesheet');
});
