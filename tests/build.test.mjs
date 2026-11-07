import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import test from 'node:test';

const read = (file) => readFile(new URL(`../dist/${file}`, import.meta.url), 'utf8');
const html = await read('index.html');
const notFound = await read('404.html');
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

test('the social preview image is published with the site', async () => {
  const image = attr(tags.find(tag => attr(tag, 'property') === 'og:image') ?? '', 'content');
  const prefix = 'https://nightshade14.github.io/satyamchatrola/';
  assert.ok(image?.startsWith(prefix), `og:image ${image} is outside the site`);
  assert.ok((await stat(new URL(`../dist/${image.slice(prefix.length)}`, import.meta.url))).isFile(), image);
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

test('pages have no placeholder text, empty links, or third party scripts and styles', () => {
  for (const page of [html, notFound]) {
    assert.ok(!page.includes('{{'), 'unfilled placeholder');
    assert.ok(!page.includes('href="#"'), 'link with no target');
    assert.doesNotMatch(page, /<script\b[^>]*\bsrc="https?:/, 'external script');
    assert.doesNotMatch(page, /<link\b[^>]*\brel="stylesheet"[^>]*\bhref="https?:/, 'external stylesheet');
  }
});

test('the 404 page links back to the portfolio', () => {
  assert.match(notFound, /<a\b[^>]*href="\/satyamchatrola\/"/);
});
