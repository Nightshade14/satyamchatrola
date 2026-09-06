import assert from 'node:assert/strict';
import test from 'node:test';
import { createFrameLoop } from '../src/scripts/animation-loop.js';

// A deterministic display clock: callbacks only fire when the display ticks.
function display() {
  const pending = new Map();
  let id = 0;
  return {
    requestFrame: fn => { pending.set(++id, fn); return id; },
    cancelFrame: id => pending.delete(id),
    tick(time) { const batch = [...pending.values()]; pending.clear(); batch.forEach(fn => fn(time)); },
    get pending() { return pending.size; },
  };
}

test('a high-refresh display does not increase the rendering budget', () => {
  const clock = display();
  let renders = 0;
  const loop = createFrameLoop(() => renders++, clock);
  loop.start();
  for (let i = 0; i < 120; i++) clock.tick(i * 1000 / 120);
  assert.ok(renders >= 29 && renders <= 31, `expected about 30 renders, got ${renders}`);
  loop.stop();
  assert.equal(clock.pending, 0);
});

test('offscreen or paused scenes stop scheduling and resume without duplicate loops', () => {
  const clock = display();
  let renders = 0;
  const loop = createFrameLoop(() => renders++, clock);
  loop.start(); loop.start();
  assert.equal(clock.pending, 1);
  clock.tick(0);
  loop.stop();
  clock.tick(1000);
  assert.equal(renders, 1);
  assert.equal(clock.pending, 0);
  loop.start(); loop.start();
  clock.tick(2000);
  assert.equal(renders, 2);
  assert.equal(clock.pending, 1);
});

test('stopping during a frame does not leave a callback running', () => {
  const clock = display();
  const loop = createFrameLoop(() => loop.stop(), clock);
  loop.start();
  clock.tick(0);
  assert.equal(clock.pending, 0);
});
