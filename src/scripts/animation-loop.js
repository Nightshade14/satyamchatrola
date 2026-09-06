// Display scheduling is separate from the scene so visibility and pause events
// can start/stop the same loop without allocating another renderer.
export function createFrameLoop(render, {
  requestFrame = requestAnimationFrame,
  cancelFrame = cancelAnimationFrame,
} = {}) {
  let active = false;
  let frameId = null;
  let lastFrame = null;
  const interval = 1000 / 30;
  function tick(now) {
    frameId = null;
    if (!active) return;
    if (lastFrame === null || now - lastFrame >= interval - 0.5) {
      lastFrame = now;
      render(now);
    }
    if (active) frameId = requestFrame(tick);
  }
  return {
    start() {
      if (active) return;
      active = true;
      lastFrame = null;
      frameId = requestFrame(tick);
    },
    stop() {
      active = false;
      if (frameId !== null) cancelFrame(frameId);
      frameId = null;
    },
  };
}
