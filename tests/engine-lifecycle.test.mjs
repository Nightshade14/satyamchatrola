import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

// Execute the actual Astro scene script. Mock browser APIs and asynchronous
// Three module boundaries to reproduce eligibility changes during loading.
const scenePath = new URL('../src/components/GpuScene.astro', import.meta.url);
const source = fs.readFileSync(scenePath, 'utf8').split('<script>')[1].split('</script>')[0]
  .replace(/import \{ createFrameLoop \}[^;]+;/, '')
  .replace(/import\.meta\.env\.BASE_URL/g, "'/satyamchatrola/'")
  .replace(/import\(/g, 'fakeImport(');
const flush = () => new Promise(resolve => setImmediate(resolve));

function harness(initial = {}) {
  const state = { reduced: false, saveData: false, hidden: false, paused: false, offscreen: false, ...initial };
  const imports = [], requests = [], idle = [];
  let rendererCount = 0, observe;
  let releasePost;
  const postGate = new Promise(resolve => { releasePost = resolve; });
  function eventSurface() {
    const listeners = new Map();
    return {
      addEventListener(type, listener) {
        const entries = listeners.get(type) || [];
        entries.push(listener);
        listeners.set(type, entries);
      },
      emit(type, event = {}) { for (const listener of listeners.get(type) || []) listener(event); },
    };
  }
  const classes = new Set();
  const stage = {
    classList: {
      contains: key => classes.has(key),
      add: (...keys) => keys.forEach(key => classes.add(key)),
      remove: key => classes.delete(key),
    },
    appendChild() {},
  };
  const reduced = { ...eventSurface(), get matches() { return state.reduced; } };
  const connection = { ...eventSurface(), get saveData() { return state.saveData; } };
  const document = {
    ...eventSurface(), get hidden() { return state.hidden; },
    getElementById: () => stage,
    querySelector: () => ({ getBoundingClientRect: () => ({ bottom: state.offscreen ? -10 : 100 }) }),
    documentElement: { classList: { contains: key => key === 'motion-paused' && state.paused } },
  };
  class IntersectionObserver { constructor(callback) { observe = callback; } observe() {} disconnect() {} }
  const window = {
    ...eventSurface(), innerWidth: 1200, innerHeight: 800, devicePixelRatio: 1, IntersectionObserver,
    matchMedia: query => query.includes('reduced-motion') ? reduced : { matches: query.includes('fine'), addEventListener() {} },
    requestIdleCallback: callback => idle.push(callback),
  };
  class Thing {
    constructor() { this.position = { set() {} }; this.rotation = { set() {} }; this.scale = { setScalar() {} }; }
    add() {} traverse() {} dispose() {}
  }
  class Renderer {
    constructor() { rendererCount++; this.domElement = { classList: stage.classList, remove() {} }; }
    getContext() { return {}; } setPixelRatio() {} setSize() {} setClearColor() {} dispose() {}
  }
  class Composer {
    constructor() { this.passes = []; }
    setPixelRatio() {} setSize() {} addPass(pass) { this.passes.push(pass); } dispose() {}
  }
  class Loader { setDRACOLoader() {} load(url) { requests.push(url); } }
  class Draco { setDecoderPath() {} setWorkerLimit() {} dispose() {} }
  const three = {
    WebGLRenderer: Renderer, Scene: Thing, PerspectiveCamera: class extends Thing { lookAt() {} },
    PMREMGenerator: Thing, HemisphereLight: Thing, DirectionalLight: Thing, PointLight: Thing,
    Group: Thing, Vector2: Thing, Clock: class { getDelta() { return 0; } },
  };
  const modules = { GLTFLoader: Loader, DRACOLoader: Draco, HDRLoader: Loader, EffectComposer: Composer, RenderPass: Thing, UnrealBloomPass: Thing, OutputPass: Thing };
  vm.runInNewContext(source, {
    document, navigator: { connection }, window, IntersectionObserver, AbortController, setTimeout,
    createFrameLoop: () => ({ start() {}, stop() {} }),
    fakeImport: async name => {
      imports.push(name);
      if (name === 'three') return three;
      if (name.includes('postprocessing')) await postGate;
      const key = name.split('/').at(-1).replace('.js', '');
      return { [key]: modules[key] };
    },
  });
  return {
    state, imports, requests, classes, window, releasePost,
    runIdle() { const tasks = idle.splice(0); tasks.forEach(fn => fn()); },
    setOffscreen(value) { state.offscreen = value; observe([{ isIntersecting: !value }]); },
    get rendererCount() { return rendererCount; },
  };
}

for (const key of ['reduced', 'saveData', 'hidden', 'paused', 'offscreen']) {
  test(`initial ${key} skips every Three import and renderer allocation`, async () => {
    const scene = harness({ [key]: true });
    scene.runIdle(); await flush();
    assert.equal(scene.imports.length, 0);
    assert.equal(scene.rendererCount, 0);
    assert.equal(scene.requests.length, 0);
  });
}

test('pausing before the idle callback prevents imports', async () => {
  const scene = harness();
  scene.state.paused = true;
  scene.runIdle(); await flush();
  assert.equal(scene.imports.length, 0);
});

for (const key of ['reduced', 'saveData', 'hidden', 'paused', 'offscreen']) {
  test(`${key} during imports prevents allocation and permits later activation`, async () => {
    const scene = harness();
    scene.runIdle(); await flush();
    assert.ok(scene.imports.some(name => name.includes('postprocessing')));
    if (key === 'offscreen') scene.setOffscreen(true);
    else scene.state[key] = true;
    scene.releasePost(); await flush();
    assert.equal(scene.rendererCount, 0);
    assert.equal(scene.requests.length, 0);
    assert.ok(!scene.classes.has('engine-loading'));
    scene.state[key] = false;
    if (key === 'offscreen') scene.setOffscreen(false);
    else scene.window.emit('portfolio:motion');
    scene.runIdle(); await flush();
    assert.equal(scene.rendererCount, 1);
    assert.deepEqual(scene.requests, ['/satyamchatrola/models/geforce_rtx_3080_graphics_card.glb', '/satyamchatrola/hdri/studio.hdr']);
    assert.ok(!scene.classes.has('engine-failed'), 'successful initialization must not throw');
  });
}

test('pagehide during imports prevents late renderer and model creation', async () => {
  const scene = harness();
  scene.runIdle(); await flush();
  scene.window.emit('pagehide', { persisted: false });
  scene.releasePost(); await flush();
  assert.equal(scene.rendererCount, 0);
  assert.equal(scene.requests.length, 0);
});
