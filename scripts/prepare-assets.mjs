import { copyFile, mkdir } from 'node:fs/promises';

// Serve the decoder from the same origin, using the version bundled with Three.
const source = new URL('../node_modules/three/examples/jsm/libs/draco/', import.meta.url);
const target = new URL('../public/draco/', import.meta.url);
await mkdir(target, { recursive: true });
await Promise.all(['draco_wasm_wrapper.js', 'draco_decoder.wasm', 'draco_decoder.js'].map(
  name => copyFile(new URL(`gltf/${name}`, source), new URL(name, target)),
));
await copyFile(new URL('README.md', source), new URL('README.md', target));
