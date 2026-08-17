import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = resolve(root, "dist");
const mediaPipeDir = resolve(root, "node_modules/@mediapipe/tasks-vision");
const mediaPipeDist = resolve(dist, "node_modules/@mediapipe/tasks-vision");

rmSync(dist, { recursive: true, force: true });
mkdirSync(mediaPipeDist, { recursive: true });

for (const file of ["index.html", "styles.css", "app.js"]) {
  cpSync(resolve(root, file), resolve(dist, file));
}

cpSync(resolve(root, "models"), resolve(dist, "models"), { recursive: true });
cpSync(resolve(mediaPipeDir, "vision_bundle.mjs"), resolve(mediaPipeDist, "vision_bundle.mjs"));
cpSync(resolve(mediaPipeDir, "wasm"), resolve(mediaPipeDist, "wasm"), { recursive: true });

console.log("Built static site in dist/");
