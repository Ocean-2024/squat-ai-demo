import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = resolve(root, "dist");
const mediaPipeDir = resolve(root, "node_modules/@mediapipe/tasks-vision");
const vendorDist = resolve(dist, "vendor/@mediapipe/tasks-vision");

rmSync(dist, { recursive: true, force: true });
mkdirSync(vendorDist, { recursive: true });

for (const file of ["index.html", "styles.css", "app.js"]) {
  cpSync(resolve(root, file), resolve(dist, file));
}

cpSync(resolve(root, "models"), resolve(dist, "models"), { recursive: true });

const builtApp = resolve(dist, "app.js");
let appSource = readFileSync(builtApp, "utf8");
appSource = appSource.replaceAll(
  "./node_modules/@mediapipe/tasks-vision",
  "./vendor/@mediapipe/tasks-vision",
);
writeFileSync(builtApp, appSource);

cpSync(
  resolve(mediaPipeDir, "vision_bundle.mjs"),
  resolve(vendorDist, "vision_bundle.mjs"),
);
cpSync(resolve(mediaPipeDir, "wasm"), resolve(vendorDist, "wasm"), {
  recursive: true,
});

writeFileSync(
  resolve(dist, "_headers"),
  [
    "/vendor/@mediapipe/tasks-vision/vision_bundle.mjs",
    "  Content-Type: text/javascript",
    "",
    "/vendor/@mediapipe/tasks-vision/wasm/*",
    "  Content-Type: application/wasm",
    "",
  ].join("\n"),
);

console.log("Built static site in dist/");
