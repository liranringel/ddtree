import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const projectAssetsDir = resolve(rootDir, "public", "assets", "project");

const requiredAssets = [
  "speedup-bar.pdf",
  "speedup-bar.png",
  "math500-budget-tradeoff.pdf",
  "math500-budget-tradeoff.png",
  "math500-acceptance-histogram.pdf",
  "math500-acceptance-histogram.png",
  "comparison-example.mp4",
  "comparison-example-poster.png"
];

const missingAssets = requiredAssets.filter((asset) => !existsSync(resolve(projectAssetsDir, asset)));

if (missingAssets.length > 0) {
  throw new Error(`Missing website assets: ${missingAssets.join(", ")}`);
}

console.log("Website assets are present.");
