#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const htmlDir = path.join(root, "html");
const manifestPath = path.join(root, "manifest.json");

function main() {
  if (!fs.existsSync(htmlDir)) {
    throw new Error(`Missing html directory at ${htmlDir}`);
  }

  const entries = fs
    .readdirSync(htmlDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".html"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  fs.writeFileSync(manifestPath, JSON.stringify(entries, null, 2) + "\n");
  console.log(`Wrote ${entries.length} items to ${path.relative(root, manifestPath)}`);
}

main();
