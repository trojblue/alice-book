#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const contentDir = path.join(root, "content");
const htmlDir = path.join(root, "html");

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyDir(src, dest) {
  if (!fs.existsSync(src)) return;
  ensureDir(dest);
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(from, to);
    } else if (entry.isFile()) {
      ensureDir(path.dirname(to));
      fs.copyFileSync(from, to);
    }
  }
}

function main() {
  if (!fs.existsSync(contentDir)) {
    throw new Error(`Missing content directory at ${contentDir}`);
  }
  ensureDir(htmlDir);

  const assetsDirs = fs
    .readdirSync(contentDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.endsWith(".assets"))
    .map((entry) => entry.name);

  for (const dirName of assetsDirs) {
    const from = path.join(contentDir, dirName);
    const to = path.join(htmlDir, dirName);
    copyDir(from, to);
    console.log(`Copied ${dirName} to html/`);
  }
}

main();
