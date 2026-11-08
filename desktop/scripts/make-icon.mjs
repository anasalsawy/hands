// Generates src/assets/app-icon.png (1024x1024) from raw pixels + zlib, so the
// icon set is original to this app rather than copied from another product.
// Run: node scripts/make-icon.mjs   then   npx tauri icon src/assets/app-icon.png
import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SIZE = 1024;
const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, "..", "src", "assets", "app-icon.png");

const px = Buffer.alloc(SIZE * SIZE * 4);
const cx = SIZE / 2;
const cy = SIZE / 2;

for (let y = 0; y < SIZE; y += 1) {
  for (let x = 0; x < SIZE; x += 1) {
    const i = (y * SIZE + x) * 4;
    const dx = x - cx;
    const dy = y - cy;
    const r = Math.hypot(dx, dy);
    const t = y / SIZE;
    // Dark slate base with a teal-to-blue diagonal.
    let cr = 14 + t * 20;
    let cg = 18 + t * 120;
    let cb = 26 + t * 90;
    if (r > SIZE * 0.47) {
      cr = 8; cg = 9; cb = 11;
    }
    // Four "card" tiles: the deck, in one glyph.
    const tile = 150;
    const gap = 46;
    const gx = ((x - cx + gap / 2) % (tile + gap) + (tile + gap)) % (tile + gap);
    const gy = ((y - cy + gap / 2) % (tile + gap) + (tile + gap)) % (tile + gap);
    const inTile = gx < tile && gy < tile && r < SIZE * 0.4;
    if (inTile) {
      const edge = Math.min(gx, gy, tile - gx, tile - gy);
      const lit = edge < 12 ? 1 : 0;
      cr = lit ? 210 : 30;
      cg = lit ? 240 : 42;
      cb = lit ? 230 : 52;
    }
    px[i] = Math.min(255, cr);
    px[i + 1] = Math.min(255, cg);
    px[i + 2] = Math.min(255, cb);
    px[i + 3] = 255;
  }
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, "latin1"), data]);
  const crcTable = chunk.table ||= (() => {
    const t = new Int32Array(256);
    for (let n = 0; n < 256; n += 1) {
      let c = n;
      for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c;
    }
    return t;
  })();
  let crc = -1;
  for (const byte of body) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  const tail = Buffer.alloc(4);
  tail.writeUInt32BE((crc ^ -1) >>> 0, 0);
  return Buffer.concat([len, body, tail]);
}

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(SIZE, 0);
ihdr.writeUInt32BE(SIZE, 4);
ihdr[8] = 8; // bit depth
ihdr[9] = 6; // RGBA
const raw = Buffer.alloc(SIZE * (SIZE * 4 + 1));
for (let y = 0; y < SIZE; y += 1) {
  raw[y * (SIZE * 4 + 1)] = 0;
  px.copy(raw, y * (SIZE * 4 + 1) + 1, y * SIZE * 4, (y + 1) * SIZE * 4);
}
const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk("IHDR", ihdr),
  chunk("IDAT", deflateSync(raw, { level: 9 })),
  chunk("IEND", Buffer.alloc(0)),
]);

mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, png);
console.log(`wrote ${out} (${png.length} bytes)`);