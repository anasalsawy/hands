import assert from "node:assert/strict";
import test from "node:test";
import { frameBlob, latestFrames } from "../src/lib/frames.ts";

const tick = () => new Promise((resolve) => setImmediate(resolve));
const frame = (id) => new Blob([String(id)]);

function harness() {
  const decodes = [];
  const painted = [];
  const closed = [];
  const errors = [];
  const renderer = latestFrames({
    decode: (blob) => new Promise((resolve, reject) => decodes.push({ blob, resolve, reject })),
    paint: (bitmap) => painted.push(bitmap.id),
    onError: (error) => errors.push(error),
  });
  const finish = (index, id) => decodes[index].resolve({
    id, width: 1, height: 1, close: () => closed.push(id),
  });
  return { renderer, decodes, painted, closed, errors, finish };
}

test("a burst retains only the newest pending frame, with one decode in flight", async () => {
  const h = harness();
  for (let i = 0; i < 1000; i += 1) h.renderer.push(frame(i));
  assert.equal(h.decodes.length, 1);
  h.finish(0, 0);
  await tick();
  assert.equal(h.decodes.length, 2);
  assert.equal(await h.decodes[1].blob.text(), "999");
  h.finish(1, 999);
  await tick();
  assert.deepEqual(h.painted, [0, 999]);
  assert.deepEqual(h.closed, [0, 999]);
});

test("dispose closes a late bitmap without drawing it or decoding queued frames", async () => {
  const h = harness();
  h.renderer.push(frame(1));
  h.renderer.push(frame(2));
  h.renderer.dispose();
  h.finish(0, 1);
  await tick();
  h.renderer.push(frame(3));
  assert.equal(h.decodes.length, 1);
  assert.deepEqual(h.painted, []);
  assert.deepEqual(h.closed, [1]);
});

test("a corrupt frame reports an error and the newest frame still renders", async () => {
  const h = harness();
  h.renderer.push(frame(1));
  h.renderer.push(frame(2));
  h.decodes[0].reject(new Error("corrupt"));
  await tick();
  h.finish(1, 2);
  await tick();
  assert.equal(h.errors.length, 1);
  assert.deepEqual(h.painted, [2]);
  assert.deepEqual(h.closed, [2]);
});

test("a drawing failure still closes the bitmap", async () => {
  let closed = false;
  const errors = [];
  const renderer = latestFrames({
    decode: async () => ({ width: 1, height: 1, close: () => { closed = true; } }),
    paint: () => { throw new Error("draw failed"); },
    onError: (error) => errors.push(error),
  });
  renderer.push(frame(1));
  await tick();
  assert.equal(closed, true);
  assert.equal(errors.length, 1);
});

test("browser PNG and computer JPEG/PNG protocols decode with the correct MIME", async () => {
  for (const key of ["screenshot_png_b64", "jpeg_b64", "png_b64", "data_b64"]) {
    const blob = frameBlob({ [key]: btoa("pixels") });
    assert.equal(await blob.text(), "pixels");
    assert.equal(blob.type, key.includes("png") ? "image/png" : "image/jpeg");
  }
  assert.equal(frameBlob({ type: "ready" }), null);
  assert.throws(() => frameBlob({ jpeg_b64: "invalid!" }));
});
