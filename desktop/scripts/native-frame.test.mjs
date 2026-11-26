import test from "node:test";
import assert from "node:assert/strict";
import { paintBitmap } from "../src/lib/frames.ts";
test("capture retains native dimensions and draws without resampling",()=>{
 const calls=[]; const canvas={width:960,height:540,getContext:()=>({drawImage:(...args)=>calls.push(args)})};
 const bitmap={width:1920,height:1080};
 paintBitmap(canvas,bitmap); assert.equal(canvas.width,1920); assert.equal(canvas.height,1080);
 assert.deepEqual(calls,[[bitmap,0,0]]);
 const portrait={width:1080,height:1920}; paintBitmap(canvas,portrait); assert.equal(canvas.height,1920); assert.deepEqual(calls[1],[portrait,0,0]);
});