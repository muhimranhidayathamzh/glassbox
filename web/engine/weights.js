// Reads the weights.bin + manifest.json pairs written by python/glassbox/export.py.

import { LITTLE_ENDIAN, halfArrayToFloat32 } from "./fp16.js";

/** Returns Map<tensor name, Float32Array> after checking format, size and dtype. */
export function decodeTensors(manifest, buffer, format) {
  if (manifest.format !== format) throw new Error(`expected a ${format} manifest, got ${manifest.format}`);
  if (!LITTLE_ENDIAN) throw new Error("big-endian platforms are not supported");
  if (!(buffer instanceof ArrayBuffer)) throw new Error("weights must be an ArrayBuffer");
  if (buffer.byteLength !== manifest.weights_bytes) {
    throw new Error(`weights.bin has ${buffer.byteLength} bytes, manifest says ${manifest.weights_bytes}`);
  }
  let flat;
  if (manifest.dtype === "float16") flat = halfArrayToFloat32(new Uint16Array(buffer));
  else if (manifest.dtype === "float32") flat = new Float32Array(buffer);
  else throw new Error(`unsupported dtype ${manifest.dtype}`);
  const tensors = new Map();
  for (const t of manifest.tensors) tensors.set(t.name, flat.subarray(t.offset, t.offset + t.length));
  return tensors;
}
