// IEEE 754 binary16 -> float32 conversion through a 65536-entry lookup table.

let table = null;

function buildTable() {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    let v;
    if (exp === 0) v = frac * 2 ** -24; // zero and subnormals
    else if (exp === 31) v = frac ? NaN : Infinity;
    else v = (1 + frac / 1024) * 2 ** (exp - 15);
    t[h] = sign * v;
  }
  return t;
}

export function halfToFloat(h) {
  table ??= buildTable();
  return table[h];
}

/** Convert a Uint16Array of binary16 values to a new Float32Array. */
export function halfArrayToFloat32(u16) {
  table ??= buildTable();
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = table[u16[i]];
  return out;
}

export const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
