import ndarray from "ndarray";
import tile from 'ndarray-tile'

const multi = 4;

export function range(n: number) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = i;
  }
  return out;
}

export function rangeDup(n: number) {
  const out = new Float32Array(n * multi);
  for (let i = 0; i < n * multi; i++) {
    out[i] = Math.floor(i / multi);
  }
  return out;
}

export function duplicate(arr: ndarray) {
  let out = tile(arr, [1, multi]) as ndarray
  out = ndarray(out.data)
  return out
}

/**
 * Convert integer to float for shaders.
 */
export function float(i: number) {
  return i.toFixed(1);
}
