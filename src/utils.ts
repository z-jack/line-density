import ndarray from "ndarray";
import tile from 'ndarray-tile'

export function range(n: number) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = i;
  }
  return out;
}

export function rangeDup(n: number) {
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n * 2; i++) {
    out[i] = Math.floor(i / 2);
  }
  return out;
}

export function duplicate(arr: ndarray) {
  let out = tile(arr, [1, 2])
  out = ndarray(out.data)
  return out
}

/**
 * Convert integer to float for shaders.
 */
export function float(i: number) {
  return i.toFixed(1);
}
