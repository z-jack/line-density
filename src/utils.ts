import ndarray from "ndarray";
import tile from 'ndarray-tile'
import unpack from 'ndarray-unpack'
import pack from 'ndarray-pack'

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

export function makePair(arr: ndarray) {
  const offset = Math.floor(multi / 2)
  let out = arr
  out = tile(out, [1, multi]) as ndarray
  out = ndarray(out.data)
  out = tile(out, [1, offset]) as ndarray
  out = out.transpose(1, 0)
  const tmpOut = unpack(out) as number[][]
  const tailStack = []
  const headStack = []
  for (let i = 0; i < offset; i++) {
    tailStack.push(tmpOut[0].pop());
    headStack.push(tmpOut[1].shift());
  }
  for (let i = 0; i < offset; i++) {
    tmpOut[0].unshift(tailStack.pop())
    tmpOut[1].push(headStack.pop())
  }
  out = pack(tmpOut)
  out = out.transpose(1, 0)
  out = unpack(out)
  return out
}

/**
 * Convert integer to float for shaders.
 */
export function float(i: number) {
  return i.toFixed(1);
}
