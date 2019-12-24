import ndarray from "ndarray";
import regl_ from "regl";
import { MAX_REPEATS_X, MAX_REPEATS_Y } from "./constants";
import { float as f, rangeDup, duplicate, range } from "./utils";

export interface BinConfig {
  /**
   * The start of the range.
   */
  start: number;
  /**
   * The end of the range.
   */
  stop: number;
  /**
   * The size of bin steps.
   */
  step: number;
}

export interface Result {
  /**
   * Start of the time bin.
   */
  x: number;
  /**
   * Start fo teh value bin.
   */
  y: number;
  /**
   * Computed density.
   */
  value: number;
}

/**
 * Compute a density heatmap.
 * @param data The time series data as an ndarray.
 * @param binX Configuration for the binning along the time dimension.
 * @param binY Configuration for the binning along the value dimension.
 * @param canvas The canvas for the webgl context and for debug output.
 */
export default async function (
  data: ndarray,
  binX: BinConfig,
  binY: BinConfig,
  canvas?: HTMLCanvasElement,
  gaussianKernel?: number[][],
  lineWidth: number = 1,
  tangentExtent: number = 0,
  normalExtent: number = 0
) {
  const [numSeries, numDataPoints] = data.shape;

  const debugCanvas = !!canvas;
  const doGaussian = !!gaussianKernel;

  const heatmapWidth = Math.floor((binX.stop - binX.start) / binX.step);
  const heatmapHeight = Math.floor((binY.stop - binY.start) / binY.step);

  if (doGaussian && gaussianKernel.find(gaussianRow => gaussianRow.length != gaussianKernel.length)) {
    throw new Error('The input Gaussian kernal should be square matrix.')
  }
  if (doGaussian && gaussianKernel.length % 2 == 0) {
    throw new Error('The input Gaussian kernal size should be odd.')
  }
  const gaussianIndexOffset = doGaussian && Math.round((gaussianKernel.length - 1) / 2)

  console.info(`Heatmap size: ${heatmapWidth}x${heatmapHeight}`);
  doGaussian && console.info(`GaussianKernelSize: ${gaussianKernel.length}x${gaussianKernel.length}`)

  const regl = regl_({
    canvas: canvas || document.createElement("canvas"),
    extensions: ["OES_texture_float"]
  });

  // See https://github.com/regl-project/regl/issues/498
  const maxRenderbufferSize = Math.min(regl.limits.maxRenderbufferSize, 4096);

  const maxRepeatsX = Math.floor(maxRenderbufferSize / heatmapWidth);
  const maxRepeatsY = Math.floor(maxRenderbufferSize / heatmapHeight);

  const repeatsX = Math.min(
    maxRepeatsX,
    Math.ceil(numSeries / 4 - 1e-6),
    MAX_REPEATS_X
  );
  const repeatsY = Math.min(
    maxRepeatsY,
    Math.ceil(numSeries / (repeatsX * 4)),
    MAX_REPEATS_Y
  );

  console.info(
    `Can repeat ${maxRepeatsX}x${maxRepeatsY} times. Repeating ${repeatsX}x${repeatsY} times.`
  );

  const reshapedWidth = heatmapWidth * repeatsX;
  const reshapedHeight = heatmapHeight * repeatsY;

  console.info(`Canvas size ${reshapedWidth}x${reshapedHeight}.`);

  const drawLine = regl({
    vert: `
        precision mediump float;
      
        attribute float time;
        attribute float value;
        attribute float indexF;
      
        uniform float maxX;
        uniform float maxY;
        uniform float column;
        uniform float row;
        uniform float values[${numDataPoints}];

        varying float innerX;
        varying float innerY;

        int mod(int x, int y) {
          return x - x / y * y;
        }
      
        void main() {
          float repeatsX = ${f(repeatsX)};
          float repeatsY = ${f(repeatsY)};
          int index = int(indexF);
          innerX = float(mod(index, 2));
          innerY = float(index / 2 - 1);
      
          // time and value start at 0 so we can simplify the scaling
          float baseX = column / repeatsX + time / (maxX * repeatsX);
          
          // move up by 0.3 pixels so that the line is guaranteed to be drawn
          // float yOffset = row / repeatsY + 0.3 / ${f(reshapedHeight)};
          // squeeze by 0.6 pixels
          // float squeeze = 1.0 - 0.6 / ${f(heatmapHeight)};
          // float yValue = value / (maxY * repeatsY) * squeeze;
          // float y = yOffset + yValue;
          float baseY = row / repeatsY + value / (maxY * repeatsY);
          
          vec2 tangent = vec2(index / 8 + 1, values[index / 8 + 1]) - vec2(index / 8, values[index / 8]);
          tangent /= vec2(maxX * repeatsX, maxY * repeatsY);
          tangent = normalize(tangent);
          vec2 normal = vec2(-tangent.y, tangent.x);

          vec2 onePixel = vec2(1) / vec2(maxX * repeatsX, maxY * repeatsY);
          float pixelLength = length(onePixel);
          tangent *= pixelLength * ${f(tangentExtent)};
          normal *= pixelLength * ${f(normalExtent)};

          vec2 position = vec2(baseX, baseY) + tangent * float(mod((index / 2 - 1), 2) * 2 - 1) - normal * float(mod(index, 2) * 2 - 1);
      
          // squeeze y by 0.3 pixels so that the line is guaranteed to be drawn
          float yStretch = 2.0 - 0.6 / ${f(reshapedHeight)};
      
          // scale to [-1, 1]
          gl_Position = vec4(
            2.0 * (position.x - 0.5),
            2.0 * (position.y - 0.5),
            0, 1);
        }`,

    frag: `
        precision mediump float;

        varying float innerX;
        varying float innerY;
      
        void main() {
          // we will control the color with the color mask
          if (innerY < 0.0) {
            gl_FragColor = vec4(0);
            return;
          }
          gl_FragColor = vec4(1);
        }`,

    uniforms: {
      maxX: regl.prop<any, "maxX">("maxX"),
      maxY: regl.prop<any, "maxY">("maxY"),
      column: regl.prop<any, "column">("column"),
      row: regl.prop<any, "row">("row"),
      values: regl.prop<any, "rawValue">("rawValue")
    },

    attributes: {
      time: regl.prop<any, "times">("times"),
      value: regl.prop<any, "values">("values"),
      indexF: regl.prop<any, "indexes">("indexes")
    },

    colorMask: regl.prop<any, "colorMask">("colorMask"),

    depth: { enable: false, mask: false },

    count: regl.prop<any, "count">("count"),

    primitive: "triangle strip",
    //lineWidth: () => 1,

    framebuffer: regl.prop<any, "out">("out")
  });

  const computeBase = {
    vert: `
        precision mediump float;
      
        attribute vec2 position;
        varying vec2 uv;
      
        void main() {
          uv = 0.5 * (position + 1.0);
          gl_Position = vec4(position, 0, 1);
        }`,

    attributes: {
      position: [-4, -4, 4, -4, 0, 4]
    },

    depth: { enable: false, mask: false },

    count: 3
  };

  /**
   * Do Gaussian kernel density estimation
   */
  const gaussian = regl({
    ...computeBase,
    frag: `
      precision mediump float;
    
      uniform sampler2D buffer;
    
      varying vec2 uv;
    
      vec4 getColor(int offsetX, int offsetY) {
        const int canvasWidth = ${reshapedWidth};
        const int canvasHeight = ${reshapedHeight};
        const int sampleWidth = ${heatmapWidth};
        const int sampleHeight = ${heatmapHeight};
    
        int currentX = int(uv.x * float(canvasWidth) + 1e-1);
        int currentY = int(uv.y * float(canvasHeight) + 1e-1);
        int refX = currentX + offsetX;
        int refY = currentY + offsetY;
    
        if (currentX / sampleWidth == refX / sampleWidth && currentY / sampleHeight == refY / sampleHeight && refX >= 0 && refY >= 0) {
          vec2 offsetPixel = vec2(float(offsetX), float(offsetY)) / vec2(float(canvasWidth), float(canvasHeight));
          return texture2D(buffer, uv + offsetPixel);
        } else {
          return vec4(0.0, 0.0, 0.0, 0.0);
        }
      }
    
      void main() {
        gl_FragColor = ${
      doGaussian ?
        gaussianKernel.map(
          (gaussianRow, offsetY) =>
            gaussianRow.map(
              (kernelValue, offsetX) =>
                `getColor(${gaussianIndexOffset - offsetX}, ${offsetY - gaussianIndexOffset}) * ${f(kernelValue)}`)
              .join(' + '))
          .join(' + ')
        : 'getColor(0,0)'
      };
      }
    `,
    uniforms: {
      buffer: regl.prop<any, "buffer">("buffer")
    },
    framebuffer: regl.prop<any, "out">("out")
  })

  /**
   * Compute the sums of each column and put it into a framebuffer
   */
  const sum = regl({
    ...computeBase,

    frag: `
        precision mediump float;
      
        uniform sampler2D buffer;
        varying vec2 uv;
      
        void main() {
          float texelRowStart = floor(uv.y * ${f(repeatsY)}) / ${f(repeatsY)};
      
          // normalize by the column
          vec4 sum = vec4(0.0);
          for (float j = 0.0; j < ${f(heatmapHeight)}; j++) {
            float texelRow = texelRowStart + (j + 0.5) / ${f(reshapedHeight)};
            vec4 value = texture2D(buffer, vec2(uv.x, texelRow));
            sum += value;
          }
      
          // sum should be at least 1, prevents problems with empty buffers
          gl_FragColor = max(vec4(1.0), sum);
        }`,

    uniforms: {
      buffer: regl.prop<any, "buffer">("buffer")
    },

    framebuffer: regl.prop<any, "out">("out")
  });

  /**
   * Normalize the pixels in the buffer by the sums computed before.
   * Alpha blends the outputs.
   */
  const normalize = regl({
    ...computeBase,

    frag: `
        precision mediump float;
      
        uniform sampler2D buffer;
        uniform sampler2D sums;
        varying vec2 uv;
      
        void main() {
          vec4 value = texture2D(buffer, uv);
          vec4 sum = texture2D(sums, uv);
      
          gl_FragColor = value / sum;
        }`,

    uniforms: {
      sums: regl.prop<any, "sums">("sums"),
      buffer: regl.prop<any, "buffer">("buffer")
    },

    // additive blending
    blend: {
      enable: true,
      func: {
        srcRGB: "one",
        srcAlpha: 1,
        dstRGB: "one",
        dstAlpha: 1
      },
      equation: {
        rgb: "add",
        alpha: "add"
      },
      color: [0, 0, 0, 0]
    },

    framebuffer: regl.prop<any, "out">("out")
  });

  /**
   * Merge rgba from the wide buffer into one heatmap buffer
   */
  const mergeBufferHorizontally = regl({
    ...computeBase,

    frag: `
        precision mediump float;
      
        uniform sampler2D buffer;
      
        varying vec2 uv;
      
        void main() {
          vec4 color = vec4(0);
      
          // collect all columns
          for (float i = 0.0; i < ${f(repeatsX)}; i++) {
            float x = (i + uv.x) / ${f(repeatsX)};
            color += texture2D(buffer, vec2(x, uv.y));
          }
      
          gl_FragColor = color;
        }`,

    uniforms: {
      buffer: regl.prop<any, "buffer">("buffer")
    },

    framebuffer: regl.prop<any, "out">("out")
  });
  const mergeBufferVertically = regl({
    ...computeBase,

    frag: `
        precision mediump float;
      
        uniform sampler2D buffer;
      
        varying vec2 uv;
      
        void main() {
          vec4 color = vec4(0);
      
          // collect all rows
          for (float i = 0.0; i < ${f(repeatsY)}; i++) {
            float y = (i + uv.y) / ${f(repeatsY)};
            color += texture2D(buffer, vec2(uv.x, y));
          }
      
          float value = color.r + color.g + color.b + color.a;
          gl_FragColor = vec4(vec3(value), 1.0);
        }`,

    uniforms: {
      buffer: regl.prop<any, "buffer">("buffer")
    },

    framebuffer: regl.prop<any, "out">("out")
  });

  /**
   * Helper function to draw a the texture in a buffer.
   */
  const drawTexture = regl({
    ...computeBase,

    frag: `
        precision mediump float;
      
        uniform sampler2D buffer;
        
        varying vec2 uv;
        
        void main() {
          // get r and draw it
          vec3 value = texture2D(buffer, uv).rgb;
          gl_FragColor = vec4(value, 1.0);
        }`,

    colorMask: regl.prop<any, "colorMask">("colorMask"),

    uniforms: {
      buffer: regl.prop<any, "buffer">("buffer")
    }
  });

  console.time("Allocate buffers");
  const linesBuffer = regl.framebuffer({
    width: reshapedWidth,
    height: reshapedHeight,
    colorFormat: "rgba",
    colorType: "uint8"
  });

  const gaussianBuffer = regl.framebuffer({
    width: reshapedWidth,
    height: reshapedHeight,
    colorFormat: "rgba",
    colorType: "float"
  })

  const sumsBuffer = regl.framebuffer({
    width: reshapedWidth,
    height: repeatsY,
    colorFormat: "rgba",
    colorType: "float"
  });

  const resultBuffer = regl.framebuffer({
    width: reshapedWidth,
    height: reshapedHeight,
    colorFormat: "rgba",
    colorType: "float"
  });

  const preMergedBuffer = regl.framebuffer({
    width: heatmapWidth,
    height: reshapedHeight,
    colorFormat: "rgba",
    colorType: "float"
  });

  const heatBuffer = regl.framebuffer({
    width: heatmapWidth,
    height: heatmapHeight,
    colorFormat: "rgba",
    colorType: "float"
  });
  console.timeEnd("Allocate buffers");

  function colorMask(i) {
    const mask = [false, false, false, false];
    mask[i % 4] = true;
    return mask;
  }

  // For now, assume that all time series get the same time points
  const times = rangeDup(numDataPoints);

  console.time("Compute heatmap");

  // batches of 4 * repeats
  const batchSize = 4 * repeatsX * repeatsY;

  // index of series
  let series = 0;
  // how many series have already been drawn
  let finishedSeries = 0;

  for (let b = 0; b < numSeries; b += batchSize) {
    console.time("Prepare Batch");

    // array to hold the lines that should be rendered
    let lines = new Array(Math.min(batchSize, numSeries - series));

    // clear the lines buffer before the next batch
    regl.clear({
      color: [0, 0, 0, 0],
      framebuffer: linesBuffer
    });

    loop: for (let row = 0; row < repeatsY; row++) {
      for (let i = 0; i < 4 * repeatsX; i++) {
        if (series >= numSeries) {
          break loop;
        }

        // console.log(series, Math.floor(i / 4), row);

        lines[series - finishedSeries] = {
          values: duplicate(data.pick(series, null)),
          rawValue: duplicate(data.pick(series, null)),
          times: times,
          indexes: range(numDataPoints * 4),
          maxY: binY.stop,
          maxX: numDataPoints - 1,
          column: Math.floor(i / 4),
          row: row,
          colorMask: colorMask(i),
          count: numDataPoints * 4,
          out: linesBuffer
        };

        series++;
      }
    }
    console.timeEnd("Prepare Batch");

    console.info(`Drawing ${lines.length} lines.`);

    console.time("regl: drawLine");
    drawLine(lines);
    console.timeEnd("regl: drawLine");

    if (doGaussian) {
      console.time("regl: gaussian");
      gaussian({
        buffer: linesBuffer,
        out: gaussianBuffer
      })
      console.timeEnd('regl: gaussian')
    }

    finishedSeries += lines.length;

    let pendingBuffer = doGaussian ? gaussianBuffer : linesBuffer
    console.time("regl: sum");
    sum({
      buffer: pendingBuffer,
      out: sumsBuffer
    });
    console.timeEnd("regl: sum");

    console.time("regl: normalize");
    normalize({
      buffer: pendingBuffer,
      sums: sumsBuffer,
      out: resultBuffer
    });
    console.timeEnd("regl: normalize");
  }

  console.time("regl: merge");
  mergeBufferHorizontally({
    buffer: resultBuffer,
    out: preMergedBuffer
  });

  mergeBufferVertically({
    buffer: preMergedBuffer,
    out: heatBuffer
  });
  console.timeEnd("regl: merge");

  console.timeEnd("Compute heatmap");

  if (debugCanvas) {
    drawTexture({
      buffer: resultBuffer,
      colorMask: [true, true, true, true]
    });
  }

  return new Promise<Result[]>(resolve => {
    regl({ framebuffer: heatBuffer })(() => {
      const arr = regl.read();
      const out = new Float32Array(arr.length / 4);

      for (let i = 0; i < arr.length; i += 4) {
        out[i / 4] = arr[i];
      }

      const heatmap = ndarray(out, [heatmapHeight, heatmapWidth]);

      const heatmapData: Result[] = [];
      for (let x = 0; x < heatmapWidth; x++) {
        for (let y = 0; y < heatmapHeight; y++) {
          heatmapData.push({
            x: binX.start + x * binX.step,
            y: binY.start + y * binY.step,
            value: heatmap.get(y, x)
          });
        }
      }

      resolve(heatmapData);
    });
  });
}
