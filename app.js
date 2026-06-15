/* ─── State ─── */
const state = {
  layers: [],             // [0] = front-most layer
  selectedLayerId: null,
  nextLayerId: 1,
  docW: 0,
  docH: 0,
  // Effects (applied over the merged composite of all layers)
  mode: 'none',
  threshold: 128,
  thresholdSoft: 0,
  invertThreshold: false,
  chRed: 30,
  chGreen: 59,
  chBlue: 11,
  vibrance: 0,
  saturation: 0,
  sharpen: 0,
  // View
  splitX: 0.5,
  zoom: 1,
  // Mask brush
  maskEditLayerId: null,
  maskColor: 'black',
  brushSize: 50,
  brushHardness: 50,
};

/* ─── Refs ─── */
const $ = id => document.getElementById(id);
const dropZone    = $('dropZone');
const dropInner   = $('dropInner');
const fileInput   = $('fileInput');
const previewWrap = $('previewWrap');
const canvasOrig  = $('canvasOriginal');
const canvasProc  = $('canvasProcessed');
const splitDiv    = $('splitDivider');
const exportBtn   = $('exportBtn');
const resetBtn    = $('resetBtn');
const invertCheck = $('invertCheck');
const zoomBadge   = $('zoomBadge');
const ctxO        = canvasOrig.getContext('2d');

const layersPanel       = $('layersPanel');
const layerList         = $('layerList');
const layerEmpty        = $('layerEmpty');
const layerFileInput    = $('layerFileInput');
const addLayerBtn       = $('addLayerBtn');
const layerTransformGroup = $('layerTransformGroup');
const layerBlendSelect    = $('layerBlendSelect');
const layerOpacitySlider  = $('layerOpacitySlider');
const layerOpacityVal     = $('layerOpacityVal');
const layerScaleSlider    = $('layerScaleSlider');
const layerScaleVal       = $('layerScaleVal');
const layerRotationSlider = $('layerRotationSlider');
const layerRotationVal    = $('layerRotationVal');

const maskPanel          = $('maskPanel');
const maskCloseBtn       = $('maskCloseBtn');
const brushSizeSlider    = $('brushSizeSlider');
const brushSizeVal       = $('brushSizeVal');
const brushHardnessSlider= $('brushHardnessSlider');
const brushHardnessVal   = $('brushHardnessVal');
const brushCursor        = $('brushCursor');

const signed = v => (v > 0 ? '+' : '') + v;

/* ─── Icons ─── */
const EYE_ICON = `<svg viewBox="0 0 16 16" fill="none" width="14" height="14"><path d="M1 8s2.5-4.5 7-4.5S15 8 15 8s-2.5 4.5-7 4.5S1 8 1 8z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><circle cx="8" cy="8" r="1.8" stroke="currentColor" stroke-width="1.3"/></svg>`;
const EYE_OFF_ICON = `<svg viewBox="0 0 16 16" fill="none" width="14" height="14"><path d="M1 8s2.5-4.5 7-4.5S15 8 15 8s-2.5 4.5-7 4.5S1 8 1 8z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/><circle cx="8" cy="8" r="1.8" stroke="currentColor" stroke-width="1.3"/><path d="M2 2l12 12" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>`;
const UP_ICON   = `<svg viewBox="0 0 16 16" fill="none" width="11" height="11"><path d="M4 10l4-4 4 4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const DOWN_ICON = `<svg viewBox="0 0 16 16" fill="none" width="11" height="11"><path d="M4 6l4 4 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const DEL_ICON  = `<svg viewBox="0 0 16 16" fill="none" width="11" height="11"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
const MASK_ICON = `<svg viewBox="0 0 16 16" fill="none" width="11" height="11"><rect x="1.5" y="1.5" width="13" height="13" rx="2" stroke="currentColor" stroke-width="1.3"/><circle cx="8" cy="8" r="3.5" fill="currentColor"/></svg>`;

/* ──────────────────────────────────────────────────────────
   WebGL pipeline
   Each adjustment (sharpen, saturation, vibrance, channel mix,
   threshold, invert) runs as a single fragment-shader pass over
   the merged layer composite, at full resolution.
   Falls back to a CPU canvas pipeline if WebGL is unavailable.
   ────────────────────────────────────────────────────────── */

const VERT_SRC = `
attribute vec2 a_position;
varying vec2 v_texCoord;
void main() {
  v_texCoord = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FRAG_SRC = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 v_texCoord;
uniform sampler2D u_image;
uniform vec2  u_texelSize;
uniform float u_sharpen;
uniform float u_saturation;
uniform float u_vibrance;
uniform int   u_mode;          // 0 = none, 1 = threshold, 2 = b&w
uniform float u_threshold;
uniform float u_thresholdSoft;
uniform vec3  u_chWeights;
uniform float u_invert;

void main() {
  vec4 src = texture2D(u_image, v_texCoord);
  vec3 rgb = src.rgb;

  // ── Sharpen / blur (4-neighbor convolution) ──
  if (u_sharpen != 0.0) {
    vec3 top    = texture2D(u_image, v_texCoord + vec2(0.0, -u_texelSize.y)).rgb;
    vec3 bottom = texture2D(u_image, v_texCoord + vec2(0.0,  u_texelSize.y)).rgb;
    vec3 left   = texture2D(u_image, v_texCoord + vec2(-u_texelSize.x, 0.0)).rgb;
    vec3 right  = texture2D(u_image, v_texCoord + vec2( u_texelSize.x, 0.0)).rgb;
    float centerCoef, neighCoef;
    if (u_sharpen >= 0.0) {
      centerCoef = 1.0 + 4.0 * u_sharpen;
      neighCoef  = -u_sharpen;
    } else {
      centerCoef = 1.0 + u_sharpen;
      neighCoef  = -u_sharpen / 4.0;
    }
    rgb = rgb * centerCoef + (top + bottom + left + right) * neighCoef;
  }

  // ── Saturation ──
  if (u_saturation != 0.0) {
    float gray = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    float f = 1.0 + u_saturation;
    rgb = gray + f * (rgb - gray);
  }

  // ── Vibrance ──
  if (u_vibrance != 0.0) {
    float mx = max(max(rgb.r, rgb.g), rgb.b);
    float mn = min(min(rgb.r, rgb.g), rgb.b);
    float sat = (mx - mn) / max(mx, 0.0001);
    float avg = (rgb.r + rgb.g + rgb.b) / 3.0;
    float f = u_vibrance * (1.0 - sat) * 0.8;
    rgb = avg + (1.0 + f) * (rgb - avg);
  }

  // ── B&W / Threshold ──
  if (u_mode == 1 || u_mode == 2) {
    float luma = dot(rgb, u_chWeights);
    if (u_mode == 2) {
      rgb = vec3(luma);
    } else {
      float val;
      if (u_thresholdSoft <= 0.0001) {
        val = step(u_threshold, luma);
      } else {
        float lo = u_threshold - u_thresholdSoft;
        float hi = u_threshold + u_thresholdSoft;
        val = clamp((luma - lo) / (hi - lo), 0.0, 1.0);
      }
      if (u_invert > 0.5) val = 1.0 - val;
      rgb = vec3(val);
    }
  }

  gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), src.a);
}`;

let GL = null;

function compileShader(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error('Shader compile error:', gl.getShaderInfoLog(s));
    return null;
  }
  return s;
}

function initGL() {
  let gl;
  try {
    gl = canvasProc.getContext('webgl', { preserveDrawingBuffer: true })
      || canvasProc.getContext('experimental-webgl', { preserveDrawingBuffer: true });
  } catch (e) { gl = null; }
  if (!gl) return null;

  const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
  if (!vs || !fs) return null;

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
    return null;
  }

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1,-1,  1,-1,  -1,1,
    -1,1,   1,-1,   1,1,
  ]), gl.STATIC_DRAW);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  return {
    gl, program, buffer, texture,
    attribs: { position: gl.getAttribLocation(program, 'a_position') },
    uniforms: {
      image:         gl.getUniformLocation(program, 'u_image'),
      texelSize:     gl.getUniformLocation(program, 'u_texelSize'),
      sharpen:       gl.getUniformLocation(program, 'u_sharpen'),
      saturation:    gl.getUniformLocation(program, 'u_saturation'),
      vibrance:      gl.getUniformLocation(program, 'u_vibrance'),
      mode:          gl.getUniformLocation(program, 'u_mode'),
      threshold:     gl.getUniformLocation(program, 'u_threshold'),
      thresholdSoft: gl.getUniformLocation(program, 'u_thresholdSoft'),
      chWeights:     gl.getUniformLocation(program, 'u_chWeights'),
      invert:        gl.getUniformLocation(program, 'u_invert'),
    },
  };
}

function glUploadTexture(source) {
  const { gl, texture } = GL;
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
}

function glRender() {
  const { gl, program, buffer, attribs, uniforms, texture } = GL;
  gl.viewport(0, 0, canvasProc.width, canvasProc.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(attribs.position);
  gl.vertexAttribPointer(attribs.position, 2, gl.FLOAT, false, 0, 0);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.uniform1i(uniforms.image, 0);

  gl.uniform2f(uniforms.texelSize, 1 / canvasProc.width, 1 / canvasProc.height);
  gl.uniform1f(uniforms.sharpen, state.sharpen / 100);
  gl.uniform1f(uniforms.saturation, state.saturation / 100);
  gl.uniform1f(uniforms.vibrance, state.vibrance / 100);
  gl.uniform1i(uniforms.mode, state.mode === 'threshold' ? 1 : state.mode === 'bw' ? 2 : 0);
  gl.uniform1f(uniforms.threshold, state.threshold / 255);
  gl.uniform1f(uniforms.thresholdSoft, state.thresholdSoft / 255);
  const sum = (state.chRed + state.chGreen + state.chBlue) || 1;
  gl.uniform3f(uniforms.chWeights, state.chRed / sum, state.chGreen / sum, state.chBlue / sum);
  gl.uniform1f(uniforms.invert, state.invertThreshold ? 1 : 0);

  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

GL = initGL();
const useGL = !!GL;
const ctxP = useGL ? null : canvasProc.getContext('2d');

/* ──────────────────────────────────────────────────────────
   CPU fallback pipeline (used only if WebGL is unavailable)
   ────────────────────────────────────────────────────────── */
const PREVIEW_MAX = 1400;
const clamp = v => Math.max(0, Math.min(255, Math.round(v)));

function applySharpen(data, w, h, amt) {
  const src = new Uint8ClampedArray(data);
  let center, neigh;
  if (amt >= 0) { center = 1 + 4 * amt; neigh = -amt; }
  else          { center = 1 + amt;     neigh = -amt / 4; }
  for (let y = 0; y < h; y++) {
    const yUp = Math.max(y - 1, 0) * w, yDown = Math.min(y + 1, h-1) * w, yRow = y * w;
    for (let x = 0; x < w; x++) {
      const xL = Math.max(x - 1, 0), xR = Math.min(x + 1, w - 1);
      const i = (yRow + x) * 4;
      for (let c = 0; c < 3; c++) {
        const top = src[(yUp+x)*4+c], bottom = src[(yDown+x)*4+c];
        const left = src[(yRow+xL)*4+c], right = src[(yRow+xR)*4+c];
        data[i+c] = clamp(src[i+c]*center + (top+bottom+left+right)*neigh);
      }
    }
  }
}
function luma(r, g, b) {
  const t = state.chRed + state.chGreen + state.chBlue || 1;
  return (r * state.chRed + g * state.chGreen + b * state.chBlue) / t;
}
function applySat(r, g, b, a) {
  const gr = 0.2126*r + 0.7152*g + 0.0722*b, f = 1 + a;
  return [gr + f*(r-gr), gr + f*(g-gr), gr + f*(b-gr)];
}
function applyVib(r, g, b, a) {
  const mx = Math.max(r,g,b), mn = Math.min(r,g,b);
  const sat = (mx - mn) / (mx || 1);
  const avg = (r+g+b)/3;
  const f = a * (1 - sat) * 0.8;
  return [avg + (1+f)*(r-avg), avg + (1+f)*(g-avg), avg + (1+f)*(b-avg)];
}
function processInto(ctx, src, w, h) {
  if (ctx.canvas.width !== w)  ctx.canvas.width  = w;
  if (ctx.canvas.height !== h) ctx.canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(src, 0, 0, w, h);
  const id = ctx.getImageData(0, 0, w, h);
  const d = id.data;
  if (state.sharpen !== 0) applySharpen(d, w, h, state.sharpen / 100);
  for (let i = 0; i < d.length; i += 4) {
    let r = d[i], g = d[i+1], b = d[i+2];
    if (state.saturation !== 0) [r,g,b] = applySat(r, g, b, state.saturation / 100);
    if (state.vibrance   !== 0) [r,g,b] = applyVib(r, g, b, state.vibrance   / 100);
    if (state.mode === 'bw') {
      const l = luma(r, g, b); r = g = b = l;
    } else if (state.mode === 'threshold') {
      const l = luma(r, g, b), soft = state.thresholdSoft;
      let val;
      if (soft === 0) {
        val = l >= state.threshold ? 255 : 0;
      } else {
        const lo = state.threshold - soft, hi = state.threshold + soft;
        if      (l <= lo) val = 0;
        else if (l >= hi) val = 255;
        else              val = Math.round((l - lo) / (hi - lo) * 255);
      }
      if (state.invertThreshold) val = 255 - val;
      r = g = b = val;
    }
    d[i] = clamp(r); d[i+1] = clamp(g); d[i+2] = clamp(b);
  }
  ctx.putImageData(id, 0, 0);
}

/* ──────────────────────────────────────────────────────────
   Layer composite
   ────────────────────────────────────────────────────────── */
let compositeCanvas = null;
let compCtx = null;
let layerTempCanvas = null;
let layerTempCtx = null;

function ensureComposite() {
  if (!compositeCanvas) {
    compositeCanvas = document.createElement('canvas');
    compCtx = compositeCanvas.getContext('2d');
  }
  if (compositeCanvas.width !== state.docW || compositeCanvas.height !== state.docH) {
    compositeCanvas.width = state.docW;
    compositeCanvas.height = state.docH;
  }
  if (!layerTempCanvas) {
    layerTempCanvas = document.createElement('canvas');
    layerTempCtx = layerTempCanvas.getContext('2d');
  }
  if (layerTempCanvas.width !== state.docW || layerTempCanvas.height !== state.docH) {
    layerTempCanvas.width = state.docW;
    layerTempCanvas.height = state.docH;
  }
}

/* Mask: an offscreen docW×docH canvas, fully opaque white = fully visible.
   Painting "black" reduces alpha (erase); painting "white" restores it
   (destination-over). Applied via destination-in onto the rendered layer. */
function createMask(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);
  return c;
}

function compositeLayers() {
  if (!state.docW || !state.docH) return;
  ensureComposite();
  compCtx.clearRect(0, 0, state.docW, state.docH);
  for (let i = state.layers.length - 1; i >= 0; i--) {
    const layer = state.layers[i];
    if (!layer.visible) continue;

    layerTempCtx.clearRect(0, 0, state.docW, state.docH);
    layerTempCtx.save();
    layerTempCtx.translate(layer.x, layer.y);
    layerTempCtx.rotate(layer.rotation * Math.PI / 180);
    layerTempCtx.scale(layer.scale, layer.scale);
    layerTempCtx.drawImage(layer.img, -layer.img.width / 2, -layer.img.height / 2);
    layerTempCtx.restore();

    if (layer.mask) {
      layerTempCtx.globalCompositeOperation = 'destination-in';
      layerTempCtx.drawImage(layer.mask, 0, 0);
      layerTempCtx.globalCompositeOperation = 'source-over';
    }

    compCtx.save();
    compCtx.globalAlpha = layer.opacity / 100;
    compCtx.globalCompositeOperation = layer.blendMode || 'source-over';
    compCtx.drawImage(layerTempCanvas, 0, 0);
    compCtx.restore();
  }
}

/* ─── Render ─── */
function renderOriginal() {
  ctxO.clearRect(0, 0, canvasOrig.width, canvasOrig.height);
  ctxO.drawImage(compositeCanvas, 0, 0, canvasOrig.width, canvasOrig.height);
}

function renderProcessed() {
  if (!state.docW) return;
  if (useGL) {
    glUploadTexture(compositeCanvas);
    glRender();
  } else {
    processInto(ctxP, compositeCanvas, canvasProc.width, canvasProc.height);
  }
  updateSplitClip();
}

function renderAll() {
  if (!state.docW) return;
  compositeLayers();
  renderOriginal();
  renderProcessed();
}

/* ─── Render throttling ─── */
let rafPending = false;
let rafNeedsComposite = false;
function scheduleRender(needsComposite) {
  if (needsComposite) rafNeedsComposite = true;
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    if (rafNeedsComposite) { rafNeedsComposite = false; renderAll(); }
    else renderProcessed();
  });
}

/* ─── Effect slider binding ─── */
function bindSlider(sliderId, valId, key, fmt) {
  const sl = $(sliderId), vl = $(valId);
  sl.addEventListener('input', () => {
    state[key] = +sl.value;
    vl.textContent = fmt ? fmt(sl.value) : sl.value;
    if (key === 'chRed' || key === 'chGreen' || key === 'chBlue') updateChannelHint();
    scheduleRender(false);
  });
}
bindSlider('thresholdSlider',     'thresholdVal',    'threshold');
bindSlider('thresholdSoftSlider', 'thresholdSoftVal','thresholdSoft');
bindSlider('chRedSlider',         'chRedVal',        'chRed');
bindSlider('chGreenSlider',       'chGreenVal',      'chGreen');
bindSlider('chBlueSlider',        'chBlueVal',       'chBlue');
bindSlider('vibranceSlider',      'vibranceVal',     'vibrance',   signed);
bindSlider('saturationSlider',    'saturationVal',   'saturation', signed);
bindSlider('sharpenSlider',       'sharpenVal',      'sharpen',    signed);

/* ─── Invert toggle ─── */
invertCheck.addEventListener('change', () => {
  state.invertThreshold = invertCheck.checked;
  renderProcessed();
});

/* ─── Mode segment ─── */
document.querySelectorAll('.seg-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.mode = btn.dataset.mode;
    updateModeUI();
    renderProcessed();
  });
});

function updateModeUI() {
  const isT = state.mode === 'threshold';
  const isBW = state.mode === 'bw';
  const rows = [$('thresholdRow'), $('thresholdSepA'), $('thresholdSoftRow'), $('thresholdSepB'), $('invertRow')];
  rows.forEach(el => { if (el) el.style.display = isT ? '' : 'none'; });
  $('channelGroup').classList.toggle('muted', !(isT || isBW));
}

function updateChannelHint() {
  const s = state.chRed + state.chGreen + state.chBlue;
  $('channelSum').textContent = s;
  $('channelFooter').style.display = s !== 100 ? '' : 'none';
}

/* ──────────────────────────────────────────────────────────
   Layers: add / select / reorder / delete / transform
   ────────────────────────────────────────────────────────── */
function getSelectedLayer() {
  return state.layers.find(l => l.id === state.selectedLayerId);
}

function getMaskLayer() {
  return state.layers.find(l => l.id === state.maskEditLayerId);
}

function addLayer(img, name) {
  const isFirst = state.layers.length === 0;
  if (isFirst) {
    state.docW = img.width;
    state.docH = img.height;
  }
  const fitScale = Math.min(1, state.docW / img.width, state.docH / img.height);
  const id = state.nextLayerId++;
  const layer = {
    id,
    name: name || ('Layer ' + id),
    img,
    x: state.docW / 2,
    y: state.docH / 2,
    scale: isFirst ? 1 : fitScale,
    rotation: 0,
    opacity: 100,
    blendMode: 'source-over',
    visible: true,
    mask: createMask(state.docW, state.docH),
  };
  state.layers.unshift(layer);
  state.selectedLayerId = id;

  if (isFirst) {
    setupCanvases();
    dropZone.classList.add('has-image');
    dropInner.style.display = 'none';
    previewWrap.style.display = '';
    exportBtn.disabled = false;
    state.splitX = 0.5;
    state.zoom = 1;
  }

  renderAll();
  renderLayerPanel();

  if (isFirst) {
    updateDividerPos();
    applyZoom();
  }
}

function resetToEmpty() {
  state.layers = [];
  state.selectedLayerId = null;
  state.docW = 0;
  state.docH = 0;
  state.zoom = 1;
  state.splitX = 0.5;
  exitMaskEdit();
  previewWrap.style.display = 'none';
  dropInner.style.display = '';
  dropZone.classList.remove('has-image');
  exportBtn.disabled = true;
  renderLayerPanel();
}

function selectLayer(id) {
  state.selectedLayerId = id;
  renderLayerPanel();
}

function toggleVisibility(id) {
  const l = state.layers.find(x => x.id === id);
  if (!l) return;
  l.visible = !l.visible;
  renderAll();
  renderLayerPanel();
}

function moveLayer(id, dir) {
  const i = state.layers.findIndex(x => x.id === id);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= state.layers.length) return;
  [state.layers[i], state.layers[j]] = [state.layers[j], state.layers[i]];
  renderAll();
  renderLayerPanel();
}

function deleteLayer(id) {
  state.layers = state.layers.filter(x => x.id !== id);
  if (state.selectedLayerId === id) {
    state.selectedLayerId = state.layers[0] ? state.layers[0].id : null;
  }
  if (state.maskEditLayerId === id) {
    exitMaskEdit();
  }
  if (state.layers.length === 0) {
    resetToEmpty();
  } else {
    renderAll();
    renderLayerPanel();
  }
}

function drawThumb(canvas, img) {
  const ctx = canvas.getContext('2d');
  const size = 32;
  const r = Math.max(size / img.width, size / img.height);
  const w = img.width * r, h = img.height * r;
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
}

function renderLayerPanel() {
  layerList.innerHTML = '';

  if (state.layers.length === 0) {
    layerList.appendChild(layerEmpty);
    layerEmpty.style.display = '';
    layerTransformGroup.style.display = 'none';
    return;
  }
  layerEmpty.style.display = 'none';

  state.layers.forEach((layer, idx) => {
    const row = document.createElement('div');
    row.className = 'layer-row'
      + (layer.id === state.selectedLayerId ? ' selected' : '')
      + (!layer.visible ? ' hidden-layer' : '');
    row.dataset.id = layer.id;

    const visBtn = document.createElement('button');
    visBtn.className = 'layer-vis-btn';
    visBtn.dataset.action = 'vis';
    visBtn.title = layer.visible ? 'Hide layer' : 'Show layer';
    visBtn.innerHTML = layer.visible ? EYE_ICON : EYE_OFF_ICON;

    const thumb = document.createElement('canvas');
    thumb.className = 'layer-thumb';
    thumb.width = 32;
    thumb.height = 32;
    drawThumb(thumb, layer.img);

    const name = document.createElement('span');
    name.className = 'layer-name';
    name.textContent = layer.name;

    const actions = document.createElement('div');
    actions.className = 'layer-actions';

    const upBtn = document.createElement('button');
    upBtn.dataset.action = 'up';
    upBtn.title = 'Move forward';
    upBtn.disabled = idx === 0;
    upBtn.innerHTML = UP_ICON;

    const downBtn = document.createElement('button');
    downBtn.dataset.action = 'down';
    downBtn.title = 'Move backward';
    downBtn.disabled = idx === state.layers.length - 1;
    downBtn.innerHTML = DOWN_ICON;

    const delBtn = document.createElement('button');
    delBtn.dataset.action = 'del';
    delBtn.className = 'danger';
    delBtn.title = 'Delete layer';
    delBtn.innerHTML = DEL_ICON;

    const maskBtn = document.createElement('button');
    maskBtn.dataset.action = 'mask';
    maskBtn.title = 'Edit mask';
    maskBtn.className = state.maskEditLayerId === layer.id ? 'mask-active' : '';
    maskBtn.innerHTML = MASK_ICON;

    actions.append(maskBtn, upBtn, downBtn, delBtn);
    row.append(visBtn, thumb, name, actions);
    layerList.appendChild(row);
  });

  const sel = getSelectedLayer();
  if (sel) {
    layerTransformGroup.style.display = '';
    layerBlendSelect.value = sel.blendMode || 'source-over';
    layerOpacitySlider.value = sel.opacity;
    layerOpacityVal.textContent = sel.opacity;
    layerScaleSlider.value = Math.round(sel.scale * 100);
    layerScaleVal.textContent = Math.round(sel.scale * 100) + '%';
    layerRotationSlider.value = sel.rotation;
    layerRotationVal.textContent = signed(Math.round(sel.rotation)) + '°';
  } else {
    layerTransformGroup.style.display = 'none';
  }
}

layerList.addEventListener('click', e => {
  const row = e.target.closest('.layer-row');
  if (!row) return;
  const id = +row.dataset.id;
  const btn = e.target.closest('button');
  const action = btn ? btn.dataset.action : null;
  if (action === 'vis')  { toggleVisibility(id); return; }
  if (action === 'up')   { moveLayer(id, -1); return; }
  if (action === 'down') { moveLayer(id, 1); return; }
  if (action === 'del')  { deleteLayer(id); return; }
  if (action === 'mask') { toggleMaskEdit(id); return; }
  selectLayer(id);
});

/* Layer transform sliders */
layerBlendSelect.addEventListener('change', () => {
  const l = getSelectedLayer(); if (!l) return;
  l.blendMode = layerBlendSelect.value;
  scheduleRender(true);
});
layerOpacitySlider.addEventListener('input', () => {
  const l = getSelectedLayer(); if (!l) return;
  l.opacity = +layerOpacitySlider.value;
  layerOpacityVal.textContent = l.opacity;
  scheduleRender(true);
});
layerScaleSlider.addEventListener('input', () => {
  const l = getSelectedLayer(); if (!l) return;
  l.scale = +layerScaleSlider.value / 100;
  layerScaleVal.textContent = layerScaleSlider.value + '%';
  scheduleRender(true);
});
layerRotationSlider.addEventListener('input', () => {
  const l = getSelectedLayer(); if (!l) return;
  l.rotation = +layerRotationSlider.value;
  layerRotationVal.textContent = signed(l.rotation) + '°';
  scheduleRender(true);
});

/* Add-layer button */
addLayerBtn.addEventListener('click', () => layerFileInput.click());
layerFileInput.addEventListener('change', () => {
  if (layerFileInput.files[0]) loadFile(layerFileInput.files[0]);
  layerFileInput.value = '';
});

/* ─── File handling ─── */
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('drag-over'); const f = e.dataTransfer.files[0]; if (f) loadFile(f); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });
dropZone.addEventListener('click', e => {
  if (state.layers.length === 0 && !e.target.classList.contains('file-link')) fileInput.click();
});

function loadFile(file) {
  if (!file.type.match(/^image\/(webp|jpeg|png)$/)) { alert('Upload WEBP, JPG, or PNG.'); return; }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    URL.revokeObjectURL(url);
    addLayer(img, file.name.replace(/\.[a-zA-Z0-9]+$/, ''));
  };
  img.src = url;
}

/* ─── Clipboard paste ─── */
document.addEventListener('paste', e => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (file) { e.preventDefault(); loadFile(file); }
      break;
    }
  }
});

/* ─── Canvas sizing ─── */
function setupCanvases() {
  if (!state.docW || !state.docH) return;
  const stage = document.getElementById('stage');
  const rootStyle = getComputedStyle(document.documentElement);
  const rightPanelW = parseInt(rootStyle.getPropertyValue('--panel-w')) + 32;
  const leftPanelW  = parseInt(rootStyle.getPropertyValue('--panel-left-w')) + 32;
  const maxW = stage.clientWidth  - rightPanelW - leftPanelW - 40;
  const maxH = stage.clientHeight - 40;
  const r = Math.min(maxW / state.docW, maxH / state.docH, 1);
  const dw = Math.round(state.docW * r);
  const dh = Math.round(state.docH * r);

  canvasOrig.width  = state.docW;
  canvasOrig.height = state.docH;

  if (useGL) {
    canvasProc.width  = state.docW;
    canvasProc.height = state.docH;
  } else {
    const bufScale = Math.min(1, PREVIEW_MAX / Math.max(state.docW, state.docH));
    canvasProc.width  = Math.max(1, Math.round(state.docW  * bufScale));
    canvasProc.height = Math.max(1, Math.round(state.docH * bufScale));
  }

  [canvasOrig, canvasProc].forEach(c => {
    c.style.width  = dw + 'px';
    c.style.height = dh + 'px';
  });
  canvasOrig._dw = dw;
  canvasOrig._dh = dh;
}

/* ─── Zoom (scroll wheel) ─── */
const ZOOM_MIN = 1;
const ZOOM_MAX = 6;
let zoomBadgeTimer = null;

function applyZoom() {
  const t = `translate(-50%, -50%) scale(${state.zoom})`;
  canvasOrig.style.transform = t;
  canvasProc.style.transform = t;
  updateSplitClip();
  showZoomBadge();
}

function showZoomBadge() {
  zoomBadge.textContent = Math.round(state.zoom * 100) + '%';
  zoomBadge.classList.add('visible');
  clearTimeout(zoomBadgeTimer);
  zoomBadgeTimer = setTimeout(() => zoomBadge.classList.remove('visible'), 900);
}

previewWrap.addEventListener('wheel', e => {
  if (!state.docW) return;
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
  state.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, state.zoom * factor));
  applyZoom();
}, { passive: false });

/* ─── Split divider ─── */
function updateDividerPos() {
  if (!state.docW) return;
  const px = state.splitX * previewWrap.clientWidth;
  splitDiv.style.left = px + 'px';
  updateSplitClip();
}

function updateSplitClip() {
  if (!state.docW) return;
  const wW = previewWrap.clientWidth;
  const dw = canvasOrig._dw || wW;
  const center = wW / 2;
  const splitPx = state.splitX * wW;
  // clip-path % is relative to the canvas's unscaled box, but transform:
  // scale(zoom) is applied AFTER the clip — divide out the zoom factor
  // to keep the visible split line aligned with the divider.
  const pct = 50 + ((splitPx - center) / (dw * state.zoom)) * 100;
  const pctClamped = Math.max(0, Math.min(100, pct));
  canvasProc.style.clipPath = `inset(0 0 0 ${pctClamped}%)`;
  canvasOrig.style.clipPath = `inset(0 ${100-pctClamped}% 0 0)`;
}

/* Split divider drag */
let splitDragging = false;
splitDiv.addEventListener('mousedown',  e => { splitDragging = true; e.preventDefault(); });
document.addEventListener('mousemove',  e => {
  if (!splitDragging) return;
  const r = previewWrap.getBoundingClientRect();
  state.splitX = Math.max(0.02, Math.min(0.98, (e.clientX - r.left) / r.width));
  updateDividerPos();
});
document.addEventListener('mouseup',   () => { splitDragging = false; });
splitDiv.addEventListener('touchstart', e => { splitDragging = true; e.preventDefault(); }, {passive:false});
document.addEventListener('touchmove',  e => {
  if (!splitDragging) return;
  const r = previewWrap.getBoundingClientRect();
  state.splitX = Math.max(0.02, Math.min(0.98, (e.touches[0].clientX - r.left) / r.width));
  updateDividerPos();
}, {passive:false});
document.addEventListener('touchend', () => { splitDragging = false; });

/* ─── Layer drag-to-move / mask painting (on canvas) ─── */
let layerDragging = false;
let layerDragStart = { x: 0, y: 0 };
let layerDragOrigin = { x: 0, y: 0 };
let maskPainting = false;
let maskLastPoint = null;

function docDeltaFromScreen(dx, dy) {
  const fx = state.docW / (canvasOrig._dw * state.zoom);
  const fy = state.docH / (canvasOrig._dh * state.zoom);
  return [dx * fx, dy * fy];
}

function screenToDoc(clientX, clientY) {
  const rect = canvasOrig.getBoundingClientRect();
  const fx = (clientX - rect.left) / rect.width;
  const fy = (clientY - rect.top) / rect.height;
  return [fx * state.docW, fy * state.docH];
}

previewWrap.addEventListener('mousedown', e => {
  if (splitDiv.contains(e.target)) return;

  if (state.maskEditLayerId) {
    const layer = getMaskLayer();
    if (!layer) return;
    maskPainting = true;
    maskLastPoint = screenToDoc(e.clientX, e.clientY);
    paintMaskAt(layer, maskLastPoint[0], maskLastPoint[1]);
    scheduleRender(true);
    e.preventDefault();
    return;
  }

  const layer = getSelectedLayer();
  if (!layer) return;
  layerDragging = true;
  layerDragStart = { x: e.clientX, y: e.clientY };
  layerDragOrigin = { x: layer.x, y: layer.y };
  e.preventDefault();
});
document.addEventListener('mousemove', e => {
  if (maskPainting) {
    const layer = getMaskLayer();
    if (!layer) return;
    const pt = screenToDoc(e.clientX, e.clientY);
    paintMaskStroke(layer, maskLastPoint[0], maskLastPoint[1], pt[0], pt[1]);
    maskLastPoint = pt;
    scheduleRender(true);
    return;
  }
  if (!layerDragging) return;
  const layer = getSelectedLayer();
  if (!layer) return;
  const [dx, dy] = docDeltaFromScreen(e.clientX - layerDragStart.x, e.clientY - layerDragStart.y);
  layer.x = layerDragOrigin.x + dx;
  layer.y = layerDragOrigin.y + dy;
  scheduleRender(true);
});
document.addEventListener('mouseup', () => { layerDragging = false; maskPainting = false; });

previewWrap.addEventListener('touchstart', e => {
  if (splitDiv.contains(e.target)) return;
  const t = e.touches[0];

  if (state.maskEditLayerId) {
    const layer = getMaskLayer();
    if (!layer) return;
    maskPainting = true;
    maskLastPoint = screenToDoc(t.clientX, t.clientY);
    paintMaskAt(layer, maskLastPoint[0], maskLastPoint[1]);
    scheduleRender(true);
    return;
  }

  const layer = getSelectedLayer();
  if (!layer) return;
  layerDragging = true;
  layerDragStart = { x: t.clientX, y: t.clientY };
  layerDragOrigin = { x: layer.x, y: layer.y };
}, { passive: true });
document.addEventListener('touchmove', e => {
  const t = e.touches[0];
  if (maskPainting) {
    const layer = getMaskLayer();
    if (!layer) return;
    const pt = screenToDoc(t.clientX, t.clientY);
    paintMaskStroke(layer, maskLastPoint[0], maskLastPoint[1], pt[0], pt[1]);
    maskLastPoint = pt;
    scheduleRender(true);
    return;
  }
  if (!layerDragging) return;
  const layer = getSelectedLayer();
  if (!layer) return;
  const [dx, dy] = docDeltaFromScreen(t.clientX - layerDragStart.x, t.clientY - layerDragStart.y);
  layer.x = layerDragOrigin.x + dx;
  layer.y = layerDragOrigin.y + dy;
  scheduleRender(true);
}, { passive: true });
document.addEventListener('touchend', () => { layerDragging = false; maskPainting = false; });

/* ─── Mask brush ─── */
function exitMaskEdit() {
  state.maskEditLayerId = null;
  maskPanel.style.display = 'none';
  brushCursor.style.display = 'none';
  renderLayerPanel();
}

function toggleMaskEdit(id) {
  if (state.maskEditLayerId === id) {
    exitMaskEdit();
    return;
  }
  state.maskEditLayerId = id;
  state.selectedLayerId = id;
  updateMaskPanel();
  maskPanel.style.display = '';
  renderLayerPanel();
}

function updateMaskPanel() {
  brushSizeSlider.value = state.brushSize;
  brushSizeVal.textContent = state.brushSize;
  brushHardnessSlider.value = state.brushHardness;
  brushHardnessVal.textContent = state.brushHardness;
  document.querySelectorAll('.mask-color-swatch').forEach(sw => {
    sw.classList.toggle('active', sw.dataset.color === state.maskColor);
  });
}

function paintMaskAt(layer, x, y) {
  const ctx = layer.mask.getContext('2d');
  const r = Math.max(0.5, state.brushSize / 2);
  const hardness = Math.min(0.999, state.brushHardness / 100);
  const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
  if (state.maskColor === 'black') {
    ctx.globalCompositeOperation = 'destination-out';
    grad.addColorStop(0, 'rgba(0,0,0,1)');
    grad.addColorStop(hardness, 'rgba(0,0,0,1)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
  } else {
    ctx.globalCompositeOperation = 'destination-over';
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(hardness, 'rgba(255,255,255,1)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
  }
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = 'source-over';
}

function paintMaskStroke(layer, x0, y0, x1, y1) {
  const dist = Math.hypot(x1 - x0, y1 - y0);
  const step = Math.max(1.5, state.brushSize / 4);
  const n = Math.max(1, Math.ceil(dist / step));
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    paintMaskAt(layer, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
  }
}

function updateBrushCursor(clientX, clientY) {
  if (!state.maskEditLayerId) { brushCursor.style.display = 'none'; return; }
  const rect = canvasOrig.getBoundingClientRect();
  const wrapRect = previewWrap.getBoundingClientRect();
  const scale = rect.width / state.docW;
  const size = Math.max(2, state.brushSize * scale);
  brushCursor.style.width  = size + 'px';
  brushCursor.style.height = size + 'px';
  brushCursor.style.left = (clientX - wrapRect.left) + 'px';
  brushCursor.style.top  = (clientY - wrapRect.top) + 'px';
  brushCursor.style.display = 'block';
}

previewWrap.addEventListener('mousemove', e => updateBrushCursor(e.clientX, e.clientY));
previewWrap.addEventListener('mouseleave', () => { brushCursor.style.display = 'none'; });

document.addEventListener('keydown', e => {
  if (!state.maskEditLayerId) return;
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (e.key.toLowerCase() === 'x') {
    state.maskColor = state.maskColor === 'black' ? 'white' : 'black';
    updateMaskPanel();
  } else if (e.key === 'Escape') {
    exitMaskEdit();
  }
});

maskCloseBtn.addEventListener('click', exitMaskEdit);
document.querySelectorAll('.mask-color-swatch').forEach(sw => {
  sw.addEventListener('click', () => {
    state.maskColor = sw.dataset.color;
    updateMaskPanel();
  });
});
brushSizeSlider.addEventListener('input', () => {
  state.brushSize = +brushSizeSlider.value;
  brushSizeVal.textContent = state.brushSize;
});
brushHardnessSlider.addEventListener('input', () => {
  state.brushHardness = +brushHardnessSlider.value;
  brushHardnessVal.textContent = state.brushHardness;
});

/* ─── Panel drag ─── */
function makePanelDraggable(panelEl, handleEl) {
  let dragging = false, start = {x:0,y:0}, orig = {x:0,y:0};
  handleEl.addEventListener('mousedown', e => {
    dragging = true;
    const r = panelEl.getBoundingClientRect();
    start = { x: e.clientX, y: e.clientY };
    orig  = { x: r.left,    y: r.top };
    panelEl.style.right = 'auto';
    panelEl.style.bottom = 'auto';
    panelEl.style.transform = 'none';
    panelEl.style.left = r.left + 'px';
    panelEl.style.top  = r.top  + 'px';
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    panelEl.style.left = Math.max(0, orig.x + dx) + 'px';
    panelEl.style.top  = Math.max(0, Math.min(window.innerHeight - 80, orig.y + dy)) + 'px';
  });
  document.addEventListener('mouseup', () => { dragging = false; });
}
makePanelDraggable($('panel'), $('panelHandle'));
makePanelDraggable(layersPanel, $('layersPanelHandle'));
makePanelDraggable(maskPanel, $('maskPanelHandle'));

/* ─── Export ─── */
exportBtn.addEventListener('click', () => {
  if (!state.docW) return;

  if (useGL) {
    renderAll();
    const a = document.createElement('a');
    a.download = 'processed.png';
    a.href = canvasProc.toDataURL('image/png');
    a.click();
    return;
  }

  const originalLabel = exportBtn.textContent;
  exportBtn.textContent = 'Exporting…';
  exportBtn.disabled = true;
  requestAnimationFrame(() => setTimeout(() => {
    compositeLayers();
    const off = document.createElement('canvas');
    const octx = off.getContext('2d');
    processInto(octx, compositeCanvas, state.docW, state.docH);
    const a = document.createElement('a');
    a.download = 'processed.png';
    a.href = off.toDataURL('image/png');
    a.click();
    exportBtn.textContent = originalLabel;
    exportBtn.disabled = false;
  }, 0));
});

/* ─── Reset (effects only) ─── */
resetBtn.addEventListener('click', () => {
  Object.assign(state, {
    mode:'none', threshold:128, thresholdSoft:0, invertThreshold:false,
    chRed:30, chGreen:59, chBlue:11, vibrance:0, saturation:0, sharpen:0,
  });
  [
    ['thresholdSlider','thresholdVal',128],
    ['thresholdSoftSlider','thresholdSoftVal',0],
    ['chRedSlider','chRedVal',30],
    ['chGreenSlider','chGreenVal',59],
    ['chBlueSlider','chBlueVal',11],
    ['vibranceSlider','vibranceVal',0],
    ['saturationSlider','saturationVal',0],
    ['sharpenSlider','sharpenVal',0],
  ].forEach(([s,v,val]) => { $(s).value = val; $(v).textContent = val; });
  invertCheck.checked = false;
  document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('.seg-btn[data-mode="none"]').classList.add('active');
  updateModeUI();
  updateChannelHint();
  renderProcessed();
});

/* ─── Resize ─── */
window.addEventListener('resize', () => {
  if (state.docW) {
    setupCanvases();
    renderAll();
    updateDividerPos();
    applyZoom();
  }
});

/* ─── Init ─── */
updateModeUI();
renderLayerPanel();
if (!useGL) console.warn('WebGL unavailable — using CPU fallback pipeline.');
