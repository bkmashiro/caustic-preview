/**
 * ui.js — UI controller for Caustic Lens Preview
 * Wires all HTML controls to CausticApp.setParam()
 */

'use strict';

document.addEventListener('DOMContentLoaded', () => {

  const app = window.CausticApp;

  // ─── Slider + display wiring ───────────────────────────────────────────────

  function wireSlider(id, paramKey, decimals = 1, suffix = '') {
    const slider = document.getElementById(id);
    const display = document.getElementById(id + '-val');
    if (!slider) return;

    function update() {
      const v = parseFloat(slider.value);
      if (isNaN(v)) return;
      if (display) display.textContent = v.toFixed(decimals) + suffix;
      app.setParam(paramKey, v);
    }

    slider.addEventListener('input', update);
    slider.addEventListener('change', update); // catches number-input blur/enter
    update(); // initialize display
  }

  wireSlider('azimuth',    'azimuth',    0, '°');
  wireSlider('elevation',  'elevation',  0, '°');
  wireSlider('intensity',  'intensity',  1);
  wireSlider('ior',        'ior',        2);
  wireSlider('exposure',   'exposure',   1);
  wireSlider('sigma',      'sigma',      3);
  wireSlider('block-w',    'blockW',     1);
  wireSlider('block-d',    'blockD',     1);
  wireSlider('block-h',    'blockH',     1);
  wireSlider('ground-dist','groundDist', 1);
  wireSlider('ground-y',   'groundY',   2);
  wireSlider('bump-amp',   'bumpAmp',    3);
  wireSlider('bump-freq',  'bumpFreq',   1);
  wireSlider('surface-res','surfaceRes', 0);

  // ─── Select ────────────────────────────────────────────────────────────────

  document.getElementById('surface-mode').addEventListener('change', e => {
    app.setParam('surfaceMode', e.target.value);
  });

  // ─── Color pickers ─────────────────────────────────────────────────────────

  function hexToRgb(hex) {
    const r = parseInt(hex.slice(1,3), 16) / 255;
    const g = parseInt(hex.slice(3,5), 16) / 255;
    const b = parseInt(hex.slice(5,7), 16) / 255;
    return [r, g, b];
  }

  function wireColor(id, paramKey) {
    const input = document.getElementById(id);
    if (!input) return;
    input.addEventListener('input', () => {
      app.setParam(paramKey, hexToRgb(input.value));
    });
    // Initialize
    app.setParam(paramKey, hexToRgb(input.value));
  }

  wireColor('caustic-color', 'causticColor');
  wireColor('ground-color',  'groundColor');
  wireColor('block-color',   'blockColor');

  // ─── Checkboxes ────────────────────────────────────────────────────────────

  document.getElementById('show-block').addEventListener('change', e => {
    app.setParam('showBlock', e.target.checked);
  });

  document.getElementById('show-grid').addEventListener('change', e => {
    app.setParam('showGrid', e.target.checked);
  });

  document.getElementById('show-caustic-only').addEventListener('change', e => {
    app.setParam('showCausticOnly', e.target.checked);
  });

  // ─── OBJ File Upload ──────────────────────────────────────────────────────

  document.getElementById('obj-file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;
    const status = document.getElementById('obj-status');
    status.textContent = `Loading ${file.name}…`;
    const reader = new FileReader();
    reader.onload = evt => {
      app.loadOBJ(evt.target.result);
    };
    reader.onerror = () => {
      status.textContent = 'Error reading file';
    };
    reader.readAsText(file);
  });

  // ─── Generate from Image (WASM pipeline) ──────────────────────────────────

  // ── Device presets ───────────────────────────────────────────────────────────
  const DEVICE_PRESETS = {
    form4_fast:  { resW: 48,  thicknessRatio: 0.15 },
    form4_fine:  { resW: 80,  thicknessRatio: 0.12 },
    form4_ultra: { resW: 128, thicknessRatio: 0.10 },
    hubs_sla:    { resW: 64,  thicknessRatio: 0.15 },
    preview:     { resW: 32,  thicknessRatio: 0.20 },
  };

  function updateComputedParams() {
    const presetKey = document.getElementById('device-preset').value;
    const preset    = DEVICE_PRESETS[presetKey] || DEVICE_PRESETS.form4_fine;
    const lensSize  = parseInt(document.getElementById('lens-size-mm').value, 10);
    const projDist  = parseInt(document.getElementById('proj-dist-mm').value, 10);

    // Update display labels
    document.getElementById('lens-size-mm-val').textContent = lensSize + ' mm';
    document.getElementById('proj-dist-mm-val').textContent = projDist + ' mm';

    // Derived parameters (normalised to lens width = 1.0)
    const focalL    = (projDist / lensSize).toFixed(2);
    const thickness = preset.thicknessRatio.toFixed(2);
    const resW      = preset.resW;

    // Write to hidden inputs (used by generate handler)
    document.getElementById('wasm-res-w').value      = resW;
    document.getElementById('wasm-focal-l').value    = focalL;
    document.getElementById('wasm-thickness').value  = thickness;

    // Update info display
    document.getElementById('computed-params-display').textContent =
      `res_w: ${resW} · focal_l: ${focalL} · thickness: ${thickness} · est. vertices: ${resW}²=${resW*resW}`;
  }

  ['device-preset','lens-size-mm','proj-dist-mm'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', updateComputedParams);
    if (el) el.addEventListener('input', updateComputedParams);
  });
  updateComputedParams(); // init

  // State
  let pngFileData = null;        // Uint8Array of the uploaded PNG
  let wasmWorker = null;         // active Web Worker (null when idle)
  let causticModuleReady = false;// becomes true once CausticModule is available
  let lastGeneratedObjText = null; // last successfully generated OBJ text

  // UI helpers
  const btnGenerate    = document.getElementById('btn-generate');
  const btnDownloadObj = document.getElementById('btn-download-obj');
  const progressWrap   = document.getElementById('wasm-progress-wrap');
  const progressBar    = document.getElementById('wasm-progress-bar');
  const wasmStatusEl   = document.getElementById('wasm-status');
  const imgPreviewWrap = document.getElementById('img-preview-wrap');
  const imgPreview     = document.getElementById('img-preview');

  function setWasmStatus(msg, isError = false) {
    wasmStatusEl.textContent = msg;
    wasmStatusEl.className = isError ? 'error' : '';
  }

  function setProgress(pct) {
    // pct === null → indeterminate spinner; 0–100 → determinate
    if (pct === null) {
      progressBar.style.width = '40%';
      progressBar.classList.add('indeterminate');
    } else {
      progressBar.classList.remove('indeterminate');
      progressBar.style.width = pct + '%';
    }
  }

  function showProgress(visible) {
    progressWrap.classList.toggle('visible', visible);
  }

  function updateGenerateBtn() {
    btnGenerate.disabled = !(pngFileData && causticModuleReady && !wasmWorker);
  }

  // PNG upload handler
  document.getElementById('png-file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = evt => {
      pngFileData = new Uint8Array(evt.target.result);

      // Show thumbnail and read natural image size to fix block aspect ratio
      const blob = new Blob([pngFileData], { type: 'image/png' });
      const url  = URL.createObjectURL(blob);
      imgPreview.onload = () => {
        URL.revokeObjectURL(url);
        // Match block W:D ratio to image W:H so caustic pattern isn't stretched
        const imgW = imgPreview.naturalWidth;
        const imgH = imgPreview.naturalHeight;
        if (imgW > 0 && imgH > 0) {
          const aspect = imgW / imgH;
          const curW = parseFloat(document.getElementById('block-w').value);
          const newD = +(curW / aspect).toFixed(2);
          const dSlider = document.getElementById('block-d');
          dSlider.value = Math.max(0.5, Math.min(4, newD));
          dSlider.dispatchEvent(new Event('input'));
        }
      };
      imgPreview.src = url;
      imgPreviewWrap.classList.add('has-image');

      setWasmStatus('Image loaded — click Generate to compute');
      updateGenerateBtn();
    };
    reader.onerror = () => setWasmStatus('Failed to read image file', true);
    reader.readAsArrayBuffer(file);
  });

  // Poll for CausticModule being available (loaded by wasm/caustic.js script tag)
  (function pollForWasm() {
    if (typeof CausticModule === 'function') {
      causticModuleReady = true;
      updateGenerateBtn();
      setWasmStatus('WASM module ready');
      showProgress(true);
      setProgress(100);
      setTimeout(() => {
        setWasmStatus('');
        showProgress(false);
        setProgress(null);
      }, 800);
    } else {
      setTimeout(pollForWasm, 200);
    }
  })();

  // ── Web Worker script (inlined as a blob URL) ──────────────────────────────
  //
  // The worker receives { pngData, resW, focalL, thickness } and the WASM
  // module script URL so it can importScripts() it.  It calls callMain(),
  // reads the output OBJ, and posts it back.
  //
  // We can't pass the already-instantiated CausticModule across the thread
  // boundary, so the worker loads it fresh.

  const WORKER_SRC = `
'use strict';

let mod = null;  // instantiated CausticModule

self.addEventListener('message', async (e) => {
  const { type, payload } = e.data;

  if (type === 'init') {
    // Import the Emscripten-generated JS glue
    try {
      self.postMessage({ type: 'status', text: 'Loading WASM module…' });
      importScripts(payload.wasmJsUrl);

      self.postMessage({ type: 'status', text: 'Instantiating WASM…' });
      mod = await CausticModule({
        // Tell Emscripten where to find the .wasm file
        locateFile(path) {
          if (path.endsWith('.wasm')) return payload.wasmBinUrl;
          return path;
        },
        // Suppress stdout/stderr spam to console
        print: () => {},
        printErr: () => {},
      });

      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', text: 'WASM init failed: ' + err.message });
    }
    return;
  }

  if (type === 'generate') {
    const { pngData, resW, focalL, thickness } = payload;
    try {
      self.postMessage({ type: 'status', text: 'Writing input image to WASM FS…' });
      mod.FS.writeFile('/input.png', pngData);

      self.postMessage({ type: 'status', text: 'Running caustic solver…' });
      const args = [
        '--input_png',  '/input.png',
        '--output',     '/',
        '--res_w',      String(resW),
        '--focal_l',    String(focalL),
        '--thickness',  String(thickness),
      ];
      // callMain throws ExitStatus on normal program exit — catch and ignore it,
      // then continue to read the output file.
      try { mod.callMain(args); } catch(e) {
        // ExitStatus is normal — program finished. Any other exception is real.
        const isExit = (e && (e.name === 'ExitStatus' || (e.message && e.message.toLowerCase().includes('exit'))));
        self.postMessage({ type: 'status', text: 'Solver exited: ' + (isExit ? 'OK (exit code)' : 'ERROR: ' + (e && e.message)) });
        if (!isExit) throw e;
      }

      // Check what files are in root FS
      let fsRoot = [];
      try { fsRoot = mod.FS.readdir('/'); } catch(_) {}
      self.postMessage({ type: 'status', text: 'FS root: ' + JSON.stringify(fsRoot) });

      self.postMessage({ type: 'status', text: 'Reading output mesh…' });
      const objText = mod.FS.readFile('/output.obj', { encoding: 'utf8' });

      // Clean up FS entries to avoid bloat on repeated runs
      try { mod.FS.unlink('/input.png');  } catch(_) {}
      try { mod.FS.unlink('/output.obj'); } catch(_) {}

      self.postMessage({ type: 'done', objText });
    } catch (err) {
      self.postMessage({ type: 'error', text: 'Generation failed: ' + err.message });
    }
    return;
  }
});
`;

  function makeWorkerUrl(src) {
    const blob = new Blob([src], { type: 'application/javascript' });
    return URL.createObjectURL(blob);
  }

  // Derive absolute URLs for the WASM assets so the worker can fetch them
  function resolveUrl(relativePath) {
    return new URL(relativePath, window.location.href).href;
  }

  // ── Generate button handler ────────────────────────────────────────────────

  btnGenerate.addEventListener('click', () => {
    if (!pngFileData || !causticModuleReady || wasmWorker) return;

    const resW      = parseInt(document.getElementById('wasm-res-w').value, 10);
    const focalL    = parseFloat(document.getElementById('wasm-focal-l').value);
    const thickness = parseFloat(document.getElementById('wasm-thickness').value);

    // Disable button during generation
    wasmWorker = true; // placeholder until real worker is assigned
    updateGenerateBtn();
    showProgress(true);
    setProgress(null); // indeterminate
    setWasmStatus('Starting Web Worker…');

    const workerUrl = makeWorkerUrl(WORKER_SRC);
    const worker = new Worker(workerUrl);
    wasmWorker = worker;

    const wasmJsUrl  = resolveUrl('wasm/caustic.js');
    const wasmBinUrl = resolveUrl('wasm/caustic.wasm');

    worker.onmessage = (e) => {
      const { type, text, objText } = e.data;

      if (type === 'status') {
        setWasmStatus(text);
        return;
      }

      if (type === 'ready') {
        setWasmStatus('Sending image data to worker…');
        // Send a transferable copy of the PNG buffer so it's fast
        const buf = pngFileData.buffer.slice(0);
        worker.postMessage(
          { type: 'generate', payload: { pngData: new Uint8Array(buf), resW, focalL, thickness } },
          [buf]
        );
        return;
      }

      if (type === 'done') {
        setWasmStatus('Parsing OBJ mesh…');
        setProgress(90);
        // Load the OBJ into the WebGL renderer
        try {
          // Auto-configure preview FIRST so params are correct when OBJ is parsed:
          //   groundDist = focalL × blockW   (projection distance)
          //   blockH     = thickness × blockW (lens thickness)
          const bW = parseFloat(document.getElementById('block-w').value);
          const targetGroundDist = +(focalL    * bW).toFixed(2);
          const targetBlockH     = +(thickness * bW).toFixed(2);

          const gdSlider = document.getElementById('ground-dist');
          gdSlider.value = Math.max(0, Math.min(12, targetGroundDist));
          gdSlider.dispatchEvent(new Event('input'));

          const bhSlider = document.getElementById('block-h');
          bhSlider.value = Math.max(0.1, Math.min(8, targetBlockH));
          bhSlider.dispatchEvent(new Event('input'));

          // Now parse OBJ — params.groundDist / blockH are already updated above
          const loadInfo = app.loadCausticOBJ(objText);
          lastGeneratedObjText = objText;
          if (btnDownloadObj) btnDownloadObj.disabled = false;

          // Auto-increase blockH if the actual lens depth exceeds our preset estimate.
          // This happens when the solver pushes some vertices slightly deeper than the
          // nominal thickness (numerical overshoot), causing those surface points to
          // fall below blockBottom → t1<0 in ray trace → missing caustic regions.
          let finalBlockH = targetBlockH;
          if (loadInfo && loadInfo.requiredBlockH > targetBlockH) {
            finalBlockH = +loadInfo.requiredBlockH.toFixed(2);
            const bhSlider = document.getElementById('block-h');
            bhSlider.value = Math.max(0.1, Math.min(8, finalBlockH));
            bhSlider.dispatchEvent(new Event('input'));
          }

          setProgress(100);
          setWasmStatus(`Done! groundDist→${targetGroundDist}, blockH→${finalBlockH}`);
        } catch (err) {
          setWasmStatus('OBJ parse error: ' + err.message, true);
        }
        cleanup();
        return;
      }

      if (type === 'error') {
        setWasmStatus(text, true);
        setProgress(null);
        cleanup(false);
        return;
      }
    };

    worker.onerror = (err) => {
      setWasmStatus('Worker error: ' + (err.message || 'unknown'), true);
      cleanup(false);
    };

    // Kick off WASM init in worker
    worker.postMessage({ type: 'init', payload: { wasmJsUrl, wasmBinUrl } });

    function cleanup(hideProgress = true) {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      wasmWorker = null;
      updateGenerateBtn();
      if (hideProgress) {
        setTimeout(() => {
          showProgress(false);
          setProgress(null);
        }, 2000);
      }
    }
  });

  // ─── Download OBJ button ─────────────────────────────────────────────────

  if (btnDownloadObj) {
    btnDownloadObj.addEventListener('click', () => {
      if (!lastGeneratedObjText) return;
      const blob = new Blob([lastGeneratedObjText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'caustic.obj';
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // ─── Hemisphere light direction UI ────────────────────────────────────────

  const hemiCanvas = document.getElementById('hemisphere-canvas');
  const hemiCtx = hemiCanvas.getContext('2d');
  const R = hemiCanvas.width / 2;

  let hemiDragging = false;

  function drawHemisphere() {
    const az = parseFloat(document.getElementById('azimuth').value) * Math.PI / 180;
    const el = parseFloat(document.getElementById('elevation').value) * Math.PI / 180;

    hemiCtx.clearRect(0, 0, hemiCanvas.width, hemiCanvas.height);

    // Background circle
    const grad = hemiCtx.createRadialGradient(R, R, 0, R, R, R);
    grad.addColorStop(0, '#1a2030');
    grad.addColorStop(1, '#0d0d1a');
    hemiCtx.beginPath();
    hemiCtx.arc(R, R, R - 1, 0, Math.PI * 2);
    hemiCtx.fillStyle = grad;
    hemiCtx.fill();
    hemiCtx.strokeStyle = '#2a3550';
    hemiCtx.lineWidth = 1;
    hemiCtx.stroke();

    // Elevation rings
    hemiCtx.strokeStyle = '#1e2840';
    hemiCtx.lineWidth = 0.5;
    for (let deg = 15; deg < 90; deg += 15) {
      const r = R * (1 - deg / 90);
      hemiCtx.beginPath();
      hemiCtx.arc(R, R, r, 0, Math.PI * 2);
      hemiCtx.stroke();
    }

    // Cardinal lines
    hemiCtx.beginPath();
    hemiCtx.moveTo(R, 1); hemiCtx.lineTo(R, R*2-1);
    hemiCtx.moveTo(1, R); hemiCtx.lineTo(R*2-1, R);
    hemiCtx.strokeStyle = '#1e2840';
    hemiCtx.lineWidth = 0.5;
    hemiCtx.stroke();

    // Direction labels
    hemiCtx.fillStyle = '#3a5070';
    hemiCtx.font = '9px system-ui';
    hemiCtx.textAlign = 'center';
    hemiCtx.fillText('N', R, 11);
    hemiCtx.fillText('S', R, R*2 - 3);
    hemiCtx.fillText('E', R*2 - 5, R + 4);
    hemiCtx.fillText('W', 5, R + 4);

    // Light dot position
    const projR = R * Math.cos(el); // projected radius (0 = zenith, R = horizon)
    const dotX = R + projR * Math.sin(az);
    const dotY = R - projR * Math.cos(az);

    // Shadow line
    hemiCtx.beginPath();
    hemiCtx.moveTo(R, R);
    hemiCtx.lineTo(dotX, dotY);
    hemiCtx.strokeStyle = 'rgba(255, 220, 100, 0.25)';
    hemiCtx.lineWidth = 1;
    hemiCtx.stroke();

    // Glow
    const grd = hemiCtx.createRadialGradient(dotX, dotY, 0, dotX, dotY, 12);
    grd.addColorStop(0, 'rgba(255, 220, 80, 0.9)');
    grd.addColorStop(0.4, 'rgba(255, 180, 40, 0.4)');
    grd.addColorStop(1, 'rgba(255, 180, 40, 0)');
    hemiCtx.beginPath();
    hemiCtx.arc(dotX, dotY, 12, 0, Math.PI * 2);
    hemiCtx.fillStyle = grd;
    hemiCtx.fill();

    // Dot
    hemiCtx.beginPath();
    hemiCtx.arc(dotX, dotY, 5, 0, Math.PI * 2);
    hemiCtx.fillStyle = '#ffdc50';
    hemiCtx.fill();
    hemiCtx.strokeStyle = '#ffa020';
    hemiCtx.lineWidth = 1.5;
    hemiCtx.stroke();

    // Elevation arc label
    hemiCtx.fillStyle = '#5a7090';
    hemiCtx.font = '9px system-ui';
    hemiCtx.textAlign = 'left';
    const elDeg = Math.round(parseFloat(document.getElementById('elevation').value));
    hemiCtx.fillText(`${elDeg}°`, dotX + 7, dotY - 5);
  }

  function hemiPointerToAngles(cx, cy) {
    const rect = hemiCanvas.getBoundingClientRect();
    const px = cx - rect.left - R;
    const py = cy - rect.top - R;
    const dist = Math.sqrt(px*px + py*py);
    if (dist > R) return null;

    // Az: angle from north (+Y = up in canvas = north)
    let az = Math.atan2(px, -py) * 180 / Math.PI;
    if (az < 0) az += 360;

    // El: distance from center maps to elevation (center=90°, edge=0°)
    const el = Math.max(5, Math.min(85, 90 * (1 - dist / R)));

    return { az: Math.round(az), el: Math.round(el) };
  }

  function updateFromHemi(e) {
    const { clientX, clientY } = e.touches ? e.touches[0] : e;
    const angles = hemiPointerToAngles(clientX, clientY);
    if (!angles) return;

    document.getElementById('azimuth').value = angles.az;
    document.getElementById('elevation').value = angles.el;
    document.getElementById('azimuth-val').textContent = angles.az + '°';
    document.getElementById('elevation-val').textContent = angles.el + '°';
    app.setParam('azimuth', angles.az);
    app.setParam('elevation', angles.el);
    drawHemisphere();
  }

  hemiCanvas.addEventListener('mousedown', e => {
    hemiDragging = true;
    updateFromHemi(e);
    e.preventDefault();
  });

  window.addEventListener('mousemove', e => {
    if (hemiDragging) updateFromHemi(e);
  });

  window.addEventListener('mouseup', () => { hemiDragging = false; });

  hemiCanvas.addEventListener('touchstart', e => {
    hemiDragging = true;
    updateFromHemi(e);
    e.preventDefault();
  }, { passive: false });

  hemiCanvas.addEventListener('touchmove', e => {
    if (hemiDragging) updateFromHemi(e);
    e.preventDefault();
  }, { passive: false });

  hemiCanvas.addEventListener('touchend', () => { hemiDragging = false; });

  // Keep hemisphere in sync when sliders change
  document.getElementById('azimuth').addEventListener('input', drawHemisphere);
  document.getElementById('elevation').addEventListener('input', drawHemisphere);

  drawHemisphere();

  // ─── Keyboard shortcuts ────────────────────────────────────────────────────

  document.addEventListener('keydown', e => {
    switch (e.key) {
      case '1': setCameraPreset('persp'); break;
      case '2': setCameraPreset('side');  break;
      case '3': setCameraPreset('top');   break;
      case 'b':
      case 'B': {
        const cb = document.getElementById('show-block');
        cb.checked = !cb.checked;
        app.setParam('showBlock', cb.checked);
        break;
      }
      case 'g':
      case 'G': {
        const cb = document.getElementById('show-grid');
        cb.checked = !cb.checked;
        app.setParam('showGrid', cb.checked);
        break;
      }
    }
  });

  // Update info bar
  document.getElementById('info-bar').textContent =
    'Drag to orbit · Scroll to zoom · Right-drag to pan (incl. vertical) · Keys: 1/2/3 = views · B = block · G = grid';

  // Fade info bar after 6s
  setTimeout(() => {
    const bar = document.getElementById('info-bar');
    bar.style.transition = 'opacity 1.5s';
    bar.style.opacity = '0.3';
  }, 6000);

});
