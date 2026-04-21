/**
 * caustic.js — WebGL2 Caustic Renderer
 *
 * Approach: Forward photon mapping on GPU
 *  Pass 1 (COMPUTE): For each surface sample on the block top, shoot a ray.
 *    - Compute surface normal from height field
 *    - Snell's law refraction into block (n1=1 → n2=IOR)
 *    - Snell's law refraction out of block bottom (n2=IOR → n1=1)
 *    - Project exit ray onto ground plane
 *    - Output UV on ground plane as gl_Position
 *    - Splat a soft disk with additive blending → caustic accumulation texture
 *
 *  Pass 2 (DISPLAY): Draw ground quad textured with accumulated caustic.
 *
 *  Pass 3 (SCENE): Render the transparent block + ground plane with
 *    Phong shading + the caustic texture on the ground.
 */

'use strict';

const CausticRenderer = (() => {

  // ─── State ────────────────────────────────────────────────────────────────
  let gl, canvas;
  let causticFBO, causticTex;
  let causticW = 1024, causticH = 1024;

  // Shader programs
  let progCompute, progDisplay, progScene, progGround, progGrid, progBlit;

  // Geometry buffers
  let surfaceVAO, surfaceVBO, surfaceIBO; // surface sample points + triangle index buffer
  let surfaceTriCount = 0;
  let quadVAO; // fullscreen quad
  let blockVAO, blockIBO, blockVertCount; // block mesh
  let groundVAO; // ground quad

  // Current params (updated by UI)
  let params = {
    azimuth: 45,
    elevation: 45,
    intensity: 3.0,
    ior: 1.5,
    exposure: 4.0,
    spread: 0.0,
    blockW: 2.0,
    blockD: 2.0,
    blockH: 1.0,
    groundDist: 1.5,
    surfaceMode: 'sinusoidal',
    bumpAmp: 0.05,
    bumpFreq: 4.0,
    surfaceRes: 128,
    causticColor: [1.0, 0.878, 0.565],
    groundColor: [0.125, 0.125, 0.157],
    blockColor: [0.541, 0.706, 0.816],
    showBlock: true,
    showGrid: true,
    showCausticOnly: false,
  };

  // Camera
  let camera = {
    theta: 0.6,   // azimuth (radians)
    phi: 0.9,     // elevation (radians)
    radius: 6.0,
    target: [0, 0, 0],
    fov: 45,
  };

  // OBJ data
  let objSurface = null; // { positions: Float32Array, normals: Float32Array, gridW, gridH }

  // Surface geometry cache
  let surfacePositions = null;
  let surfaceNormals = null;
  let surfaceGridW = 0, surfaceGridH = 0;
  let surfaceDirty = true;

  // ─── Math helpers ─────────────────────────────────────────────────────────
  const mat4 = {
    identity: () => new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
    multiply: (a, b) => {
      // Column-major multiply: C = A*B means C[col*4+row] = sum_k A[k*4+row] * B[col*4+k]
      const r = new Float32Array(16);
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
          for (let k = 0; k < 4; k++)
            r[i*4+j] += a[k*4+j] * b[i*4+k];
      return r;
    },
    perspective: (fov, aspect, near, far) => {
      const f = 1.0 / Math.tan(fov * Math.PI / 360);
      const nf = 1 / (near - far);
      const r = new Float32Array(16);
      r[0] = f / aspect; r[5] = f;
      r[10] = (far + near) * nf; r[11] = -1;
      r[14] = 2 * far * near * nf;
      return r;
    },
    lookAt: (eye, center, up) => {
      const f = normalize(sub3(center, eye));
      const s = normalize(cross3(f, up));
      const u = cross3(s, f);
      const r = new Float32Array(16);
      r[0]=s[0]; r[4]=s[1]; r[8]=s[2];
      r[1]=u[0]; r[5]=u[1]; r[9]=u[2];
      r[2]=-f[0]; r[6]=-f[1]; r[10]=-f[2];
      r[12]=-dot3(s,eye); r[13]=-dot3(u,eye); r[14]=dot3(f,eye); r[15]=1;
      return r;
    },
    translate: (tx, ty, tz) => {
      const r = mat4.identity();
      r[12]=tx; r[13]=ty; r[14]=tz;
      return r;
    },
  };

  const sub3 = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const add3 = (a,b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
  const scale3 = (a,s) => [a[0]*s, a[1]*s, a[2]*s];
  const dot3 = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const cross3 = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const normalize = (v) => { const l = Math.sqrt(dot3(v,v)); return l>1e-10?[v[0]/l,v[1]/l,v[2]/l]:[0,1,0]; };
  const len3 = (v) => Math.sqrt(dot3(v,v));

  // ─── Shader sources ────────────────────────────────────────────────────────

  // Pass 1: Compute caustic hit positions on ground plane
  // Each vertex = one surface sample; outputs a point on the caustic accumulation texture
  const VS_COMPUTE = `#version 300 es
precision highp float;

// Per-sample: position on block top surface + pre-computed normal
layout(location=0) in vec3 aSurfPos;   // world pos on block top surface
layout(location=1) in vec3 aSurfNorm;  // surface normal (pointing up-ish)

uniform vec3  uLightDir;      // direction FROM light TO surface (normalized, downward)
uniform float uIOR;           // refractive index of block (n2), air is n1=1.0
uniform float uBlockBottom;   // Y of block's flat bottom surface
uniform float uGroundY;       // Y of ground plane
uniform float uBlockW;        // block full width X
uniform float uBlockD;        // block full depth Z
// Caustic texture covers world XZ in range [-uGroundHalf, +uGroundHalf]
uniform float uGroundHalf;    // half-size of the ground region mapped to [0,1] UV

uniform float uSpread;
uniform float uIntensity;

out float vIntensity;
out vec2  vCausticUV;

vec3 snellRefract(vec3 I, vec3 N, float eta) {
  // I: incident direction (unit, toward surface)
  // N: surface normal (unit, pointing away from surface into incident medium)
  // eta: n1/n2
  float cosI = -dot(N, I);
  float sin2T = eta * eta * (1.0 - cosI * cosI);
  if (sin2T > 1.0) return vec3(0.0); // TIR — no transmission
  float cosT = sqrt(1.0 - sin2T);
  return eta * I + (eta * cosI - cosT) * N;
}

void main() {
  // --- Step 1: Refract into block at top surface ---
  vec3 N_top = normalize(aSurfNorm);
  vec3 I = normalize(uLightDir);
  float eta1 = 1.0 / uIOR;
  vec3 refracted1 = snellRefract(I, N_top, eta1);

  if (length(refracted1) < 0.01) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); gl_PointSize = 0.0; vIntensity = 0.0; vCausticUV = vec2(0.0); return;
  }

  // --- Step 2: Trace to block bottom ---
  vec3 pos = aSurfPos;
  float t1 = (abs(refracted1.y) > 1e-6) ? (uBlockBottom - pos.y) / refracted1.y : -1.0;
  if (t1 < 0.0) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); gl_PointSize = 0.0; vIntensity = 0.0; vCausticUV = vec2(0.0); return;
  }
  vec3 posBottom = pos + t1 * refracted1;
  if (abs(posBottom.x) > uBlockW * 0.5 + 0.01 || abs(posBottom.z) > uBlockD * 0.5 + 0.01) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); gl_PointSize = 0.0; vIntensity = 0.0; vCausticUV = vec2(0.0); return;
  }

  // --- Step 3: Refract out of block bottom (N into glass = up) ---
  vec3 N_bot = vec3(0.0, 1.0, 0.0);
  vec3 refracted2 = snellRefract(refracted1, N_bot, uIOR);
  if (length(refracted2) < 0.01) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); gl_PointSize = 0.0; vIntensity = 0.0; vCausticUV = vec2(0.0); return;
  }

  // --- Step 4: Intersect with ground ---
  float t2 = (abs(refracted2.y) > 1e-6) ? (uGroundY - posBottom.y) / refracted2.y : -1.0;
  if (t2 < 0.0) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); gl_PointSize = 0.0; vIntensity = 0.0; vCausticUV = vec2(0.0); return;
  }
  vec3 groundHit = posBottom + t2 * refracted2;

  // --- Step 5: Map to caustic FBO clip space ---
  vec2 uv = groundHit.xz / (uGroundHalf * 2.0) + 0.5;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    gl_Position = vec4(2.0, 2.0, 0.0, 1.0); gl_PointSize = 0.0; vIntensity = 0.0; vCausticUV = vec2(0.0); return;
  }

  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uSpread;
  vIntensity = uIntensity;
  vCausticUV = uv;
}
`;

  const FS_COMPUTE = `#version 300 es
precision highp float;
in float vIntensity;
in vec2  vCausticUV;
out vec4 fragColor;

void main() {
  // Soft disk falloff (POINTS mode)
  vec2 pc = gl_PointCoord - 0.5;
  float d = length(pc) * 2.0;
  float alpha = max(0.0, 1.0 - d * d);
  float energy = alpha * vIntensity * 0.008;
  fragColor = vec4(energy, energy, energy, energy);
}
`;

  // Pass 2: Render ground plane with caustic texture
  const VS_GROUND = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos; // -1..1 quad

uniform mat4 uMVP;
uniform vec3 uGroundCorner; // world space corner of ground quad (-half, y, -half)
uniform vec2 uGroundSize;   // world space size of ground quad (full size)
uniform float uGroundHalf;  // half-size of caustic texture coverage (matches Pass 1)

out vec2 vUV;
out vec3 vWorldPos;

void main() {
  // Map aPos (-1..1) to world XZ
  vec2 t = aPos * 0.5 + 0.5;
  vec3 world = vec3(
    uGroundCorner.x + t.x * uGroundSize.x,
    uGroundCorner.y,
    uGroundCorner.z + t.y * uGroundSize.y
  );
  vWorldPos = world;
  // Match Pass 1 UV mapping: world.xz in [-uGroundHalf, +uGroundHalf] maps to [0, 1]
  vUV = world.xz / (uGroundHalf * 2.0) + 0.5;
  gl_Position = uMVP * vec4(world, 1.0);
}
`;

  const FS_GROUND = `#version 300 es
precision highp float;
in vec2 vUV;
in vec3 vWorldPos;

uniform sampler2D uCausticTex;
uniform vec3 uCausticColor;
uniform vec3 uGroundColor;
uniform float uExposure;
uniform vec3 uLightDir; // from surface to light (negated)
uniform bool uShowGrid;

out vec4 fragColor;

float grid(vec2 p, float size) {
  vec2 g = abs(fract(p / size - 0.5) - 0.5) / fwidth(p / size);
  return 1.0 - min(min(g.x, g.y), 1.0);
}

// Soft 5-tap tent filter: smooths triangle-boundary seams in the caustic FBO
float sampleCaustic(sampler2D tex, vec2 uv) {
  vec2 px = 1.0 / vec2(textureSize(tex, 0));
  float c  = texture(tex, uv).r * 4.0;
  c += texture(tex, uv + vec2( px.x,  0.0)).r;
  c += texture(tex, uv + vec2(-px.x,  0.0)).r;
  c += texture(tex, uv + vec2( 0.0,  px.y)).r;
  c += texture(tex, uv + vec2( 0.0, -px.y)).r;
  return c / 8.0;
}

void main() {
  // Sample accumulated caustic with soft filter
  float caustic = sampleCaustic(uCausticTex, vUV);
  caustic = 1.0 - exp(-caustic * uExposure);

  // Ground base color with simple diffuse
  vec3 groundN = vec3(0.0, 1.0, 0.0);
  vec3 L = normalize(-uLightDir);
  float diff = max(dot(groundN, L), 0.0) * 0.3 + 0.15;
  vec3 col = uGroundColor * (diff + 0.1);

  // Add caustic
  col += uCausticColor * caustic;

  // Grid overlay
  if (uShowGrid) {
    float g = grid(vWorldPos.xz, 0.5) * 0.06;
    col += vec3(g);
  }

  fragColor = vec4(col, 1.0);
}
`;

  // Pass 3: Scene — render the glass block
  const VS_SCENE = `#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;

out vec3 vNorm;
out vec3 vWorldPos;
out vec3 vViewPos;

void main() {
  vWorldPos = (uModel * vec4(aPos, 1.0)).xyz;
  vNorm = normalize(uNormalMat * aNorm);
  gl_Position = uMVP * vec4(aPos, 1.0);
}
`;

  const FS_SCENE = `#version 300 es
precision highp float;
in vec3 vNorm;
in vec3 vWorldPos;

uniform vec3 uLightDir;    // toward light
uniform vec3 uCameraPos;
uniform vec3 uBlockColor;
uniform float uIOR;

out vec4 fragColor;

void main() {
  vec3 N = normalize(vNorm);
  vec3 L = normalize(-uLightDir);
  vec3 V = normalize(uCameraPos - vWorldPos);
  vec3 H = normalize(L + V);

  float diff = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), 64.0);

  // Fresnel (Schlick)
  float F0 = ((uIOR - 1.0) / (uIOR + 1.0));
  F0 *= F0;
  float fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);

  vec3 col = uBlockColor * (diff * 0.4 + 0.05);
  col += vec3(1.0) * spec * 0.8;
  col = mix(col, vec3(0.9, 0.95, 1.0), fresnel * 0.5);

  fragColor = vec4(col, 0.35 + fresnel * 0.4);
}
`;

  // Pass: Blit caustic FBO directly to screen (debug / caustic-only view)
  const VS_BLIT = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

  const FS_BLIT = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform float uExposure;
uniform vec3 uCausticColor;
out vec4 fragColor;
void main() {
  float v = texture(uTex, vUV).r;
  // tone-map same as ground shader
  v = 1.0 - exp(-v * uExposure);
  fragColor = vec4(uCausticColor * v, 1.0);
}
`;

  // ─── GL Utilities ──────────────────────────────────────────────────────────

  function createShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`Shader compile error:\n${err}`);
    }
    return s;
  }

  function createProgram(vsSrc, fsSrc) {
    const p = gl.createProgram();
    gl.attachShader(p, createShader(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, createShader(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const err = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error(`Program link error:\n${err}`);
    }
    return p;
  }

  function ul(prog, name) { return gl.getUniformLocation(prog, name); }

  // ─── Surface generation ───────────────────────────────────────────────────

  function heightField(x, z, mode, amp, freq) {
    switch (mode) {
      case 'sinusoidal':
        return amp * (Math.sin(x * freq) * Math.cos(z * freq));
      case 'concentric': {
        const r = Math.sqrt(x*x + z*z);
        return amp * Math.cos(r * freq * 1.5);
      }
      case 'diagonal':
        return amp * Math.sin((x + z) * freq * 0.7071);
      case 'random': {
        // Simple smooth noise approximation
        const px = x * freq * 0.3;
        const pz = z * freq * 0.3;
        const h = Math.sin(px*2.1+1.3)*Math.cos(pz*1.7+0.8)
                + Math.sin(px*4.3+2.1)*Math.cos(pz*3.9+1.4)*0.5
                + Math.sin(px*8.7+0.5)*Math.cos(pz*7.3+2.0)*0.25;
        return amp * h / 1.75;
      }
      case 'flat':
        return 0;
      default:
        return 0;
    }
  }

  function buildSurface() {
    const res = params.surfaceRes;
    const hw = params.blockW / 2;
    const hd = params.blockD / 2;
    const topY = params.groundDist + params.blockH; // Y of block top

    let positions, normals, gridW, gridH;

    if (params.surfaceMode === 'obj' && objSurface) {
      // Bilinear upsample OBJ grid to TARGET_RES×TARGET_RES so we always
      // have enough photon samples regardless of solver resolution setting.
      const TARGET_RES = 128;
      const srcW = objSurface.gridW, srcH = objSurface.gridH;
      const srcPos = objSurface.positions, srcNrm = objSurface.normals;

      if (srcW >= TARGET_RES && srcH >= TARGET_RES) {
        positions = srcPos; normals = srcNrm; gridW = srcW; gridH = srcH;
      } else {
        gridW = TARGET_RES; gridH = TARGET_RES;
        positions = new Float32Array(gridW * gridH * 3);
        normals   = new Float32Array(gridW * gridH * 3);

        for (let dj = 0; dj < gridH; dj++) {
          for (let di = 0; di < gridW; di++) {
            const fx = di / (gridW - 1) * (srcW - 1);
            const fy = dj / (gridH - 1) * (srcH - 1);
            const ix = Math.min(Math.floor(fx), srcW - 2);
            const iy = Math.min(Math.floor(fy), srcH - 2);
            const tx = fx - ix, ty = fy - iy;

            const i00 = (iy * srcW + ix) * 3;
            const i10 = (iy * srcW + ix + 1) * 3;
            const i01 = ((iy + 1) * srcW + ix) * 3;
            const i11 = ((iy + 1) * srcW + ix + 1) * 3;
            const out  = (dj * gridW + di) * 3;

            const w00 = (1-tx)*(1-ty), w10 = tx*(1-ty);
            const w01 = (1-tx)*ty,     w11 = tx*ty;

            for (let c = 0; c < 3; c++) {
              positions[out+c] = srcPos[i00+c]*w00 + srcPos[i10+c]*w10
                               + srcPos[i01+c]*w01 + srcPos[i11+c]*w11;
              normals[out+c]   = srcNrm[i00+c]*w00 + srcNrm[i10+c]*w10
                               + srcNrm[i01+c]*w01 + srcNrm[i11+c]*w11;
            }
            // Re-normalise the interpolated normal
            const nx=normals[out], ny=normals[out+1], nz=normals[out+2];
            const nl = Math.sqrt(nx*nx+ny*ny+nz*nz) || 1;
            normals[out] /= nl; normals[out+1] /= nl; normals[out+2] /= nl;
          }
        }
      }
    } else {
      gridW = res;
      gridH = res;
      const N = gridW * gridH;
      positions = new Float32Array(N * 3);
      normals = new Float32Array(N * 3);

      const mode = params.surfaceMode === 'obj' ? 'sinusoidal' : params.surfaceMode;
      const amp = params.bumpAmp;
      const freq = params.bumpFreq;
      const dx = params.blockW / (gridW - 1);
      const dz = params.blockD / (gridH - 1);

      for (let j = 0; j < gridH; j++) {
        for (let i = 0; i < gridW; i++) {
          const idx = j * gridW + i;
          const wx = -hw + i * dx;
          const wz = -hd + j * dz;
          const wy = topY + heightField(wx, wz, mode, amp, freq);

          positions[idx*3+0] = wx;
          positions[idx*3+1] = wy;
          positions[idx*3+2] = wz;

          // Finite-difference normal
          const eps = dx * 0.5;
          const hL = heightField(wx - eps, wz, mode, amp, freq);
          const hR = heightField(wx + eps, wz, mode, amp, freq);
          const hD = heightField(wx, wz - eps, mode, amp, freq);
          const hU = heightField(wx, wz + eps, mode, amp, freq);

          const nx = -(hR - hL) / (2 * eps);
          const nz = -(hU - hD) / (2 * eps);
          const ny = 1.0;
          const nl = Math.sqrt(nx*nx + ny*ny + nz*nz);
          normals[idx*3+0] = nx / nl;
          normals[idx*3+1] = ny / nl;
          normals[idx*3+2] = nz / nl;
        }
      }
    }

    surfacePositions = positions;
    surfaceNormals = normals;
    surfaceGridW = gridW;
    surfaceGridH = gridH;

    // Build triangle index buffer for the grid
    const indices = new Uint32Array((gridW - 1) * (gridH - 1) * 6);
    let idx = 0;
    for (let j = 0; j < gridH - 1; j++) {
      for (let i = 0; i < gridW - 1; i++) {
        const a = j * gridW + i;
        const b = a + 1;
        const c = a + gridW;
        const d = c + 1;
        indices[idx++] = a; indices[idx++] = b; indices[idx++] = d;
        indices[idx++] = a; indices[idx++] = d; indices[idx++] = c;
      }
    }
    surfaceTriCount = idx;

    // Upload to GPU
    if (!surfaceVAO) {
      surfaceVAO = gl.createVertexArray();
      surfaceVBO = [gl.createBuffer(), gl.createBuffer()];
      surfaceIBO = gl.createBuffer();
    }
    gl.bindVertexArray(surfaceVAO);

    gl.bindBuffer(gl.ARRAY_BUFFER, surfaceVBO[0]);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, surfaceVBO[1]);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, surfaceIBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);

    gl.bindVertexArray(null);

    surfaceDirty = false;
  }

  // ─── Block mesh ────────────────────────────────────────────────────────────

  function buildBlock() {
    if (blockVAO) {
      gl.deleteVertexArray(blockVAO);
      gl.deleteBuffer(blockIBO);
    }
    blockVAO = gl.createVertexArray();
    const vbo = gl.createBuffer();
    blockIBO = gl.createBuffer();

    const hw = params.blockW / 2;
    const hd = params.blockD / 2;
    const bot = params.groundDist;
    const top = params.groundDist + params.blockH;
    const res = Math.min(params.surfaceRes, 64);
    const mode = params.surfaceMode === 'obj' ? 'sinusoidal' : params.surfaceMode;
    const amp  = params.surfaceMode === 'obj' ? 0 : params.bumpAmp; // flat top when OBJ loaded
    const freq = params.bumpFreq;

    const verts = [];
    const indices = [];

    function pushFace(positions, normals) {
      const base = verts.length / 6;
      for (let i = 0; i < positions.length; i++) {
        verts.push(positions[i][0], positions[i][1], positions[i][2],
                   normals[i][0],  normals[i][1],  normals[i][2]);
      }
      return base;
    }

    // Bottom face (flat)
    {
      const base = verts.length / 6;
      verts.push(-hw, bot, -hd,  0,-1,0,
                  hw, bot, -hd,  0,-1,0,
                  hw, bot,  hd,  0,-1,0,
                 -hw, bot,  hd,  0,-1,0);
      indices.push(base, base+1, base+2, base, base+2, base+3);
    }

    // Top surface (bumpy)
    {
      const dx = params.blockW / (res - 1);
      const dz = params.blockD / (res - 1);
      const base = verts.length / 6;
      for (let j = 0; j < res; j++) {
        for (let i = 0; i < res; i++) {
          const wx = -hw + i * dx;
          const wz = -hd + j * dz;
          const wy = top + heightField(wx, wz, mode, amp, freq);
          const eps = dx * 0.5;
          const hL = heightField(wx-eps, wz, mode, amp, freq);
          const hR = heightField(wx+eps, wz, mode, amp, freq);
          const hD = heightField(wx, wz-eps, mode, amp, freq);
          const hU = heightField(wx, wz+eps, mode, amp, freq);
          const nx = -(hR-hL)/(2*eps), nz = -(hU-hD)/(2*eps), ny = 1.0;
          const nl = Math.sqrt(nx*nx+ny*ny+nz*nz);
          verts.push(wx, wy, wz, nx/nl, ny/nl, nz/nl);
        }
      }
      for (let j = 0; j < res-1; j++) {
        for (let i = 0; i < res-1; i++) {
          const a = base + j*res+i;
          const b = a+1, c = a+res, d = a+res+1;
          indices.push(a,b,d, a,d,c);
        }
      }
    }

    // Side faces (flat)
    const sides = [
      // -X face
      [[-hw,bot,-hd,  -1,0,0],  [-hw,bot,hd,  -1,0,0],  [-hw,top,-hd, -1,0,0],  [-hw,top,hd,  -1,0,0]],
      // +X face
      [[hw,bot,hd,   1,0,0],   [hw,bot,-hd,  1,0,0],   [hw,top,hd,  1,0,0],   [hw,top,-hd, 1,0,0]],
      // -Z face
      [[hw,bot,-hd,  0,0,-1],  [-hw,bot,-hd, 0,0,-1],  [hw,top,-hd, 0,0,-1],  [-hw,top,-hd, 0,0,-1]],
      // +Z face
      [[-hw,bot,hd,  0,0,1],   [hw,bot,hd,  0,0,1],   [-hw,top,hd, 0,0,1],   [hw,top,hd, 0,0,1]],
    ];
    for (const side of sides) {
      const base = verts.length / 6;
      for (const v of side) verts.push(...v);
      indices.push(base, base+1, base+2, base+1, base+3, base+2);
    }

    const va = new Float32Array(verts);
    const ia = new Uint32Array(indices);
    blockVertCount = indices.length;

    gl.bindVertexArray(blockVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, va, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, blockIBO);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ia, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
  }

  // ─── Caustic FBO ──────────────────────────────────────────────────────────

  let useHalfFloat = true;

  function buildCausticFBO() {
    if (causticFBO) {
      gl.deleteFramebuffer(causticFBO);
      gl.deleteTexture(causticTex);
    }
    causticTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, causticTex);
    if (useHalfFloat) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, causticW, causticH, 0, gl.RGBA, gl.HALF_FLOAT, null);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, causticW, causticH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    causticFBO = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, causticFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, causticTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // ─── Ground & fullscreen quad ──────────────────────────────────────────────

  function buildQuad() {
    const data = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
    groundVAO = gl.createVertexArray();
    gl.bindVertexArray(groundVAO);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    quadVAO = groundVAO; // same geometry
  }

  // ─── Light direction ──────────────────────────────────────────────────────

  function getLightDir() {
    const az = params.azimuth * Math.PI / 180;
    const el = params.elevation * Math.PI / 180;
    // Direction FROM light TO surface (downward into scene)
    return [
      -Math.cos(el) * Math.sin(az),
      -Math.sin(el),
      -Math.cos(el) * Math.cos(az),
    ];
  }

  // ─── Camera matrices ───────────────────────────────────────────────────────

  function getCameraPos() {
    const { theta, phi, radius, target } = camera;
    return [
      target[0] + radius * Math.cos(phi) * Math.sin(theta),
      target[1] + radius * Math.sin(phi),
      target[2] + radius * Math.cos(phi) * Math.cos(theta),
    ];
  }

  function getViewMatrix() {
    const eye = getCameraPos();
    return mat4.lookAt(eye, camera.target, [0,1,0]);
  }

  function getProjMatrix() {
    const aspect = canvas.width / canvas.height;
    return mat4.perspective(camera.fov, aspect, 0.01, 100);
  }

  function getMVP(model) {
    const V = getViewMatrix();
    const P = getProjMatrix();
    if (model) return mat4.multiply(P, mat4.multiply(V, model));
    return mat4.multiply(P, V);
  }

  function getNormalMatrix(model) {
    // Return upper-left 3x3 of model (no non-uniform scaling assumed)
    return new Float32Array([
      model[0], model[1], model[2],
      model[4], model[5], model[6],
      model[8], model[9], model[10],
    ]);
  }

  // ─── Render ────────────────────────────────────────────────────────────────

  let frameCount = 0;
  let lastFpsTime = 0;
  let lastFrameMs = 0;
  let debugLogged = false;

  function render(now) {
    if (surfaceDirty) {
      buildSurface();
      buildBlock();
    }

    if (!debugLogged && surfacePositions) {
      const nPts = surfaceGridW * surfaceGridH;
      const ys = Array.from({ length: nPts }, (_, i) => surfacePositions[i * 3 + 1]);
      console.log('[caustic debug] Surface points:', nPts);
      console.log('[caustic debug] Surface Y range:', Math.min(...ys).toFixed(4), 'to', Math.max(...ys).toFixed(4));
      console.log('[caustic debug] Camera pos:', getCameraPos().map(x => x.toFixed(3)));
      debugLogged = true;
    }

    const t0 = performance.now();

    const lightDir = getLightDir();
    const groundY = 0;
    const blockBottom = params.groundDist;
    const groundSize = Math.max(params.blockW, params.blockD) * 5;

    // ── Pass 1: Accumulate caustics into FBO ──────────────────────────────
    gl.bindFramebuffer(gl.FRAMEBUFFER, causticFBO);
    gl.viewport(0, 0, causticW, causticH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Additive blending — photon energy accumulates
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.DEPTH_TEST);

    gl.useProgram(progCompute);
    gl.uniform3fv(ul(progCompute, 'uLightDir'), lightDir);
    gl.uniform1f(ul(progCompute, 'uIOR'), params.ior);
    gl.uniform1f(ul(progCompute, 'uBlockBottom'), blockBottom);
    gl.uniform1f(ul(progCompute, 'uGroundY'), groundY);
    gl.uniform1f(ul(progCompute, 'uBlockW'), params.blockW);
    gl.uniform1f(ul(progCompute, 'uBlockD'), params.blockD);
    gl.uniform1f(ul(progCompute, 'uGroundHalf'), groundSize / 2);
    // Scale intensity to normalise for point count vs 128×128 reference
    const numPts = surfaceGridW * surfaceGridH;
    const refPts = 128 * 128;
    const ptScale = refPts / Math.max(numPts, 1);
    gl.uniform1f(ul(progCompute, 'uIntensity'), params.intensity * ptScale);

    // Auto-compute minimum spread so adjacent photons always overlap on the FBO.
    // Point spacing on ground (in FBO pixels): blockW / (numPtsX-1) / groundSize * causticW
    const ptSpacingPx = (params.blockW / Math.max(surfaceGridW - 1, 1)) / groundSize * causticW;
    const autoMinSpread = ptSpacingPx * 2.5; // diameter = 2.5× spacing → comfortable overlap
    // User 'spread' param adds softness on top; auto ensures gap-free floor.
    gl.uniform1f(ul(progCompute, 'uSpread'), Math.max(params.spread, autoMinSpread));

    gl.bindVertexArray(surfaceVAO);
    gl.drawArrays(gl.POINTS, 0, numPts);
    gl.bindVertexArray(null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ── Caustic-only view: blit FBO texture directly to screen ────────────
    if (params.showCausticOnly) {
      const W = canvas.width, H = canvas.height;
      gl.viewport(0, 0, W, H);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);

      gl.useProgram(progBlit);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, causticTex);
      gl.uniform1i(ul(progBlit, 'uTex'), 0);
      gl.uniform1f(ul(progBlit, 'uExposure'), params.exposure);
      gl.uniform3fv(ul(progBlit, 'uCausticColor'), params.causticColor);

      gl.bindVertexArray(quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);

      lastFrameMs = performance.now() - t0;
      frameCount++;
      const nPtsDisp = surfaceGridW * surfaceGridH;
      document.getElementById('perf-display').textContent =
        `⏱ ${lastFrameMs.toFixed(1)}ms | ${nPtsDisp.toLocaleString()} pts`;
      requestAnimationFrame(render);
      return;
    }

    // ── Pass 2 + 3: Render scene ──────────────────────────────────────────
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width, H = canvas.height;
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    const model = mat4.identity();
    const mvp = getMVP(model);
    const cameraPos = getCameraPos();

    // Ground plane
    gl.useProgram(progGround);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, causticTex);
    gl.uniform1i(ul(progGround, 'uCausticTex'), 0);
    gl.uniform3fv(ul(progGround, 'uGroundCorner'), [-groundSize/2, groundY, -groundSize/2]);
    gl.uniform2fv(ul(progGround, 'uGroundSize'), [groundSize, groundSize]);
    gl.uniform1f(ul(progGround, 'uGroundHalf'), groundSize / 2);
    gl.uniform3fv(ul(progGround, 'uCausticColor'), params.causticColor);
    gl.uniform3fv(ul(progGround, 'uGroundColor'), params.groundColor);
    gl.uniform1f(ul(progGround, 'uExposure'), params.exposure);
    gl.uniform3fv(ul(progGround, 'uLightDir'), lightDir);
    gl.uniform1i(ul(progGround, 'uShowGrid'), params.showGrid ? 1 : 0);
    gl.uniformMatrix4fv(ul(progGround, 'uMVP'), false, mvp);

    gl.bindVertexArray(groundVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);

    // Glass block (transparent, rendered last)
    if (params.showBlock) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);

      gl.useProgram(progScene);
      gl.uniformMatrix4fv(ul(progScene, 'uMVP'), false, mvp);
      gl.uniformMatrix4fv(ul(progScene, 'uModel'), false, model);
      gl.uniformMatrix3fv(ul(progScene, 'uNormalMat'), false, getNormalMatrix(model));
      gl.uniform3fv(ul(progScene, 'uLightDir'), lightDir);
      gl.uniform3fv(ul(progScene, 'uCameraPos'), cameraPos);
      gl.uniform3fv(ul(progScene, 'uBlockColor'), params.blockColor);
      gl.uniform1f(ul(progScene, 'uIOR'), params.ior);

      gl.bindVertexArray(blockVAO);
      gl.drawElements(gl.TRIANGLES, blockVertCount, gl.UNSIGNED_INT, 0);
      gl.bindVertexArray(null);

      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }

    lastFrameMs = performance.now() - t0;
    frameCount++;

    // Update perf display
    document.getElementById('perf-display').textContent =
      `⏱ ${lastFrameMs.toFixed(1)}ms | ${numPts.toLocaleString()} pts`;

    requestAnimationFrame(render);
  }

  // ─── OBJ Loader ───────────────────────────────────────────────────────────

  function parseOBJ(text) {
    const rawVerts = [];
    const rawNorms = [];
    const faces = [];
    const lines = text.split('\n');

    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts[0] === 'v') {
        rawVerts.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
      } else if (parts[0] === 'vn') {
        rawNorms.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
      } else if (parts[0] === 'f') {
        const face = [];
        for (let i = 1; i < parts.length; i++) {
          const sp = parts[i].split('/');
          face.push({ v: parseInt(sp[0])-1, n: sp[2] ? parseInt(sp[2])-1 : -1 });
        }
        faces.push(face);
      }
    }

    if (rawVerts.length === 0) return null;

    // Find the top surface: vertices with highest Y values (top surface of block)
    let maxY = -Infinity;
    for (const v of rawVerts) maxY = Math.max(maxY, v[1]);
    const topThresh = maxY - 0.001;
    const topVerts = rawVerts.filter(v => v[1] >= topThresh - (maxY - Math.min(...rawVerts.map(v=>v[1]))) * 0.1);

    // Actually use ALL vertices in a grid-like arrangement
    // Sort by XZ to form a grid
    const allVerts = [...rawVerts];
    allVerts.sort((a,b) => a[2] === b[2] ? a[0]-b[0] : a[2]-b[2]);

    // Estimate grid dimensions
    const xs = [...new Set(allVerts.map(v => Math.round(v[0]*1000)))].sort((a,b)=>a-b);
    const zs = [...new Set(allVerts.map(v => Math.round(v[2]*1000)))].sort((a,b)=>a-b);
    const gridW = xs.length;
    const gridH = zs.length;

    if (gridW < 2 || gridH < 2) {
      // Fall back to just using all verts as unstructured
      const N = Math.min(rawVerts.length, 65536);
      const positions = new Float32Array(N * 3);
      const normals = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        positions[i*3+0] = rawVerts[i][0];
        positions[i*3+1] = rawVerts[i][1];
        positions[i*3+2] = rawVerts[i][2];
        normals[i*3+0] = 0; normals[i*3+1] = 1; normals[i*3+2] = 0;
      }
      return { positions, normals, gridW: N, gridH: 1 };
    }

    // Build structured grid
    const N = gridW * gridH;
    const positions = new Float32Array(N * 3);
    const normals = new Float32Array(N * 3);

    const vertMap = new Map();
    for (const v of rawVerts) {
      const key = `${Math.round(v[0]*1000)},${Math.round(v[2]*1000)}`;
      vertMap.set(key, v);
    }

    for (let j = 0; j < gridH; j++) {
      for (let i = 0; i < gridW; i++) {
        const idx = j * gridW + i;
        const key = `${xs[i]},${zs[j]}`;
        const v = vertMap.get(key) || [xs[i]/1000, maxY, zs[j]/1000];
        positions[idx*3+0] = v[0];
        positions[idx*3+1] = v[1];
        positions[idx*3+2] = v[2];

        // Compute normal by finite differences in grid
        const vL = j*gridW + Math.max(i-1, 0);
        const vR = j*gridW + Math.min(i+1, gridW-1);
        const vD = Math.max(j-1, 0)*gridW + i;
        const vU = Math.min(j+1, gridH-1)*gridW + i;
        // Will compute after all positions are filled
        normals[idx*3+0] = 0;
        normals[idx*3+1] = 1;
        normals[idx*3+2] = 0;
      }
    }

    // Compute normals via cross product
    for (let j = 1; j < gridH-1; j++) {
      for (let i = 1; i < gridW-1; i++) {
        const idx = j*gridW+i;
        const L = positions.slice((j*gridW+i-1)*3, (j*gridW+i-1)*3+3);
        const R = positions.slice((j*gridW+i+1)*3, (j*gridW+i+1)*3+3);
        const D = positions.slice(((j-1)*gridW+i)*3, ((j-1)*gridW+i)*3+3);
        const U = positions.slice(((j+1)*gridW+i)*3, ((j+1)*gridW+i)*3+3);
        const dx = [R[0]-L[0], R[1]-L[1], R[2]-L[2]];
        const dz = [U[0]-D[0], U[1]-D[1], U[2]-D[2]];
        const n = cross3(dz, dx);
        const nl = Math.sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]) || 1;
        normals[idx*3+0] = n[0]/nl;
        normals[idx*3+1] = Math.abs(n[1]/nl);
        normals[idx*3+2] = n[2]/nl;
      }
    }

    return { positions, normals, gridW, gridH };
  }

  // ─── Caustic OBJ Loader ───────────────────────────────────────────────────

  function parseCausticOBJ(text) {
    const rawVerts = [];
    for (const line of text.split('\n')) {
      const parts = line.trim().split(/\s+/);
      if (parts[0] === 'v') {
        rawVerts.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
      }
    }

    if (rawVerts.length === 0) return null;

    // First half = top curved surface (X,Y in [0,1], Z = height deformation ≤ 0)
    const N = Math.floor(rawVerts.length / 2);
    const topVerts = rawVerts.slice(0, N);

    // Build grid from unique X and Y values
    const xSet = [...new Set(topVerts.map(v => Math.round(v[0] * 100000)))].sort((a, b) => a - b);
    const ySet = [...new Set(topVerts.map(v => Math.round(v[1] * 100000)))].sort((a, b) => a - b);
    const gridW = xSet.length;
    const gridH = ySet.length;

    if (gridW < 2 || gridH < 2) return null;

    const { blockW, blockD, blockH } = params;

    const positions = new Float32Array(gridW * gridH * 3);
    const normals   = new Float32Array(gridW * gridH * 3);

    // Map each vertex into a lookup by (xi, yi) index
    const vertMap = new Map();
    for (const v of topVerts) {
      const xi = Math.round(v[0] * 100000);
      const yi = Math.round(v[1] * 100000);
      vertMap.set(`${xi},${yi}`, v);
    }

    for (let j = 0; j < gridH; j++) {
      for (let i = 0; i < gridW; i++) {
        const idx = j * gridW + i;
        const key = `${xSet[i]},${ySet[j]}`;
        const v = vertMap.get(key) || [xSet[i] / 100000, ySet[j] / 100000, 0];
        const obj_x = v[0];
        const obj_y = v[1];
        const obj_z = v[2]; // negative height deformation

        // Map to renderer coordinates
        // Caustic OBJ: X,Y in [0,1] = 2D grid; Z ≤ 0 = height deformation (in units of lens_width=1.0)
        // Renderer: Y=up, top of block at groundDist+blockH
        positions[idx*3+0] = (obj_x - 0.5) * blockW;
        positions[idx*3+1] = (params.groundDist + blockH) + obj_z * blockW;
        positions[idx*3+2] = (obj_y - 0.5) * blockD;

        normals[idx*3+0] = 0;
        normals[idx*3+1] = 1;
        normals[idx*3+2] = 0;
      }
    }

    // Compute normals via finite differences
    for (let j = 1; j < gridH - 1; j++) {
      for (let i = 1; i < gridW - 1; i++) {
        const idx = j * gridW + i;
        const L = positions.slice((j*gridW + i - 1)*3, (j*gridW + i - 1)*3 + 3);
        const R = positions.slice((j*gridW + i + 1)*3, (j*gridW + i + 1)*3 + 3);
        const D = positions.slice(((j-1)*gridW + i)*3, ((j-1)*gridW + i)*3 + 3);
        const U = positions.slice(((j+1)*gridW + i)*3, ((j+1)*gridW + i)*3 + 3);
        const dx = [R[0]-L[0], R[1]-L[1], R[2]-L[2]];
        const dz = [U[0]-D[0], U[1]-D[1], U[2]-D[2]];
        const n = cross3(dz, dx);
        const nl = Math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]) || 1;
        normals[idx*3+0] = n[0] / nl;
        normals[idx*3+1] = Math.abs(n[1] / nl);
        normals[idx*3+2] = n[2] / nl;
      }
    }

    return { positions, normals, gridW, gridH };
  }

  function loadCausticOBJ(text) {
    const result = parseCausticOBJ(text);
    if (!result) {
      document.getElementById('obj-status').textContent = 'Failed to parse caustic OBJ';
      return;
    }
    objSurface = result;
    params.surfaceMode = 'obj';
    document.getElementById('surface-mode').value = 'obj';
    surfaceDirty = true;
    const pts = result.gridW * result.gridH;
    document.getElementById('obj-status').textContent =
      `Loaded: ${pts.toLocaleString()} surface points (${result.gridW}×${result.gridH})`;
  }

  // ─── Init ──────────────────────────────────────────────────────────────────

  function init() {
    canvas = document.getElementById('main-canvas');

    // Size canvas
    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
    }
    resize();
    window.addEventListener('resize', () => { resize(); });

    // Get WebGL2
    gl = canvas.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) {
      alert('WebGL2 not supported. Please use a modern browser.');
      return;
    }

    // Check for half-float FBO support
    const ext = gl.getExtension('EXT_color_buffer_half_float') ||
                gl.getExtension('EXT_color_buffer_float');
    if (!ext) {
      useHalfFloat = false;
      causticW = 512; causticH = 512;
      console.warn('Half-float FBO not supported; falling back to RGBA8');
    }

    // Compile shaders
    try {
      progCompute = createProgram(VS_COMPUTE, FS_COMPUTE);
      progScene   = createProgram(VS_SCENE, FS_SCENE);
      progGround  = createProgram(VS_GROUND, FS_GROUND);
      progBlit    = createProgram(VS_BLIT, FS_BLIT);
    } catch(e) {
      console.error(e);
      alert('Shader compilation failed:\n' + e.message);
      return;
    }

    buildCausticFBO();
    buildQuad();
    buildSurface();
    buildBlock();

    setupInteraction();

    requestAnimationFrame(render);
  }

  // ─── Mouse / Touch interaction ────────────────────────────────────────────

  function setupInteraction() {
    let dragging = false;
    let rightDrag = false;
    let lastX = 0, lastY = 0;

    canvas.addEventListener('mousedown', e => {
      dragging = true;
      rightDrag = e.button === 2;
      lastX = e.clientX;
      lastY = e.clientY;
      e.preventDefault();
    });

    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;

      if (rightDrag) {
        // Pan
        const scale = camera.radius * 0.003;
        const t = camera.theta;
        camera.target[0] -= (Math.cos(t)*dx + Math.sin(t)*0) * scale;
        camera.target[2] -= (Math.sin(t)*dx - Math.cos(t)*0) * scale;
        camera.target[1] += dy * scale;
      } else {
        // Orbit
        camera.theta -= dx * 0.007;
        camera.phi = Math.max(0.05, Math.min(Math.PI/2 - 0.02, camera.phi - dy * 0.007));
      }
    });

    window.addEventListener('mouseup', () => { dragging = false; });
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    canvas.addEventListener('wheel', e => {
      camera.radius = Math.max(1, Math.min(20, camera.radius + e.deltaY * 0.01));
      e.preventDefault();
    }, { passive: false });

    // Touch support
    let touches = [];
    canvas.addEventListener('touchstart', e => {
      touches = [...e.touches];
      e.preventDefault();
    }, { passive: false });

    canvas.addEventListener('touchmove', e => {
      if (e.touches.length === 1) {
        const dx = e.touches[0].clientX - touches[0].clientX;
        const dy = e.touches[0].clientY - touches[0].clientY;
        camera.theta -= dx * 0.007;
        camera.phi = Math.max(0.05, Math.min(Math.PI/2 - 0.02, camera.phi - dy * 0.007));
      } else if (e.touches.length === 2) {
        const d0 = Math.hypot(touches[0].clientX - touches[1].clientX, touches[0].clientY - touches[1].clientY);
        const d1 = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
        camera.radius = Math.max(1, Math.min(20, camera.radius * (d0/d1)));
      }
      touches = [...e.touches];
      e.preventDefault();
    }, { passive: false });
  }

  // ─── Public API ────────────────────────────────────────────────────────────

  function setParam(key, value) {
    const geometryKeys = new Set([
      'blockW', 'blockD', 'blockH', 'groundDist',
      'surfaceMode', 'bumpAmp', 'bumpFreq', 'surfaceRes'
    ]);
    params[key] = value;
    if (geometryKeys.has(key)) surfaceDirty = true;
  }

  function loadOBJ(text) {
    const result = parseOBJ(text);
    if (!result) {
      document.getElementById('obj-status').textContent = 'Failed to parse OBJ';
      return;
    }
    objSurface = result;
    params.surfaceMode = 'obj';
    document.getElementById('surface-mode').value = 'obj';
    surfaceDirty = true;
    const pts = result.gridW * result.gridH;
    document.getElementById('obj-status').textContent =
      `Loaded: ${pts.toLocaleString()} surface points (${result.gridW}×${result.gridH})`;
  }

  function setCameraPreset(preset) {
    if (preset === 'top') {
      camera.phi = Math.PI / 2 - 0.01;
      camera.theta = 0;
      camera.radius = 7;
      camera.target = [0, 0, 0];
    } else if (preset === 'side') {
      camera.phi = 0.1;
      camera.theta = 0;
      camera.radius = 6;
      camera.target = [0, 0.5, 0];
    } else {
      camera.phi = 0.8;
      camera.theta = 0.6;
      camera.radius = 6;
      camera.target = [0, 0.2, 0];
    }
  }

  // Expose globals needed by ui.js
  window.CausticApp = { setParam, loadOBJ, loadCausticOBJ, setCameraPreset, init };
  window.setCameraPreset = setCameraPreset;

  document.addEventListener('DOMContentLoaded', init);

})();
