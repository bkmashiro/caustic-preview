/**
 * caustic.js — WebGL2 Caustic Renderer
 *
 * Approach: Per-pixel backward ray tracing in the ground fragment shader.
 *  For each ground pixel, trace a ray backward through the glass block to
 *  determine where light from the source would have refracted and hit,
 *  then compute a Gaussian kernel contribution as the caustic intensity.
 *
 *  Pass 1 (GROUND): Draw ground quad; FS_GROUND does per-pixel backward RT.
 *  Pass 2 (SCENE):  Render the transparent glass block with Phong shading.
 */

'use strict';

const CausticRenderer = (() => {

  // ─── State ────────────────────────────────────────────────────────────────
  let gl, canvas;

  // Shader programs
  let progScene, progGround, progCaustic;

  // Geometry buffers
  let quadVAO;          // fullscreen quad (also used as groundVAO)
  let groundVAO;
  let blockVAO, blockIBO, blockVertCount;
  let emptyVAO;         // attribute-less VAO for instanced splat draw

  // Caustic FBO (forward splat accumulation)
  let causticFBO = null;
  let causticTex = null;
  const CAUSTIC_W = 1024, CAUSTIC_H = 1024;
  let causticUseFloat = false; // true = R32F (needs EXT_float_blend); false = RGBA8 normalized

  // Surface texture (RGBA32F: nx, ny, nz, heightOff)
  let surfaceTex = null;
  let floatLinearSupported = false;

  // Current params (updated by UI)
  let params = {
    azimuth: 45,
    elevation: 45,
    intensity: 3.0,
    ior: 1.62,
    exposure: 4.0,
    spread: 0.0,
    sigma: 0.05,
    blockW: 2.0,
    blockD: 2.0,
    blockH: 0.4,
    groundDist: 2.0,  // = lens top surface to ground (optical focal distance)
    groundY: 0.0,     // Y position of the ground plane (move down to create gap)
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

  // Surface geometry cache (JS arrays, still needed to build surfaceTex)
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

  // Ground plane vertex shader — outputs vWorldPos for per-pixel RT in FS
  const VS_GROUND = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos; // -1..1 quad

uniform mat4 uMVP;
uniform vec3 uGroundCorner; // world space corner of ground quad (-half, y, -half)
uniform vec2 uGroundSize;   // world space size of ground quad (full size)
uniform float uGroundHalf;  // half-size of caustic texture coverage

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
  vUV = world.xz / (uGroundHalf * 2.0) + 0.5;
  gl_Position = uMVP * vec4(world, 1.0);
}
`;

  const FS_GROUND = `#version 300 es
precision highp float;
in vec2 vUV;
in vec3 vWorldPos;

uniform sampler2D uCausticTex; // R32F: accumulated caustic splats
uniform float uIntensity;
uniform float uExposure;
uniform vec3  uLightDir;
uniform vec3  uCausticColor;
uniform vec3  uGroundColor;
uniform bool  uShowGrid;
uniform bool  uShowCausticOnly;

out vec4 fragColor;

float gridLine(vec2 p, float size) {
  vec2 g = abs(fract(p / size - 0.5) - 0.5) / fwidth(p / size);
  return 1.0 - min(min(g.x, g.y), 1.0);
}

void main() {
  float causticRaw = texture(uCausticTex, vUV).r;
  float caustic    = 1.0 - exp(-causticRaw * uIntensity * uExposure);

  vec3 col;
  if (uShowCausticOnly) {
    col = uCausticColor * caustic;
  } else {
    vec3 Lup  = normalize(-uLightDir);
    float diff = max(dot(vec3(0.0, 1.0, 0.0), Lup), 0.0) * 0.3 + 0.15;
    col = uGroundColor * (diff + 0.1) + uCausticColor * caustic;
    if (uShowGrid) col += vec3(gridLine(vWorldPos.xz, 0.5) * 0.06);
  }
  fragColor = vec4(col, 1.0);
}
`;

  // Pass: Scene — render the glass block
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

  // Caustic shaders — Evan Wallace "area splatting" method:
  //   Render the lens surface as a triangle MESH with each vertex displaced to
  //   where refracted light lands on the ground (gHit).  The fragment shader uses
  //   screen-space partial derivatives (dFdx/dFdy) of the original surface XZ to
  //   compute the area-compression Jacobian → brightness.  A connected mesh has
  //   zero gaps by construction; no Gaussian splat size to tune.
  const VS_CAUSTIC = `#version 300 es
precision highp float;

uniform sampler2D uSurfTex;  // RGBA32F: nx,ny,nz,heightOff
uniform int   uSurfW;
uniform int   uSurfH;
uniform float uBlockW;
uniform float uBlockD;
uniform float uBlockTop;
uniform float uBlockBottom;
uniform float uGroundY;
uniform float uIOR;
uniform vec3  uLightDir;
uniform float uGroundHalf;  // half-size of ground coverage (world units)

out vec2 vSurfXZ;   // original surface XZ (world units) — for Jacobian in FS

vec3 snell(vec3 I, vec3 N, float eta) {
  float cosI  = -dot(N, I);
  float sin2T = eta * eta * (1.0 - cosI * cosI);
  if (sin2T > 1.0) return vec3(0.0);
  return eta * I + (eta * cosI - sqrt(1.0 - sin2T)) * N;
}

// Returns gHit.xz on the ground, or (INF,INF) on TIR/invalid
vec2 traceToGround(float wx, float wz, float wy, vec3 Nt, vec3 L) {
  // Refract air → glass at top surface
  vec3 D1 = snell(L, Nt, 1.0 / uIOR);
  if (length(D1) < 0.01 || D1.y >= 0.0) return vec2(1e9);
  D1 = normalize(D1);

  // Trace to block bottom
  float t1 = (uBlockBottom - wy) / D1.y;
  if (t1 < 0.0) return vec2(1e9);
  vec3 B = vec3(wx, wy, wz) + t1 * D1;

  // Refract glass → air at flat bottom (N into glass = upward)
  vec3 D2 = snell(D1, vec3(0.0, 1.0, 0.0), uIOR);
  if (length(D2) < 0.01 || D2.y >= 0.0) return vec2(1e9);
  D2 = normalize(D2);

  // Trace to ground plane
  float t2 = (uGroundY - B.y) / D2.y;
  if (t2 < 0.0) return vec2(1e9);
  return B.xz + t2 * D2.xz;
}

void main() {
  // Each cell = 2 triangles = 6 vertices;  grid has (W-1)×(H-1) cells
  int cellIdx = gl_VertexID / 6;
  int corner  = gl_VertexID % 6;

  int ci = cellIdx % (uSurfW - 1);
  int cj = cellIdx / (uSurfW - 1);

  if (cj >= uSurfH - 1) {
    gl_Position = vec4(2.0, 0.0, 0.0, 1.0);
    vSurfXZ = vec2(0.0);
    return;
  }

  // Map corner index → cell vertex (0,0)(1,0)(1,1) / (0,0)(1,1)(0,1)
  int di = (corner == 1 || corner == 2 || corner == 4) ? 1 : 0;
  int dj = (corner == 2 || corner == 4 || corner == 5) ? 1 : 0;

  int si = ci + di;
  int sj = cj + dj;

  // Sample surface texture at this grid corner
  vec2 uv = (vec2(float(si), float(sj)) + 0.5) / vec2(float(uSurfW), float(uSurfH));
  vec4 sd  = texture(uSurfTex, uv);
  vec3 Nt  = length(sd.rgb) > 0.01 ? normalize(sd.rgb) : vec3(0.0, 1.0, 0.0);
  float h  = sd.a;

  float wx = (float(si) / float(uSurfW - 1) - 0.5) * uBlockW;
  float wz = (float(sj) / float(uSurfH - 1) - 0.5) * uBlockD;
  float wy = uBlockTop + h;

  vec3 L = normalize(uLightDir);
  vec2 gHit = traceToGround(wx, wz, wy, Nt, L);

  vSurfXZ = vec2(wx, wz);

  // Map gHit world XZ → FBO NDC [-1,1]
  vec2 ndc = (length(gHit) < 1e8) ? (gHit / uGroundHalf) : vec2(2.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
`;

  const FS_CAUSTIC = `#version 300 es
precision highp float;
in  vec2 vSurfXZ;        // interpolated original surface XZ (world units)
uniform float uJacScale; // = (CAUSTIC_W / groundSize)^2  →  flat lens → 1.0 output
out vec4 fragColor;

void main() {
  // Screen-space partial derivatives of original surface position (world units/FBO pixel)
  // Large Jacobian = many surface cells per ground pixel = caustic focus = bright
  vec2 dsx = dFdx(vSurfXZ);
  vec2 dsy = dFdy(vSurfXZ);
  float jac = abs(dsx.x * dsy.y - dsx.y * dsy.x); // 2D cross product (area element)
  float v   = jac * uJacScale;
  fragColor = vec4(v, v, v, v); // all channels for RGBA8 compat
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

  // ─── Surface texture upload ───────────────────────────────────────────────

  function uploadSurfaceTex() {
    if (!surfacePositions) return;
    if (!surfaceTex) surfaceTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, surfaceTex);

    const N = surfaceGridW * surfaceGridH;
    const data = new Float32Array(N * 4);
    const blockTopY = params.groundDist; // groundDist = lens top to ground
    for (let i = 0; i < N; i++) {
      data[i*4+0] = surfaceNormals[i*3+0];
      data[i*4+1] = surfaceNormals[i*3+1];
      data[i*4+2] = surfaceNormals[i*3+2];
      data[i*4+3] = surfacePositions[i*3+1] - blockTopY; // height offset
    }

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, surfaceGridW, surfaceGridH, 0, gl.RGBA, gl.FLOAT, data);
    const filter = floatLinearSupported ? gl.LINEAR : gl.NEAREST;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ─── Caustic FBO ─────────────────────────────────────────────────────────

  function buildCausticFBO() {
    if (causticFBO) {
      gl.deleteFramebuffer(causticFBO);
      gl.deleteTexture(causticTex);
    }
    causticTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, causticTex);
    if (causticUseFloat) {
      // R32F: full precision additive accumulation (requires EXT_float_blend)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, CAUSTIC_W, CAUSTIC_H, 0, gl.RED, gl.FLOAT, null);
    } else {
      // RGBA8 fallback: splat contributions scaled down to avoid saturation
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, CAUSTIC_W, CAUSTIC_H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    causticFBO = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, causticFBO);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, causticTex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.error('Caustic FBO incomplete, status:', status);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

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
    const topY = params.groundDist; // Y of block top (= optical focal distance)

    let positions, normals, gridW, gridH;

    if (params.surfaceMode === 'obj' && objSurface) {
      // Bilinear upsample OBJ grid to TARGET_RES×TARGET_RES so we always
      // have enough samples regardless of solver resolution setting.
      const TARGET_RES = 128;
      const srcW = objSurface.gridW, srcH = objSurface.gridH;
      const srcPos = objSurface.positions, srcNrm = objSurface.normals;

      if (srcW >= TARGET_RES && srcH >= TARGET_RES) {
        // Always copy so we can safely shift Y in place below
        positions = new Float32Array(srcPos); normals = new Float32Array(srcNrm);
        gridW = srcW; gridH = srcH;
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

    // For OBJ mode: shift Y to reflect current blockTop (in case params changed after parse)
    if (params.surfaceMode === 'obj' && objSurface && objSurface.blockTopAtParse !== undefined) {
      const curBlockTop = params.groundDist;
      const shift = curBlockTop - objSurface.blockTopAtParse;
      if (Math.abs(shift) > 1e-6) {
        for (let i = 0; i < positions.length / 3; i++) {
          positions[i*3+1] += shift;
        }
        // Update cached blockTopAtParse so repeated calls don't double-shift
        objSurface.blockTopAtParse = curBlockTop;
      }
    }

    surfacePositions = positions;
    surfaceNormals = normals;
    surfaceGridW = gridW;
    surfaceGridH = gridH;

    // Upload surface data to GPU texture
    uploadSurfaceTex();

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
    const top = params.groundDist;                  // lens top = focal distance from ground
    const bot = params.groundDist - params.blockH;  // flat support bottom

    const verts = [];
    const indices = [];

    const useOBJ = params.surfaceMode === 'obj' && surfacePositions && surfaceGridW >= 2 && surfaceGridH >= 2;

    if (useOBJ) {
      // ── OBJ mode: use actual lens surface for top mesh ──────────────────────
      const gW = surfaceGridW, gH = surfaceGridH;

      // Top surface (actual lens profile from OBJ)
      const topBase = verts.length / 6;
      for (let j = 0; j < gH; j++) {
        for (let i = 0; i < gW; i++) {
          const idx = j * gW + i;
          verts.push(
            surfacePositions[idx*3+0], surfacePositions[idx*3+1], surfacePositions[idx*3+2],
            surfaceNormals[idx*3+0],   surfaceNormals[idx*3+1],   surfaceNormals[idx*3+2]
          );
        }
      }
      for (let j = 0; j < gH - 1; j++) {
        for (let i = 0; i < gW - 1; i++) {
          const a = topBase + j*gW + i;
          const b = a + 1, c = a + gW, d = a + gW + 1;
          indices.push(a, b, d, a, d, c);
        }
      }

      // Bottom face (flat support plate)
      {
        const b = verts.length / 6;
        verts.push(-hw, bot, -hd,  0,-1,0,
                    hw, bot, -hd,  0,-1,0,
                    hw, bot,  hd,  0,-1,0,
                   -hw, bot,  hd,  0,-1,0);
        indices.push(b, b+2, b+1, b, b+3, b+2);
      }

      // Side faces: for each boundary edge of the lens surface, drop a quad to bot.
      // Culling is disabled for the glass block so winding order doesn't matter.
      const P = surfacePositions;
      function sideQuad(i0, i1, nx, ny, nz) {
        const x0=P[i0*3], y0=P[i0*3+1], z0=P[i0*3+2];
        const x1=P[i1*3], y1=P[i1*3+1], z1=P[i1*3+2];
        const b = verts.length / 6;
        verts.push(
          x0, y0, z0,  nx, ny, nz,
          x1, y1, z1,  nx, ny, nz,
          x1, bot, z1, nx, ny, nz,
          x0, bot, z0, nx, ny, nz
        );
        indices.push(b, b+1, b+2, b, b+2, b+3);
      }

      // Front edge (j=0, normal -Z)
      for (let i = 0; i < gW - 1; i++)
        sideQuad(0*gW + i, 0*gW + i+1, 0, 0, -1);
      // Back edge (j=gH-1, normal +Z)
      for (let i = 0; i < gW - 1; i++)
        sideQuad((gH-1)*gW + i+1, (gH-1)*gW + i, 0, 0, 1);
      // Left edge (i=0, normal -X)
      for (let j = 0; j < gH - 1; j++)
        sideQuad((j+1)*gW, j*gW, -1, 0, 0);
      // Right edge (i=gW-1, normal +X)
      for (let j = 0; j < gH - 1; j++)
        sideQuad(j*gW + (gW-1), (j+1)*gW + (gW-1), 1, 0, 0);

    } else {
      // ── Procedural mode: heightfield top + flat sides ───────────────────────
      const res  = Math.min(params.surfaceRes, 64);
      const mode = params.surfaceMode;
      const amp  = params.bumpAmp;
      const freq = params.bumpFreq;

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
            indices.push(a, b, d, a, d, c);
          }
        }
      }

      // Side faces (flat, straight from top to bot)
      const sides = [
        [[-hw,bot,-hd,-1,0,0], [-hw,bot,hd,-1,0,0], [-hw,top,-hd,-1,0,0], [-hw,top,hd,-1,0,0]],
        [[ hw,bot, hd, 1,0,0], [ hw,bot,-hd, 1,0,0], [ hw,top, hd, 1,0,0], [ hw,top,-hd, 1,0,0]],
        [[ hw,bot,-hd,0,0,-1], [-hw,bot,-hd,0,0,-1], [ hw,top,-hd,0,0,-1], [-hw,top,-hd,0,0,-1]],
        [[-hw,bot, hd,0,0, 1], [ hw,bot, hd,0,0, 1], [-hw,top, hd,0,0, 1], [ hw,top, hd,0,0, 1]],
      ];
      for (const side of sides) {
        const base = verts.length / 6;
        for (const v of side) verts.push(...v);
        indices.push(base, base+1, base+2, base+1, base+3, base+2);
      }
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
    const groundY  = params.groundY;
    // groundDist = lens top surface to ground (optical focal distance)
    const blockTop    = params.groundDist;
    const blockBottom = params.groundDist - params.blockH;
    // Caustic spread depends on distance from block exit (bottom) to ground
    const groundSize  = Math.max(params.blockW, params.blockD) * 2 + Math.max(blockBottom, 0.1) * 4;
    const groundHalf  = groundSize / 2;

    const W = canvas.width, H = canvas.height;

    // ── Pass 0: Caustic forward-splat into R32F FBO ───────────────────────
    if (surfaceTex) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, causticFBO);
      gl.viewport(0, 0, CAUSTIC_W, CAUSTIC_H);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE); // additive

      gl.useProgram(progCaustic);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, surfaceTex);
      gl.uniform1i(ul(progCaustic, 'uSurfTex'),    0);
      gl.uniform1i(ul(progCaustic, 'uSurfW'),      surfaceGridW);
      gl.uniform1i(ul(progCaustic, 'uSurfH'),      surfaceGridH);
      gl.uniform1f(ul(progCaustic, 'uBlockW'),     params.blockW);
      gl.uniform1f(ul(progCaustic, 'uBlockD'),     params.blockD);
      gl.uniform1f(ul(progCaustic, 'uBlockTop'),   blockTop);
      gl.uniform1f(ul(progCaustic, 'uBlockBottom'),blockBottom);
      gl.uniform1f(ul(progCaustic, 'uGroundY'),    groundY);
      gl.uniform1f(ul(progCaustic, 'uIOR'),        params.ior);
      gl.uniform3fv(ul(progCaustic, 'uLightDir'),  lightDir);
      gl.uniform1f(ul(progCaustic, 'uGroundHalf'), groundHalf);
      // Jacobian scale: flat lens → causticRaw ≈ 1.0 everywhere
      // jac = (surfaceCell_world_area) / (groundPixel_world_area)
      // uJacScale = (CAUSTIC_W / groundSize)^2
      const jacScale = (CAUSTIC_W / groundSize) * (CAUSTIC_H / groundSize);
      gl.uniform1f(ul(progCaustic, 'uJacScale'), jacScale);

      gl.bindVertexArray(emptyVAO);
      gl.drawArrays(gl.TRIANGLES, 0, (surfaceGridW - 1) * (surfaceGridH - 1) * 6);
      gl.bindVertexArray(null);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.disable(gl.BLEND);
    }

    // ── Pass 1: Ground plane (samples causticTex) ─────────────────────────
    const model  = mat4.identity();
    const mvp    = getMVP(model);
    const cameraPos = getCameraPos();

    gl.viewport(0, 0, W, H);
    gl.clearColor(0.05, 0.05, 0.08, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);

    gl.useProgram(progGround);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, causticTex);
    gl.uniform1i(ul(progGround, 'uCausticTex'),      0);
    gl.uniform1f(ul(progGround, 'uIntensity'),        params.intensity);
    gl.uniform1f(ul(progGround, 'uExposure'),         params.exposure);
    gl.uniform3fv(ul(progGround, 'uLightDir'),        lightDir);
    gl.uniform3fv(ul(progGround, 'uCausticColor'),    params.causticColor);
    gl.uniform3fv(ul(progGround, 'uGroundColor'),     params.groundColor);
    gl.uniform1i(ul(progGround, 'uShowGrid'),         params.showGrid ? 1 : 0);
    gl.uniform1i(ul(progGround, 'uShowCausticOnly'),  params.showCausticOnly ? 1 : 0);
    gl.uniform3fv(ul(progGround, 'uGroundCorner'),    [-groundHalf, groundY, -groundHalf]);
    gl.uniform2fv(ul(progGround, 'uGroundSize'),      [groundSize, groundSize]);
    gl.uniform1f(ul(progGround, 'uGroundHalf'),       groundHalf);
    gl.uniformMatrix4fv(ul(progGround, 'uMVP'), false, mvp);

    gl.bindVertexArray(groundVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);

    // ── Pass 2: Glass block (transparent, rendered last) ──────────────────
    if (params.showBlock && !params.showCausticOnly) {
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
    document.getElementById('perf-display').textContent = `⏱ ${lastFrameMs.toFixed(1)}ms · RT`;

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

    // First half = top curved surface (X,Y in [0,1], Z = height deformation <= 0)
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
        // Caustic OBJ: X,Y in [0,1] = 2D grid; Z <= 0 = height deformation (in units of lens_width=1.0)
        // Renderer: Y=up, lens top surface at params.groundDist (= optical focal distance)
        positions[idx*3+0] = (obj_x - 0.5) * blockW;
        positions[idx*3+1] = params.groundDist + obj_z * blockW;  // blockTop = groundDist
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

    // Compute actual maximum deformation depth from raw OBJ data (normalized units)
    // so we can tell the caller the minimum blockH needed.
    let maxAbsObjZ = 0;
    for (const v of topVerts) {
      maxAbsObjZ = Math.max(maxAbsObjZ, Math.abs(v[2]));
    }
    // requiredBlockH = depth in world units + small safety margin
    const requiredBlockH = maxAbsObjZ * blockW + 0.02;

    // Safety clamp: ensure no surface point falls below block bottom
    const blockBottomY = params.groundDist - params.blockH;
    for (let i = 0; i < gridW * gridH; i++) {
      if (positions[i*3+1] < blockBottomY) {
        positions[i*3+1] = blockBottomY;
      }
    }

    return { positions, normals, gridW, gridH,
             blockTopAtParse: params.groundDist,  // blockTop = groundDist (focal distance)
             requiredBlockH };
  }

  function loadCausticOBJ(text) {
    const result = parseCausticOBJ(text);
    if (!result) {
      document.getElementById('obj-status').textContent = 'Failed to parse caustic OBJ';
      return null;
    }
    objSurface = result;
    params.surfaceMode = 'obj';
    document.getElementById('surface-mode').value = 'obj';
    surfaceDirty = true;
    const pts = result.gridW * result.gridH;
    document.getElementById('obj-status').textContent =
      `Loaded: ${pts.toLocaleString()} surface points (${result.gridW}×${result.gridH})`;
    // Re-center camera on the new geometry
    setCameraPreset('persp');
    // Return info so ui.js can auto-adjust blockH if needed
    return { requiredBlockH: result.requiredBlockH };
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

    // EXT_color_buffer_float: needed for R32F/RGBA32F render targets + surface texture
    gl.getExtension('EXT_color_buffer_float');

    // EXT_float_blend: required to blend (additive accumulate) into float framebuffers.
    // Without this, WebGL2 silently disables blending on float FBOs — last splat wins,
    // all previous splats discarded → dots.  Fall back to RGBA8 if not available.
    causticUseFloat = !!gl.getExtension('EXT_float_blend');
    console.log('[caustic] EXT_float_blend:', causticUseFloat ? 'YES' : 'NO — RGBA8 fallback');

    // OES_texture_float_linear: smooth filtering of float textures
    const floatLinExt = gl.getExtension('OES_texture_float_linear');
    floatLinearSupported = !!floatLinExt;

    try {
      progScene   = createProgram(VS_SCENE,   FS_SCENE);
      progGround  = createProgram(VS_GROUND,  FS_GROUND);
      progCaustic = createProgram(VS_CAUSTIC, FS_CAUSTIC);
    } catch(e) {
      console.error(e);
      alert('Shader compilation failed:\n' + e.message);
      return;
    }

    buildCausticFBO();

    // Attribute-less VAO for the caustic splat draw (uses only gl_VertexID)
    emptyVAO = gl.createVertexArray();

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
        // Pan: move target in camera's right/up plane
        const scale = camera.radius * 0.003;
        const t = camera.theta;
        // Horizontal pan along ground plane (rotate with camera azimuth)
        camera.target[0] -= Math.cos(t) * dx * scale;
        camera.target[2] -= Math.sin(t) * dx * scale;
        // Vertical pan: drag down → look down (target Y decreases)
        camera.target[1] -= dy * scale;
      } else {
        // Orbit
        camera.theta -= dx * 0.007;
        camera.phi = Math.max(0.01, Math.min(Math.PI / 2, camera.phi - dy * 0.007));
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
        camera.phi = Math.max(0.01, Math.min(Math.PI / 2, camera.phi - dy * 0.007));
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
    // Centre the view vertically between ground (0) and block top (groundDist)
    const midY = params.groundDist * 0.4;
    const sceneR = Math.max(params.blockW, params.blockD, params.groundDist) * 1.8;

    if (preset === 'top') {
      camera.phi = Math.PI / 2 - 0.01;
      camera.theta = 0;
      camera.radius = sceneR * 1.2;
      camera.target = [0, midY, 0];
    } else if (preset === 'side') {
      camera.phi = 0.08;
      camera.theta = 0;
      camera.radius = sceneR * 1.4;
      camera.target = [0, midY, 0];
    } else {
      camera.phi = 0.75;
      camera.theta = 0.6;
      camera.radius = sceneR * 1.3;
      camera.target = [0, midY, 0];
    }
  }

  // Expose globals needed by ui.js
  window.CausticApp = { setParam, loadOBJ, loadCausticOBJ, setCameraPreset, init };
  window.setCameraPreset = setCameraPreset;

  document.addEventListener('DOMContentLoaded', init);

})();
