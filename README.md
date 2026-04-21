# Caustic Lens Preview

A browser-based interactive tool to preview caustic light projection effects from a transparent resin block. Built for previewing 3D-printed caustic lenses before printing.

## Usage

Open `index.html` in any modern browser (Chrome, Firefox, Edge, Safari). No build step or server required — just open the file.

## How It Works

The renderer uses WebGL2 with a three-pass forward ray tracing approach:

### Pass 1 — Caustic Accumulation (GPU Photon Tracing)
For each sample point on the block's top surface (up to 256×256 = 65,536 points):
1. Compute the surface normal at that point
2. Apply **Snell's law** to refract the incoming light ray into the block (air n=1.0 → resin n≈1.5)
3. Ray travels in a straight line through the block to the flat bottom surface
4. Apply **Snell's law** again at the flat bottom to get the exit ray direction (resin → air)
5. Find where the exit ray intersects the ground plane
6. Draw a soft-disk "photon splat" at that location using **additive blending** → accumulates caustic intensity

### Pass 2 — Ground Plane
Renders the ground quad textured with the accumulated caustic map. Tone-mapped with exposure control.

### Pass 3 — Glass Block
Renders the transparent block with Phong shading + Fresnel reflectance.

## Controls

| Control | Effect |
|---------|--------|
| Hemisphere dial | Drag to set light direction interactively |
| Azimuth / Elevation sliders | Fine-tune light angle |
| Refractive Index | n≈1.5 glass/resin, n≈1.33 water, n≈2.4 diamond |
| Caustic Exposure | Brightness of the projected caustic pattern |
| Spread / Blur | Size of each photon splat |
| Surface Mode | Sinusoidal, concentric rings, diagonal, Perlin-like, or flat |
| Bump Amplitude | Height of surface features |
| Bump Frequency | Spatial frequency of bumps |
| Resolution | Grid density — higher = more accurate caustic but slower |
| Load OBJ | Load a surface from `poisson_caustic_design` output |

**Keyboard shortcuts:**
- `1` — Perspective view
- `2` — Side view
- `3` — Top-down view
- `B` — Toggle block visibility
- `G` — Toggle ground grid

**Mouse:**
- Left-drag: orbit
- Right-drag: pan
- Scroll: zoom

## Loading a Caustic Lens OBJ

When you have the output from [poisson_caustic_design](https://github.com/dylanmsu/poisson_caustic_design):

1. Click **Choose OBJ…** in the sidebar
2. Select the `.obj` file (the top surface of the computed lens)
3. The tool will parse the vertex grid and use it directly for caustic tracing
4. Switch the Surface Mode dropdown to **Loaded OBJ** if not automatic

The parser looks for the grid structure in the vertex list (sorted by XZ position) and computes normals via cross product of neighboring vertices.

## Physics

**Snell's law:** `n₁ sin(θ₁) = n₂ sin(θ₂)`

- Air: n₁ = 1.000
- Glass/resin: n₂ ≈ 1.49–1.52
- Water: n₂ ≈ 1.333
- Diamond: n₂ ≈ 2.417

At the flat bottom surface, with steep enough angles, **total internal reflection** (TIR) occurs and the ray does not exit — those photons are discarded.

The surface normal affects where each refracted ray hits the ground. Carefully designed surface normals (as computed by poisson_caustic_design) can project an arbitrary target image as a caustic pattern.

## WebAssembly Note

WASM compilation of `poisson_caustic_design` was not attempted for this preview tool. The Ceres Solver dependency (required for the optimization step) is extremely complex to build with Emscripten due to its dependency on LAPACK/BLAS and template-heavy C++ code. The preview renderer is the priority, and the OBJ loading provides the integration point once a desktop build produces the lens geometry.

## Browser Requirements

- WebGL2 (Chrome 56+, Firefox 51+, Safari 15+, Edge 79+)
- `EXT_color_buffer_half_float` extension (for high-precision caustic accumulation)
  - Falls back to 512×512 texture if unavailable

## Files

```
index.html   — Main page, layout, HTML controls
caustic.js   — WebGL2 renderer: shaders, passes, geometry, OBJ parser
ui.js        — UI wiring: sliders, color pickers, hemisphere dial
README.md    — This file
```
