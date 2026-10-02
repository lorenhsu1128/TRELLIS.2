import gradio as gr

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime
import shutil
import cv2
from typing import *
import torch
import numpy as np
from PIL import Image
import base64
import io
import uuid
import time
import json
import subprocess
import re
import gc
import ctypes
from collections import OrderedDict
import trimesh
from trellis2.modules.sparse import SparseTensor
from trellis2.modules import image_feature_extractor
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


# Traditional Chinese, LAN-accessible version of app.py (mirrors the official HF Space).
SERVER_NAME = os.environ.get("APP_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("APP_PORT", "7860"))
# The official DINOv3 repo is gated; fall back to a public mirror with an identical
# model.safetensors (sha256 dcb2e451...8179). Override with DINOV3_MODEL.
DINOV3_OFFICIAL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_MODEL = os.environ.get("DINOV3_MODEL", "visualbruno/dinov3-vitl16-pretrain-lvd1689m")

_dinov3_init = image_feature_extractor.DinoV3FeatureExtractor.__init__
def _dinov3_init_with_mirror(self, model_name, *args, **kwargs):
    if model_name == DINOV3_OFFICIAL:
        model_name = DINOV3_MODEL
    _dinov3_init(self, model_name, *args, **kwargs)
image_feature_extractor.DinoV3FeatureExtractor.__init__ = _dinov3_init_with_mirror

MAX_SEED = np.iinfo(np.int32).max
# Keep per-session outputs outside the repo (.gitignore does not exclude tmp/).
TMP_DIR = os.environ.get("APP_TMP_DIR", os.path.expanduser("~/.cache/trellis2_app_zh/tmp"))
# Vendored three.js (r170) for the live GLB viewer, served locally so LAN clients need no internet.
THREE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'app', 'three')
# glb-shrink pipeline (Node.js), vendored in tools/glb_shrink
SHRINK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'glb_shrink')
SHRINK_MAX_BYTES = 200 * 1024 * 1024
SHRINK_PRESETS = {
    "📦 最小檔案 — 遠景或畫面上很小": 0,
    "⚖️ 平衡 — 大多數專案（建議）": 50,
    "✨ 最清晰 — 近距離或主角物件": 100,
}
VIEW_STATIC = "靜態渲染預覽"
VIEW_LIVE = "即時 3D 檢視"
LIVE_IDLE_TEXT = "切換到「即時 3D 檢視」時，會自動以目前的減面與貼圖設定匯出 GLB 並載入"
# Real-world size / orientation of the exported GLB (glTF unit = meter, Y-up).
# Generated assets are normalized so the longest side is ~1 m regardless of the object.
SCALE_NONE = "不縮放（最長邊約 100 cm）"
SCALE_AXES = {"最長邊": None, "X 軸（寬）": 0, "Y 軸（高）": 1, "Z 軸（深）": 2}
SCALE_MODES = [SCALE_NONE] + list(SCALE_AXES)
ORIGIN_CENTER = "幾何中心"
ORIGIN_BOTTOM = "底部中心"
ORIGIN_TOP = "頂部中心"
ORIGIN_CUSTOM = "自訂（在檢視器點選）"
ORIGIN_MODES = [ORIGIN_CENTER, ORIGIN_BOTTOM, ORIGIN_TOP, ORIGIN_CUSTOM]
# Raw (untransformed) GLB mesh per session, so size/rotation changes skip the slow remesh + bake.
# Bounded LRU; an entry is also dropped on a new generation and when the session ends.
RAW_GLB = OrderedDict()
RAW_GLB_MAX_SESSIONS = 2
# Generated candidates per session: [{'state', 'html', 'seed'}, ...] (num_samples > 1)
CANDIDATES = {}
ALPHA_MODES = {
    "不透明（OPAQUE）": "OPAQUE",
    "半透明混合（BLEND）": "BLEND",
    "透明裁切（MASK）": "MASK",
}
# Keeping every model resident on a 32GB GPU leaves too little room for 1024+ generation,
# so models are swapped CPU<->GPU per request (low_vram). The launcher pins glibc's mmap
# threshold so freed weight copies go back to the OS instead of fragmenting the heap.
LOW_VRAM = os.environ.get("APP_LOW_VRAM", "1") == "1"

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def release_memory():
    """Free Python garbage, cached CUDA blocks, and return freed heap pages to the OS."""
    gc.collect()
    torch.cuda.empty_cache()
    if _libc is not None:
        _libc.malloc_trim(0)
MODES = [
    {"name": "法線", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "黏土渲染", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "基礎色", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "HDRI 森林", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "HDRI 夕陽", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "HDRI 庭院", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}

/* Previewer */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: clamp(420px, calc(100vh - 380px), 722px);  /* fit the sticky middle panel */
    margin: 0 auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 300px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.2;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

/* Row 1: Display Modes */
.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.1); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.1);
}

/* Row 2: Display Image */
.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 400px;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible {
    display: block;
}

/* Row 3: Custom HTML Slider */
.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 400px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

/* Overwrite Previewer Block Style */
.gradio-container .padded:has(.previewer-container) {
    padding: 0 !important;
}

.gradio-container:has(.previewer-container) [data-testid="block-label"] {
    position: absolute;
    top: 0;
    left: 0;
}

/* Live three.js viewer */
#live-glb-url, #origin-pick-box, #origin-pick-btn, #shrink-view-json { display: none !important; }

/* 圖片轉 3D: three panels. Side panels stick to the viewport and scroll on their own,
   so the preview in the middle stays visible while adjusting parameters. */
#col-left, #col-mid, #col-right {
    position: sticky;
    top: 8px;
    align-self: flex-start;
}
#col-left, #col-mid, #col-right {
    max-height: calc(100vh - 16px);
    overflow-y: auto;
    overflow-x: hidden;
    overscroll-behavior: contain;
    padding-right: 4px;
    /* Gradio columns wrap by default: with a max-height, overflowing children would
       wrap into a second column instead of scrolling */
    flex-wrap: nowrap !important;
}
#col-left > *, #col-mid > *, #col-right > * { flex-shrink: 0; }

/* GLB 壓縮 tab (styled after glb-shrink) */
.shrink-hero {
    background: #0b0d14;
    border-radius: 12px;
    padding: 18px 16px 16px;
    text-align: center;
    color: #e5e7eb;
}
.shrink-hero-label {
    font-size: 12px;
    letter-spacing: 0.2em;
    color: #9ca3af;
    margin-bottom: 6px;
}
.shrink-hero-sizes {
    display: flex;
    justify-content: center;
    align-items: baseline;
    gap: 18px;
    flex-wrap: wrap;
    font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 700;
    font-size: clamp(32px, 5vw, 60px);
    line-height: 1.1;
}
.shrink-before { color: #f87171; text-decoration: line-through; text-decoration-thickness: 3px; }
.shrink-before.plain { text-decoration: none; color: #e5e7eb; }
.shrink-arrow { color: #6b7280; font-size: 0.7em; }
.shrink-after { color: #4ade80; }
.shrink-badge {
    display: inline-block;
    margin-top: 10px;
    padding: 4px 14px;
    border-radius: 999px;
    background: rgba(74, 222, 128, 0.14);
    color: #4ade80;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-weight: 700;
    font-size: 18px;
}
.shrink-hero-sub { margin-top: 8px; font-size: 13px; color: #9ca3af; }

.shrink-viewers {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    height: 640px;
}
@media (max-width: 900px) {
    .shrink-viewers { grid-template-columns: 1fr; height: auto; }
    .shrink-card { height: 420px; }
}
.shrink-card {
    position: relative;
    background: #0e1018;
    border-radius: 12px;
    overflow: hidden;
}
.shrink-canvas { position: absolute; inset: 0; }
.shrink-canvas canvas { display: block; width: 100% !important; height: 100% !important; cursor: grab; }
.shrink-card-header {
    position: absolute;
    top: 10px;
    left: 10px;
    right: 10px;
    display: flex;
    justify-content: space-between;
    z-index: 5;
    pointer-events: none;
}
.shrink-tag {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 999px;
}
.shrink-tag.before { color: #f87171; background: rgba(248, 113, 113, 0.15); }
.shrink-tag.after { color: #4ade80; background: rgba(74, 222, 128, 0.15); }
.shrink-stat {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 12px;
    color: #d1d5db;
    background: rgba(255, 255, 255, 0.06);
    padding: 3px 8px;
    border-radius: 6px;
}
.shrink-empty {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #6b7280;
    font-size: 15px;
    pointer-events: none;
}
.shrink-empty.hidden { display: none; }

body:not(.live-mode) #preview-live { display: none !important; }
body.live-mode #preview-static { display: none !important; }

.gradio-container .padded:has(.live-viewer-container) {
    padding: 0 !important;
}

.live-viewer-container {
    position: relative;
    width: 100%;
    height: clamp(420px, calc(100vh - 380px), 722px);  /* fit the sticky middle panel */
    overflow: hidden;
    border-radius: var(--block-radius);
    background: #404040;
}

.live-viewer-container canvas {
    display: block;
    width: 100% !important;
    height: 100% !important;
    cursor: grab;
    touch-action: none;
}

.live-viewer-container canvas:active { cursor: grabbing; }

.live-viewer-container .live-viewer-hint {
    position: absolute;
    left: 10px;
    bottom: 10px;
    color: #fff;
    font-size: 13px;
    opacity: 0.75;
    pointer-events: none;
    user-select: none;
}

.live-viewer-container .live-viewer-reset.live-viewer-pick { right: 96px; }
.live-viewer-container .live-viewer-reset.live-viewer-pick.active { background-color: #2b8a3e; }
.live-viewer-container.picking canvas { cursor: crosshair; }
.live-viewer-container.picking .live-viewer-hint { opacity: 1; color: #ffd54a; font-size: 15px; font-weight: bold; }

.live-viewer-container .live-viewer-dims {
    position: absolute;
    left: 10px;
    top: 36px;
    color: #fff;
    font-size: 13px;
    line-height: 1.6;
    background: rgba(0, 0, 0, 0.45);
    border-radius: 8px;
    padding: 4px 8px;
    pointer-events: none;
    user-select: none;
}

.live-viewer-container .live-viewer-dims:empty { display: none; }

.live-viewer-container .live-viewer-status {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 16px;
    text-align: center;
    padding: 20px;
    pointer-events: none;
    user-select: none;
}

.live-viewer-container .live-viewer-reset {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border: none;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 10px;
    cursor: pointer;
}
"""


head = """
<script>
    function refreshView(mode, step) {
        // 1. Find current mode and step
        const allImgs = document.querySelectorAll('.previewer-main-image');
        for (let i = 0; i < allImgs.length; i++) {
            const img = allImgs[i];
            if (img.classList.contains('visible')) {
                const id = img.id;
                const [_, m, s] = id.split('-');
                if (mode === -1) mode = parseInt(m.slice(1));
                if (step === -1) step = parseInt(s.slice(1));
                break;
            }
        }
        
        // 2. Hide ALL images
        // We select all elements with class 'previewer-main-image'
        allImgs.forEach(img => img.classList.remove('visible'));

        // 3. Construct the specific ID for the current state
        // Format: view-m{mode}-s{step}
        const targetId = 'view-m' + mode + '-s' + step;
        const targetImg = document.getElementById(targetId);

        // 4. Show ONLY the target
        if (targetImg) {
            targetImg.classList.add('visible');
        }

        // 5. Update Button Highlights
        const allBtns = document.querySelectorAll('.mode-btn');
        allBtns.forEach((btn, idx) => {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }
    
    // --- Action: Switch Mode ---
    function selectMode(mode) {
        refreshView(mode, -1);
    }
    
    // --- Action: Slider Change ---
    function onSliderChange(val) {
        refreshView(-1, parseInt(val));
    }
</script>
<script type="module">
    import * as THREE from '__THREE_BASE__/three.module.js';
    import { OrbitControls } from '__THREE_BASE__/addons/controls/OrbitControls.js';
    import { GLTFLoader } from '__THREE_BASE__/addons/loaders/GLTFLoader.js';
    import { DRACOLoader } from '__THREE_BASE__/addons/loaders/DRACOLoader.js';

    // Draco decoder served locally (compressed GLBs from GLB 壓縮 use KHR_draco_mesh_compression)
    const dracoLoader = new DRACOLoader();
    dracoLoader.setDecoderPath('__THREE_BASE__/draco/');
    function makeGltfLoader() {
        const loader = new GLTFLoader();
        loader.setDRACOLoader(dracoLoader);
        return loader;
    }
    import { RoomEnvironment } from '__THREE_BASE__/addons/environments/RoomEnvironment.js';

    const IDLE_TEXT = '__IDLE_TEXT__';
    // Match the static previewer's default view (render_snapshot yaw=119deg, pitch=20deg, Z-up),
    // mapped into the GLB's Y-up frame: (x, y, z)_zup -> (x, z, -y)_yup.
    const HOME_YAW = THREE.MathUtils.degToRad(__HOME_YAW__), HOME_PITCH = THREE.MathUtils.degToRad(20);
    const HOME_DIR = new THREE.Vector3(
        Math.sin(HOME_YAW) * Math.cos(HOME_PITCH),
        Math.sin(HOME_PITCH),
        -Math.cos(HOME_YAW) * Math.cos(HOME_PITCH),
    ).normalize();
    const V = { renderer: null, scene: null, camera: null, controls: null, model: null, url: null, home: null, container: null };

    function getContainer() {
        return document.getElementById('live-viewer');
    }

    function setStatus(text) {
        const el = getContainer();
        const s = el && el.querySelector('.live-viewer-status');
        if (!s) return;
        s.textContent = text || '';
        s.style.display = text ? 'flex' : 'none';
    }

    function resize() {
        const el = V.container;
        if (!el || !V.renderer) return;
        const w = el.clientWidth, h = el.clientHeight;
        if (!w || !h) return;
        V.renderer.setSize(w, h, false);
        V.camera.aspect = w / h;
        V.camera.updateProjectionMatrix();
    }

    function init() {
        const el = getContainer();
        if (!el) return false;
        if (V.renderer) {
            // Gradio may re-mount the HTML block; re-attach the canvas if needed
            if (V.container !== el) {
                V.container = el;
                el.prepend(V.renderer.domElement);
                V.resizeObserver.disconnect();
                V.resizeObserver.observe(el);
            }
            resize();
            return true;
        }
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.setClearColor(0x404040, 1);
        el.prepend(renderer.domElement);

        const scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        const light = new THREE.DirectionalLight(0xffffff, 1.0);
        light.position.set(1, 2, 3);
        scene.add(light);

        const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
        camera.position.set(0, 0, 2.5);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.zoomSpeed = 1.2;

        // Orientation gizmo: unit axes rendered in the bottom-right corner, following the camera
        const gizmoScene = new THREE.Scene();
        gizmoScene.add(makeAxes([1, 1, 1], false, 0.5));
        const gizmoCamera = new THREE.OrthographicCamera(-1.55, 1.55, 1.55, -1.55, 0.1, 10);

        Object.assign(V, { renderer, scene, camera, controls, container: el, gizmoScene, gizmoCamera });

        // Surface picking for "設定原點": a click (not a drag) while in pick mode
        let down = null;
        renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
        renderer.domElement.addEventListener('pointerup', (e) => {
            if (!V.picking || !down || e.button !== 0) return;
            if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
            pickAt(e);
        });
        window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && V.picking) setPicking(false); });
        V.resizeObserver = new ResizeObserver(resize);
        V.resizeObserver.observe(el);
        resize();
        renderer.autoClear = false;
        renderer.setAnimationLoop(() => {
            if (!V.container || !V.container.offsetParent) return;  // hidden: skip rendering
            controls.update();
            const w = V.container.clientWidth, h = V.container.clientHeight;
            renderer.setViewport(0, 0, w, h);
            renderer.setScissorTest(false);
            renderer.clear();
            renderer.render(scene, camera);
            // Gizmo camera looks at the origin from the main camera's viewing direction
            const dir = camera.position.clone().sub(controls.target).normalize();
            gizmoCamera.position.copy(dir.multiplyScalar(4));
            gizmoCamera.up.copy(camera.up);
            gizmoCamera.lookAt(0, 0, 0);
            const g = GIZMO_SIZE;
            renderer.setScissorTest(true);
            renderer.setScissor(w - g - 8, 8, g, g);
            renderer.setViewport(w - g - 8, 8, g, g);
            renderer.clearDepth();
            renderer.render(gizmoScene, gizmoCamera);
            renderer.setScissorTest(false);
        });
        return true;
    }

    const GIZMO_SIZE = 130;
    const AXES = [
        { name: 'X', color: 0xff5555, css: '#ff6b6b', dir: new THREE.Vector3(1, 0, 0) },
        { name: 'Y', color: 0x55dd55, css: '#6be06b', dir: new THREE.Vector3(0, 1, 0) },
        { name: 'Z', color: 0x5599ff, css: '#6ba6ff', dir: new THREE.Vector3(0, 0, 1) },
    ];

    function makeLabel(text, css) {
        const c = document.createElement('canvas');
        c.width = c.height = 64;
        const ctx = c.getContext('2d');
        ctx.font = 'bold 46px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.lineWidth = 6;
        ctx.strokeStyle = 'rgba(0,0,0,0.7)';
        ctx.strokeText(text, 32, 34);
        ctx.fillStyle = css;
        ctx.fillText(text, 32, 34);
        const tex = new THREE.CanvasTexture(c);
        tex.colorSpace = THREE.SRGBColorSpace;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, depthWrite: false, toneMapped: false }));
        sprite.renderOrder = 1001;
        return sprite;
    }

    // Labeled X/Y/Z arrows from the origin (+ origin dot); onTop draws them through the model.
    // lens: per-axis arrow lengths [x, y, z]; labelScale: label size relative to the shortest arrow.
    function makeAxes(lens, onTop, labelScale = 0.22) {
        const group = new THREE.Group();
        const unit = Math.min(...lens);
        AXES.forEach((a, i) => {
            const len = lens[i];
            const arrow = new THREE.ArrowHelper(a.dir, new THREE.Vector3(), len, a.color, unit * 0.12, unit * 0.06);
            [arrow.line, arrow.cone].forEach((o) => {
                o.material.depthTest = !onTop;
                o.material.toneMapped = false;
                o.renderOrder = 1000;
            });
            group.add(arrow);
            const label = makeLabel(a.name, a.css);
            label.position.copy(a.dir).multiplyScalar(len + unit * labelScale * 0.6);
            label.scale.setScalar(unit * labelScale);
            group.add(label);
        });
        const dot = new THREE.Mesh(
            new THREE.SphereGeometry(unit * 0.04, 16, 12),
            new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: !onTop, toneMapped: false }),
        );
        dot.renderOrder = 1002;
        group.add(dot);
        return group;
    }

    function disposeObject(obj) {
        obj.traverse((o) => {
            if (o.geometry) o.geometry.dispose();
            if (o.material) {
                if (o.material.map) o.material.map.dispose();
                o.material.dispose();
            }
        });
    }

    function disposeModel() {
        if (!V.model) return;
        V.scene.remove(V.model);
        V.model.traverse((obj) => {
            if (obj.geometry) obj.geometry.dispose();
            const mats = obj.material ? (Array.isArray(obj.material) ? obj.material : [obj.material]) : [];
            mats.forEach((m) => {
                Object.values(m).forEach((v) => { if (v && v.isTexture) v.dispose(); });
                m.dispose();
            });
        });
        V.model = null;
    }

    // The model is kept at its true GLB coordinates (so the axes show the real origin);
    // the camera orbits around the model's bounding-box center instead.
    function fitView(model) {
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1;
        const dist = radius / Math.sin(THREE.MathUtils.degToRad(V.camera.fov / 2)) * 1.2;  // margin for axis labels
        V.camera.near = dist / 100;
        V.camera.far = dist * 100;
        V.camera.position.copy(HOME_DIR).multiplyScalar(dist).add(center);
        V.camera.updateProjectionMatrix();
        V.controls.target.copy(center);
        V.controls.minDistance = radius * 0.2;
        V.controls.maxDistance = dist * 5;
        V.controls.update();
        V.home = { position: V.camera.position.clone(), target: V.controls.target.clone() };
    }

    // "Nice" grid cell (1/2/5 x 10^n meters) giving roughly 8 cells along the longest side
    function niceStep(length) {
        const target = length / 8;
        const pow = Math.pow(10, Math.floor(Math.log10(target)));
        const n = target / pow;
        return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * pow;
    }

    function formatCm(meters) {
        const cm = meters * 100;
        return (cm >= 10 ? cm.toFixed(1) : cm.toPrecision(3)) + ' cm';
    }

    // Signed extent of the model along one axis, relative to the origin
    function rangeCm(lo, hi) {
        const f = (m) => { const v = (m * 100).toFixed(1); return v === '-0.0' ? '0.0' : v; };
        return f(lo) + ' ~ ' + f(hi) + ' cm';
    }

    // Ground grid under the model + size readout (GLB unit = meter)
    const HINT_TEXT = '左鍵拖曳：旋轉　滾輪：縮放　右鍵拖曳：平移';
    const PICK_HINT_TEXT = '請點選模型表面，作為新的原點（按 Esc 取消）';

    function setPicking(on) {
        V.picking = on && !!V.model;
        const el = getContainer();
        if (!el) return;
        el.classList.toggle('picking', V.picking);
        const btn = el.querySelector('.live-viewer-pick');
        if (btn) btn.classList.toggle('active', V.picking);
        const hint = el.querySelector('.live-viewer-hint');
        if (hint) hint.textContent = V.picking ? PICK_HINT_TEXT : HINT_TEXT;
    }

    function togglePick() {
        if (!V.model) { setStatus('請先載入模型'); setTimeout(() => setStatus(''), 1500); return; }
        setPicking(!V.picking);
    }

    function pickAt(e) {
        const rect = V.renderer.domElement.getBoundingClientRect();
        const ndc = new THREE.Vector2(
            ((e.clientX - rect.left) / rect.width) * 2 - 1,
            -((e.clientY - rect.top) / rect.height) * 2 + 1,
        );
        const ray = new THREE.Raycaster();
        ray.setFromCamera(ndc, V.camera);
        const hit = ray.intersectObject(V.model, true)[0];
        if (!hit) return;  // missed the model: stay in pick mode
        V.picked = hit.point.toArray();
        setPicking(false);
        setStatus('正在以點選位置作為原點重新匯出…');
        const btn = document.getElementById('origin-pick-btn');
        if (btn) btn.click();
    }

    function removeHelpers() {
        for (const key of ['grid', 'axes', 'parts']) {
            if (V[key]) {
                V.scene.remove(V[key]);
                disposeObject(V[key]);
                V[key] = null;
            }
        }
    }

    // Ground grid centered on the origin's XZ, origin + axes, and size readout (GLB unit = meter)
    function updateHelpers(model) {
        removeHelpers();
        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const maxSide = Math.max(size.x, size.y, size.z);
        const step = niceStep(maxSide);
        const reach = Math.max(Math.abs(box.min.x), Math.abs(box.max.x), Math.abs(box.min.z), Math.abs(box.max.z));
        const half = Math.ceil(reach * 1.5 / step) + 2;
        V.grid = new THREE.GridHelper(step * half * 2, half * 2, 0x999999, 0x5a5a5a);
        V.grid.position.y = box.min.y;
        V.scene.add(V.grid);

        // Each arrow pokes out of the model along its own axis
        const lens = [0, 1, 2].map((i) => {
            const half = Math.max(Math.abs(box.min.getComponent(i)), Math.abs(box.max.getComponent(i)));
            return Math.max(half * 1.3, maxSide * 0.3);
        });
        V.axes = makeAxes(lens, true, 0.3);
        V.scene.add(V.axes);

        // Part numbers (nodes named part_N by the server) above each part, when there are several
        const parts = new Map();
        model.traverse((o) => {
            const m = /^part_(\d+)/.exec(o.name || '');
            if (m && o.isMesh) {
                const b = new THREE.Box3().setFromObject(o);
                parts.set(m[1], parts.has(m[1]) ? parts.get(m[1]).union(b) : b);
            }
        });
        if (parts.size > 1) {
            V.parts = new THREE.Group();
            parts.forEach((b, num) => {
                const label = makeLabel(num, '#ffd54a');
                const c = b.getCenter(new THREE.Vector3());
                label.scale.setScalar(maxSide * 0.12);
                label.position.set(c.x, b.max.y + maxSide * 0.08, c.z);
                V.parts.add(label);
            });
            V.scene.add(V.parts);
        }

        const el = getContainer();
        const dims = el && el.querySelector('.live-viewer-dims');
        if (dims) {
            const c = (i) => '<b style="color:' + AXES[i].css + '">' + AXES[i].name + '</b>';
            dims.innerHTML = '寬（' + c(0) + '）' + formatCm(size.x) + '　高（' + c(1) + '）' + formatCm(size.y)
                + '　深（' + c(2) + '）' + formatCm(size.z)
                + '<br>地面網格每格 ' + formatCm(step)
                + '<br>⚪ 原點 (0, 0, 0)　' + c(0) + ' 紅　' + c(1) + ' 綠（上）　' + c(2) + ' 藍'
                + '<br>模型相對原點範圍：' + [0, 1, 2].map((i) => c(i) + ' ' + rangeCm(box.min.getComponent(i), box.max.getComponent(i))).join('　')
                + (parts.size > 1 ? '<br><b style="color:#ffd54a">黃色數字</b> = 部件編號（' + parts.size + ' 個）' : '');
        }
    }

    function load(url) {
        if (!url || !init()) return;
        if (url === V.url && V.model) { setStatus(''); return; }
        setStatus('載入模型中… 0%');
        makeGltfLoader().load(
            url,
            (gltf) => {
                disposeModel();
                V.model = gltf.scene;
                V.scene.add(V.model);
                fitView(V.model);
                updateHelpers(V.model);
                V.url = url;
                setStatus('');
            },
            (xhr) => {
                if (xhr.total) setStatus('載入模型中… ' + Math.round(xhr.loaded / xhr.total * 100) + '%');
            },
            (err) => setStatus('模型載入失敗：' + (err && err.message ? err.message : err)),
        );
    }

    function clear() {
        if (V.renderer) { disposeModel(); removeHelpers(); }
        const dims = getContainer() && getContainer().querySelector('.live-viewer-dims');
        if (dims) dims.innerHTML = '';
        V.url = null;
        setStatus(IDLE_TEXT);
    }

    function resetView() {
        if (!V.home) return;
        V.camera.position.copy(V.home.position);
        V.controls.target.copy(V.home.target);
        V.controls.update();
    }

    // --- GLB 壓縮 before/after viewers: port of glb-shrink's ModelViewer (auto-rotate, framed) ---
    class CompareViewer {
        constructor(side) {
            this.side = side;
            this.model = null;
            this.url = null;
            this.renderer = null;
        }

        el() { return document.getElementById('shrink-canvas-' + this.side); }

        ensure() {
            const el = this.el();
            if (!el) return false;
            if (this.renderer) {
                if (this.container !== el) {  // re-mounted by Gradio
                    this.container = el;
                    el.prepend(this.renderer.domElement);
                    this.ro.disconnect();
                    this.ro.observe(el);
                }
                return true;
            }
            this.container = el;
            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.1;
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            el.prepend(renderer.domElement);

            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0e1018);
            const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
            camera.position.set(1.8, 1.2, 1.8);
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.autoRotate = true;
            controls.autoRotateSpeed = 1.2;
            const pmrem = new THREE.PMREMGenerator(renderer);
            scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
            pmrem.dispose();
            const key = new THREE.DirectionalLight(0xffffff, 1.4);
            key.position.set(3, 5, 2);
            scene.add(key);
            scene.add(new THREE.AmbientLight(0xffffff, 0.35));

            Object.assign(this, { renderer, scene, camera, controls });
            this.ro = new ResizeObserver(() => this.resize());
            this.ro.observe(el);
            this.resize();
            renderer.setAnimationLoop(() => {
                if (!this.container || !this.container.offsetParent) return;
                controls.update();
                renderer.render(scene, camera);
            });
            return true;
        }

        resize() {
            const w = this.container.clientWidth, h = this.container.clientHeight;
            if (w <= 0 || h <= 0) return;
            this.renderer.setSize(w, h, false);
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        }

        setEmpty(text) {
            const e = document.getElementById('shrink-empty-' + this.side);
            if (!e) return;
            if (text) e.textContent = text;
            e.classList.toggle('hidden', !text);
        }

        clear() {
            if (!this.model) return;
            this.scene.remove(this.model);
            this.model.traverse((o) => {
                if (o.isMesh) {
                    o.geometry.dispose();
                    (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
                }
            });
            this.model = null;
        }

        frame(object) {
            const box = new THREE.Box3().setFromObject(object);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());
            const dist = Math.max(size.x, size.y, size.z) * 1.8 || 1;
            object.position.sub(center);
            this.camera.near = dist / 200;
            this.camera.far = dist * 50;
            this.camera.updateProjectionMatrix();
            this.camera.position.set(dist * 0.7, dist * 0.55, dist * 0.7);
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }

        load(url, emptyText, tries = 0) {
            if (!this.ensure()) {
                // The tab may not be mounted yet right after switching to it
                if (tries < 30) setTimeout(() => this.load(url, emptyText, tries + 1), 100);
                return;
            }
            if (!url) {
                this.clear();
                this.url = null;
                this.setEmpty(emptyText);
                return;
            }
            if (url === this.url && this.model) return;
            this.setEmpty('載入中…');
            makeGltfLoader().load(
                url,
                (gltf) => {
                    this.clear();
                    this.model = gltf.scene;
                    this.frame(this.model);
                    this.scene.add(this.model);
                    this.url = url;
                    this.setEmpty('');
                },
                undefined,
                (err) => this.setEmpty('載入失敗：' + (err && err.message ? err.message : err)),
            );
        }
    }

    const compare = { before: new CompareViewer('before'), after: new CompareViewer('after') };
    window.shrinkView = {
        load(payload) {
            let d = {};
            try { d = JSON.parse(payload || '{}'); } catch (e) { return; }
            compare.before.load(d.before, '原始模型');
            compare.after.load(d.after, '壓縮後預覽');
            const sb = document.getElementById('shrink-stat-before');
            const sa = document.getElementById('shrink-stat-after');
            if (sb) sb.textContent = d.beforeStat || '—';
            if (sa) sa.textContent = d.afterStat || '—';
        },
    };

    window.trellisViewer = {
        init, load, clear, resetView, setStatus, togglePick,
        hasModel: () => !!V.model,
        getPicked: () => V.picked || null,
    };
</script>
""".replace('__THREE_BASE__', f'/gradio_api/file={THREE_DIR}').replace('__IDLE_TEXT__', LIVE_IDLE_TEXT).replace(
    '__HOME_YAW__', str(round(np.degrees(DEFAULT_STEP * 2 * np.pi / STEPS - 16 / 180 * np.pi), 3))
)


live_viewer_html = f"""
<div id="live-viewer" class="live-viewer-container">
    <button class="live-viewer-reset" onclick="window.trellisViewer && window.trellisViewer.resetView()">重設視角</button>
    <button class="live-viewer-reset live-viewer-pick" onclick="window.trellisViewer && window.trellisViewer.togglePick()">設定原點</button>
    <div class="live-viewer-dims"></div>
    <div class="live-viewer-status">{LIVE_IDLE_TEXT}</div>
    <div class="live-viewer-hint">左鍵拖曳：旋轉　滾輪：縮放　右鍵拖曳：平移</div>
</div>
"""

# Client-side handlers for the preview mode toggle
# Runs before prepare_live_glb and passes its inputs through unchanged.
TOGGLE_VIEW_JS = f"""
(mode, ...rest) => {{
    const live = mode === '{VIEW_LIVE}';
    document.body.classList.toggle('live-mode', live);
    const v = window.trellisViewer;
    if (live && v) {{
        v.init();
        if (!v.hasModel()) v.setStatus('正在匯出 GLB，約需半分鐘…');
    }}
    return [mode, ...rest];
}}
"""
LOAD_GLB_JS = "(url) => { if (url && window.trellisViewer) window.trellisViewer.load(url); return url; }"
SHRINK_VIEW_JS = "(v) => { if (window.shrinkView) window.shrinkView.load(v); return v; }"

# Before/after viewers for the GLB 壓縮 tab (layout from glb-shrink)
shrink_viewers_html = """
<div class="shrink-viewers">
    <div class="shrink-card">
        <div class="shrink-card-header"><span class="shrink-tag before">壓縮前</span><span class="shrink-stat" id="shrink-stat-before">—</span></div>
        <div class="shrink-canvas" id="shrink-canvas-before"></div>
        <div class="shrink-empty" id="shrink-empty-before">原始模型</div>
    </div>
    <div class="shrink-card">
        <div class="shrink-card-header"><span class="shrink-tag after">壓縮後</span><span class="shrink-stat" id="shrink-stat-after">—</span></div>
        <div class="shrink-canvas" id="shrink-canvas-after"></div>
        <div class="shrink-empty" id="shrink-empty-after">壓縮後預覽</div>
    </div>
</div>
"""
# Replaces the hidden textbox input with the point picked in the viewer
PICK_ORIGIN_JS = "(_, ...rest) => [JSON.stringify(window.trellisViewer ? window.trellisViewer.getPicked() : null), ...rest]"
RESET_VIEW_JS = f"""
() => {{
    document.body.classList.remove('live-mode');
    if (window.trellisViewer) window.trellisViewer.clear();
}}
"""


empty_html = f"""
<div class="previewer-container">
    <svg style=" opacity: .5; height: var(--size-5); color: var(--body-text-color);"
    xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="feather feather-image"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
</div>
"""


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)
    RAW_GLB.pop(str(req.session_hash), None)
    CANDIDATES.pop(str(req.session_hash), None)
    release_memory()


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
        'gen_id': uuid.uuid4().hex,
    }
    
    
def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    num_candidates: int,
    ss_interval_start: float,
    ss_interval_end: float,
    shape_interval_start: float,
    shape_interval_end: float,
    tex_interval_start: float,
    tex_interval_end: float,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate one or more candidates (seed, seed+1, ...) and show the first one.
    Candidates are kept server-side so switching between them needs no re-generation.
    """
    if image is None:
        raise gr.Error("請先上傳圖片。")

    def interval(a, b):
        a, b = float(a), float(b)
        if a >= b:
            raise gr.Error(f"引導區間的起點（{a}）必須小於終點（{b}）。")
        return [a, b]

    params = dict(
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
            "guidance_interval": interval(ss_interval_start, ss_interval_end),
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
            "guidance_rescale": shape_slat_guidance_rescale,
            "rescale_t": shape_slat_rescale_t,
            "guidance_interval": interval(shape_interval_start, shape_interval_end),
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
            "guidance_rescale": tex_slat_guidance_rescale,
            "rescale_t": tex_slat_rescale_t,
            "guidance_interval": interval(tex_interval_start, tex_interval_end),
        },
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[resolution],
    )
    session = str(req.session_hash)
    # The previous generation's raw GLB mesh and candidates are now stale
    RAW_GLB.pop(session, None)
    CANDIDATES.pop(session, None)
    release_memory()

    candidates = []
    for i in range(int(num_candidates)):
        outputs, latents = pipeline.run(image, seed=int(seed) + i, preprocess_image=False, return_latent=True, **params)
        mesh = outputs[0]
        mesh.simplify(16777216)  # nvdiffrast limit
        images = render_utils.render_snapshot(mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=envmap)
        candidates.append({'state': pack_state(latents), 'html': build_preview_html(images), 'seed': int(seed) + i})
        del outputs, latents, mesh, images
        release_memory()
    CANDIDATES[session] = candidates

    labels = candidate_labels(candidates)
    return candidates[0]['state'], candidates[0]['html'], gr.update(choices=labels, value=labels[0])


def candidate_labels(candidates: List[dict]) -> List[str]:
    return [f"候選 {i + 1}（種子 {c['seed']}）" for i, c in enumerate(candidates)]


def select_candidate(label: str, req: gr.Request):
    """Switch the active candidate; resets the export/edit state like a new generation."""
    candidates = CANDIDATES.get(str(req.session_hash))
    if not candidates or not label:
        return (gr.skip(),) * 2
    labels = candidate_labels(candidates)
    c = candidates[labels.index(label)] if label in labels else candidates[0]
    RAW_GLB.pop(str(req.session_hash), None)
    return c['state'], c['html']


def build_preview_html(images: dict) -> str:
    # --- HTML Construction ---
    # The Stack of 48 Images
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            # ID Naming Convention: view-m{mode}-s{step}
            unique_id = f"view-m{m_idx}-s{s_idx}"
            
            # Logic: Only Mode 0, Step 0 is visible initially
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            
            # Image Source
            img_base64 = image_to_base64(Image.fromarray(images[mode['render_key']][s_idx]))
            
            # Render the Tag
            images_html += f"""
                <img id="{unique_id}" 
                     class="previewer-main-image {vis_class}" 
                     src="{img_base64}" 
                     loading="eager">
            """
    
    # Button Row HTML
    btns_html = ""
    for idx, mode in enumerate(MODES):        
        active_class = "active" if idx == DEFAULT_MODE else ""
        # Note: onclick calls the JS function defined in Head
        btns_html += f"""
            <img src="{mode['icon_base64']}" 
                 class="mode-btn {active_class}" 
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """
    
    # Assemble the full component
    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡提示</div>
            <div class="tips-text">
                <p>● <b>渲染模式</b> - 點選下方圓形按鈕切換不同的渲染模式。</p>
                <p>● <b>視角</b> - 拖曳滑桿改變觀看角度。</p>
            </div>
        </div>
        
        <!-- Row 1: Viewport containing 48 static <img> tags -->
        <div class="display-row">
            {images_html}
        </div>
        
        <!-- Row 2 -->
        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <!-- Row 3: Slider -->
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    
    return full_html


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    scale_mode: str,
    target_cm: float,
    rot_x: int,
    rot_y: int,
    rot_z: int,
    keep_parts: List[str],
    origin_mode: str,
    origin_point: Optional[List[float]],
    origin_dx: float,
    origin_dy: float,
    origin_dz: float,
    exp_remesh_project: float,
    exp_cone_deg: float,
    exp_refine_iters: int,
    exp_global_iters: int,
    exp_smooth: float,
    exp_alpha_mode: str,
    glb_cache: Optional[dict],
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.
        scale_mode, target_cm: Which axis to scale to which real-world length.
        rot_x, rot_y, rot_z: Orientation correction in degrees, applied before scaling.
        keep_parts: Labels of the separated parts to keep (empty = all).
        origin_mode, origin_point: Where the GLB origin goes (origin_point is in the raw mesh frame).
        glb_cache (dict): The GLB already exported for the current generation, if any.

    Returns:
        str: The path to the extracted GLB file.
    """
    if state is None:
        raise gr.Error("請先按「生成」建立 3D 素材。")
    edit = make_edit(scale_mode, target_cm, rot_x, rot_y, rot_z, keep_parts, origin_mode, origin_point,
                     origin_dx, origin_dy, origin_dz)
    glb_cache = get_or_export_glb(state, decimation_target, texture_size, edit, glb_cache, req,
                                  make_export_opts(exp_remesh_project, exp_cone_deg, exp_refine_iters,
                                                   exp_global_iters, exp_smooth, exp_alpha_mode))
    path = glb_cache['path']
    return path, path, glb_cache, format_dims(glb_cache['dims']), parts_update(glb_cache)


def prepare_live_glb(
    mode: str,
    state: dict,
    decimation_target: int,
    texture_size: int,
    scale_mode: str,
    target_cm: float,
    rot_x: int,
    rot_y: int,
    rot_z: int,
    keep_parts: List[str],
    origin_mode: str,
    origin_point: Optional[List[float]],
    origin_dx: float,
    origin_dy: float,
    origin_dz: float,
    exp_remesh_project: float,
    exp_cone_deg: float,
    exp_refine_iters: int,
    exp_global_iters: int,
    exp_smooth: float,
    exp_alpha_mode: str,
    glb_cache: Optional[dict],
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Export (or reuse) the GLB for the live three.js viewer, and sync it to the Extract step.
    """
    if mode != VIEW_LIVE:
        return (gr.skip(),) * 6
    if state is None:
        gr.Info("請先按「生成」建立 3D 素材，再切換到即時 3D 檢視。")
        return gr.skip(), "", gr.skip(), gr.skip(), gr.skip(), gr.skip()
    edit = make_edit(scale_mode, target_cm, rot_x, rot_y, rot_z, keep_parts, origin_mode, origin_point,
                     origin_dx, origin_dy, origin_dz)
    glb_cache = get_or_export_glb(state, decimation_target, texture_size, edit, glb_cache, req,
                                  make_export_opts(exp_remesh_project, exp_cone_deg, exp_refine_iters,
                                                   exp_global_iters, exp_smooth, exp_alpha_mode))
    path = glb_cache['path']
    return glb_cache, f"/gradio_api/file={path}", path, path, format_dims(glb_cache['dims']), parts_update(glb_cache)


def refresh_transformed_glb(
    mode: str,
    state: dict,
    decimation_target: int,
    texture_size: int,
    scale_mode: str,
    target_cm: float,
    rot_x: int,
    rot_y: int,
    rot_z: int,
    keep_parts: List[str],
    origin_mode: str,
    origin_point: Optional[List[float]],
    origin_dx: float,
    origin_dy: float,
    origin_dz: float,
    exp_remesh_project: float,
    exp_cone_deg: float,
    exp_refine_iters: int,
    exp_global_iters: int,
    exp_smooth: float,
    exp_alpha_mode: str,
    glb_cache: Optional[dict],
    req: gr.Request,
):
    """
    Re-apply part selection / size / rotation / origin to an already exported GLB
    (fast: reuses the raw mesh). Does nothing until a GLB has been exported for the current generation.
    """
    if state is None or glb_cache is None:
        return (gr.skip(),) * 6
    if not keep_parts and len(glb_cache.get('labels', [])) > 1:
        gr.Warning("至少要保留一個部件，已恢復為保留全部。")
    if origin_mode == ORIGIN_CUSTOM and origin_point is None:
        gr.Info("請在「即時 3D 檢視」按「設定原點」，再點選模型表面。目前暫用幾何中心。")
    edit = make_edit(scale_mode, target_cm, rot_x, rot_y, rot_z, keep_parts, origin_mode, origin_point,
                     origin_dx, origin_dy, origin_dz)
    glb_cache = get_or_export_glb(state, decimation_target, texture_size, edit, glb_cache, req,
                                  make_export_opts(exp_remesh_project, exp_cone_deg, exp_refine_iters,
                                                   exp_global_iters, exp_smooth, exp_alpha_mode))
    path = glb_cache['path']
    url = f"/gradio_api/file={path}" if mode == VIEW_LIVE else gr.skip()
    return glb_cache, url, path, path, format_dims(glb_cache['dims']), parts_update(glb_cache)


def set_origin_from_pick(
    picked: str,
    mode: str,
    state: dict,
    decimation_target: int,
    texture_size: int,
    scale_mode: str,
    target_cm: float,
    rot_x: int,
    rot_y: int,
    rot_z: int,
    keep_parts: List[str],
    origin_mode: str,
    origin_point: Optional[List[float]],
    origin_dx: float,
    origin_dy: float,
    origin_dz: float,
    exp_remesh_project: float,
    exp_cone_deg: float,
    exp_refine_iters: int,
    exp_global_iters: int,
    exp_smooth: float,
    exp_alpha_mode: str,
    glb_cache: Optional[dict],
    req: gr.Request,
):
    """
    Use a point clicked in the live viewer (current GLB coordinates) as the new origin.
    The point is stored in the raw mesh frame so it follows later size/rotation changes.
    """
    try:
        point = np.array(json.loads(picked), dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        point = None
    if state is None or glb_cache is None or point is None:
        return (gr.skip(),) * 11
    raw_point = trimesh.transform_points(point[None], np.linalg.inv(np.array(glb_cache['matrix'])))[0].tolist()
    edit = make_edit(scale_mode, target_cm, rot_x, rot_y, rot_z, keep_parts, ORIGIN_CUSTOM, raw_point, 0, 0, 0)
    glb_cache = get_or_export_glb(state, decimation_target, texture_size, edit, glb_cache, req,
                                  make_export_opts(exp_remesh_project, exp_cone_deg, exp_refine_iters,
                                                   exp_global_iters, exp_smooth, exp_alpha_mode))
    path = glb_cache['path']
    url = f"/gradio_api/file={path}" if mode == VIEW_LIVE else gr.skip()
    return (glb_cache, url, path, path, format_dims(glb_cache['dims']), parts_update(glb_cache),
            ORIGIN_CUSTOM, raw_point, 0, 0, 0)


def make_edit(scale_mode, target_cm, rot_x, rot_y, rot_z, keep_parts, origin_mode, origin_point,
              origin_dx=0, origin_dy=0, origin_dz=0) -> dict:
    keep = sorted({int(m) - 1 for m in re.findall(r"部件 (\d+)", " ".join(keep_parts or []))})
    if origin_mode == ORIGIN_CUSTOM and origin_point is None:
        origin_mode = ORIGIN_CENTER
    return {
        'scale_mode': scale_mode, 'target_cm': float(target_cm or 0),
        'rot': [round(float(r or 0), 3) for r in (rot_x, rot_y, rot_z)],
        'keep': keep,  # empty = keep all parts
        'origin_mode': origin_mode,
        'origin_point': [round(float(v), 6) for v in origin_point] if origin_mode == ORIGIN_CUSTOM else None,
        'origin_offset': [round(float(v or 0), 3) for v in (origin_dx, origin_dy, origin_dz)],  # cm
    }


def parts_update(glb_cache: dict):
    labels = glb_cache.get('labels', [])
    kept = [labels[i] for i in glb_cache.get('kept', []) if i < len(labels)]
    return gr.update(choices=labels, value=kept)


def format_dims(dims: Optional[List[float]]) -> str:
    if not dims:
        return ""
    x, y, z = (d * 100 for d in dims)
    return (f"**模型尺寸**：寬（X）{x:.1f} cm × 高（Y）{y:.1f} cm × 深（Z）{z:.1f} cm"
            f"　*（glTF 單位為公尺，three.js 中 1 單位 = 1 m）*")


def transform_matrix(points: np.ndarray, scale_mode: str, target_cm: float,
                     rot_x: float, rot_y: float, rot_z: float) -> np.ndarray:
    """
    Rotation by arbitrary angles in degrees (X, then Y, then Z) about the bbox center, then
    uniform scaling so the requested axis of the *rotated* geometry has the real-world length.
    Extents are measured on the actual vertices, so oblique angles are exact.
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    T = trimesh.transformations.translation_matrix(-(lo + hi) / 2)
    for angle, axis in ((rot_x, [1, 0, 0]), (rot_y, [0, 1, 0]), (rot_z, [0, 0, 1])):
        if float(angle) % 360:
            T = trimesh.transformations.rotation_matrix(np.radians(float(angle)), axis) @ T
    if scale_mode in SCALE_AXES and target_cm and target_cm > 0:
        moved = trimesh.transform_points(points, T)
        extents = moved.max(axis=0) - moved.min(axis=0)
        axis = SCALE_AXES[scale_mode]
        current = extents.max() if axis is None else extents[axis]
        if current > 0:
            T = trimesh.transformations.scale_matrix(target_cm / 100 / current) @ T
    return T


def origin_matrix(points: np.ndarray, T: np.ndarray, origin_mode: str, origin_point: Optional[List[float]],
                  origin_offset: Optional[List[float]] = None) -> np.ndarray:
    """
    Translation (applied after T) that moves the requested origin to (0, 0, 0).
    """
    moved = trimesh.transform_points(points, T)
    lo, hi = moved.min(axis=0), moved.max(axis=0)
    center = (lo + hi) / 2
    if origin_mode == ORIGIN_BOTTOM:
        p = [center[0], lo[1], center[2]]
    elif origin_mode == ORIGIN_TOP:
        p = [center[0], hi[1], center[2]]
    elif origin_mode == ORIGIN_CUSTOM and origin_point is not None:
        p = trimesh.transform_points(np.array([origin_point]), T)[0]
    else:
        p = center
    # Fine adjustment in cm along the final (rotated + scaled) axes
    p = np.asarray(p, dtype=np.float64) + np.asarray(origin_offset or [0, 0, 0], dtype=np.float64) / 100
    return trimesh.transformations.translation_matrix(-p)


def split_parts(mesh: trimesh.Trimesh, gap_ratio: float = 0.02, min_frac: float = 0.01,
                max_parts: int = 9, tiny_faces: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group faces into spatially separate objects (e.g. the three copies generated from a
    three-view sheet). Connected components whose bounding boxes come within gap_ratio of
    the model size are merged; tiny fragments and small groups join the nearest large part.

    Returns:
        face_group: part index per face (0 = largest part).
        fracs: face fraction of each part.
    """
    n = len(mesh.faces)
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=n)
    k = labels.max() + 1
    tri = mesh.vertices[mesh.faces]
    cmin = np.full((k, 3), np.inf)
    np.minimum.at(cmin, labels, tri.min(axis=1))
    cmax = np.full((k, 3), -np.inf)
    np.maximum.at(cmax, labels, tri.max(axis=1))
    counts = np.bincount(labels, minlength=k)
    margin = gap_ratio * mesh.extents.max()

    # Union-find over the non-tiny components with overlapping (expanded) bounding boxes
    main = np.nonzero(counts >= tiny_faces)[0]
    if len(main) == 0:
        main = np.array([int(np.argmax(counts))])
    lo, hi = cmin[main] - margin, cmax[main] + margin
    parent = np.arange(len(main))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(main) - 1):
        overlap = np.all((lo[i] <= hi[i + 1:]) & (hi[i] >= lo[i + 1:]), axis=1)
        for j in np.nonzero(overlap)[0] + i + 1:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
    roots = np.array([find(i) for i in range(len(main))])
    _, main_group = np.unique(roots, return_inverse=True)
    g = main_group.max() + 1
    gcount = np.bincount(main_group, weights=counts[main], minlength=g)
    gmin = np.full((g, 3), np.inf)
    np.minimum.at(gmin, main_group, cmin[main])
    gmax = np.full((g, 3), -np.inf)
    np.maximum.at(gmax, main_group, cmax[main])
    gcenter = (gmin + gmax) / 2

    # Keep the largest groups as parts; everything else joins the nearest part
    order = np.argsort(-gcount)
    big = [int(o) for o in order if gcount[o] >= min_frac * n][:max_parts] or [int(order[0])]
    remap = np.array([int(np.argmin(np.linalg.norm(gcenter[big] - gcenter[o], axis=1))) for o in range(g)])
    for rank, o in enumerate(big):
        remap[o] = rank
    comp_part = np.empty(k, dtype=np.int64)
    comp_part[main] = remap[main_group]
    tiny = np.setdiff1d(np.arange(k), main)
    if len(tiny):
        tcenter = (cmin[tiny] + cmax[tiny]) / 2
        pcenter = gcenter[big]
        comp_part[tiny] = np.argmin(np.linalg.norm(tcenter[:, None] - pcenter[None], axis=2), axis=1)
    face_group = comp_part[labels]
    fracs = np.bincount(face_group, minlength=len(big)) / n
    return face_group, fracs


def get_or_export_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    edit: dict,
    glb_cache: Optional[dict],
    req: gr.Request,
    export_opts: Optional[dict] = None,
) -> dict:
    """
    Reuse the GLB exported for the current generation when the export settings are unchanged.
    The raw remeshed + baked mesh (and its part split) is kept per session, so editing
    parts / size / rotation / origin only re-exports (~1-3 s).
    """
    export_opts = export_opts or make_export_opts()
    raw_opts = {k: v for k, v in export_opts.items() if k != 'alpha_mode'}  # need a new remesh/bake
    key = [state['gen_id'], int(decimation_target), int(texture_size), raw_opts, edit, export_opts['alpha_mode']]
    if glb_cache and glb_cache['key'] == key and os.path.exists(glb_cache['path']):
        return glb_cache
    t_start = time.time()
    session = str(req.session_hash)
    raw_key = key[:4]
    raw = RAW_GLB.get(session)
    if raw is None or raw['key'] != raw_key:
        RAW_GLB.pop(session, None)
        print(f"[export] {datetime.now():%T} remesh+bake start opts={raw_opts}", flush=True)
        mesh = build_raw_glb(state, decimation_target, texture_size, raw_opts)
        print(f"[export] {datetime.now():%T} remesh+bake done {time.time() - t_start:.1f}s", flush=True)
        face_group, fracs = split_parts(mesh)
        labels = [f"部件 {i + 1}（{f * 100:.0f}%）" for i, f in enumerate(fracs)]
        raw = {'key': raw_key, 'mesh': mesh, 'face_group': face_group, 'labels': labels}
        RAW_GLB[session] = raw
        while len(RAW_GLB) > RAW_GLB_MAX_SESSIONS:
            RAW_GLB.popitem(last=False)
    RAW_GLB.move_to_end(session)

    labels = raw['labels']
    keep = [i for i in edit['keep'] if i < len(labels)] or list(range(len(labels)))
    parts = {i: raw['mesh'].submesh([np.nonzero(raw['face_group'] == i)[0]], append=True) for i in keep}
    points = np.vstack([p.vertices for p in parts.values()])
    T = transform_matrix(points, edit['scale_mode'], edit['target_cm'], *edit['rot'])
    T = origin_matrix(points, T, edit['origin_mode'], edit['origin_point'], edit['origin_offset']) @ T
    del points
    scene = trimesh.Scene()
    for i, part in parts.items():
        part.apply_transform(T)
        apply_alpha_mode(part, export_opts['alpha_mode'])
        scene.add_geometry(part, node_name=f"part_{i + 1}", geom_name=f"part_{i + 1}")
    glb_path = save_glb(scene, req)
    dims = scene.extents.tolist()
    del parts, scene
    release_memory()
    print(f"[export] {datetime.now():%T} glb written {time.time() - t_start:.1f}s edit={edit} alpha={export_opts['alpha_mode']}", flush=True)
    return {'key': key, 'path': glb_path, 'dims': dims, 'matrix': T.tolist(), 'labels': labels, 'kept': keep}


def make_export_opts(remesh_project=0.0, cone_deg=90.0, refine_iters=0, global_iters=1, smooth=1.0,
                     alpha_mode="不透明（OPAQUE）") -> dict:
    """Expert GLB export options (defaults = the original app.py behavior)."""
    return {
        'remesh_project': round(float(remesh_project), 3),
        'cone_deg': round(float(cone_deg), 2),
        'refine_iters': int(refine_iters),
        'global_iters': max(1, int(global_iters)),
        'smooth': round(float(smooth), 3),
        'alpha_mode': ALPHA_MODES.get(alpha_mode, alpha_mode if alpha_mode in ALPHA_MODES.values() else 'OPAQUE'),
    }


def apply_alpha_mode(mesh: trimesh.Trimesh, alpha_mode: str):
    """to_glb always writes OPAQUE although the alpha channel is baked; expose it."""
    material = getattr(mesh.visual, 'material', None)
    if material is None:
        return
    # The raw mesh's material is shared across exports, so set every field explicitly
    material.alphaMode = alpha_mode
    material.alphaCutoff = 0.5 if alpha_mode == 'MASK' else None
    material.doubleSided = alpha_mode == 'BLEND'  # remeshed output is single-sided otherwise


def save_glb(mesh: Union[trimesh.Trimesh, trimesh.Scene], req: gr.Request) -> str:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    mesh.export(glb_path, extension_webp=True)
    return glb_path


def build_raw_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    opts: Optional[dict] = None,
) -> trimesh.Trimesh:
    opts = opts or {k: v for k, v in make_export_opts().items() if k != 'alpha_mode'}
    shape_slat, tex_slat, res = unpack_state(state)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=opts['remesh_project'],
        mesh_cluster_threshold_cone_half_angle_rad=np.radians(opts['cone_deg']),
        mesh_cluster_refine_iterations=opts['refine_iters'],
        mesh_cluster_global_iterations=opts['global_iters'],
        mesh_cluster_smooth_strength=opts['smooth'],
        use_tqdm=True,
    )
    torch.cuda.empty_cache()
    return glb


# ---------------------------------------------------------------------------
# GLB 壓縮 (glb-shrink integration): the Node pipeline lives in tools/glb_shrink
# ---------------------------------------------------------------------------

def shrink_hint(quality: float) -> str:
    """Traditional Chinese version of glb-shrink's getPresetHint()."""
    q = max(0.0, min(100.0, float(quality)))
    if q <= 20:
        return "極小檔案：適合幾乎不會注意到的背景道具。"
    if q <= 40:
        return "小檔案：適合場景中較遠的物件。"
    if q <= 60:
        return "平衡：大多數專案的最佳選擇。"
    if q <= 80:
        return "更多細節：邊緣與貼圖保持較銳利。"
    return "最高細節：適合近距離觀看，檔案較大。"


def format_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def format_tris(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M 面"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k 面"
    return f"{n} 面"


def run_shrink(*args: str) -> dict:
    """Run tools/glb_shrink/cli.mjs and return its JSON result."""
    try:
        proc = subprocess.run(
            ["node", os.path.join(SHRINK_DIR, "cli.mjs"), *args],
            cwd=SHRINK_DIR, capture_output=True, text=True, timeout=900,
        )
    except FileNotFoundError:
        raise gr.Error("找不到 Node.js，請先在 WSL 安裝 Node.js 18 以上版本。")
    except subprocess.TimeoutExpired:
        raise gr.Error("GLB 壓縮逾時（超過 15 分鐘）。")
    try:
        result = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        result = {"error": (proc.stderr or proc.stdout or "未知錯誤").strip()[-500:]}
    if proc.returncode != 0 or "error" in result:
        raise gr.Error(f"GLB 壓縮失敗：{result.get('error', proc.stderr.strip()[-500:])}")
    return result


def shrink_hero_html(src: Optional[dict] = None, out: Optional[dict] = None) -> str:
    if not src:
        return ('<div class="shrink-hero"><div class="shrink-hero-label">檔案大小</div>'
                '<div class="shrink-hero-sub">上傳 GLB，或在「圖片轉 3D」按「送到 GLB 壓縮」</div></div>')
    before = format_bytes(src['size'])
    if not out:
        return (f'<div class="shrink-hero"><div class="shrink-hero-label">檔案大小</div>'
                f'<div class="shrink-hero-sizes"><span class="shrink-before plain">{before}</span>'
                f'<span class="shrink-arrow">→</span><span class="shrink-after">—</span></div>'
                f'<div class="shrink-badge">可以開始壓縮</div>'
                f'<div class="shrink-hero-sub">{format_tris(src["tris"])} · 選擇品質後按「壓縮模型」</div></div>')
    pct = (src['size'] - out['size']) / src['size'] * 100 if src['size'] else 0
    return (f'<div class="shrink-hero"><div class="shrink-hero-label">檔案大小</div>'
            f'<div class="shrink-hero-sizes"><span class="shrink-before">{before}</span>'
            f'<span class="shrink-arrow">→</span><span class="shrink-after">{format_bytes(out["size"])}</span></div>'
            f'<div class="shrink-badge">−{pct:.0f}% 更小</div>'
            f'<div class="shrink-hero-sub">{format_tris(src["tris"])} → {format_tris(out["tris"])}'
            f' · Draco 幾何 + WebP 貼圖</div></div>')


def shrink_view_json(src: Optional[dict], out: Optional[dict] = None) -> str:
    """Payload for the before/after three.js viewers (read by client-side js)."""
    return json.dumps({
        'before': f"/gradio_api/file={src['path']}" if src else "",
        'after': f"/gradio_api/file={out['path']}" if out else "",
        'beforeStat': format_bytes(src['size']) if src else "—",
        'afterStat': format_bytes(out['size']) if out else "—",
    })


def shrink_ingest(path: str, name: str, req: gr.Request):
    """Copy a GLB into the session's shrink folder, inspect it, and reset the output side."""
    if not name.lower().endswith('.glb'):
        raise gr.Error("請上傳 .glb 檔案。")
    if os.path.getsize(path) > SHRINK_MAX_BYTES:
        raise gr.Error("檔案超過 200 MB 上限。")
    shrink_dir = os.path.join(TMP_DIR, str(req.session_hash), 'shrink')
    os.makedirs(shrink_dir, exist_ok=True)
    src_path = os.path.join(shrink_dir, f"{uuid.uuid4().hex[:8]}_{os.path.basename(name)}")
    shutil.copyfile(path, src_path)
    info = run_shrink("inspect", src_path)
    src = {'path': src_path, 'name': os.path.basename(name), 'size': info['fileSize'], 'tris': info['tris']}
    meta = f"**{src['name']}**　`{format_tris(src['tris'])} · {format_bytes(src['size'])}`"
    return src, shrink_hero_html(src), shrink_view_json(src), meta, gr.update(value=None, interactive=False)


def shrink_upload(file_path: Optional[str], req: gr.Request):
    if not file_path:
        return None, shrink_hero_html(), shrink_view_json(None), "", gr.update(value=None, interactive=False)
    return shrink_ingest(file_path, os.path.basename(file_path), req)


def shrink_compress(src: Optional[dict], quality: float, req: gr.Request, progress=gr.Progress()):
    if not src:
        raise gr.Error("請先上傳 GLB 檔案。")
    progress(0.1, desc="壓縮中…（減面 → WebP 貼圖 → Draco）")
    stem = re.sub(r'\.glb$', '', src['name'], flags=re.IGNORECASE)
    out_path = os.path.join(os.path.dirname(src['path']), f"{uuid.uuid4().hex[:8]}", f"{stem}-draco.glb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = run_shrink("compress", src['path'], out_path, str(int(quality)))
    out = {'path': out_path, 'size': result['outputSize'], 'tris': result['stats']['finalTris']}
    return shrink_hero_html(src, out), shrink_view_json(src, out), gr.update(value=out_path, interactive=True)


def send_to_shrink(
    state: dict,
    decimation_target: int,
    texture_size: int,
    scale_mode: str,
    target_cm: float,
    rot_x: float,
    rot_y: float,
    rot_z: float,
    keep_parts: List[str],
    origin_mode: str,
    origin_point: Optional[List[float]],
    origin_dx: float,
    origin_dy: float,
    origin_dz: float,
    exp_remesh_project: float,
    exp_cone_deg: float,
    exp_refine_iters: int,
    exp_global_iters: int,
    exp_smooth: float,
    exp_alpha_mode: str,
    glb_cache: Optional[dict],
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """Export the current (edited) GLB if needed and open it in the GLB 壓縮 tab."""
    if state is None:
        raise gr.Error("請先按「生成」建立 3D 素材。")
    edit = make_edit(scale_mode, target_cm, rot_x, rot_y, rot_z, keep_parts, origin_mode, origin_point,
                     origin_dx, origin_dy, origin_dz)
    glb_cache = get_or_export_glb(state, decimation_target, texture_size, edit, glb_cache, req,
                                  make_export_opts(exp_remesh_project, exp_cone_deg, exp_refine_iters,
                                                   exp_global_iters, exp_smooth, exp_alpha_mode))
    name = f"trellis_{datetime.now():%Y%m%d_%H%M%S}.glb"
    # Clear the upload widget so it does not show a previously uploaded file
    return (glb_cache, gr.Tabs(selected="shrink"), None, *shrink_ingest(glb_cache['path'], name, req))


with gr.Blocks(delete_cache=(600, 600), title="TRELLIS.2 圖片轉 3D") as demo:
    gr.Markdown("""
    ## 使用 [TRELLIS.2](https://microsoft.github.io/TRELLIS.2) 將圖片轉成 3D 素材
    * 上傳一張圖片（建議使用已去背、帶透明通道的前景物件），按下「生成」即可建立 3D 素材。
    * 對結果滿意的話，按「匯出 GLB」即可匯出並下載 GLB 檔；不滿意可以換個隨機種子再試一次。
    * *註：上傳的圖片只會暫存在本機伺服器，工作階段結束後即刪除，不會上傳到任何外部服務。*
    """)

    with gr.Tabs(selected="gen") as tabs:
        with gr.Tab("圖片轉 3D", id="gen"):
            with gr.Row():
                # Three panels: input/generation | preview | editing & settings (side panels scroll independently)
                with gr.Column(scale=3, min_width=320, elem_id="col-left"):
                    image_prompt = gr.Image(label="輸入圖片", format="png", image_mode="RGBA", type="pil", sources=["upload", "clipboard"], height=400)
                    gr.Markdown(
                        "*上傳後會自動去背並裁切到物件範圍。若圖片已有透明背景（PNG 帶 alpha），會直接使用不再去背。"
                        "**請上傳單一視角**：正面或 3/4 角度效果最好；三視圖整張上傳會被生成成三個物件。*"
                    )

                    resolution = gr.Radio(
                        ["512", "1024", "1536"], label="解析度", value="1024",
                        info="生成的體素解析度，決定幾何與材質的細緻程度。"
                             "512：最快（約 15 秒），細節較少，適合快速試圖；"
                             "1024：建議值（生成約 45 秒、匯出約 30 秒），細節與速度平衡；"
                             "1536：細節最多，但生成約 2 分鐘、匯出 GLB 約 5 分鐘且會用滿 32GB 顯存。",
                    )
                    seed = gr.Slider(
                        0, MAX_SEED, label="隨機種子", value=0, step=1,
                        info="決定生成時的隨機雜訊。相同圖片 + 相同種子 + 相同參數會得到相同結果；"
                             "換種子會得到不同的形狀與細節變化。找到滿意的結果時記下種子，之後可重現。",
                    )
                    randomize_seed = gr.Checkbox(
                        label="每次隨機產生種子", value=True,
                        info="勾選：每次按「生成」都換一個新種子（探索不同結果）；"
                             "取消勾選：使用上方固定的種子（重現或微調參數比較差異）。",
                    )
                    decimation_target = gr.Slider(
                        10, 1000000, label="減面目標（面數）", value=500000, step=10,
                        info="匯出 GLB 時重新網格化後要保留的三角面數上限（可直接在右側數字框輸入）。"
                             "越高：細節越多、檔案越大、遊戲中渲染越吃效能；越低：越輕量但小細節會被抹平。"
                             "下限 10 是實測仍可運算的最低值，但形狀會隨面數急速崩壞：約 2 萬面仍完整、"
                             "5 千面大致完整、1 千面細部開始消失、100 面以下嚴重變形。"
                             "遊戲角色/機甲部件建議 1～5 萬（主角可到 10～30 萬），展示用 50～100 萬。"
                             "也是 GLB 壓縮能降到多少面數的主要因素。",
                    )
                    texture_size = gr.Slider(
                        1024, 4096, label="貼圖尺寸", value=2048, step=1024,
                        info="烘焙出的 PBR 貼圖（基礎色、金屬度、粗糙度）邊長像素。"
                             "越大：表面花紋與文字越清晰，但檔案與顯存用量約隨邊長平方成長；"
                             "1024 適合遠景或小物件、2048 為一般建議、4096 適合主角近拍。",
                    )

                    generate_btn = gr.Button("生成", variant="primary")

                with gr.Column(scale=6, min_width=480, elem_id="col-mid"):
                    with gr.Walkthrough(selected=0) as walkthrough:
                        with gr.Step("預覽", id=0):
                            candidate_pick = gr.Radio(
                                [], label="候選模型",
                                info="「專家設定 → 一次生成幾個候選」大於 1 時，可在這裡切換要使用哪一個（會重設 GLB 編輯狀態）。",
                            )
                            view_mode = gr.Radio(
                                [VIEW_STATIC, VIEW_LIVE], value=VIEW_STATIC, label="預覽方式",
                                info="靜態渲染預覽：生成後立即可看，6 種渲染模式 × 8 個視角；"
                                     "即時 3D 檢視：自動匯出 GLB（約 30 秒）後用 three.js 自由旋轉縮放，並可使用 GLB 編輯功能。",
                            )
                            preview_output = gr.HTML(empty_html, label="3D 素材預覽", show_label=True, container=True, elem_id="preview-static")
                            live_view = gr.HTML(live_viewer_html, label="即時 3D 檢視（three.js）", show_label=True, container=True, elem_id="preview-live")
                            with gr.Row():
                                extract_btn = gr.Button("匯出 GLB")
                                to_shrink_btn = gr.Button("送到 GLB 壓縮 ➜")
                            gr.Markdown("*「匯出 GLB」：依目前的減面、貼圖、GLB 編輯與專家設定輸出可下載的 GLB；"
                                        "「送到 GLB 壓縮」：把目前編輯好的模型帶到 GLB 壓縮分頁做 Draco + WebP 壓縮。*")
                        with gr.Step("匯出", id=1):
                            glb_output = gr.Model3D(label="已匯出的 GLB", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                            download_btn = gr.DownloadButton(label="下載 GLB")
                            dims_md = gr.Markdown("")
                            to_shrink_btn2 = gr.Button("送到 GLB 壓縮 ➜")
                            gr.Markdown("*匯出 GLB 需要重新網格化、減面與材質烘焙，通常需要半分鐘以上，請耐心等候。*")

                with gr.Column(scale=3, min_width=340, elem_id="col-right"):
                    with gr.Accordion(label="GLB 編輯（部件、尺寸、方向、原點）", open=True):
                        gr.Markdown("匯出 GLB（或切換到即時 3D 檢視）後即可編輯，每次調整約 1–3 秒自動重新匯出。"
                                    "*套用順序：保留部件 → 旋轉 → 等比例縮放 → 移動原點。*")
                        keep_parts = gr.CheckboxGroup(
                            [], label="保留的部件",
                            info="匯出時會自動把空間上分開的物件分成「部件 1、2、3…」（依大小排序，括號內為面數占比），"
                                 "編號會以黃色數字標示在即時 3D 檢視中。取消勾選即從 GLB 刪除該部件，"
                                 "例如三視圖生成的三個模型只保留一個。至少需保留一個。",
                        )
                        scale_mode = gr.Dropdown(
                            SCALE_MODES, value=SCALE_NONE, label="縮放依據", min_width=160,
                            info="TRELLIS 會把每個模型正規化成最長邊約 100 cm，不同部件比例不一致。"
                                 "選擇以哪一軸為基準，等比例縮放成右側指定的實際長度（旋轉後的軸向）。",
                        )
                        target_cm = gr.Number(
                            value=100, minimum=0.1, label="目標長度（cm）", min_width=120,
                            info="縮放依據那一軸要變成的實際長度。GLB 單位為公尺，three.js 中 1 單位 = 1 m。"
                                 "按 Enter 或點到其他地方套用。",
                        )
                        with gr.Row():
                            rot_x = gr.Number(value=0, step=1, label="繞 X 軸旋轉（°）", min_width=100,
                                              info="繞紅色 X 軸旋轉，常用於把向前/向後倒的模型扶正。")
                            rot_y = gr.Number(value=0, step=1, label="繞 Y 軸旋轉（°）", min_width=100,
                                              info="繞綠色 Y 軸（朝上）旋轉，用來調整模型面向的方向。")
                            rot_z = gr.Number(value=0, step=1, label="繞 Z 軸旋轉（°）", min_width=100,
                                              info="繞藍色 Z 軸旋轉，常用於把側躺的模型扶正。")
                        gr.Markdown("*旋轉可輸入任意角度（可含小數、負數），固定依 X → Y → Z 順序套用；"
                                    "依右手定則，從軸的正方向看過去正值為逆時針。按 Enter 或點到其他地方即套用。*")
                        origin_mode = gr.Dropdown(
                            ORIGIN_MODES, value=ORIGIN_CENTER, label="原點位置",
                            info="GLB 的 (0,0,0) 放在哪裡，決定在 three.js 中 position 的基準點與旋轉的支點。"
                                 "幾何中心：外框正中央；底部中心：腳底，適合站在地面的物件；"
                                 "頂部中心：適合吊掛的部件（如手臂從肩膀往下）；"
                                 "自訂：在即時 3D 檢視按「設定原點」再點選模型表面（例如關節處）。",
                        )
                        with gr.Row():
                            origin_dx = gr.Number(value=0, step=0.1, label="原點微調 X（cm）", min_width=100,
                                                  info="原點沿 X 軸再移動的距離，正值往紅色箭頭方向。")
                            origin_dy = gr.Number(value=0, step=0.1, label="原點微調 Y（cm）", min_width=100,
                                                  info="原點沿 Y 軸再移動的距離，正值往上。")
                            origin_dz = gr.Number(value=0, step=0.1, label="原點微調 Z（cm）", min_width=100,
                                                  info="原點沿 Z 軸再移動的距離，正值往藍色箭頭方向。")
                        with gr.Row():
                            gr.Markdown("*微調是在上方原點位置的基礎上，沿最終（旋轉、縮放後）的軸向移動；"
                                        "點選模型設定原點時會自動歸零。即時 3D 檢視左上角會顯示模型相對原點的範圍。*")
                            origin_offset_reset = gr.Button("微調歸零", size="sm", scale=0, min_width=90)

                    with gr.Accordion(label="進階設定（三階段取樣參數）", open=False):
                        gr.Markdown(
                            "TRELLIS.2 分三個階段生成，每個階段都用 flow matching 從雜訊逐步去噪：\n"
                            "1. **稀疏結構**：決定哪些體素被佔據，也就是整體輪廓與大致形狀。\n"
                            "2. **形狀**：在這些體素上生成精細幾何（表面細節、邊角）。\n"
                            "3. **材質**：依形狀生成 PBR 材質（顏色、金屬度、粗糙度）。\n\n"
                            "四個參數的意義：**引導強度**（CFG）越高越貼近輸入圖片，過高會過度銳化、破面或顏色過飽和，1 = 不使用引導；"
                            "**引導重新縮放**用來抑制高引導強度造成的過度飽和與爆亮，0 = 不抑制、1 = 完全抑制；"
                            "**取樣步數**越多越穩定、細節越完整，但時間大致與步數成正比；"
                            "**時間重新縮放**越大越把步數集中在高雜訊（決定大結構）的階段，通常讓整體結構更穩，"
                            "過大可能損失細節。預設值為官方建議值。"
                        )
                        gr.Markdown("**階段 1：稀疏結構生成**（影響輪廓與整體形狀）")
                        with gr.Row():
                            ss_guidance_strength = gr.Slider(1.0, 10.0, label="引導強度", value=7.5, step=0.1,
                                                             info="預設 7.5。調高：輪廓更貼近圖片；調低：形狀更自由但可能偏離圖片。")
                            ss_guidance_rescale = gr.Slider(0.0, 1.0, label="引導重新縮放", value=0.7, step=0.01,
                                                            info="預設 0.7。出現多餘碎塊或過度膨脹時可調高。")
                        with gr.Row():
                            ss_sampling_steps = gr.Slider(1, 50, label="取樣步數", value=12, step=1,
                                                          info="預設 12。輪廓不穩或有缺塊時可增加到 20～30。")
                            ss_rescale_t = gr.Slider(1.0, 6.0, label="時間重新縮放（Rescale T）", value=5.0, step=0.1,
                                                     info="預設 5.0。此階段以大結構為主，建議維持較高值。")
                        gr.Markdown("**階段 2：形狀生成**（影響幾何細節）")
                        with gr.Row():
                            shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="引導強度", value=7.5, step=0.1,
                                                                     info="預設 7.5。調高：表面細節更貼近圖片；過高會出現尖刺或破面。")
                            shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="引導重新縮放", value=0.5, step=0.01,
                                                                    info="預設 0.5。表面出現雜訊或過度銳化時可調高。")
                        with gr.Row():
                            shape_slat_sampling_steps = gr.Slider(1, 50, label="取樣步數", value=12, step=1,
                                                                  info="預設 12。增加可讓細節更完整，生成時間隨之增加。")
                            shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="時間重新縮放（Rescale T）", value=3.0, step=0.1,
                                                             info="預設 3.0。調低會保留更多細部，調高結構更穩。")
                        gr.Markdown("**階段 3：材質生成**（影響顏色與金屬/粗糙質感）")
                        with gr.Row():
                            tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="引導強度", value=1.0, step=0.1,
                                                                   info="預設 1.0（不使用引導）。調高可讓顏色更接近圖片，過高易過飽和。")
                            tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="引導重新縮放", value=0.0, step=0.01,
                                                                  info="預設 0。調高引導強度後若顏色過飽和，再調高此值。")
                        with gr.Row():
                            tex_slat_sampling_steps = gr.Slider(1, 50, label="取樣步數", value=12, step=1,
                                                                info="預設 12。材質出現斑駁或雜訊時可增加。")
                            tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="時間重新縮放（Rescale T）", value=3.0, step=0.1,
                                                           info="預設 3.0。一般不需調整。")

                    with gr.Accordion(label="專家設定", open=False):
                        gr.Markdown("#### 生成時生效（下次按「生成」才會套用）")
                        num_candidates = gr.Slider(
                            1, 4, value=1, step=1, label="一次生成幾個候選",
                            info="以種子、種子+1、種子+2…各生成一個模型，生成後可在預覽上方切換挑選最好的。"
                                 "時間約與數量成正比（1024 每個約 45 秒），候選只保留到下次生成。",
                        )
                        gr.Markdown(
                            "**引導區間**：只在去噪過程的這段時間內套用「引導強度」（0 = 去噪結束、1 = 純雜訊開始）。"
                            "區間越大，越多步驟受圖片引導，結果越貼近圖片但也越容易過度銳化；"
                            "起點調低會讓引導延伸到細節階段，終點調低則讓開頭的大結構更自由。起點必須小於終點。"
                        )
                        with gr.Row():
                            ss_interval_start = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="階段 1 區間起點",
                                                          info="預設 0.6")
                            ss_interval_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="階段 1 區間終點",
                                                        info="預設 1.0")
                        with gr.Row():
                            shape_interval_start = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="階段 2 區間起點",
                                                             info="預設 0.6")
                            shape_interval_end = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="階段 2 區間終點",
                                                           info="預設 1.0")
                        with gr.Row():
                            tex_interval_start = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="階段 3 區間起點",
                                                           info="預設 0.6（材質引導強度為 1 時此設定無作用）")
                            tex_interval_end = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="階段 3 區間終點",
                                                         info="預設 0.9")

                        gr.Markdown("#### 匯出 GLB 時生效（變更後若已匯出，會自動重新匯出，約 30～45 秒）")
                        exp_remesh_project = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05, label="頂點吸附原表面（remesh_project）",
                            info="重新網格化後，把頂點往原始高解析表面拉回的程度。0：維持重建後較圓滑的表面（預設）；"
                                 "越接近 1：稜角、刻線越銳利，適合機甲、武器等硬邊造型，但可能出現細小鋸齒。建議硬邊模型試 0.6～0.9。",
                        )
                        exp_cone_deg = gr.Slider(
                            10, 180, value=90, step=5, label="UV 分塊角度（°）",
                            info="UV 展開時，表面法線方向差異在此角度內的面會分在同一塊。"
                                 "調大：UV 塊數與接縫變少，貼圖空間利用率較高，但大塊 UV 的貼圖可能較扭曲；"
                                 "調小：接縫多但每塊扭曲小。實測對 GLB 壓縮後的面數影響很小（90°→150° 約少 2%），"
                                 "匯出時間略增。預設 90。",
                        )
                        exp_smooth = gr.Slider(
                            0, 10, value=1, step=0.5, label="UV 分塊平滑度",
                            info="分塊邊界的平滑強度。調高可讓分塊邊界更整齊、減少零碎小塊。預設 1。",
                        )
                        exp_refine_iters = gr.Slider(
                            0, 10, value=0, step=1, label="UV 分塊細化次數",
                            info="對分塊結果做局部細化的次數。增加可減少不規則的小碎塊，匯出時間略增。預設 0。",
                        )
                        exp_global_iters = gr.Slider(
                            1, 10, value=1, step=1, label="UV 分塊全域迭代次數",
                            info="整體重新分配分塊的次數。增加可讓分塊更合理、塊數更少，匯出時間略增。預設 1。",
                        )
                        exp_alpha_mode = gr.Dropdown(
                            list(ALPHA_MODES), value=list(ALPHA_MODES)[0], label="透明度模式",
                            info="TRELLIS 會烘焙出透明度，但原版一律以不透明輸出。"
                                 "不透明：忽略透明度（預設，效能最好）；"
                                 "半透明混合：玻璃座艙罩、能量罩等半透明部件會正確顯示，但半透明物件在遊戲中排序與效能成本較高；"
                                 "透明裁切：透明度低於 50% 的部分直接挖空（適合鏤空網格、葉片），不需排序。",
                        )
                        expert_reset_btn = gr.Button("還原專家設定預設值", size="sm")

        with gr.Tab("GLB 壓縮", id="shrink"):
            # Mirrors github.com/lorenhsu1128/glb-shrink: hero strip, controls, before/after viewers
            shrink_hero = gr.HTML(shrink_hero_html(), elem_id="shrink-hero")
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    gr.Markdown("### ◆ GLB 壓縮\nDraco + WebP 壓縮，產出適合遊戲、App、AR/VR 與網頁的輕量 3D 素材")
                    shrink_file = gr.File(label="拖放 GLB 檔（或點擊瀏覽 · 最大 200 MB）", file_types=[".glb"],
                                          type="filepath", height=140)
                    shrink_meta = gr.Markdown("")
                    shrink_preset = gr.Radio(
                        list(SHRINK_PRESETS), value=list(SHRINK_PRESETS)[1], label="要多清晰？",
                        info="快速選擇壓縮強度，會同步設定下方滑桿（0 / 50 / 100）。"
                             "最小檔案：貼圖縮到 256 px、減面最多，適合遠景道具；"
                             "平衡：貼圖 384 px，多數遊戲物件適用；最清晰：貼圖 512 px、保留較多面，適合近距離主角。",
                    )
                    shrink_quality = gr.Slider(
                        0, 100, value=50, step=1, label="檔案更小 ↔ 外觀更清晰",
                        info="在三個預設之間連續微調：數值越低，減面比例越大、貼圖越小（256→512 px），檔案越小；"
                             "越高細節越多、檔案越大。幾何一律用 Draco 壓縮、貼圖一律轉 WebP。",
                    )
                    shrink_hint_md = gr.Markdown(shrink_hint(50))
                    shrink_btn = gr.Button("壓縮模型", variant="primary")
                    # Always rendered: a DownloadButton that starts hidden loses its file value when shown (Gradio 6)
                    shrink_download = gr.DownloadButton("下載壓縮後的 GLB", interactive=False)
                    gr.Markdown(
                        "*流程：移除舊壓縮擴充 → 合併頂點 → meshoptimizer 減面 → 重算平滑法線 → "
                        "貼圖轉 WebP 並縮小 → Draco 幾何壓縮。輸出使用 `KHR_draco_mesh_compression` 與 "
                        "`EXT_texture_webp`，three.js 載入需搭配 `DRACOLoader`。*\n\n"
                        "*TRELLIS 生成的模型因 UV 接縫多，面數主要由「圖片轉 3D」的「減面目標」決定，"
                        "glb-shrink 主要壓縮檔案大小。*"
                    )
                with gr.Column(scale=3):
                    gr.HTML(shrink_viewers_html)

                    
    output_buf = gr.State()
    glb_cache = gr.State()  # GLB exported for the current generation: {'key': [decimation, texture], 'path': str}
    # Rendered but hidden via CSS: visible=False components do not deliver values to client-side js
    live_glb_url = gr.Textbox(elem_id="live-glb-url", container=False, show_label=False)
    origin_point = gr.State()  # custom origin in the raw mesh frame
    # Hidden channel for the point clicked in the live viewer ("設定原點")
    origin_pick_box = gr.Textbox(elem_id="origin-pick-box", container=False, show_label=False)
    origin_pick_btn = gr.Button("origin-pick", elem_id="origin-pick-btn")
    shrink_src = gr.State()  # {'path', 'name', 'size', 'tris'} of the GLB loaded in the GLB 壓縮 tab
    shrink_view = gr.Textbox(elem_id="shrink-view-json", container=False, show_label=False)  # viewer payload


    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    # Reset export/edit state for a new model (new generation or another candidate)
    def reset_for_new_model():
        return (gr.Walkthrough(selected=0), VIEW_STATIC, None, "", "", gr.update(choices=[], value=[]), None, 0, 0, 0)

    reset_outputs = [walkthrough, view_mode, glb_cache, live_glb_url, dims_md, keep_parts, origin_point,
                     origin_dx, origin_dy, origin_dz]

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        reset_for_new_model, outputs=reset_outputs, js=RESET_VIEW_JS,
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
            shape_slat_guidance_strength, shape_slat_guidance_rescale, shape_slat_sampling_steps, shape_slat_rescale_t,
            tex_slat_guidance_strength, tex_slat_guidance_rescale, tex_slat_sampling_steps, tex_slat_rescale_t,
            num_candidates, ss_interval_start, ss_interval_end, shape_interval_start, shape_interval_end,
            tex_interval_start, tex_interval_end,
        ],
        outputs=[output_buf, preview_output, candidate_pick],
    )

    # Switching candidates behaves like a new generation for the export/edit state
    candidate_pick.input(
        reset_for_new_model, outputs=reset_outputs, js=RESET_VIEW_JS,
    ).then(
        select_candidate, inputs=[candidate_pick], outputs=[output_buf, preview_output],
    )

    expert_export = [exp_remesh_project, exp_cone_deg, exp_refine_iters, exp_global_iters, exp_smooth, exp_alpha_mode]
    export_inputs = [output_buf, decimation_target, texture_size, scale_mode, target_cm, rot_x, rot_y, rot_z,
                     keep_parts, origin_mode, origin_point, origin_dx, origin_dy, origin_dz, *expert_export, glb_cache]
    live_outputs = [glb_cache, live_glb_url, glb_output, download_btn, dims_md, keep_parts]

    view_mode.input(
        prepare_live_glb,
        inputs=[view_mode, *export_inputs],
        outputs=live_outputs,
        js=TOGGLE_VIEW_JS,
    ).then(
        # Paired with a no-op backend fn: js-only events are not reliably chained in Gradio 6
        lambda url: gr.skip(), inputs=[live_glb_url], outputs=[live_glb_url], js=LOAD_GLB_JS,
    )

    # Editing parts/size/rotation/origin re-exports an already exported GLB (fast) and refreshes the live view
    for trigger in [keep_parts.input, scale_mode.input, origin_mode.input,
                    rot_x.submit, rot_x.blur, rot_y.submit, rot_y.blur, rot_z.submit, rot_z.blur,
                    origin_dx.submit, origin_dx.blur, origin_dy.submit, origin_dy.blur,
                    origin_dz.submit, origin_dz.blur, target_cm.submit, target_cm.blur,
                    # Expert export options (remesh / UV changes rebuild the raw GLB, ~30 s)
                    exp_remesh_project.release, exp_cone_deg.release, exp_refine_iters.release,
                    exp_global_iters.release, exp_smooth.release, exp_alpha_mode.input]:
        trigger(
            refresh_transformed_glb,
            inputs=[view_mode, *export_inputs],
            outputs=live_outputs,
        ).then(
            lambda url: gr.skip(), inputs=[live_glb_url], outputs=[live_glb_url], js=LOAD_GLB_JS,
        )

    expert_reset_btn.click(
        lambda: (1, 0.6, 1.0, 0.6, 1.0, 0.6, 0.9, 0.0, 90, 0, 1, 1, list(ALPHA_MODES)[0]),
        outputs=[num_candidates, ss_interval_start, ss_interval_end, shape_interval_start, shape_interval_end,
                 tex_interval_start, tex_interval_end, *expert_export],
    ).then(
        refresh_transformed_glb,
        inputs=[view_mode, *export_inputs],
        outputs=live_outputs,
    ).then(
        lambda url: gr.skip(), inputs=[live_glb_url], outputs=[live_glb_url], js=LOAD_GLB_JS,
    )

    origin_offset_reset.click(
        lambda: (0, 0, 0), outputs=[origin_dx, origin_dy, origin_dz],
    ).then(
        refresh_transformed_glb,
        inputs=[view_mode, *export_inputs],
        outputs=live_outputs,
    ).then(
        lambda url: gr.skip(), inputs=[live_glb_url], outputs=[live_glb_url], js=LOAD_GLB_JS,
    )

    # The viewer clicks the hidden button after a surface pick; js swaps in the picked point
    origin_pick_btn.click(
        set_origin_from_pick,
        inputs=[origin_pick_box, view_mode, *export_inputs],
        outputs=[*live_outputs, origin_mode, origin_point, origin_dx, origin_dy, origin_dz],
        js=PICK_ORIGIN_JS,
    ).then(
        lambda url: gr.skip(), inputs=[live_glb_url], outputs=[live_glb_url], js=LOAD_GLB_JS,
    )

    extract_btn.click(
        lambda: gr.Walkthrough(selected=1), outputs=walkthrough
    ).then(
        extract_glb,
        inputs=export_inputs,
        outputs=[glb_output, download_btn, glb_cache, dims_md, keep_parts],
    )

    # --- GLB 壓縮 tab ---
    shrink_ingest_outputs = [shrink_src, shrink_hero, shrink_view, shrink_meta, shrink_download]
    load_shrink_view = dict(
        fn=lambda v: gr.skip(), inputs=[shrink_view], outputs=[shrink_view], js=SHRINK_VIEW_JS,
    )

    shrink_file.upload(shrink_upload, inputs=[shrink_file], outputs=shrink_ingest_outputs).then(**load_shrink_view)
    shrink_file.clear(shrink_upload, inputs=[shrink_file], outputs=shrink_ingest_outputs).then(**load_shrink_view)

    # Preset cards <-> fine-tune slider, as in glb-shrink
    shrink_preset.input(
        lambda p: (SHRINK_PRESETS[p], shrink_hint(SHRINK_PRESETS[p])),
        inputs=[shrink_preset], outputs=[shrink_quality, shrink_hint_md],
    )
    shrink_quality.input(
        lambda q: (next((k for k, v in SHRINK_PRESETS.items() if v == int(q)), None), shrink_hint(q)),
        inputs=[shrink_quality], outputs=[shrink_preset, shrink_hint_md],
    )

    shrink_btn.click(
        shrink_compress, inputs=[shrink_src, shrink_quality], outputs=[shrink_hero, shrink_view, shrink_download],
    ).then(**load_shrink_view)

    # Send the edited model from 圖片轉 3D straight into GLB 壓縮
    for btn in [to_shrink_btn, to_shrink_btn2]:
        btn.click(
            send_to_shrink,
            inputs=export_inputs,
            outputs=[glb_cache, tabs, shrink_file, *shrink_ingest_outputs],
        ).then(**load_shrink_view)
        

# Launch the Gradio app
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    # Construct ui components
    btn_img_base64_strs = {}
    for i in range(len(MODES)):
        icon = Image.open(MODES[i]['icon'])
        MODES[i]['icon_base64'] = image_to_base64(icon)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.low_vram = LOW_VRAM
    pipeline.cuda()
    release_memory()  # drop the CPU copies of weights now resident on the GPU
    
    envmap = {
        'forest': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'sunset': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'courtyard': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
    }
    
    demo.queue(default_concurrency_limit=1)  # one GPU job at a time across all LAN users
    demo.launch(
        css=css, head=head,
        server_name=SERVER_NAME, server_port=SERVER_PORT,
        allowed_paths=[TMP_DIR, THREE_DIR],
        max_file_size="200mb",  # glb-shrink's upload limit
    )
