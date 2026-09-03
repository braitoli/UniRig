/**
 * 🎨 UniRig 3D Statue Studio - Interactive WebGL Frontend
 * Features:
 * - 3D Statue Viewer & Studio Lighting
 * - Interactive Live Painting (Bucket Fill, Brush, Eraser, Finish Selector)
 * - Real-time Pipeline Progress Tracking
 * - Multi-Format GLB & ZIP Downloads
 * - Direct Client-side Painted GLB Export
 * - Full Automation & Webhook Hub Management
 */

// Global State
const state = {
    selectedFile: null,
    activeJobId: null,
    currentJobData: null,
    activeMode: 'painted', // 'painted', 'plaster', 'segmented', 'textured', 'wireframe'
    currentGlbUrl: null,
    paintTool: 'bucket',   // 'bucket', 'brush', 'eraser'
    currentColor: '#FF5722',
    currentFinish: 'ceramic_glossy',
    isPainting: false,
    originalMaterialsMap: new Map(),
    paintedMaterialsMap: new Map(),
    submeshPartsMap: new Map(),
    animMixer: null,
    animClips: [],
    activeAction: null,
    isAnimPlaying: false,
    automationConfig: null,
    modelRotation: { rx: 0, ry: 0, rz: 0 }
};

// 24 Standard Vibrant Statue Painting Colors
const STATUE_COLORS = [
    '#FFE0BD', '#795548', '#4FC3F7', '#81C784', '#FFB74D', '#BA68C8',
    '#FFD54F', '#90A4AE', '#E57373', '#4DD0E1', '#F44336', '#E91E63',
    '#9C27B0', '#673AB7', '#3F51B5', '#2196F3', '#00BCD4', '#009688',
    '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FF9800', '#FFFFFF'
];

// Rich Sample Presets for Instant 2D Image Selection & Instant 3D Painting Preview (100% Matched 2D & 3D)
const SAMPLE_PRESETS = {
    mythical_beast: {
        name: 'Linh Thú Chibi',
        image: '/static/sample_presets/mythical_beast.png',
        filename: 'mythical_beast.png',
        model: '/static/sample_presets/models/mythical_beast/statue_segmented.glb',
        parts: [
            { name: "Đầu & Tai Linh Thú", hex_color: "#4FC3F7", face_count: 1829 },
            { name: "Bờm Tóc & Hoa Văn", hex_color: "#2196F3", face_count: 5534 },
            { name: "Thân & Giáp Ngực", hex_color: "#FFE0BD", face_count: 11342 },
            { name: "Đuôi Lửa Thần", hex_color: "#F44336", face_count: 1434 },
            { name: "Tay Chân & Móng Vuốt", hex_color: "#FF9800", face_count: 4861 }
        ],
        vertices: 12502,
        faces: 25000
    },
    cyber_turtle: {
        name: 'Rùa Máy Cyber',
        image: '/static/sample_presets/cyber_turtle.png',
        filename: 'cyber_turtle.png',
        model: '/static/sample_presets/models/cyber_turtle/statue_segmented.glb',
        parts: [
            { name: "Đầu Rùa & Mắt Robot", hex_color: "#00BCD4", face_count: 1269 },
            { name: "Cổ & Khớp Máy", hex_color: "#90A4AE", face_count: 3314 },
            { name: "Mai Rùa Bọc Giáp", hex_color: "#2196F3", face_count: 4884 },
            { name: "Khung Thân Dưới", hex_color: "#FFD54F", face_count: 12104 },
            { name: "Chân Máy Trước", hex_color: "#4CAF50", face_count: 2094 },
            { name: "Chân Máy Sau", hex_color: "#8BC34A", face_count: 4727 },
            { name: "Đế Tượng Tròn", hex_color: "#795548", face_count: 2562 }
        ],
        vertices: 15480,
        faces: 30954
    },
    mushroom_house: {
        name: 'Nhà Nấm Cổ Tích',
        image: '/static/sample_presets/mushroom_house.png',
        filename: 'mushroom_house.png',
        model: '/static/sample_presets/models/mushroom_house/statue_segmented.glb',
        parts: [
            { name: "Chóp Nấm Cao", hex_color: "#E91E63", face_count: 1256 },
            { name: "Mái Vòm Nấm Lớn", hex_color: "#FF5722", face_count: 5225 },
            { name: "Thân Nhà Cửa Sổ", hex_color: "#FFE0BD", face_count: 8071 },
            { name: "Cửa Ra Vào & Lò Sưởi", hex_color: "#FFEB3B", face_count: 6944 },
            { name: "Vành Nấm Phụ", hex_color: "#9C27B0", face_count: 3459 },
            { name: "Bệ Đất Rêu Phong", hex_color: "#81C784", face_count: 44 }
        ],
        vertices: 12500,
        faces: 24999
    },
    fox_girl: {
        name: 'Cô Bé Cáo Anime',
        image: '/static/sample_presets/fox_girl.jpeg',
        filename: 'fox_girl.jpeg',
        model: '/static/sample_presets/models/fox_girl/statue_segmented.glb',
        parts: [
            { name: "Khuôn Mặt & Nụ Cười", hex_color: "#FFE0BD", face_count: 1960 },
            { name: "Tai Cáo & Tóc Cam", hex_color: "#FF9800", face_count: 4719 },
            { name: "Mũ & Khăn Choàng", hex_color: "#FFB74D", face_count: 5428 },
            { name: "Áo Khoác & Cáo Con", hex_color: "#E57373", face_count: 8287 },
            { name: "Tay Áo", hex_color: "#4FC3F7", face_count: 2326 },
            { name: "Thân Dưới & Quần", hex_color: "#3F51B5", face_count: 1105 },
            { name: "Bệ Đứng", hex_color: "#90A4AE", face_count: 1220 }
        ],
        vertices: 12524,
        faces: 25045
    },
    gentleman: {
        name: 'Quý Ông Chibi 3D',
        image: '/static/sample_presets/gentleman.jpeg',
        filename: 'gentleman.jpeg',
        model: '/static/sample_presets/models/gentleman/statue_segmented.glb',
        parts: [
            { name: "Khuôn Mặt & Râu", hex_color: "#FFE0BD", face_count: 976 },
            { name: "Mái Tóc Nâu", hex_color: "#795548", face_count: 2306 },
            { name: "Áo Da Cổ Lông & Sơ Mi", hex_color: "#8D6E63", face_count: 13669 },
            { name: "Thân Dưới & Thắt Lưng", hex_color: "#5D4037", face_count: 2191 },
            { name: "Cánh Tay Áo Da", hex_color: "#A1887F", face_count: 5858 }
        ],
        vertices: 12502,
        faces: 25000
    }
};

// Three.js Components
let scene, camera, renderer, controls, currentModel, gridHelper, dirLight;
let raycaster, mouse;

document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initUIEvents();
    initPalette();
    loadAutomationStatus();
    loadStatueHistory();
    setInterval(loadAutomationStatus, 4000);
});

/* ===================================================
   1. Three.js Viewport & Lighting Setup
   =================================================== */
function initThreeJS() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e121a);

    // Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.05, 100);
    camera.position.set(0, 1.2, 3.2);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Orbit Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.05; // Don't go below floor
    controls.minDistance = 0.5;
    controls.maxDistance = 15;
    controls.target.set(0, 0.8, 0);

    // Studio Lighting - Multi-point illumination for bright plaster & vibrant paint
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.1);
    scene.add(ambientLight);

    dirLight = new THREE.DirectionalLight(0xfffaed, 1.4);
    dirLight.position.set(3, 6, 4);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 15;
    dirLight.shadow.bias = -0.0005;
    const d = 2.5;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d;
    dirLight.shadow.camera.bottom = -d;
    scene.add(dirLight);

    const fillLight = new THREE.DirectionalLight(0xdbeafe, 0.8);
    fillLight.position.set(-4, 4, -2);
    scene.add(fillLight);

    const frontLight = new THREE.DirectionalLight(0xffffff, 0.6);
    frontLight.position.set(0, 2, 5);
    scene.add(frontLight);

    const rimLight = new THREE.DirectionalLight(0xfff0f5, 0.7);
    rimLight.position.set(0, 5, -4);
    scene.add(rimLight);

    // Ground Grid & Shadow Receiver Plane
    const floorGeo = new THREE.PlaneGeometry(20, 20);
    const floorMat = new THREE.ShadowMaterial({ opacity: 0.35 });
    const floorPlane = new THREE.Mesh(floorGeo, floorMat);
    floorPlane.rotation.x = -Math.PI / 2;
    floorPlane.position.y = 0;
    floorPlane.receiveShadow = true;
    scene.add(floorPlane);

    gridHelper = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
    gridHelper.position.y = 0.001;
    scene.add(gridHelper);

    // Raycaster for Painting Click Detection
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer.domElement.addEventListener('pointerdown', onCanvasPointerDown);
    renderer.domElement.addEventListener('pointermove', onCanvasPointerMove);
    renderer.domElement.addEventListener('pointerleave', clearHoverHighlight);
    window.addEventListener('pointerup', onCanvasPointerUp);
    window.addEventListener('pointercancel', onCanvasPointerUp);

    // Window Resize
    window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    // Animation Render Loop
    const clock = new THREE.Clock();
    function animate() {
        requestAnimationFrame(animate);
        const delta = clock.getDelta();
        if (state.animMixer) {
            state.animMixer.update(delta);
        }
        controls.update();
        renderer.render(scene, camera);
    }
    animate();
}

/* ===================================================
   2. Palette & Tool Bar Initialization
   =================================================== */
function initPalette() {
    const paletteContainer = document.getElementById('palette-colors');
    paletteContainer.innerHTML = '';

    STATUE_COLORS.forEach((colorHex, idx) => {
        const swatch = document.createElement('div');
        swatch.className = 'color-swatch' + (idx === 0 ? ' active' : '');
        swatch.style.backgroundColor = colorHex;
        swatch.dataset.color = colorHex;
        swatch.addEventListener('click', () => {
            document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
            swatch.classList.add('active');
            state.currentColor = colorHex;
            document.getElementById('custom-color-input').value = colorHex;
            if (state.activeMode !== 'painted') {
                updateViewMode('painted', true);
            }
        });
        paletteContainer.appendChild(swatch);
    });

    // Custom Color Input
    const customPicker = document.getElementById('custom-color-input');
    customPicker.addEventListener('input', (e) => {
        state.currentColor = e.target.value;
        document.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('active'));
        if (state.activeMode !== 'painted') {
            updateViewMode('painted', true);
        }
    });

    // Finish Material Preset
    const finishSelect = document.getElementById('material-finish-select');
    finishSelect.addEventListener('change', (e) => {
        state.currentFinish = e.target.value;
        applyCurrentFinishToPaintedModel();
    });
}

let isPointerDraggingPaint = false;

function onCanvasPointerDown(event) {
    if (!currentModel) return;
    if (event.button !== 0) return; // Only left click

    // If currently in another view mode (e.g. textured/segmented/plaster), auto switch to painted mode
    if (state.activeMode !== 'painted') {
        updateViewMode('painted', true);
    }

    const hitMesh = getIntersectedMesh(event);
    if (hitMesh) {
        applyPaintAction(hitMesh);
        if (state.paintTool === 'brush' || state.paintTool === 'eraser') {
            isPointerDraggingPaint = true;
            if (controls) controls.enabled = false; // Disable orbit camera during brush drag
        }
    }
}

let hoveredMesh = null;
let originalHoverEmissive = null;

function onCanvasPointerMove(event) {
    if (!currentModel) return;
    if (isPointerDraggingPaint && (state.paintTool === 'brush' || state.paintTool === 'eraser')) {
        const hitMesh = getIntersectedMesh(event);
        if (hitMesh) {
            applyPaintAction(hitMesh);
        }
    } else {
        onCanvasPointerHover(event);
    }
}

function onCanvasPointerHover(event) {
    if (!currentModel || !renderer) return;
    const hitMesh = getIntersectedMesh(event);
    const tooltip = document.getElementById('part-hover-tooltip');

    if (hitMesh && hitMesh !== hoveredMesh) {
        clearHoverHighlight();

        hoveredMesh = hitMesh;
        if (hoveredMesh.material) {
            originalHoverEmissive = hoveredMesh.material.emissive ? hoveredMesh.material.emissive.clone() : new THREE.Color(0x000000);
            hoveredMesh.material.emissive = new THREE.Color(0x38bdf8);
            hoveredMesh.material.emissiveIntensity = 0.4;
        }

        if (tooltip) {
            const rect = renderer.domElement.getBoundingClientRect();
            tooltip.style.display = 'block';
            tooltip.style.left = `${event.clientX - rect.left}px`;
            tooltip.style.top = `${event.clientY - rect.top}px`;
            const partName = hoveredMesh.name || 'Phân vùng chi tiết';
            const faceCount = hoveredMesh.geometry?.index ? hoveredMesh.geometry.index.count / 3 : (hoveredMesh.geometry?.attributes?.position?.count / 3 || 0);
            tooltip.innerHTML = `🎯 <strong>${partName}</strong> <span style="opacity:0.75;">(${Math.round(faceCount).toLocaleString()} mặt)</span>`;
        }

        highlightSidebarPart(hoveredMesh.name);
    } else if (!hitMesh && hoveredMesh) {
        clearHoverHighlight();
    } else if (hitMesh && tooltip) {
        const rect = renderer.domElement.getBoundingClientRect();
        tooltip.style.left = `${event.clientX - rect.left}px`;
        tooltip.style.top = `${event.clientY - rect.top}px`;
    }
}

function clearHoverHighlight() {
    if (hoveredMesh && hoveredMesh.material && originalHoverEmissive) {
        hoveredMesh.material.emissive.copy(originalHoverEmissive);
        hoveredMesh.material.emissiveIntensity = 0.0;
    }
    hoveredMesh = null;
    originalHoverEmissive = null;
    const tooltip = document.getElementById('part-hover-tooltip');
    if (tooltip) tooltip.style.display = 'none';
    document.querySelectorAll('.part-item').forEach(i => i.classList.remove('active-hover'));
}

function highlightSidebarPart(name) {
    document.querySelectorAll('.part-item').forEach(item => {
        const isMatch = name && item.innerText.includes(name);
        item.classList.toggle('active-hover', isMatch);
    });
}

function onCanvasPointerUp() {
    if (isPointerDraggingPaint) {
        isPointerDraggingPaint = false;
        if (controls) controls.enabled = true; // Re-enable orbit camera
    }
}

function getIntersectedMesh(event) {
    if (!currentModel || !renderer) return null;
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(currentModel.children, true);
    if (intersects.length > 0) {
        return intersects[0].object;
    }
    return null;
}

function applyPaintAction(mesh) {
    if (!mesh || !mesh.isMesh) return;
    if (state.paintTool === 'bucket' || state.paintTool === 'brush') {
        paintSubmesh(mesh, state.currentColor);
    } else if (state.paintTool === 'eraser') {
        resetSubmeshToPlaster(mesh);
    }
}

function getPBRMaterialForColor(hexColor, finishType) {
    const color = new THREE.Color(hexColor);
    let roughness = 0.88;
    let metalness = 0.02;

    if (finishType === 'ceramic_glossy') {
        roughness = 0.18;
        metalness = 0.08;
    } else if (finishType === 'clay') {
        roughness = 0.75;
        metalness = 0.0;
    } else if (finishType === 'metallic_gold') {
        roughness = 0.25;
        metalness = 0.95;
    }

    return new THREE.MeshStandardMaterial({
        color: color,
        roughness: roughness,
        metalness: metalness,
        side: THREE.DoubleSide
    });
}

function paintSubmesh(mesh, hexColor) {
    if (!mesh || !mesh.isMesh) return;
    const newMat = getPBRMaterialForColor(hexColor, state.currentFinish);
    mesh.material = newMat;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    state.paintedMaterialsMap.set(mesh.uuid, newMat);

    // Trigger subtle bounce feedback on painted part
    const origScale = mesh.scale.clone();
    mesh.scale.multiplyScalar(1.02);
    setTimeout(() => { mesh.scale.copy(origScale); }, 100);
}

function resetSubmeshToPlaster(mesh) {
    if (!mesh || !mesh.isMesh) return;
    const plasterMat = getPBRMaterialForColor('#F0EFEB', 'plaster');
    mesh.material = plasterMat;
    state.paintedMaterialsMap.set(mesh.uuid, plasterMat);
}

function resetAllPaintedColors() {
    if (!currentModel) return;
    currentModel.traverse((child) => {
        if (child.isMesh) {
            resetSubmeshToPlaster(child);
        }
    });
}

function applyCurrentFinishToPaintedModel() {
    if (!currentModel) return;
    currentModel.traverse((child) => {
        if (child.isMesh && child.material && child.material.color) {
            const hex = '#' + child.material.color.getHexString();
            child.material = getPBRMaterialForColor(hex, state.currentFinish);
        }
    });
}

function load3DStatueModel(glbUrl, mode = 'painted', statueName = '') {
    showCanvasLoader('Đang tải tượng 3D...');
    state.currentGlbUrl = glbUrl;
    if (statueName) state.currentStatueName = statueName;
    else if (state.currentPresetKey && SAMPLE_PRESETS[state.currentPresetKey]) {
        state.currentStatueName = SAMPLE_PRESETS[state.currentPresetKey].name;
    }
    syncModelTo3DPaintingIframe();

    const loader = new THREE.GLTFLoader();
    loader.load(glbUrl, (gltf) => {
        if (currentModel) {
            scene.remove(currentModel);
        }

        currentModel = gltf.scene;
        state.originalMaterialsMap.clear();
        state.paintedMaterialsMap.clear();

        // Enable shadows, compute normals, and store materials
        currentModel.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
                if (child.geometry) {
                    if (!child.geometry.attributes.normal) {
                        child.geometry.computeVertexNormals();
                    }
                    child.geometry.normalizeNormals();
                }
                if (child.material) {
                    child.material.side = THREE.DoubleSide;
                    state.originalMaterialsMap.set(child.uuid, child.material.clone());
                    // In painted mode, if no user painting exists yet, initialize with pure plaster white
                    if (!child.material.map && mode === 'painted') {
                        const plasterMat = getPBRMaterialForColor('#F0EFEB', state.currentFinish);
                        child.material = plasterMat;
                        state.paintedMaterialsMap.set(child.uuid, plasterMat);
                    }
                }
            }
        });

        // Frame model to center
        const bbox = new THREE.Box3().setFromObject(currentModel);
        const center = bbox.getCenter(new THREE.Vector3());
        const size = bbox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        currentModel.position.set(-center.x, -bbox.min.y, -center.z);
        scene.add(currentModel);

        // Adjust camera
        camera.position.set(0, size.y * 0.8, maxDim * 2.2);
        controls.target.set(0, size.y * 0.5, 0);
        controls.update();

        // Setup Animations if present
        if (gltf.animations && gltf.animations.length > 0) {
            setupAnimations(gltf);
        } else {
            document.getElementById('animation-bar').style.display = 'none';
        }

        document.getElementById('empty-viewport-hint').style.display = 'none';
        document.getElementById('painting-bar').style.display = 'flex';

        try {
            updateViewMode(mode, false);
        } catch (e) {
            console.error('Error in updateViewMode:', e);
        } finally {
            hideCanvasLoader();
        }
    }, (progress) => {
        if (progress.total > 0) {
            const pct = Math.round((progress.loaded / progress.total) * 100);
            document.getElementById('canvas-loader-text').innerText = `Đang tải: ${pct}%`;
        }
    }, (error) => {
        hideCanvasLoader();
        console.error('Error loading GLB:', error);
        alert('Không thể tải file GLB: ' + error);
    });
}

function updateViewMode(mode, reloadModel = false) {
    state.activeMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    if (!currentModel) return;

    // If returning from textured mode to painted/segmented/plaster, reload segmented model if needed
    if ((mode === 'painted' || mode === 'segmented' || mode === 'plaster') && reloadModel) {
        let segUrl = null;
        if (state.activeJobId) {
            segUrl = `/api/statue/jobs/${state.activeJobId}/files/segmented_glb`;
        } else if (state.currentPresetKey && SAMPLE_PRESETS[state.currentPresetKey]) {
            segUrl = SAMPLE_PRESETS[state.currentPresetKey].model;
        }
        if (segUrl && state.currentGlbUrl && state.currentGlbUrl.includes('_textured.glb') && state.currentGlbUrl !== segUrl) {
            load3DStatueModel(segUrl, mode);
            return;
        }
    }

    if (mode === 'painted') {
        currentModel.traverse((child) => {
            if (child.isMesh) {
                if (child.material) child.material.wireframe = false;
                const painted = state.paintedMaterialsMap.get(child.uuid);
                if (painted) child.material = painted;
                else child.material = getPBRMaterialForColor('#F0EFEB', state.currentFinish);
            }
        });
    } else if (mode === 'plaster') {
        currentModel.traverse((child) => {
            if (child.isMesh) {
                if (child.material) child.material.wireframe = false;
                child.material = getPBRMaterialForColor('#F0EFEB', 'plaster');
            }
        });
    } else if (mode === 'segmented') {
        currentModel.traverse((child) => {
            if (child.isMesh) {
                if (child.material) child.material.wireframe = false;
                const orig = state.originalMaterialsMap.get(child.uuid);
                if (orig) {
                    child.material = orig.clone();
                    child.material.side = THREE.DoubleSide;
                }
            }
        });
    } else if (mode === 'textured') {
        if (reloadModel) {
            let texUrl = null;
            if (state.activeJobId) {
                texUrl = `/api/statue/jobs/${state.activeJobId}/files/textured_glb`;
            } else if (state.currentPresetKey && SAMPLE_PRESETS[state.currentPresetKey]) {
                texUrl = SAMPLE_PRESETS[state.currentPresetKey].model.replace('statue_segmented.glb', 'statue_textured.glb');
            }
            if (texUrl && state.currentGlbUrl !== texUrl) {
                load3DStatueModel(texUrl, 'textured');
                return;
            }
        }
        currentModel.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.wireframe = false;
            }
        });
    } else if (mode === 'wireframe') {
        currentModel.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.wireframe = true;
            }
        });
    }
}

function setupAnimations(gltf) {
    state.animMixer = new THREE.AnimationMixer(currentModel);
    state.animClips = gltf.animations;

    const animSelect = document.getElementById('anim-select');
    animSelect.innerHTML = '';

    state.animClips.forEach((clip, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = clip.name || `Animation ${idx + 1}`;
        animSelect.appendChild(opt);
    });

    document.getElementById('animation-bar').style.display = 'block';
    playAnimation(0);
}

function playAnimation(index) {
    if (!state.animMixer || !state.animClips[index]) return;
    if (state.activeAction) state.activeAction.stop();

    state.activeAction = state.animMixer.clipAction(state.animClips[index]);
    state.activeAction.play();
    state.isAnimPlaying = true;
    document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-pause"></i>';
}

/* ===================================================
   5. Export Client-Side Painted GLB
   =================================================== */
function exportPaintedGLB() {
    if (!currentModel) {
        alert('Chưa có tượng 3D để xuất!');
        return;
    }
    const exporter = new THREE.GLTFExporter();
    exporter.parse(currentModel, (glbBuffer) => {
        const blob = new Blob([glbBuffer], { type: 'model/gltf-binary' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `statue_custom_painted_${Date.now()}.glb`;
        link.click();
    }, { binary: true });
}

/* ===================================================
   6. UI Event Listeners & Job Handling
   =================================================== */
function initUIEvents() {
    // Drop Zone Upload
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const dropContent = document.getElementById('drop-content');
    const btnRemove = document.getElementById('btn-remove-preview');

    dropZone.addEventListener('click', (e) => {
        if (e.target !== btnRemove && !btnRemove.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelected(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        state.selectedFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        dropContent.style.display = 'block';
    });

    // Preset Chips - Fast 2D Image Selection & Instant 3D Painting Demo
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', async () => {
            const presetKey = chip.dataset.preset;
            const preset = SAMPLE_PRESETS[presetKey];
            if (!preset) return;

            // 1. Highlight active chip
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');

            try {
                // 2. Fetch the 2D sample image and populate upload preview
                const res = await fetch(preset.image);
                if (res.ok) {
                    const blob = await res.blob();
                    const file = new File([blob], preset.filename, { type: blob.type || 'image/png' });
                    handleFileSelected(file);
                }
            } catch (err) {
                console.warn('Could not load preset 2D image:', err);
            }

            state.currentPresetKey = presetKey;
            state.activeJobId = null;

            // 3. If preset has ready-made 3D statue model, load into Three.js immediately!
            if (preset.model) {
                load3DStatueModel(preset.model, 'painted', preset.name);

                // Update Stats Panel
                document.getElementById('stat-vertices').innerText = (preset.vertices || 0).toLocaleString();
                document.getElementById('stat-faces').innerText = (preset.faces || 0).toLocaleString();
                document.getElementById('stat-parts').innerText = (preset.parts ? preset.parts.length : 6);
                document.getElementById('stat-duration').innerText = 'Sẵn sàng';

                // Render Parts List in right sidebar
                renderPartsList(preset.parts || []);

                // Enable direct downloads for preset model
                document.getElementById('dl-plaster').onclick = () => window.open(preset.model.replace('statue_segmented.glb', 'statue_plaster.glb'), '_blank');
                document.getElementById('dl-segmented').onclick = () => window.open(preset.model, '_blank');
                document.getElementById('dl-textured').onclick = () => window.open(preset.model.replace('statue_segmented.glb', 'statue_textured.glb'), '_blank');
                document.getElementById('dl-shell').onclick = () => window.open(preset.model.replace('statue_segmented.glb', 'statue_shell.glb'), '_blank');
                document.getElementById('dl-shell-optimized').onclick = () => window.open(preset.model.replace('statue_segmented.glb', 'statue_shell_optimized.glb'), '_blank');
            }
        });
    });

    // Mesh Detail Segment Control
    document.querySelectorAll('#mesh-detail-control .segment-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#mesh-detail-control .segment-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Generator Radio Cards
    document.querySelectorAll('.radio-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.radio-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            card.querySelector('input').checked = true;
        });
    });

    // Generate Statue Action Button
    document.getElementById('btn-generate-statue').addEventListener('click', startStatueGeneration);

    // View Modes
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => updateViewMode(btn.dataset.mode, true));
    });

    // Painting Tools
    document.getElementById('tool-bucket').addEventListener('click', () => setPaintTool('bucket'));
    document.getElementById('tool-brush').addEventListener('click', () => setPaintTool('brush'));
    document.getElementById('tool-eraser').addEventListener('click', () => setPaintTool('eraser'));
    document.getElementById('btn-reset-paint').addEventListener('click', resetAllPaintedColors);
    document.getElementById('dl-painted').addEventListener('click', (e) => {
        e.preventDefault();
        exportPaintedGLB();
    });

    // Viewport Helper Buttons
    document.getElementById('btn-reset-camera').addEventListener('click', () => {
        if (currentModel) {
            const bbox = new THREE.Box3().setFromObject(currentModel);
            const size = bbox.getSize(new THREE.Vector3());
            camera.position.set(0, size.y * 0.8, Math.max(size.x, size.y, size.z) * 2.2);
            controls.target.set(0, size.y * 0.5, 0);
            controls.update();
        }
    });

    document.getElementById('btn-toggle-grid').addEventListener('click', () => {
        gridHelper.visible = !gridHelper.visible;
    });

    document.getElementById('btn-screenshot').addEventListener('click', () => {
        renderer.render(scene, camera);
        const imgData = renderer.domElement.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `statue_preview_${Date.now()}.png`;
        link.href = imgData;
        link.click();
    });

    // Animation Controls
    document.getElementById('btn-play-pause').addEventListener('click', () => {
        if (!state.activeAction) return;
        if (state.isAnimPlaying) {
            state.activeAction.paused = true;
            state.isAnimPlaying = false;
            document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-play"></i>';
        } else {
            state.activeAction.paused = false;
            state.isAnimPlaying = true;
            document.getElementById('btn-play-pause').innerHTML = '<i class="fa-solid fa-pause"></i>';
        }
    });

    document.getElementById('anim-select').addEventListener('change', (e) => {
        playAnimation(parseInt(e.target.value));
    });

    // 3DPainting Studio Engine Switcher
    document.getElementById('btn-engine-preview').addEventListener('click', () => {
        setStudioEngine('preview');
    });
    document.getElementById('btn-engine-3dpainting').addEventListener('click', () => {
        setStudioEngine('3dpainting');
    });
    document.getElementById('btn-open-3dpainting-tab').addEventListener('click', () => {
        const url = state.currentGlbUrl || '/static/sample_presets/models/cyber_turtle/statue_segmented.glb';
        const name = state.currentStatueName || 'Tượng 3D UniRig';
        window.open(`/painting?model=${encodeURIComponent(url)}&name=${encodeURIComponent(name)}`, '_blank');
    });

    const iframeEl = document.getElementById('iframe-3dpainting');
    if (iframeEl) {
        iframeEl.addEventListener('load', () => {
            const url = state.currentGlbUrl || '/static/sample_presets/models/cyber_turtle/statue_segmented.glb';
            const name = state.currentStatueName || 'Tượng 3D UniRig';
            try {
                iframeEl.contentWindow?.postMessage({ type: 'LOAD_3D_MODEL', url, name }, '*');
            } catch (e) {}
        });
    }

    // Automation Modal
    document.getElementById('btn-open-auto-modal').addEventListener('click', () => {
        openAutomationModal();
    });
    document.getElementById('btn-close-auto-modal').addEventListener('click', () => {
        document.getElementById('automation-modal').style.display = 'none';
    });
    document.getElementById('btn-save-auto-config').addEventListener('click', saveAutomationConfig);
    document.getElementById('btn-scan-now').addEventListener('click', triggerManualScan);
    document.getElementById('btn-test-webhook').addEventListener('click', testWebhookPing);
    // Axis Orientation Popover & Rotation Controls
    const btnToggleOrientation = document.getElementById('btn-toggle-orientation-menu');
    const orientationPopover = document.getElementById('orientation-menu-popover');
    if (btnToggleOrientation && orientationPopover) {
        btnToggleOrientation.addEventListener('click', (e) => {
            e.stopPropagation();
            const isShown = orientationPopover.style.display === 'block';
            orientationPopover.style.display = isShown ? 'none' : 'block';
        });
        document.addEventListener('click', (e) => {
            if (!orientationPopover.contains(e.target) && e.target !== btnToggleOrientation) {
                orientationPopover.style.display = 'none';
            }
        });
    }

    document.getElementById('btn-rotate-car-horizontal')?.addEventListener('click', () => {
        rotateCurrentModel(-90, 0, 0);
    });
    document.getElementById('btn-rotate-upright')?.addEventListener('click', () => {
        rotateCurrentModel(90, 0, 0);
    });
    document.getElementById('btn-rotate-x-90')?.addEventListener('click', () => {
        rotateCurrentModel(90, 0, 0);
    });
    document.getElementById('btn-rotate-y-90')?.addEventListener('click', () => {
        rotateCurrentModel(0, 90, 0);
    });
    document.getElementById('btn-rotate-z-90')?.addEventListener('click', () => {
        rotateCurrentModel(0, 0, 90);
    });
    document.getElementById('btn-reset-rotation')?.addEventListener('click', () => {
        resetModelRotation();
    });
    document.getElementById('btn-save-model-rotation')?.addEventListener('click', () => {
        saveModelRotationToBackend();
    });
}

function setStudioEngine(engine) {
    state.activeEngine = engine;
    const is3D = engine === '3dpainting';
    document.getElementById('btn-engine-preview').classList.toggle('active', !is3D);
    document.getElementById('btn-engine-3dpainting').classList.toggle('active', is3D);

    const canvasContainer = document.getElementById('canvas-container');
    const iframeContainer = document.getElementById('iframe-3dpainting-container');
    const paintingBar = document.getElementById('painting-bar');
    const viewModes = document.getElementById('view-modes-container');
    const viewLabel = document.getElementById('view-mode-label');
    const previewControls = document.getElementById('preview-controls-group');
    const btnOpenTab = document.getElementById('btn-open-3dpainting-tab');

    if (is3D) {
        canvasContainer.style.display = 'none';
        paintingBar.style.display = 'none';
        iframeContainer.style.display = 'flex';
        viewModes.style.display = 'none';
        if (viewLabel) viewLabel.style.display = 'none';
        if (previewControls) previewControls.style.display = 'none';
        if (btnOpenTab) btnOpenTab.style.display = 'inline-flex';
        syncModelTo3DPaintingIframe();
    } else {
        canvasContainer.style.display = 'block';
        paintingBar.style.display = 'flex';
        iframeContainer.style.display = 'none';
        viewModes.style.display = 'flex';
        if (viewLabel) viewLabel.style.display = 'inline-flex';
        if (previewControls) previewControls.style.display = 'flex';
        if (btnOpenTab) btnOpenTab.style.display = 'none';
    }
}

function rotateCurrentModel(rxDeg, ryDeg, rzDeg) {
    if (!currentModel) return;
    state.modelRotation.rx = (state.modelRotation.rx + rxDeg) % 360;
    state.modelRotation.ry = (state.modelRotation.ry + ryDeg) % 360;
    state.modelRotation.rz = (state.modelRotation.rz + rzDeg) % 360;

    // Apply rotation around world axes so rotation is intuitive
    if (rxDeg) currentModel.rotateOnWorldAxis(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(rxDeg));
    if (ryDeg) currentModel.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(ryDeg));
    if (rzDeg) currentModel.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), THREE.MathUtils.degToRad(rzDeg));

    // Re-center horizontally and place base flat at ground Y = 0
    const bbox = new THREE.Box3().setFromObject(currentModel);
    const center = bbox.getCenter(new THREE.Vector3());
    currentModel.position.x -= center.x;
    currentModel.position.z -= center.z;
    currentModel.position.y -= bbox.min.y;

    const size = bbox.getSize(new THREE.Vector3());
    controls.target.set(0, size.y * 0.5, 0);
    controls.update();

    // Send rotation notification to 3DPainting iframe
    try {
        const iframe = document.getElementById('iframe-3dpainting');
        iframe?.contentWindow?.postMessage({
            type: 'ROTATE_MODEL',
            rx: rxDeg,
            ry: ryDeg,
            rz: rzDeg
        }, '*');
    } catch (e) {}
}

function resetModelRotation() {
    state.modelRotation = { rx: 0, ry: 0, rz: 0 };
    if (state.currentGlbUrl) {
        load3DStatueModel(state.currentGlbUrl, state.activeMode, state.currentStatueName);
    }
}

async function saveModelRotationToBackend() {
    if (!state.activeJobId && !state.currentPresetKey) {
        alert("Vui lòng chọn một tác phẩm hoặc preset để lưu hướng trục!");
        return;
    }
    const btnSave = document.getElementById('btn-save-model-rotation');
    const origText = btnSave.innerHTML;
    btnSave.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang lưu...';
    btnSave.disabled = true;

    try {
        const payload = {
            job_id: state.activeJobId || null,
            preset_key: state.activeJobId ? null : state.currentPresetKey,
            rx: state.modelRotation.rx,
            ry: state.modelRotation.ry,
            rz: state.modelRotation.rz,
            re_ground: true
        };
        const res = await fetch('/api/statue/rotate_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success) {
            btnSave.innerHTML = '<i class="fa-solid fa-check"></i> Đã lưu thành công!';
            state.modelRotation = { rx: 0, ry: 0, rz: 0 };
            // Sync updated model to 3DPainting iframe
            syncModelTo3DPaintingIframe(true);
            setTimeout(() => {
                btnSave.innerHTML = origText;
                btnSave.disabled = false;
                const pop = document.getElementById('orientation-menu-popover');
                if (pop) pop.style.display = 'none';
            }, 1200);
        } else {
            alert("Lỗi khi lưu: " + (data.detail || data.message));
            btnSave.innerHTML = origText;
            btnSave.disabled = false;
        }
    } catch (err) {
        console.error('Save rotation error:', err);
        alert("Không thể kết nối đến máy chủ để lưu trục!");
        btnSave.innerHTML = origText;
        btnSave.disabled = false;
    }
}

function syncModelTo3DPaintingIframe(forceReload = false) {
    const iframe = document.getElementById('iframe-3dpainting');
    if (!iframe) return;
    const url = state.currentGlbUrl || '/static/sample_presets/models/cyber_turtle/statue_segmented.glb';
    const name = state.currentStatueName || 'Tượng 3D UniRig';

    const cacheParam = forceReload ? `&_t=${Date.now()}` : '';
    const targetSrc = `/3dpainting/index.html?model=${encodeURIComponent(url)}&name=${encodeURIComponent(name)}&embedded=1${cacheParam}`;
    const currentSrc = iframe.getAttribute('src') || '';
    if (forceReload || !currentSrc || !currentSrc.includes(encodeURIComponent(url))) {
        iframe.src = targetSrc;
    }
    try {
        iframe.contentWindow?.postMessage({ type: 'LOAD_3D_MODEL', url, name }, '*');
    } catch (e) {}
}

function setPaintTool(tool) {
    state.paintTool = tool;
    document.querySelectorAll('.paint-tool-btn').forEach(btn => btn.classList.remove('active'));
    if (tool === 'bucket') {
        document.getElementById('tool-bucket').classList.add('active');
        if (renderer && renderer.domElement) renderer.domElement.style.cursor = 'default';
    }
    if (tool === 'brush') {
        document.getElementById('tool-brush').classList.add('active');
        if (renderer && renderer.domElement) renderer.domElement.style.cursor = 'crosshair';
    }
    if (tool === 'eraser') {
        document.getElementById('tool-eraser').classList.add('active');
        if (renderer && renderer.domElement) renderer.domElement.style.cursor = 'cell';
    }

    // Auto-switch to painted mode if currently in another view mode (e.g. textured/segmented/plaster)
    if (state.activeMode !== 'painted') {
        updateViewMode('painted', true);
    }
}

function handleFileSelected(file) {
    state.selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('drop-content').style.display = 'none';
        document.getElementById('preview-container').style.display = 'flex';
    };
    reader.readAsDataURL(file);
}

/* ===================================================
   7. Statue Generation API Request & Progress Loop
   =================================================== */
async function startStatueGeneration() {
    if (!state.selectedFile) {
        alert('Vui lòng chọn hoặc kéo thả 1 ảnh 2D trước khi bắt đầu tạo tượng!');
        return;
    }

    const generator = document.querySelector('input[name="generator"]:checked').value;
    const meshDetail = document.querySelector('#mesh-detail-control .segment-btn.active').dataset.val;
    const targetFaces = document.getElementById('target-faces-select').value;
    const pedestalShape = document.getElementById('pedestal-shape-select').value;
    const enableRigging = document.getElementById('enable-rigging-checkbox').checked;

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    formData.append('generator', generator);
    formData.append('mesh_detail', meshDetail);
    formData.append('texture_detail', 'high');
    formData.append('target_faces', targetFaces);
    formData.append('pedestal_shape', pedestalShape);
    formData.append('orientation', document.getElementById('statue-orientation-select')?.value || 'auto');
    formData.append('enable_rigging', enableRigging);

    const btn = document.getElementById('btn-generate-statue');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang Xử Lý Pipeline...';

    const progressBox = document.getElementById('progress-box');
    progressBox.style.display = 'block';

    try {
        const resp = await fetch('/api/statue/generate', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error(await resp.text());
        const job = await resp.json();
        state.activeJobId = job.id;

        pollStatueProgress(job.id);
    } catch (err) {
        alert('Lỗi tạo tượng: ' + err.message);
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Bắt Đầu Tạo Tượng 3D';
        progressBox.style.display = 'none';
    }
}

async function pollStatueProgress(jobId) {
    const pollTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/statue/jobs/${jobId}`);
            if (!res.ok) return;
            const job = await res.json();
            const meta = job.metadata || {};
            const prog = meta.progress || { pct: 10, step_name: 'Đang xử lý...' };

            document.getElementById('progress-title').innerText = `Giai đoạn ${prog.step_idx || 1}/${prog.total_steps || 5}`;
            document.getElementById('progress-pct').innerText = `${prog.pct || 0}%`;
            document.getElementById('progress-bar-fill').style.width = `${prog.pct || 0}%`;
            document.getElementById('progress-detail').innerText = prog.step_name || 'Đang xử lý...';

            if (job.status === 'completed') {
                clearInterval(pollTimer);
                onStatueJobCompleted(job);
            } else if (job.status === 'failed') {
                clearInterval(pollTimer);
                alert('Tạo tượng thất bại: ' + (job.error_message || 'Unknown error'));
                resetGenerateButton();
            }
        } catch (e) {
            console.error(e);
        }
    }, 1500);
}

function onStatueJobCompleted(job) {
    resetGenerateButton();
    state.currentJobData = job;

    // Update stats card
    document.getElementById('stat-vertices').innerText = (job.num_vertices || 0).toLocaleString();
    document.getElementById('stat-faces').innerText = (job.num_faces || 0).toLocaleString();
    document.getElementById('stat-parts').innerText = job.num_parts || 0;
    document.getElementById('stat-duration').innerText = `${job.duration_sec || 0}s`;

    // Update download buttons
    document.getElementById('dl-plaster').href = `/api/statue/jobs/${job.id}/files/plaster_glb`;
    document.getElementById('dl-segmented').href = `/api/statue/jobs/${job.id}/files/segmented_glb`;
    document.getElementById('dl-textured').href = `/api/statue/jobs/${job.id}/files/textured_glb`;
    document.getElementById('dl-shell').href = `/api/statue/jobs/${job.id}/files/shell_glb`;
    document.getElementById('dl-shell-optimized').href = `/api/statue/jobs/${job.id}/files/shell_optimized_glb`;
    document.getElementById('dl-package').href = `/api/statue/jobs/${job.id}/files/package_zip`;

    // Update detected parts list
    const parts = (job.metadata && job.metadata.mesh_stats && job.metadata.mesh_stats.parts) || [];
    renderPartsList(parts);

    // Load segmented or plaster model into 3D view
    load3DStatueModel(`/api/statue/jobs/${job.id}/files/segmented_glb`, 'painted');
    loadStatueHistory();
}

function renderPartsList(parts) {
    const partsContainer = document.getElementById('parts-list');
    const partsSection = document.getElementById('parts-section');
    if (!partsContainer) return;
    partsContainer.innerHTML = '';
    if (parts && parts.length > 0) {
        if (partsSection) partsSection.style.display = 'block';
        parts.forEach(p => {
            const item = document.createElement('div');
            item.className = 'part-item';
            item.innerHTML = `
                <span><span class="part-color-indicator" style="background-color: ${p.hex_color};"></span>${p.name}</span>
                <span style="color: var(--text-muted);">${(p.face_count || 0).toLocaleString()} mặt</span>
            `;
            partsContainer.appendChild(item);
        });
    } else if (partsSection) {
        partsSection.style.display = 'none';
    }
}

function resetGenerateButton() {
    const btn = document.getElementById('btn-generate-statue');
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Bắt Đầu Tạo Tượng 3D';
    setTimeout(() => {
        document.getElementById('progress-box').style.display = 'none';
    }, 2000);
}

/* ===================================================
   8. History & Automation Hub
   ================================================== */
async function loadStatueHistory() {
    try {
        const res = await fetch('/api/statue/jobs?limit=20');
        const jobs = await res.json();
        const listContainer = document.getElementById('history-list');
        listContainer.innerHTML = '';

        if (!jobs || jobs.length === 0) {
            listContainer.innerHTML = '<div class="history-empty">Chưa có tượng nào được tạo.</div>';
            return;
        }

        jobs.forEach(job => {
            const item = document.createElement('div');
            item.className = 'history-item' + (job.id === state.activeJobId ? ' active' : '');
            const dateStr = new Date(job.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            item.innerHTML = `
                <span><strong>${job.input_filename}</strong> <small style="color:var(--text-muted);">(${dateStr})</small></span>
                <span class="badge ${job.status === 'completed' ? 'badge-success' : ''}">${job.status}</span>
            `;
            item.addEventListener('click', () => {
                state.activeJobId = job.id;
                onStatueJobCompleted(job);
            });
            listContainer.appendChild(item);
        });
    } catch (e) {
        console.error(e);
    }
}

async function loadAutomationStatus() {
    try {
        const res = await fetch('/api/statue/automation/status');
        const status = await res.json();
        const badge = document.getElementById('auto-status-indicator');
        const text = document.getElementById('auto-status-text');

        if (status.is_running) {
            badge.classList.add('active');
            text.innerText = `Automation: Đang Quét (${status.active_jobs_count || 0} đang chạy)`;
        } else {
            badge.classList.remove('active');
            text.innerText = 'Automation: Tắt';
        }
    } catch (e) {
        console.error(e);
    }
}

async function openAutomationModal() {
    try {
        const res = await fetch('/api/statue/automation/config');
        const cfg = await res.json();
        state.automationConfig = cfg;

        document.getElementById('auto-enabled-checkbox').checked = !!cfg.enabled;
        document.getElementById('auto-input-folder').value = cfg.input_folder || '';
        document.getElementById('auto-output-folder').value = cfg.output_folder || '';
        document.getElementById('auto-poll-interval').value = cfg.poll_interval_sec || 5;
        document.getElementById('auto-webhook-url').value = cfg.webhook_url || '';
        document.getElementById('auto-webhook-secret').value = cfg.webhook_secret || '';

        document.getElementById('webhook-test-result').innerText = '';
        document.getElementById('automation-modal').style.display = 'flex';
    } catch (e) {
        alert('Lỗi tải cấu hình: ' + e);
    }
}

async function saveAutomationConfig() {
    const config = {
        enabled: document.getElementById('auto-enabled-checkbox').checked,
        input_folder: document.getElementById('auto-input-folder').value.trim(),
        output_folder: document.getElementById('auto-output-folder').value.trim(),
        poll_interval_sec: parseInt(document.getElementById('auto-poll-interval').value) || 5,
        webhook_url: document.getElementById('auto-webhook-url').value.trim(),
        webhook_secret: document.getElementById('auto-webhook-secret').value.trim()
    };

    try {
        const res = await fetch('/api/statue/automation/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        if (!res.ok) throw new Error(await res.text());
        alert('Đã lưu cấu hình tự động hóa thành công!');
        document.getElementById('automation-modal').style.display = 'none';
        loadAutomationStatus();
    } catch (e) {
        alert('Lỗi lưu cấu hình: ' + e.message);
    }
}

async function triggerManualScan() {
    const btn = document.getElementById('btn-scan-now');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang quét...';

    try {
        const res = await fetch('/api/statue/automation/scan-now', { method: 'POST' });
        const data = await res.json();
        alert(`Đã tìm thấy và đưa vào xử lý ${data.count} ảnh mới!`);
    } catch (e) {
        alert('Lỗi quét thư mục: ' + e);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Quét Thư Mục Ngay';
    }
}

async function testWebhookPing() {
    const url = document.getElementById('auto-webhook-url').value.trim();
    const secret = document.getElementById('auto-webhook-secret').value.trim();
    const resultSpan = document.getElementById('webhook-test-result');

    if (!url) {
        alert('Vui lòng nhập Webhook URL trước khi test!');
        return;
    }

    resultSpan.innerText = 'Đang ping...';
    resultSpan.style.color = 'var(--text-secondary)';

    try {
        const res = await fetch('/api/statue/automation/test-webhook', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ webhook_url: url, webhook_secret: secret })
        });
        const data = await res.json();

        if (data.status === 'success') {
            resultSpan.innerText = `✅ Thành công (HTTP ${data.status_code}, ${data.response_time_ms}ms)`;
            resultSpan.style.color = 'var(--success)';
        } else {
            resultSpan.innerText = `❌ Thất bại: ${data.error || 'HTTP ' + data.status_code}`;
            resultSpan.style.color = 'var(--accent)';
        }
    } catch (e) {
        resultSpan.innerText = `❌ Lỗi: ${e.message}`;
        resultSpan.style.color = 'var(--accent)';
    }
}

function showCanvasLoader(msg) {
    document.getElementById('canvas-loader-text').innerText = msg;
    document.getElementById('canvas-loader').style.display = 'flex';
}

function hideCanvasLoader() {
    document.getElementById('canvas-loader').style.display = 'none';
}
