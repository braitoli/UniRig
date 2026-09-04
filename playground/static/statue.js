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
    activeMode: 'segmented', // 'plaster', 'segmented', 'textured', 'wireframe'
    currentGlbUrl: null,
    isPainting: false,
    originalMaterialsMap: new Map(),
    submeshPartsMap: new Map(),
    animMixer: null,
    animClips: [],
    activeAction: null,
    isAnimPlaying: false,
    automationConfig: null,
    modelRotation: { rx: 0, ry: 0, rz: 0 }
};

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

// Độ sáng khung nhìn 3D: nhân cường độ TẤT CẢ nguồn sáng với 1 hệ số, không đụng vật liệu model.
const BRIGHTNESS_STORAGE_KEY = 'statue_brightness_factor';
let brightnessLights = []; // [{ light, baseIntensity }] — nạp lại ở initThreeJS()
let brightnessFactor = 1.0;

document.addEventListener('DOMContentLoaded', () => {
    // Mỗi bước bọc try/catch riêng: nếu một hàm ném lỗi (ví dụ tham chiếu tới
    // phần tử DOM đã bị đổi/xóa), các bước còn lại - đặc biệt loadStatueHistory() -
    // vẫn phải chạy, thay vì cả DOMContentLoaded bị chặn đứng giữa chừng và
    // khiến khối "Lịch Sử Tạo Tượng" trông như trống rỗng dù dữ liệu vẫn còn.
    try { initThreeJS(); } catch (e) { console.error('initThreeJS failed:', e); }
    try { initUIEvents(); } catch (e) { console.error('initUIEvents failed:', e); }
    try { setupDownloadPreviewHovers(); } catch (e) { console.error('setupDownloadPreviewHovers failed:', e); }
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

    // Ghi lại cường độ gốc của từng nguồn sáng để thanh trượt "Độ sáng" nhân hệ số lên,
    // không đụng tới vật liệu model.
    brightnessLights = [ambientLight, dirLight, fillLight, frontLight, rimLight]
        .map((light) => ({ light, baseIntensity: light.intensity }));
    applyBrightnessFactor(brightnessFactor);

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

    // Raycaster for hover tooltip / part highlight detection
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    renderer.domElement.addEventListener('pointermove', onCanvasPointerHover);
    renderer.domElement.addEventListener('pointerleave', clearHoverHighlight);

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

function applyBrightnessFactor(factor) {
    brightnessFactor = factor;
    brightnessLights.forEach(({ light, baseIntensity }) => {
        light.intensity = baseIntensity * factor;
    });
}

let hoveredMesh = null;
let originalHoverEmissive = null;

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

function load3DStatueModel(glbUrl, mode = 'segmented', statueName = '', fallbackUrl = null, fallbackMode = null) {
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

        // Áp lại hệ số Độ sáng đang chọn — phòng trường hợp lights bị tạo/gán lại sau này,
        // để kéo xong rồi nạp model mới không bị mất lựa chọn.
        applyBrightnessFactor(brightnessFactor);

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
        // Vi du: job co san textured_glb (HEAD 2xx) nhung file thuc te khong doc duoc
        // (hong, khong dung dinh dang GLB...) -> lui ve ban du phong thay vi vo trang.
        if (fallbackUrl) {
            console.warn(`Không nạp được ${glbUrl} (GLTFLoader lỗi), lùi về bản dự phòng: ${fallbackUrl}`, error);
            load3DStatueModel(fallbackUrl, fallbackMode, statueName);
            return;
        }
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

    // Nếu model đang nạp KHÔNG PHẢI bản segmented_glb (ví dụ đang ở textured_glb — chỉ có
    // 1 mesh node, không có 7 phân vùng), phải nạp lại đúng bản segmented mới có phân vùng
    // để hiện. Trước đây có thêm điều kiện `.includes('_textured.glb')` để chỉ reload khi
    // rời từ textured — nhưng URL textured của job là `/files/textured_glb` (không có đuôi
    // .glb) nên điều kiện đó không bao giờ khớp, làm mode segmented bị kẹt hiện textured.
    // Bỏ điều kiện thừa đó, dùng đúng cách so sánh URL đơn giản như nhánh 'textured' đang làm.
    if ((mode === 'segmented' || mode === 'plaster') && reloadModel) {
        let segUrl = null;
        if (state.activeJobId) {
            segUrl = `/api/statue/jobs/${state.activeJobId}/files/segmented_glb`;
        } else if (state.currentPresetKey && SAMPLE_PRESETS[state.currentPresetKey]) {
            segUrl = SAMPLE_PRESETS[state.currentPresetKey].model;
        }
        if (segUrl && state.currentGlbUrl !== segUrl) {
            // Dự phòng: nếu segmented_glb tải lỗi, quay lại đúng model đang hiển thị hiện tại
            // (mode textured) thay vì để khung nhìn trống/vỡ.
            const fallbackUrl = state.currentGlbUrl;
            load3DStatueModel(segUrl, mode, '', fallbackUrl, 'textured');
            return;
        }
    }

    if (mode === 'plaster') {
        hideBoundaryLines();
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
        ensureBoundaryLines();
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
            if (child.isMesh) {
                // Khôi phục vật liệu gốc (có texture) của model đang nạp — nếu chỉ tắt
                // wireframe trên child.material hiện tại thì khi vừa ở mode plaster
                // material đã bị ghi đè thành thạch cao trắng, tượng sẽ kẹt trắng vĩnh viễn.
                const orig = state.originalMaterialsMap.get(child.uuid);
                if (orig) {
                    child.material = orig.clone();
                    child.material.side = THREE.DoubleSide;
                }
                if (child.material) child.material.wireframe = false;
            }
        });
        hideBoundaryLines();
    } else if (mode === 'wireframe') {
        currentModel.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.wireframe = true;
            }
        });
        hideBoundaryLines();
    }
}

// ===== Đường viền ranh giới các phân vùng (chỉ hiện ở mode 'segmented') =====
// Với mỗi submesh, cạnh nào chỉ thuộc đúng 1 tam giác trong CHÍNH submesh đó là biên của
// vùng (biên ngoài của submesh HOẶC ranh giới với submesh khác — đúng cái cần vẽ).
let boundaryLinesForModel = null; // currentModel đã tính đường viền, dùng để cache
let boundaryLineObjects = [];     // mỗi phần tử là 1 THREE.LineSegments, làm con của submesh tương ứng
let boundaryLineMaterial = null;  // dùng chung 1 material cho mọi đường viền, không tạo lại mỗi lần

function getBoundaryLineMaterial() {
    if (!boundaryLineMaterial) {
        // Màu tối, tương phản với các vật liệu phân vùng thường sáng/pastel.
        boundaryLineMaterial = new THREE.LineBasicMaterial({ color: 0x11151f });
    }
    return boundaryLineMaterial;
}

function computeBoundaryEdgesForMesh(mesh) {
    const geom = mesh.geometry;
    const posAttr = geom.attributes.position;
    const normAttr = geom.attributes.normal;
    const index = geom.index ? geom.index.array : null;
    const idxCount = index ? index.length : posAttr.count;

    if (!geom.boundingSphere) geom.computeBoundingSphere();
    // Đẩy đỉnh đường viền ra ngoài theo pháp tuyến một khoảng rất nhỏ để chống z-fighting
    // với chính mặt tam giác — tỉ lệ theo kích thước mesh để hợp lý ở mọi model.
    const eps = Math.max((geom.boundingSphere ? geom.boundingSphere.radius : 1) * 0.0015, 0.0002);

    // key = toạ độ 2 đầu mút đã làm tròn (gộp các đỉnh trùng vị trí nhưng tách rời do UV seam)
    const edgeMap = new Map();
    const ROUND = 1e4;
    const vKey = (x, y, z) => Math.round(x * ROUND) + ',' + Math.round(y * ROUND) + ',' + Math.round(z * ROUND);

    function addEdge(ia, ib) {
        const ax = posAttr.getX(ia), ay = posAttr.getY(ia), az = posAttr.getZ(ia);
        const bx = posAttr.getX(ib), by = posAttr.getY(ib), bz = posAttr.getZ(ib);
        const ka = vKey(ax, ay, az), kb = vKey(bx, by, bz);
        const key = ka < kb ? ka + '|' + kb : kb + '|' + ka;
        const existing = edgeMap.get(key);
        if (existing) {
            existing.count++;
        } else {
            edgeMap.set(key, {
                count: 1,
                ax, ay, az, bx, by, bz,
                anx: normAttr ? normAttr.getX(ia) : 0, any: normAttr ? normAttr.getY(ia) : 0, anz: normAttr ? normAttr.getZ(ia) : 1,
                bnx: normAttr ? normAttr.getX(ib) : 0, bny: normAttr ? normAttr.getY(ib) : 0, bnz: normAttr ? normAttr.getZ(ib) : 1
            });
        }
    }

    for (let t = 0; t < idxCount; t += 3) {
        const i0 = index ? index[t] : t;
        const i1 = index ? index[t + 1] : t + 1;
        const i2 = index ? index[t + 2] : t + 2;
        addEdge(i0, i1);
        addEdge(i1, i2);
        addEdge(i2, i0);
    }

    const positions = [];
    edgeMap.forEach((e) => {
        if (e.count === 1) {
            positions.push(
                e.ax + e.anx * eps, e.ay + e.any * eps, e.az + e.anz * eps,
                e.bx + e.bnx * eps, e.by + e.bny * eps, e.bz + e.bnz * eps
            );
        }
    });
    return positions;
}

function disposeBoundaryLines() {
    boundaryLineObjects.forEach((obj) => {
        if (obj.parent) obj.parent.remove(obj);
        obj.geometry.dispose();
    });
    boundaryLineObjects = [];
    boundaryLinesForModel = null;
}

function ensureBoundaryLines() {
    if (!currentModel) return;
    // Đã tính cho đúng model này rồi — chỉ cần hiện lại, không tính lại (mesh có thể tới
    // hàng trăm nghìn mặt, không nên tính mỗi lần bấm qua lại chế độ).
    if (boundaryLinesForModel === currentModel) {
        boundaryLineObjects.forEach((obj) => { obj.visible = true; });
        return;
    }
    disposeBoundaryLines();

    const startTime = performance.now();
    let totalEdges = 0;
    const material = getBoundaryLineMaterial();

    currentModel.traverse((child) => {
        if (child.isMesh && child.geometry) {
            const positions = computeBoundaryEdgesForMesh(child);
            if (positions.length === 0) return;
            const geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            const lines = new THREE.LineSegments(geom, material);
            lines.renderOrder = 1;
            child.add(lines);
            boundaryLineObjects.push(lines);
            totalEdges += positions.length / 6;
        }
    });

    boundaryLinesForModel = currentModel;
    const elapsedMs = performance.now() - startTime;
    console.log(`[boundary] Đã dựng ${totalEdges} cạnh biên cho ${boundaryLineObjects.length} phân vùng trong ${elapsedMs.toFixed(1)}ms`);
}

function hideBoundaryLines() {
    boundaryLineObjects.forEach((obj) => { obj.visible = false; });
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
                load3DStatueModel(preset.model, 'segmented', preset.name);

                // Update Stats Panel
                document.getElementById('stat-vertices').innerText = (preset.vertices || 0).toLocaleString();
                document.getElementById('stat-faces').innerText = (preset.faces || 0).toLocaleString();
                document.getElementById('stat-parts').innerText = (preset.parts ? preset.parts.length : 6);
                document.getElementById('stat-duration').innerText = 'Sẵn sàng';

                // Render Parts List in right sidebar
                renderPartsList(preset.parts || []);

                // Enable direct downloads for preset model
                // Gán thêm .href = đúng URL đã truyền cho window.open, để hover-preview đọc được;
                // onclick vẫn preventDefault() rồi tự window.open như cũ, tránh href thật (thay vì "#")
                // bị anchor target="_blank" tự mở thêm 1 tab thứ hai của cùng file.
                const dlPresetUrls = {
                    'dl-plaster': preset.model.replace('statue_segmented.glb', 'statue_plaster.glb'),
                    'dl-segmented': preset.model,
                    'dl-textured': preset.model.replace('statue_segmented.glb', 'statue_textured.glb'),
                    'dl-shell': preset.model.replace('statue_segmented.glb', 'statue_shell.glb'),
                    'dl-shell-optimized': preset.model.replace('statue_segmented.glb', 'statue_shell_optimized.glb'),
                };
                Object.entries(dlPresetUrls).forEach(([id, url]) => {
                    const el = document.getElementById(id);
                    el.href = url;
                    el.onclick = (e) => {
                        e.preventDefault();
                        window.open(url, '_blank');
                    };
                });
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

    // Thanh trượt Độ sáng khung nhìn 3D — khôi phục từ localStorage nếu có, mặc định 1.0
    // (đúng hiện trạng cường độ đèn ban đầu, không đổi trải nghiệm ngoài ý muốn).
    const brightnessSlider = document.getElementById('brightness-slider');
    const brightnessValueLabel = document.getElementById('brightness-value');
    const savedBrightness = parseFloat(localStorage.getItem(BRIGHTNESS_STORAGE_KEY));
    const initialBrightness = (!isNaN(savedBrightness) && savedBrightness >= 0.4 && savedBrightness <= 2.5)
        ? savedBrightness : 1.0;
    brightnessSlider.value = initialBrightness;
    brightnessValueLabel.textContent = initialBrightness.toFixed(1) + 'x';
    applyBrightnessFactor(initialBrightness);
    brightnessSlider.addEventListener('input', () => {
        const factor = parseFloat(brightnessSlider.value);
        brightnessValueLabel.textContent = factor.toFixed(1) + 'x';
        applyBrightnessFactor(factor);
        localStorage.setItem(BRIGHTNESS_STORAGE_KEY, String(factor));
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
    const viewModes = document.getElementById('view-modes-container');
    const viewLabel = document.getElementById('view-mode-label');
    const previewControls = document.getElementById('preview-controls-group');
    const btnOpenTab = document.getElementById('btn-open-3dpainting-tab');

    if (is3D) {
        canvasContainer.style.display = 'none';
        iframeContainer.style.display = 'flex';
        viewModes.style.display = 'none';
        if (viewLabel) viewLabel.style.display = 'none';
        if (previewControls) previewControls.style.display = 'none';
        if (btnOpenTab) btnOpenTab.style.display = 'inline-flex';
        syncModelTo3DPaintingIframe();
    } else {
        canvasContainer.style.display = 'block';
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

    const formData = new FormData();
    formData.append('file', state.selectedFile);
    formData.append('generator', generator);
    formData.append('mesh_detail', meshDetail);
    formData.append('texture_detail', 'high');
    formData.append('target_faces', targetFaces);
    formData.append('pedestal_shape', pedestalShape);
    formData.append('orientation', document.getElementById('statue-orientation-select')?.value || 'auto');

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

async function onStatueJobCompleted(job) {
    resetGenerateButton();
    state.currentJobData = job;

    // Update stats card
    document.getElementById('stat-vertices').innerText = (job.num_vertices || 0).toLocaleString();
    document.getElementById('stat-faces').innerText = (job.num_faces || 0).toLocaleString();
    document.getElementById('stat-parts').innerText = job.num_parts || 0;
    document.getElementById('stat-duration').innerText = `${job.duration_sec || 0}s`;

    // Update download buttons (xoá onclick còn sót lại từ preset chip, nếu không nó sẽ
    // tiếp tục mở file preset thay vì file của job đang chọn)
    ['dl-plaster', 'dl-segmented', 'dl-textured', 'dl-shell', 'dl-shell-optimized', 'dl-package']
        .forEach(id => { document.getElementById(id).onclick = null; });
    document.getElementById('dl-plaster').href = `/api/statue/jobs/${job.id}/files/plaster_glb`;
    document.getElementById('dl-segmented').href = `/api/statue/jobs/${job.id}/files/segmented_glb`;
    document.getElementById('dl-textured').href = `/api/statue/jobs/${job.id}/files/textured_glb`;
    document.getElementById('dl-shell').href = `/api/statue/jobs/${job.id}/files/shell_glb`;
    document.getElementById('dl-shell-optimized').href = `/api/statue/jobs/${job.id}/files/shell_optimized_glb`;
    document.getElementById('dl-package').href = `/api/statue/jobs/${job.id}/files/package_zip`;

    // Update detected parts list
    const parts = (job.metadata && job.metadata.mesh_stats && job.metadata.mesh_stats.parts) || [];
    renderPartsList(parts);

    // Mặc định xem bản chất lượng cao nhất (có texture AI thật) khi mở job/lịch sử.
    // Không phải job nào cũng có bản texture (generator xuất mesh trắng, hoặc job cũ
    // thiếu file) nên phải HEAD-check trước, và luôn truyền fallback về bản phân vùng
    // (segmented) cho load3DStatueModel để tự lùi nếu GLTFLoader nạp thất bại.
    const segmentedUrl = `/api/statue/jobs/${job.id}/files/segmented_glb`;
    const texturedUrl = `/api/statue/jobs/${job.id}/files/textured_glb`;
    try {
        const headRes = await fetch(texturedUrl, { method: 'HEAD' });
        if (headRes.ok) {
            load3DStatueModel(texturedUrl, 'textured', '', segmentedUrl, 'segmented');
        } else {
            console.warn(`Job ${job.id} không có bản textured_glb (HTTP ${headRes.status}), dùng bản phân vùng.`);
            load3DStatueModel(segmentedUrl, 'segmented');
        }
    } catch (e) {
        console.warn(`Không kiểm tra được textured_glb cho job ${job.id}, dùng bản phân vùng.`, e);
        load3DStatueModel(segmentedUrl, 'segmented');
    }
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

/* ===================================================
   9. Hover Preview Cho Các Nút Tải Về GLB (render 3D thật)
   =================================================== */
const DL_PREVIEW_IDS = ['dl-plaster', 'dl-segmented', 'dl-textured', 'dl-shell', 'dl-shell-optimized'];
const dlPreviewImgCache = new Map();  // url -> data URL (PNG render)
const dlPreviewSizeCache = new Map(); // url -> chuỗi dung lượng đã format
let dlPreviewRenderer = null;
let dlPreviewScene = null;
let dlPreviewCamera = null;
let dlPreviewToken = 0; // tăng dần mỗi lần hover mục mới, để huỷ kết quả nạp cũ nếu người dùng đã rê đi

function ensureDlPreviewRenderer() {
    if (dlPreviewRenderer) return;
    dlPreviewRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    dlPreviewRenderer.setSize(360, 360);
    dlPreviewRenderer.outputEncoding = THREE.sRGBEncoding;
    dlPreviewRenderer.setClearColor(0x11151f, 1);

    dlPreviewScene = new THREE.Scene();
    dlPreviewScene.background = new THREE.Color(0x11151f);
    dlPreviewScene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(2, 3, 4); // đặt cùng phía camera
    dlPreviewScene.add(dirLight);

    dlPreviewCamera = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
}

function formatDlPreviewSize(bytes) {
    if (!bytes || bytes <= 0) return '';
    const mb = bytes / (1024 * 1024);
    if (mb >= 0.1) return mb.toFixed(1).replace('.', ',') + ' MB';
    return Math.round(bytes / 1024) + ' KB';
}

function renderDlPreviewImage(url) {
    return new Promise((resolve, reject) => {
        ensureDlPreviewRenderer();
        const loader = new THREE.GLTFLoader();
        loader.load(url, (gltf) => {
            const obj = gltf.scene;
            dlPreviewScene.add(obj);

            const bbox = new THREE.Box3().setFromObject(obj);
            const center = bbox.getCenter(new THREE.Vector3());
            const size = bbox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z) || 1;

            obj.position.set(-center.x, -center.y, -center.z);
            dlPreviewCamera.position.set(0, maxDim * 0.15, maxDim * 2.2);
            dlPreviewCamera.lookAt(0, 0, 0);

            dlPreviewRenderer.render(dlPreviewScene, dlPreviewCamera);
            const imgData = dlPreviewRenderer.domElement.toDataURL('image/png');

            // Dọn dẹp model tạm khỏi scene dùng chung, tránh chồng lên lần preview kế tiếp
            dlPreviewScene.remove(obj);
            obj.traverse((child) => {
                if (child.isMesh) {
                    if (child.geometry) child.geometry.dispose();
                    const mats = Array.isArray(child.material) ? child.material : [child.material];
                    mats.forEach((m) => {
                        if (!m) return;
                        if (m.map) m.map.dispose();
                        m.dispose();
                    });
                }
            });

            resolve(imgData);
        }, undefined, (err) => reject(err));
    });
}

function positionDlPreviewPanel(anchorEl) {
    const panel = document.getElementById('dl-preview-panel');
    const rect = anchorEl.getBoundingClientRect();
    const panelWidth = panel.offsetWidth || 300;
    const panelHeight = panel.offsetHeight || 340;

    // Cột tải nằm sát mép phải màn hình -> panel phải mở sang trái, không tràn viewport
    let left = rect.left - panelWidth - 14;
    if (left < 8) left = 8;

    let top = rect.top + rect.height / 2 - panelHeight / 2;
    top = Math.max(8, Math.min(top, window.innerHeight - panelHeight - 8));

    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
}

function hideDlPreviewPanel() {
    document.getElementById('dl-preview-panel').style.display = 'none';
}

async function showDlPreviewFor(btnEl) {
    const url = btnEl.getAttribute('href');
    if (!url || url === '#') return;

    const myToken = ++dlPreviewToken;
    const panel = document.getElementById('dl-preview-panel');
    const imgEl = document.getElementById('dl-preview-img');
    const loadingEl = document.getElementById('dl-preview-loading');
    const nameEl = document.getElementById('dl-preview-name');
    const sizeEl = document.getElementById('dl-preview-size');

    nameEl.textContent = btnEl.querySelector('.dl-title')?.textContent || '';
    sizeEl.textContent = dlPreviewSizeCache.get(url) || '';
    panel.style.display = 'flex';
    positionDlPreviewPanel(btnEl);

    if (!dlPreviewSizeCache.has(url)) {
        fetch(url, { method: 'HEAD' }).then((res) => {
            const len = parseInt(res.headers.get('content-length') || '0', 10);
            const sizeText = formatDlPreviewSize(len);
            dlPreviewSizeCache.set(url, sizeText);
            if (myToken === dlPreviewToken) sizeEl.textContent = sizeText;
        }).catch(() => { /* bỏ qua, không để vỡ preview */ });
    }

    if (dlPreviewImgCache.has(url)) {
        imgEl.src = dlPreviewImgCache.get(url);
        imgEl.style.display = 'block';
        loadingEl.style.display = 'none';
        positionDlPreviewPanel(btnEl);
        return;
    }

    imgEl.style.display = 'none';
    loadingEl.style.display = 'block';
    loadingEl.textContent = 'Đang dựng xem trước...';

    try {
        const imgData = await renderDlPreviewImage(url);
        if (myToken !== dlPreviewToken) return; // người dùng đã rê sang mục khác, bỏ kết quả cũ
        dlPreviewImgCache.set(url, imgData);
        imgEl.src = imgData;
        imgEl.style.display = 'block';
        loadingEl.style.display = 'none';
        positionDlPreviewPanel(btnEl);
    } catch (err) {
        if (myToken !== dlPreviewToken) return;
        loadingEl.textContent = 'Không tạo được xem trước';
        console.warn('Không tải được model để xem trước:', url, err);
    }
}

function setupDownloadPreviewHovers() {
    DL_PREVIEW_IDS.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('mouseenter', () => showDlPreviewFor(el));
        el.addEventListener('mouseleave', () => {
            dlPreviewToken++; // huỷ kết quả của lần nạp đang dang dở
            hideDlPreviewPanel();
        });
    });
}
