// UniRig 3D Web Playground Frontend Logic

let scene, camera, renderer, controls;
let currentMeshGroup = null;
let skeletonGroup = null;
let animSkeletonHelper = null;
let currentMixer = null;
let currentActions = {};
let currentActiveAction = null;
let clock = new THREE.Clock();

let activeJobId = localStorage.getItem('unirig_active_job_id') || null;
let currentJobData = null;
let currentStage = 1;
let selectedBoneIndex = 0;
let isPlaying = true;
let animSpeed = 1.0;
let wireframeMode = false;
let showSkeleton = true;
let showGrid = true;
let pollTimer = null;

// ARKit 52 Facial Blendshapes State & Presets
let currentMorphMeshes = [];
let currentMorphDictionary = {};
let activeExpressionPreset = 'neutral';
// Preset metadata from the server, keyed the same way the buttons are. The clip a button
// plays is named after the preset, and that name is defined once in
// pipeline/facial_blendshapes.py -- fetching it keeps the button, the exported clip and the
// animation dropdown from drifting apart.
let expressionMeta = {};

const ARKIT_CATEGORIES = {
  eyes: [
    "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight",
    "eyeSquintLeft", "eyeSquintRight"
  ],
  brows: [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight"
  ]
};

const EXPRESSION_PRESETS = {
  wink_right: {
    eyeBlinkRight: 1.0,
    eyeSquintRight: 0.3
  },
  wink_left: {
    eyeBlinkLeft: 1.0,
    eyeSquintLeft: 0.3
  },
  blink: {
    eyeBlinkLeft: 1.0,
    eyeBlinkRight: 1.0
  },
  squint: {
    eyeSquintLeft: 0.85,
    eyeSquintRight: 0.85
  },
  wide: {
    eyeWideLeft: 1.0,
    eyeWideRight: 1.0,
    browInnerUp: 0.7
  },
  frown: {
    browDownLeft: 1.0,
    browDownRight: 1.0,
    eyeSquintLeft: 0.5,
    eyeSquintRight: 0.5
  },
  look_up: {
    eyeLookUpLeft: 0.9,
    eyeLookUpRight: 0.9,
    browInnerUp: 0.4
  },
  look_left: {
    eyeLookInRight: 0.85,
    eyeLookOutLeft: 0.85
  },
  look_right: {
    eyeLookInLeft: 0.85,
    eyeLookOutRight: 0.85
  },
  neutral: {}
};

// Initialize 3D Scene
function init3D() {
  const container = document.getElementById('canvas-wrapper');
  const width = container.clientWidth;
  const height = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x080c14);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
  camera.position.set(0, 1.2, 3.2);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0.5, 0);

  // Studio PBR Multi-directional Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambient);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x334466, 0.8);
  hemi.position.set(0, 10, 0);
  scene.add(hemi);

  const keyLight = new THREE.DirectionalLight(0xfff8f0, 1.3);
  keyLight.position.set(5, 8, 6);
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x88ccff, 0.8);
  fillLight.position.set(-6, 4, -5);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
  rimLight.position.set(0, -6, -4);
  scene.add(rimLight);

  // Grid
  const grid = new THREE.GridHelper(10, 20, 0x00f2fe, 0x233150);
  grid.position.y = -0.001;
  grid.name = "gridHelper";
  scene.add(grid);

  window.addEventListener('resize', onWindowResize);
  animate();
}

function onWindowResize() {
  const container = document.getElementById('canvas-wrapper');
  if (!container) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  if (currentMixer && isPlaying) {
    currentMixer.update(delta);
  }
  controls.update();
  renderer.render(scene, camera);
}

// Weight to RGB Jet Colormap
function weightToColor(w) {
  w = Math.max(0.0, Math.min(1.0, w));
  let r, g, b;
  if (w < 0.25) {
    r = 0;
    g = 4 * w;
    b = 1;
  } else if (w < 0.5) {
    r = 0;
    g = 1;
    b = 1 - 4 * (w - 0.25);
  } else if (w < 0.75) {
    r = 4 * (w - 0.5);
    g = 1;
    b = 0;
  } else {
    r = 1;
    g = 1 - 4 * (w - 0.75);
    b = 0;
  }
  return new THREE.Color(r, g, b);
}

// Clear current loaded 3D models
function clearSceneObjects() {
  if (currentMeshGroup) {
    scene.remove(currentMeshGroup);
    currentMeshGroup = null;
  }
  if (skeletonGroup) {
    if (skeletonGroup.parent) skeletonGroup.parent.remove(skeletonGroup);
    scene.remove(skeletonGroup);
    skeletonGroup = null;
  }
  if (animSkeletonHelper) {
    scene.remove(animSkeletonHelper);
    animSkeletonHelper = null;
  }
  if (currentMixer) {
    currentMixer.stopAllAction();
    currentMixer = null;
  }
  currentActions = {};
  currentActiveAction = null;
}

// Fetch & Display System Info (LAN IP)
async function loadSystemInfo() {
  try {
    const res = await fetch('/api/info');
    const data = await res.json();
    const lanElem = document.getElementById('lan-address');
    
    // Ưu tiên IP vật lý của máy host (e.g. 192.168.1.43), loại bỏ IP container / VPN (10.8.x.x)
    let hostIp = (data.lan_ips && data.lan_ips.length > 0) ? data.lan_ips[0] : null;
    const currentHost = window.location.hostname;
    if (currentHost && currentHost !== 'localhost' && currentHost !== '127.0.0.1' && !currentHost.startsWith('10.8.') && !currentHost.startsWith('172.')) {
      hostIp = currentHost;
    }

    if (hostIp) {
      lanElem.textContent = `LAN: ${hostIp}:${data.port}`;
      document.getElementById('lan-pill').onclick = () => {
        const url = `http://${hostIp}:${data.port}`;
        navigator.clipboard.writeText(url);
        alert(`Đã copy link LAN vào clipboard: ${url}`);
      };
    } else {
      lanElem.textContent = `Local: localhost:${data.port}`;
    }
  } catch (e) {
    console.error("Failed to load info", e);
  }
}

// Fetch & Render History List
async function loadHistory() {
  try {
    const res = await fetch('/api/jobs');
    const jobs = await res.json();
    const container = document.getElementById('history-list');
    if (!jobs || jobs.length === 0) {
      container.innerHTML = `<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px 0;">Chưa có lịch sử.</div>`;
      return;
    }

    container.innerHTML = jobs.map(j => {
      const isSel = j.id === activeJobId ? 'active' : '';
      const stClass = j.status === 'completed' ? 'completed' : (j.status === 'completed_3d_only' ? 'completed' : (j.status === 'failed' ? 'failed' : 'running'));
      const statusLabel = j.status === 'completed_3d_only' ? '3D Ready' : j.status;
      const timeStr = new Date(j.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const hasRiggedGlb = j.status === 'completed';
      const has3DGlb = j.status === 'completed' || j.status === 'completed_3d_only' || j.stage >= 0;

      return `
        <div class="history-item ${isSel}" data-job-id="${j.id}">
          <div class="history-header">
            <div class="history-title">${j.title}</div>
            <span class="status-badge ${stClass}">${statusLabel}</span>
          </div>
          <div class="history-meta" style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; gap: 8px;">
              <span><i class="fa-regular fa-clock"></i> ${timeStr}</span>
              <span><i class="fa-solid fa-bone"></i> ${j.num_bones} bones</span>
              <span><i class="fa-solid fa-stopwatch"></i> ${j.duration_sec ? j.duration_sec + 's' : '--'}</span>
            </div>
            <div style="display: flex; gap: 4px;">
              ${has3DGlb ? `<a href="/api/jobs/${j.id}/files/generated_3d_glb" target="_blank" class="tool-btn-sm" style="color: var(--accent-cyan); font-size: 11px; padding: 2px 5px; text-decoration: none;" title="Tải Model 3D .GLB" onclick="event.stopPropagation()"><i class="fa-solid fa-cube"></i></a>` : ''}
              ${hasRiggedGlb ? `<a href="/api/jobs/${j.id}/files/rigged_glb" target="_blank" class="tool-btn-sm" style="color: var(--accent-emerald); font-size: 11px; padding: 2px 5px; text-decoration: none;" title="Tải Rigged Animated .GLB" onclick="event.stopPropagation()"><i class="fa-solid fa-file-arrow-down"></i></a>` : ''}
            </div>
          </div>
        </div>
      `;
    }).join('');

    // Attach click events
    container.querySelectorAll('.history-item').forEach(item => {
      item.onclick = () => {
        const jid = item.getAttribute('data-job-id');
        selectJob(jid);
      };
    });
  } catch (e) {
    console.error("Failed to load history", e);
  }
}

// Select & Restore a Job (State Persistence)
async function selectJob(jobId) {
  if (!jobId) return;
  activeJobId = jobId;
  localStorage.setItem('unirig_active_job_id', jobId);

  // Update history items active class
  document.querySelectorAll('.history-item').forEach(item => {
    item.classList.toggle('active', item.getAttribute('data-job-id') === jobId);
  });

  await fetchAndRenderJob(jobId);
}

let smoothProgressVal = 0;
let progressInterpolationTimer = null;

function applyProgressUI(pct) {
  const overlayPct = document.getElementById('overlay-progress-pct');
  const overlayBar = document.getElementById('overlay-progress-bar');
  const step0Badge = document.getElementById('step-0-badge');

  if (overlayPct) overlayPct.textContent = `${pct}%`;
  if (overlayBar) overlayBar.style.width = `${Math.max(5, Math.min(100, pct))}%`;
  if (step0Badge && currentJobData && currentJobData.status === 'processing_image_to_3d') {
    step0Badge.textContent = `${pct}%`;
  }
}

// Fetch single job status & render appropriate stage
async function fetchAndRenderJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}?_t=${Date.now()}`);
    if (!res.ok) throw new Error("Job not found");
    const job = await res.json();
    currentJobData = job;

    updateUIWithJob(job);

    // If job still processing, start polling
    if (job.status.startsWith('processing') || job.status === 'queued') {
      startPolling(jobId);
    } else {
      stopPolling();
      // Load 3D model (default to stage 4 Animation if completed, or stage 0 if completed_3d_only)
      const stageToLoad = job.status === 'completed' ? 4 : (job.status === 'completed_3d_only' ? 0 : currentStage);
      load3DForStage(stageToLoad);
    }
  } catch (e) {
    console.error("Error fetching job", e);
  }
}

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  if (progressInterpolationTimer) clearInterval(progressInterpolationTimer);

  // Micro-tick animation to keep progress bar smoothly interpolating to backend target
  progressInterpolationTimer = setInterval(() => {
    if (!currentJobData) return;
    const st = currentJobData.status;
    if (st.startsWith('processing') || st === 'queued') {
      const targetPct = currentJobData.metadata?.progress?.pct || 10;
      if (smoothProgressVal < targetPct) {
        smoothProgressVal += (targetPct - smoothProgressVal) * 0.35 + 0.2;
        if (smoothProgressVal >= targetPct) smoothProgressVal = targetPct;
        applyProgressUI(Math.round(smoothProgressVal));
      } else {
        applyProgressUI(Math.round(smoothProgressVal));
      }
    }
  }, 100);

  const doPoll = async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}?_t=${Date.now()}`);
      if (res.ok) {
        const job = await res.json();
        currentJobData = job;
        updateUIWithJob(job);
        
        if (job.status === 'completed' || job.status === 'completed_3d_only' || job.status === 'failed') {
          stopPolling();
          smoothProgressVal = 100;
          applyProgressUI(100);
          loadHistory();
          const stageToLoad = job.status === 'completed' ? 4 : (job.status === 'completed_3d_only' ? 0 : currentStage);
          load3DForStage(stageToLoad);
        }
      }
    } catch (e) {
      console.warn("Polling error:", e);
    }
  };

  doPoll();
  pollTimer = setInterval(doPoll, 400);
}


function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (progressInterpolationTimer) {
    clearInterval(progressInterpolationTimer);
    progressInterpolationTimer = null;
  }
}

// Update UI badges, steppers, and tree
function updateUIWithJob(job) {
  const badge = document.getElementById('global-status-badge');
  badge.style.display = 'block';
  badge.textContent = job.status === 'completed_3d_only' ? '3D Model Ready' : job.status;
  badge.className = `status-badge ${job.status.includes('completed') ? 'completed' : (job.status === 'failed' ? 'failed' : 'running')}`;

  // Check if input is a 2D image
  const ext = job.input_filename ? job.input_filename.split('.').pop().toLowerCase() : '';
  const isImageInput = ['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(ext) || !!job.metadata?.stage0;
  
  const step0Card = document.getElementById('step-card-0');
  const step0Desc = document.getElementById('step-0-desc');
  const imgPreviewCard = document.getElementById('image-preview-card');
  const imgPreviewElem = document.getElementById('image-preview-img');

  const genChoice = job.metadata?.generator || '';
  const modelUsed = job.metadata?.stage0?.model_used || '';
  const genName = (genChoice === 'pixal3d' || modelUsed.includes('Pixal3D'))
    ? 'Pixal3D (SIGGRAPH 2026)'
    : (genChoice === 'trellis' || modelUsed.includes('TRELLIS'))
      ? 'TRELLIS.2-4B'
      : 'Tencent Hunyuan3D-2.1';

  // Progress info from metadata
  const progress = job.metadata?.progress || {};
  const currentPct = progress.pct !== undefined ? progress.pct : (job.status === 'completed' || job.status === 'completed_3d_only' ? 100 : (job.status === 'queued' ? 5 : 20));
  const currentStepName = progress.step_name || (job.status === 'processing_image_to_3d' ? `Đang tạo 3D bằng ${genName}...` : `Giai đoạn ${job.stage}/4: ${job.status.replace('processing_', '')}...`);
  const stepIdx = progress.step_idx || (job.stage || 1);
  const totalSteps = progress.total_steps || (job.status === 'processing_image_to_3d' ? 5 : 4);

  if (job.status === 'completed' || job.status === 'completed_3d_only') {
    smoothProgressVal = 100;
  } else if (smoothProgressVal < currentPct) {
    smoothProgressVal = currentPct;
  }

  const step0Badge = document.getElementById('step-0-badge');
  if (isImageInput) {
    if (step0Card) step0Card.style.display = 'flex';
    if (step0Desc) step0Desc.textContent = `${genName} Model`;
    if (step0Badge) {
      step0Badge.style.display = 'inline-block';
      if (job.status === 'processing_image_to_3d') {
        step0Badge.textContent = `${Math.round(smoothProgressVal)}%`;
        step0Badge.className = 'status-badge running';
      } else if (job.stage >= 1 || job.status === 'completed_3d_only' || job.status === 'completed') {
        step0Badge.textContent = '100%';
        step0Badge.className = 'status-badge completed';
      } else {
        step0Badge.textContent = '0%';
        step0Badge.className = 'status-badge running';
      }
    }
    if (imgPreviewCard && imgPreviewElem) {
      imgPreviewCard.style.display = 'block';
      imgPreviewElem.src = `/api/jobs/${job.id}/files/input_image`;
    }
  } else {
    if (step0Card) step0Card.style.display = 'none';
    if (imgPreviewCard) imgPreviewCard.style.display = 'none';
  }

  // Viewport Overlays Handling
  const overlay = document.getElementById('viewport-processing-overlay');
  const overlayTitle = document.getElementById('overlay-title');
  const overlayStepBadge = document.getElementById('overlay-step-badge');
  const overlayPct = document.getElementById('overlay-progress-pct');
  const overlayBar = document.getElementById('overlay-progress-bar');
  const overlayDescText = document.getElementById('overlay-desc-text');
  const bannerCompleted = document.getElementById('viewport-completed-banner');
  const bannerTitle = document.getElementById('banner-completed-title');

  if (job.status.startsWith('processing') || job.status === 'queued') {
    if (overlay) overlay.style.display = 'block';
    if (bannerCompleted) bannerCompleted.style.display = 'none';

    if (job.status === 'processing_image_to_3d' || job.status === 'queued') {
      if (overlayTitle) overlayTitle.textContent = "✨ Đang tạo Model 3D từ Ảnh 2D...";
    } else {
      if (overlayTitle) overlayTitle.textContent = "🦴 Đang xử lý Rigging & Motion 3D...";
    }

    if (overlayStepBadge) overlayStepBadge.textContent = `Bước ${stepIdx}/${totalSteps}`;
    applyProgressUI(Math.round(smoothProgressVal));
    if (overlayDescText) overlayDescText.textContent = currentStepName;

  } else {
    if (overlay) overlay.style.display = 'none';
    if (job.status === 'completed_3d_only' && bannerCompleted) {
      if (bannerTitle) {
        bannerTitle.innerHTML = `<i class="fa-solid fa-circle-check"></i> Đã tạo xong Model 3D từ Ảnh 2D (${genName})!`;
      }
      bannerCompleted.style.display = 'block';
    } else if (bannerCompleted) {
      bannerCompleted.style.display = 'none';
    }
  }



  // Update stepper cards
  const stageMap = {
    'processing_image_to_3d': 0,
    'processing_prep': 1,
    'processing_skeleton': 2,
    'processing_skin': 3,
    'processing_rig': 4
  };
  const activeStep = stageMap[job.status];

  for (let s = 0; s <= 4; s++) {
    const card = document.getElementById(`step-card-${s}`);
    if (!card) continue;
    const check = card.querySelector('.step-check');
    
    // Clear old state classes
    card.classList.remove('completed', 'running');

    if (s === activeStep) {
      card.classList.add('running');
      if (check) check.style.display = 'none';
    } else if (job.stage > s || (job.status === 'completed_3d_only' && s === 0) || (job.status === 'completed' && s <= 4)) {
      card.classList.add('completed');
      if (check) check.style.display = 'inline-block';
    } else {
      if (check) check.style.display = 'none';
    }
  }

  // Update buttons
  const isDone = job.status === 'completed';
  const is3DModelAvailable = job.status === 'completed' || job.status === 'completed_3d_only' || job.stage >= 1 || !!job.metadata?.stage0 || !!job.metadata?.prep;
  
  // Download buttons
  const btn3d = document.getElementById('btn-download-3d-stage0');
  if (btn3d) {
    btn3d.disabled = !is3DModelAvailable;
  }

  const btnPreview3d = document.getElementById('btn-preview-download-3d');
  if (btnPreview3d) {
    btnPreview3d.disabled = !is3DModelAvailable;
    btnPreview3d.style.opacity = is3DModelAvailable ? '1' : '0.5';
  }

  const btnGlb = document.getElementById('btn-download-glb');
  if (btnGlb) {
    btnGlb.disabled = !isDone;
  }

  const btnObj = document.getElementById('btn-download-obj');
  if (btnObj) {
    btnObj.disabled = job.stage < 2;
  }

  const btnContinue = document.getElementById('btn-continue-rigging');
  if (btnContinue) {
    if (job.status === 'completed_3d_only' || (job.stage === 0 && !job.status.startsWith('processing'))) {
      btnContinue.style.display = 'block';
      btnContinue.disabled = false;
    } else {
      btnContinue.style.display = 'none';
    }
  }

  // Update Bone Tree
  const skelMeta = job.metadata?.skel;
  const treeContainer = document.getElementById('bone-tree');
  if (skelMeta && skelMeta.tree) {
    document.getElementById('bone-count-badge').textContent = `${skelMeta.num_bones} Bones`;
    treeContainer.innerHTML = skelMeta.tree.map(b => {
      const isSel = b.index === selectedBoneIndex ? 'selected' : '';
      return `
        <div class="tree-node ${isSel}" data-bone-index="${b.index}">
          <span><i class="fa-solid fa-bone" style="font-size: 10px; margin-right: 4px;"></i> [${b.index}] ${b.name}</span>
          <span style="font-size: 10px; color: var(--text-muted);">${b.parent !== null ? 'p:' + b.parent : 'root'}</span>
        </div>
      `;
    }).join('');

    treeContainer.querySelectorAll('.tree-node').forEach(n => {
      n.onclick = () => {
        const bIdx = parseInt(n.getAttribute('data-bone-index'));
        selectBone(bIdx);
      };
    });
  }
}

// 3D Rendering for each stage
async function load3DForStage(stage) {
  if (!currentJobData) return;
  currentStage = stage;

  // Switch tabs active state
  document.querySelectorAll('.stage-tab').forEach(tab => {
    tab.classList.toggle('active', parseInt(tab.getAttribute('data-tab')) === stage);
  });

  const animBar = document.getElementById('anim-bar');
  const legend = document.getElementById('heatmap-legend');

  animBar.style.display = stage === 4 ? 'flex' : 'none';
  legend.style.display = stage === 3 ? 'block' : 'none';

  clearSceneObjects();

  const loader = new THREE.GLTFLoader();

  if (stage === 0 || stage === 1 || stage === 2 || stage === 3) {
    // Load Input Normalized GLB
    const url = `/api/jobs/${currentJobData.id}/files/input_model`;
    loader.load(url, (gltf) => {
      currentMeshGroup = gltf.scene;
      currentMeshGroup.position.set(0, 0, 0);
      currentMeshGroup.updateMatrixWorld(true);

      // Apply materials based on stage
      currentMeshGroup.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          const hasColors = !!(child.geometry && child.geometry.attributes && child.geometry.attributes.color);
          const hasPBRMap = !!(child.material && (child.material.map || child.material.metalnessMap || child.material.roughnessMap));

          if (stage === 0 || stage === 1) {
            if (child.material) {
              child.material.wireframe = wireframeMode;
              if (child.material.map) {
                child.material.map.encoding = THREE.sRGBEncoding;
                child.material.map.needsUpdate = true;
              }
              if (child.material.roughnessMap || child.material.metalnessMap) {
                child.material.roughness = 1.0;
                child.material.metalness = 1.0;
                child.material.needsUpdate = true;
              } else if (!hasPBRMap && !hasColors) {
                child.material = new THREE.MeshStandardMaterial({
                  color: 0xd1d5db,
                  roughness: 0.45,
                  metalness: 0.1,
                  wireframe: wireframeMode
                });
              }
              if (hasColors) {
                child.material.vertexColors = true;
                child.material.needsUpdate = true;
              }
            }
          } else if (stage === 2) {
            child.material = new THREE.MeshStandardMaterial({
              color: 0x778899,
              transparent: true,
              opacity: 0.6,
              roughness: 0.5,
              wireframe: wireframeMode
            });
          }
        }
      });

      scene.add(currentMeshGroup);
      setupMorphTargetsControls(currentMeshGroup);

      if (stage === 2 || stage === 3) {
        buildSkeletonVisualizer(currentJobData.metadata?.skel);
      }
      if (stage === 3) {
        loadBoneHeatmap(selectedBoneIndex);
      }
    });
  } else if (stage === 4) {
    // Load Final Rigged and Animated GLB. Cache-busted like the JSON fetches above: the URL
    // never changes when a job is re-animated, so without this the browser keeps replaying
    // the GLB it downloaded before and the new clips never show up in the picker.
    const url = `/api/jobs/${currentJobData.id}/files/rigged_glb?_t=${Date.now()}`;
    loader.load(url, (gltf) => {
      currentMeshGroup = gltf.scene;
      currentMeshGroup.position.set(0, 0, 0);
      currentMeshGroup.updateMatrixWorld(true);

      currentMeshGroup.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          if (child.material) {
            child.material.wireframe = wireframeMode;
            if (child.material.map) {
              child.material.map.encoding = THREE.sRGBEncoding;
              child.material.map.needsUpdate = true;
            }
            if (child.material.roughnessMap || child.material.metalnessMap) {
              child.material.roughness = 1.0;
              child.material.metalness = 1.0;
              child.material.needsUpdate = true;
            }
            const hasColors = !!(child.geometry && child.geometry.attributes && child.geometry.attributes.color);
            if (hasColors) {
              child.material.vertexColors = true;
              child.material.needsUpdate = true;
            }
          }
        }
      });

      scene.add(currentMeshGroup);
      setupMorphTargetsControls(currentMeshGroup);

      // Setup Animation Mixer
      if (gltf.animations && gltf.animations.length > 0) {
        currentMixer = new THREE.AnimationMixer(currentMeshGroup);
        currentActions = {};
        
        const animSelect = document.getElementById('anim-select');
        animSelect.innerHTML = gltf.animations.map(a => `<option value="${a.name}">${a.name}</option>`).join('');

        gltf.animations.forEach(clip => {
          const action = currentMixer.clipAction(clip);
          currentActions[clip.name] = action;
        });

        // Play first animation (Idle)
        const defaultAnim = gltf.animations[0].name;
        playAnimation(defaultAnim);
      }

      // Add SkeletonHelper in world space
      if (animSkeletonHelper) {
        scene.remove(animSkeletonHelper);
        animSkeletonHelper = null;
      }
      animSkeletonHelper = new THREE.SkeletonHelper(currentMeshGroup);
      animSkeletonHelper.material.linewidth = 2;
      animSkeletonHelper.material.color = new THREE.Color(0x00f2fe);
      animSkeletonHelper.material.depthTest = false;
      animSkeletonHelper.renderOrder = 999;
      animSkeletonHelper.visible = showSkeleton;
      scene.add(animSkeletonHelper);
    });
  }
}

// Build 3D Skeleton (Spheres & Cylinders in world space)
function buildSkeletonVisualizer(skelMeta) {
  if (!skelMeta || !skelMeta.tree) return;
  if (skeletonGroup) {
    if (skeletonGroup.parent) skeletonGroup.parent.remove(skeletonGroup);
    skeletonGroup = null;
  }

  skeletonGroup = new THREE.Group();
  skeletonGroup.name = "visualizerSkeletonGroup";
  const tree = skelMeta.tree;
  const jointMatDefault = new THREE.MeshStandardMaterial({
    color: 0x00f2fe,
    roughness: 0.2,
    metalness: 0.8,
    depthTest: false,
    transparent: true,
    opacity: 0.95
  });
  const jointMatSelected = new THREE.MeshStandardMaterial({
    color: 0xffd700,
    roughness: 0.1,
    metalness: 0.9,
    depthTest: false,
    transparent: true,
    opacity: 1.0
  });
  const boneMat = new THREE.MeshStandardMaterial({
    color: 0xff0055,
    roughness: 0.3,
    depthTest: false,
    transparent: true,
    opacity: 0.9
  });

  const sphereGeo = new THREE.SphereGeometry(0.02, 16, 16);

  tree.forEach(node => {
    const isSel = node.index === selectedBoneIndex;
    const pos = new THREE.Vector3(node.position[0], node.position[1], node.position[2]);
    const jointMesh = new THREE.Mesh(sphereGeo, isSel ? jointMatSelected : jointMatDefault);
    jointMesh.position.copy(pos);
    jointMesh.renderOrder = 999;
    jointMesh.userData = { boneIndex: node.index };
    skeletonGroup.add(jointMesh);

    if (node.parent !== null && node.parent >= 0) {
      const parentNode = tree[node.parent];
      const pPos = new THREE.Vector3(parentNode.position[0], parentNode.position[1], parentNode.position[2]);
      
      const dist = pPos.distanceTo(pos);
      if (dist > 1e-4) {
        const cylGeo = new THREE.CylinderGeometry(0.008, 0.008, dist, 8);
        const cylMesh = new THREE.Mesh(cylGeo, boneMat);
        const mid = pPos.clone().add(pos).multiplyScalar(0.5);
        cylMesh.position.copy(mid);
        cylMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), pos.clone().sub(pPos).normalize());
        cylMesh.renderOrder = 999;
        skeletonGroup.add(cylMesh);
      }
    }
  });

  skeletonGroup.visible = showSkeleton;
  scene.add(skeletonGroup);
}

// Load Skin Heatmap for a bone
async function loadBoneHeatmap(boneIndex) {
  if (!currentJobData || !currentMeshGroup) return;
  try {
    const res = await fetch(`/api/jobs/${currentJobData.id}/weights/${boneIndex}`);
    if (!res.ok) return;
    const data = await res.json();
    const weights = data.weights;

    currentMeshGroup.traverse((child) => {
      if (child.isMesh) {
        const geo = child.geometry;
        const count = geo.attributes.position.count;
        const colors = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
          const w = weights[i] || 0.0;
          const col = weightToColor(w);
          colors[i * 3] = col.r;
          colors[i * 3 + 1] = col.g;
          colors[i * 3 + 2] = col.b;
        }

        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        child.material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.5,
          metalness: 0.1,
          wireframe: wireframeMode
        });
      }
    });
  } catch (e) {
    console.error("Failed to load bone heatmap", e);
  }
}

function selectBone(index) {
  selectedBoneIndex = index;
  document.querySelectorAll('.tree-node').forEach(n => {
    n.classList.toggle('selected', parseInt(n.getAttribute('data-bone-index')) === index);
  });
  if (currentStage === 2 || currentStage === 3) {
    buildSkeletonVisualizer(currentJobData?.metadata?.skel);
  }
  if (currentStage === 3) {
    loadBoneHeatmap(index);
  }
}

// Animation controls
function playAnimation(name) {
  if (!currentActions[name]) return;
  if (currentActiveAction) {
    currentActiveAction.fadeOut(0.2);
  }
  const action = currentActions[name];
  action.reset().fadeIn(0.2).play();
  currentActiveAction = action;
  document.getElementById('anim-select').value = name;
}

// Facial Morph Targets & ARKit Blendshapes Controller
function setupMorphTargetsControls(meshGroup) {
  currentMorphMeshes = [];
  currentMorphDictionary = {};

  if (!meshGroup) return;

  meshGroup.traverse((child) => {
    if (child.isMesh && child.morphTargetDictionary && child.morphTargetInfluences) {
      currentMorphMeshes.push(child);
      Object.assign(currentMorphDictionary, child.morphTargetDictionary);
    }
  });

  const countBadge = document.getElementById('blendshape-count-badge');
  const morphNames = Object.keys(currentMorphDictionary);

  if (morphNames.length === 0) {
    if (countBadge) countBadge.textContent = '0 Blendshapes';
    clearMorphSliders();
    return;
  }

  if (countBadge) countBadge.textContent = `${morphNames.length} Blendshapes`;

  populateMorphSliders(morphNames);
  applyExpressionPreset(activeExpressionPreset);
}

function clearMorphSliders() {
  ['eyes', 'brows'].forEach(cat => {
    const list = document.getElementById(`bs-list-${cat}`);
    if (list) list.innerHTML = '<div style="color: var(--text-muted); font-size: 11px; text-align: center; padding: 4px;">Chưa có biểu cảm mắt trên model này</div>';
  });
}

function populateMorphSliders(availableNames) {
  const nameSet = new Set(availableNames);

  Object.entries(ARKIT_CATEGORIES).forEach(([cat, targetNames]) => {
    const list = document.getElementById(`bs-list-${cat}`);
    if (!list) return;

    const matchedNames = targetNames.filter(name => nameSet.has(name));
    if (matchedNames.length === 0) {
      list.innerHTML = '<div style="color: var(--text-muted); font-size: 11px; text-align: center; padding: 4px;">Không có target khớp</div>';
      return;
    }

    list.innerHTML = '';
    matchedNames.forEach(name => {
      const row = document.createElement('div');
      row.className = 'bs-slider-row';

      const label = document.createElement('span');
      label.className = 'bs-slider-label';
      label.textContent = name;
      label.title = name;

      const input = document.createElement('input');
      input.type = 'range';
      input.className = 'bs-slider-input';
      input.min = '0';
      input.max = '1';
      input.step = '0.01';
      input.value = '0';
      input.dataset.morph = name;

      const valBadge = document.createElement('span');
      valBadge.className = 'bs-slider-val';
      valBadge.id = `bs-val-${name}`;
      valBadge.textContent = '0%';

      input.oninput = (e) => {
        const val = parseFloat(e.target.value);
        valBadge.textContent = `${Math.round(val * 100)}%`;
        setMorphInfluence(name, val);
        document.querySelectorAll('.expr-btn').forEach(b => b.classList.remove('active'));
      };

      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(valBadge);
      list.appendChild(row);
    });
  });
}

function setMorphInfluence(name, value) {
  currentMorphMeshes.forEach(mesh => {
    if (mesh.morphTargetDictionary && mesh.morphTargetInfluences) {
      const idx = mesh.morphTargetDictionary[name];
      if (idx !== undefined && idx < mesh.morphTargetInfluences.length) {
        mesh.morphTargetInfluences[idx] = value;
      }
    }
  });
}

async function loadExpressionMeta() {
  try {
    const res = await fetch('/api/facial_blendshapes/presets');
    const data = await res.json();
    expressionMeta = data.presets || {};
  } catch (e) {
    expressionMeta = {};
  }
}

// Play an expression as a clip rather than snapping to its pose. The body animation keeps
// running: this clip only drives morph weights and the body clip only drives bones, so the
// two do not compete for the same property and neither has to be stopped.
function playExpressionClip(presetKey) {
  const meta = expressionMeta[presetKey];
  const action = meta && currentActions ? currentActions[meta.name] : null;
  if (!action) return false;
  action.reset();
  action.setLoop(THREE.LoopOnce, 1);
  action.clampWhenFinished = false;
  action.play();
  return true;
}

function applyExpressionPreset(presetKey) {
  activeExpressionPreset = presetKey;
  const targetWeights = EXPRESSION_PRESETS[presetKey] || {};

  document.querySelectorAll('.expr-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-preset') === presetKey);
  });

  // A clip shows the expression happening; the static pose is the fallback for a model
  // exported before these clips existed, or one whose eyes were never located.
  if (presetKey !== 'neutral' && playExpressionClip(presetKey)) {
    return;
  }

  const allMorphNames = Object.keys(currentMorphDictionary);

  allMorphNames.forEach(name => {
    const targetVal = targetWeights[name] || 0.0;
    setMorphInfluence(name, targetVal);

    const slider = document.querySelector(`.bs-slider-input[data-morph="${name}"]`);
    const valBadge = document.getElementById(`bs-val-${name}`);
    if (slider) slider.value = targetVal;
    if (valBadge) valBadge.textContent = `${Math.round(targetVal * 100)}%`;
  });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  init3D();
  loadSystemInfo();
  loadHistory();
  loadExpressionMeta();

  // Mode select checkbox initialization & persistence
  const chkAutoFull = document.getElementById('chk-auto-full');
  const modeBadge = document.getElementById('mode-badge');
  const savedAutoFull = localStorage.getItem('unirig_auto_full');
  if (chkAutoFull) {
    chkAutoFull.checked = savedAutoFull !== null ? (savedAutoFull === 'true') : true;
    const updateModeUI = () => {
      if (modeBadge) {
        if (chkAutoFull.checked) {
          modeBadge.textContent = '3D + Rig + Motion';
          modeBadge.className = 'status-badge completed';
        } else {
          modeBadge.textContent = 'Chỉ tạo Model 3D';
          modeBadge.className = 'status-badge running';
        }
      }
      localStorage.setItem('unirig_auto_full', chkAutoFull.checked ? 'true' : 'false');
    };
    chkAutoFull.onchange = updateModeUI;
    updateModeUI();
  }

  if (activeJobId) {
    selectJob(activeJobId);
  }

  // Presets click
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.onclick = async () => {
      const preset = btn.getAttribute('data-preset');
      const form = new FormData();
      form.append('preset_name', preset);
      const res = await fetch('/api/jobs/preset', { method: 'POST', body: form });
      if (res.ok) {
        const job = await res.json();
        selectJob(job.id);
        loadHistory();
      }
    };
  });

  // Generator selection state & click handlers
  let selectedGenerator = localStorage.getItem('unirig_generator') || 'hunyuan3d';
  const genBadge = document.getElementById('generator-badge');
  const genButtons = document.querySelectorAll('.generator-btn');

  const updateGeneratorUI = (gen) => {
    selectedGenerator = gen;
    localStorage.setItem('unirig_generator', gen);
    genButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-generator') === gen);
    });
    if (genBadge) {
      if (gen === 'pixal3d') {
        genBadge.textContent = 'Pixal3D';
      } else if (gen === 'hunyuan3d') {
        genBadge.textContent = 'Hunyuan3D-2.1';
      } else {
        genBadge.textContent = 'TRELLIS.2-4B';
      }
    }
    const mvContainer = document.getElementById('mv-container');
    if (mvContainer) {
      mvContainer.style.display = (gen === 'pixal3d') ? 'block' : 'none';
      if (gen !== 'pixal3d') {
        // Reset về single view nếu chuyển sang model khác
        document.getElementById('btn-view-single')?.classList.add('active');
        document.getElementById('btn-view-multi')?.classList.remove('active');
        const optBox = document.getElementById('mv-options-box');
        if (optBox) optBox.style.display = 'none';
        const stdUpload = document.getElementById('upload-zone');
        if (stdUpload) stdUpload.style.display = 'block';
      }
    }
  };

  genButtons.forEach(btn => {
    btn.onclick = () => {
      const gen = btn.getAttribute('data-generator');
      if (gen) updateGeneratorUI(gen);
    };
  });
  updateGeneratorUI(selectedGenerator);

  // Generation detail state. Geometry and texture are independent because they cost
  // very differently -- the texture bake dominates generation time.
  const selectedDetail = {
    mesh: localStorage.getItem('unirig_mesh_detail') || 'high',
    texture: localStorage.getItem('unirig_texture_detail') || 'high',
  };
  const detailButtons = document.querySelectorAll('.detail-btn');

  const updateDetailUI = (kind, level) => {
    selectedDetail[kind] = level;
    localStorage.setItem(`unirig_${kind}_detail`, level);
    detailButtons.forEach(btn => {
      if (btn.getAttribute('data-detail-kind') !== kind) return;
      btn.classList.toggle('active', btn.getAttribute('data-detail') === level);
    });
  };

  detailButtons.forEach(btn => {
    btn.onclick = () => {
      const kind = btn.getAttribute('data-detail-kind');
      const level = btn.getAttribute('data-detail');
      if (kind && level) updateDetailUI(kind, level);
    };
  });
  updateDetailUI('mesh', selectedDetail.mesh);
  updateDetailUI('texture', selectedDetail.texture);

  // Custom File Upload & Drag-and-Drop
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');

  async function handleFileUpload(file) {
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    const isAutoFull = document.getElementById('chk-auto-full')?.checked ?? true;
    form.append('mode', isAutoFull ? 'full' : '3d_only');
    form.append('generator', selectedGenerator);
    form.append('mesh_detail', selectedDetail.mesh);
    form.append('texture_detail', selectedDetail.texture);

    try {
      const res = await fetch('/api/jobs/upload', { method: 'POST', body: form });
      if (res.ok) {
        const job = await res.json();
        activeJobId = job.id;
        localStorage.setItem('unirig_active_job_id', job.id);
        currentJobData = job;
        smoothProgressVal = 10;
        updateUIWithJob(job);
        startPolling(job.id);
        loadHistory();
      }
    } catch (err) {
      console.error("Upload error", err);
    }
  }


  uploadZone.onclick = () => fileInput.click();
  uploadZone.ondragover = (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--accent-cyan)';
    uploadZone.style.background = 'rgba(0, 242, 254, 0.08)';
  };
  uploadZone.ondragleave = () => {
    uploadZone.style.borderColor = '';
    uploadZone.style.background = '';
  };
  uploadZone.ondrop = (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = '';
    uploadZone.style.background = '';
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  fileInput.onchange = (e) => {
    if (fileInput.files.length > 0) {
      handleFileUpload(fileInput.files[0]);
    }
  };

  // Pixal3D Multi-View View Mode & Upload Handlers
  const btnViewSingle = document.getElementById('btn-view-single');
  const btnViewMulti = document.getElementById('btn-view-multi');
  const mvOptionsBox = document.getElementById('mv-options-box');
  const mvSheetDropzone = document.getElementById('mv-sheet-dropzone');
  const mvSheetInput = document.getElementById('mv-sheet-input');
  const mv4viewsGrid = document.getElementById('mv-4views-grid');
  const btnStartMv = document.getElementById('btn-start-mv');

  let currentViewMode = 'single';
  let currentMvType = 'turnaround';
  const selected4Views = { front: null, right: null, back: null, left: null };

  if (btnViewSingle && btnViewMulti) {
    btnViewSingle.onclick = () => {
      currentViewMode = 'single';
      btnViewSingle.classList.add('active');
      btnViewMulti.classList.remove('active');
      if (mvOptionsBox) mvOptionsBox.style.display = 'none';
      if (uploadZone) uploadZone.style.display = 'block';
    };

    btnViewMulti.onclick = () => {
      currentViewMode = 'multi';
      btnViewMulti.classList.add('active');
      btnViewSingle.classList.remove('active');
      if (mvOptionsBox) mvOptionsBox.style.display = 'block';
      if (uploadZone) uploadZone.style.display = 'none';
    };
  }

  document.querySelectorAll('input[name="mv_type_radio"]').forEach(radio => {
    radio.onchange = (e) => {
      currentMvType = e.target.value;
      if (currentMvType === 'turnaround') {
        if (mvSheetDropzone) mvSheetDropzone.style.display = 'block';
        if (mv4viewsGrid) mv4viewsGrid.style.display = 'none';
        if (btnStartMv) btnStartMv.style.display = 'none';
      } else {
        if (mvSheetDropzone) mvSheetDropzone.style.display = 'none';
        if (mv4viewsGrid) mv4viewsGrid.style.display = 'grid';
        if (btnStartMv) btnStartMv.style.display = 'flex';
      }
    };
  });

  if (mvSheetDropzone && mvSheetInput) {
    mvSheetDropzone.onclick = () => mvSheetInput.click();
    mvSheetInput.onchange = () => {
      if (mvSheetInput.files.length > 0) {
        handleMultiviewSheetUpload(mvSheetInput.files[0]);
      }
    };
  }

  async function handleMultiviewSheetUpload(file) {
    if (!file) return;
    const form = new FormData();
    form.append('mv_type', 'turnaround');
    form.append('file_sheet', file);
    const isAutoFull = document.getElementById('chk-auto-full')?.checked ?? true;
    form.append('mode', isAutoFull ? 'full' : '3d_only');
    form.append('generator', 'pixal3d_mv');
    form.append('mesh_detail', selectedDetail.mesh);
    form.append('texture_detail', selectedDetail.texture);

    try {
      const res = await fetch('/api/jobs/upload_multiview', { method: 'POST', body: form });
      if (res.ok) {
        const job = await res.json();
        activeJobId = job.id;
        localStorage.setItem('unirig_active_job_id', job.id);
        currentJobData = job;
        smoothProgressVal = 10;
        updateUIWithJob(job);
        startPolling(job.id);
        loadHistory();
      }
    } catch (err) {
      console.error("MV Sheet upload error", err);
    }
  }

  document.querySelectorAll('.mv-slot').forEach(slot => {
    const slotName = slot.getAttribute('data-slot');
    const input = slot.querySelector('input[type="file"]');
    slot.onclick = (e) => {
      e.stopPropagation();
      input?.click();
    };
    if (input) {
      input.onchange = () => {
        if (input.files.length > 0) {
          const f = input.files[0];
          selected4Views[slotName] = f;
          const statusElem = document.getElementById(`slot-${slotName}-status`);
          if (statusElem) {
            statusElem.textContent = f.name;
            statusElem.style.color = '#34d399';
            statusElem.style.fontWeight = '700';
          }
        }
      };
    }
  });

  if (btnStartMv) {
    btnStartMv.onclick = async () => {
      if (!selected4Views.front || !selected4Views.right || !selected4Views.back || !selected4Views.left) {
        alert("Vui lòng chọn đầy đủ cả 4 ảnh: Front (0°), Sườn phải (270°), Back (180°), Sườn trái (90°)!");
        return;
      }
      btnStartMv.disabled = true;
      btnStartMv.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tải lên & khởi chạy...';

      const form = new FormData();
      form.append('mv_type', '4views');
      form.append('file_front', selected4Views.front);
      form.append('file_right', selected4Views.right);
      form.append('file_back', selected4Views.back);
      form.append('file_left', selected4Views.left);
      const isAutoFull = document.getElementById('chk-auto-full')?.checked ?? true;
      form.append('mode', isAutoFull ? 'full' : '3d_only');
      form.append('generator', 'pixal3d_mv');
      form.append('mesh_detail', selectedDetail.mesh);
      form.append('texture_detail', selectedDetail.texture);

      try {
        const res = await fetch('/api/jobs/upload_multiview', { method: 'POST', body: form });
        if (res.ok) {
          const job = await res.json();
          activeJobId = job.id;
          localStorage.setItem('unirig_active_job_id', job.id);
          currentJobData = job;
          smoothProgressVal = 10;
          updateUIWithJob(job);
          startPolling(job.id);
          loadHistory();
        } else {
          const err = await res.json();
          alert(`Lỗi: ${err.detail || 'Không thể tạo job Multi-View'}`);
        }
      } catch (err) {
        console.error("4Views upload error", err);
      } finally {
        btnStartMv.disabled = false;
        btnStartMv.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Bắt đầu xử lý Đa Góc Nhìn 360°';
      }
    };
  }

  // Run Full Pipeline / Trigger Continuation Button
  const btnRunPipeline = document.getElementById('btn-run-pipeline');
  if (btnRunPipeline) {
    btnRunPipeline.onclick = async () => {
      if (!currentJobData) {
        fileInput.click();
        return;
      }
      const ext = currentJobData.input_filename ? currentJobData.input_filename.split('.').pop().toLowerCase() : '';
      const isImg = ['png', 'jpg', 'jpeg', 'webp', 'bmp'].includes(ext) || !!currentJobData.metadata?.stage0;

      btnRunPipeline.disabled = true;
      btnRunPipeline.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang khởi chạy...';
      try {
        let res;
        // If image input, rerun full pipeline from Stage 0 with currently selected generator & details!
        if (isImg) {
          const form = new FormData();
          form.append('generator', selectedGenerator);
          form.append('mesh_detail', selectedDetail.mesh);
          form.append('texture_detail', selectedDetail.texture);
          res = await fetch(`/api/jobs/${currentJobData.id}/rerun`, { method: 'POST', body: form });
        } else {
          res = await fetch(`/api/jobs/${currentJobData.id}/continue_rigging`, { method: 'POST' });
        }

        if (res.ok) {
          smoothProgressVal = 10;
          startPolling(currentJobData.id);
          loadHistory();
        } else {
          const err = await res.json();
          alert(`Lỗi: ${err.detail || 'Không thể khởi chạy pipeline'}`);
        }
      } catch (e) {
        console.error("Error re-triggering pipeline", e);
        alert(`Lỗi: ${e.message}`);
      } finally {
        btnRunPipeline.disabled = false;
        btnRunPipeline.innerHTML = '<i class="fa-solid fa-bolt"></i> Chạy Full Pipeline';
      }
    };
  }

  // Stage tab clicks
  document.querySelectorAll('.stage-tab').forEach(tab => {
    tab.onclick = () => {
      const s = parseInt(tab.getAttribute('data-tab'));
      load3DForStage(s);
    };
  });

  // Stage stepper clicks
  document.querySelectorAll('.stage-step').forEach(card => {
    card.onclick = () => {
      const s = parseInt(card.getAttribute('data-step'));
      load3DForStage(s);
    };
  });

  // Continue rigging from Stage 0 button
  const btnContinue = document.getElementById('btn-continue-rigging');
  if (btnContinue) {
    btnContinue.onclick = async () => {
      if (!currentJobData) return;
      btnContinue.disabled = true;
      btnContinue.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang chạy Rigging...';
      try {
        const res = await fetch(`/api/jobs/${currentJobData.id}/continue_rigging`, { method: 'POST' });
        if (res.ok) {
          smoothProgressVal = 20;
          startPolling(currentJobData.id);
          loadHistory();
        } else {
          const err = await res.json();
          alert(`Lỗi: ${err.detail}`);
        }
      } catch (e) {
        alert(`Lỗi kích hoạt Rigging: ${e.message}`);
      } finally {
        btnContinue.innerHTML = '<i class="fa-solid fa-play"></i> Chạy tiếp Pipeline Rig & Animation';
      }
    };
  }


  // Banner overlay buttons
  const bannerBtnDl = document.getElementById('banner-btn-download-3d');
  if (bannerBtnDl) {
    bannerBtnDl.onclick = () => {
      if (currentJobData) window.open(`/api/jobs/${currentJobData.id}/files/generated_3d_glb`, '_blank');
    };
  }
  const bannerBtnContinue = document.getElementById('banner-btn-continue');
  if (bannerBtnContinue) {
    bannerBtnContinue.onclick = () => {
      if (btnContinue) btnContinue.click();
    };
  }

  // Download buttons
  const btnDownload3D = document.getElementById('btn-download-3d-stage0');
  if (btnDownload3D) {
    btnDownload3D.onclick = () => {
      if (currentJobData) {
        window.open(`/api/jobs/${currentJobData.id}/files/generated_3d_glb`, '_blank');
      }
    };
  }

  const btnPreviewDownload3D = document.getElementById('btn-preview-download-3d');
  if (btnPreviewDownload3D) {
    btnPreviewDownload3D.onclick = (e) => {
      e.stopPropagation();
      if (currentJobData) {
        window.open(`/api/jobs/${currentJobData.id}/files/generated_3d_glb`, '_blank');
      }
    };
  }

  const btnDownloadGlb = document.getElementById('btn-download-glb');
  if (btnDownloadGlb) {
    btnDownloadGlb.onclick = () => {
      if (!currentJobData) return;
      if (currentJobData.status === 'completed') {
        window.open(`/api/jobs/${currentJobData.id}/files/rigged_glb`, '_blank');
      } else {
        window.open(`/api/jobs/${currentJobData.id}/files/generated_3d_glb`, '_blank');
      }
    };
  }

  const btnDownloadObj = document.getElementById('btn-download-obj');
  if (btnDownloadObj) {
    btnDownloadObj.onclick = () => {
      if (currentJobData) window.open(`/api/jobs/${currentJobData.id}/files/skeleton_obj`, '_blank');
    };
  }

  // Eye Mask Visualizer
  let eyeMaskVisualizerActive = false;
  const cachedOriginalMaterials = new Map();

  async function applyEyeMaskHighlight(active) {
    if (!currentJobData) return;

    if (active && currentStage !== 4 && (currentJobData.status === 'completed' || currentJobData.stage >= 4)) {
      // Auto switch to Tab 4 Animation 3D which contains full ARKit morph targets
      await load3DForStage(4);
      await new Promise(r => setTimeout(r, 300));
    }

    if (!currentMeshGroup) return;

    let totalEyeVerts = 0;
    let eyeCenterSum = new THREE.Vector3(0, 0, 0);

    currentMeshGroup.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const geo = child.geometry;
        const count = geo.attributes.position.count;

        if (active) {
          if (!cachedOriginalMaterials.has(child)) {
            cachedOriginalMaterials.set(child, child.material);
          }

          const colors = new Float32Array(count * 3);
          const dict = child.morphTargetDictionary || {};
          const morphAttrs = geo.morphAttributes?.position || [];

          // Collect all eye-related morph targets
          const eyeIndices = [];
          Object.keys(dict).forEach(name => {
            if (name.toLowerCase().includes('eye') || name.toLowerCase().includes('blink') || name.toLowerCase().includes('squint')) {
              const idx = dict[name];
              if (idx !== undefined && idx < morphAttrs.length) {
                eyeIndices.push(idx);
              }
            }
          });

          const eyeWeights = new Float32Array(count);
          if (eyeIndices.length > 0) {
            eyeIndices.forEach(idx => {
              const attr = morphAttrs[idx];
              for (let i = 0; i < count; i++) {
                const dx = attr.getX(i);
                const dy = attr.getY(i);
                const dz = attr.getZ(i);
                const mag = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (mag > 0.00003) {
                  eyeWeights[i] = Math.max(eyeWeights[i], Math.min(1.0, mag * 100.0));
                }
              }
            });
          }

          for (let i = 0; i < count; i++) {
            const w = eyeWeights[i];
            if (w > 0.02) {
              totalEyeVerts++;
              const px = geo.attributes.position.getX(i);
              const py = geo.attributes.position.getY(i);
              const pz = geo.attributes.position.getZ(i);
              eyeCenterSum.add(new THREE.Vector3(px, py, pz));

              // Bright Radiant Electric Cyan/Neon Green (0, 255, 180)
              colors[i * 3] = 0.0;                       // R
              colors[i * 3 + 1] = 0.7 + 0.3 * w;         // G (glowing green/cyan)
              colors[i * 3 + 2] = 0.95;                  // B (electric cyan)
            } else {
              // Dim dark slate for the rest of the body
              colors[i * 3] = 0.12;
              colors[i * 3 + 1] = 0.15;
              colors[i * 3 + 2] = 0.20;
            }
          }

          geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
          geo.attributes.color.needsUpdate = true;

          child.material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.3,
            metalness: 0.1,
            skinning: !!child.isSkinnedMesh,
            morphTargets: morphAttrs.length > 0,
            wireframe: wireframeMode
          });
          child.material.needsUpdate = true;
        } else {
          // Restore original material
          if (cachedOriginalMaterials.has(child)) {
            child.material = cachedOriginalMaterials.get(child);
            cachedOriginalMaterials.delete(child);
            child.geometry.deleteAttribute('color');
            child.material.needsUpdate = true;
          }
        }
      }
    });

    if (active) {
      if (totalEyeVerts > 0) {
        const eyeCenter = eyeCenterSum.divideScalar(totalEyeVerts);
        controls.target.copy(eyeCenter);
        camera.position.set(eyeCenter.x, eyeCenter.y + 0.05, eyeCenter.z + 0.55);
        controls.update();
      }
      showToast('👁️ Đã bật Mask Vùng Mắt (Màu Xanh Neon). Hãy xem vùng mắt trên khuôn mặt!', 'success');
    } else {
      showToast('Đã tắt Mask Vùng Mắt, trở về chế độ hiển thị gốc.', 'info');
    }
  }

  function showToast(msg, type = 'info') {
    let toast = document.getElementById('playground-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'playground-toast';
      toast.style.cssText = 'position: absolute; bottom: 80px; left: 50%; transform: translateX(-50%); z-index: 1000; padding: 10px 20px; border-radius: 8px; font-size: 13px; font-weight: 600; color: #fff; box-shadow: 0 8px 24px rgba(0,0,0,0.5); backdrop-filter: blur(10px); transition: all 0.3s ease; pointer-events: none; text-align: center;';
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    if (type === 'success') {
      toast.style.background = 'rgba(16, 185, 129, 0.95)';
      toast.style.border = '1px solid rgba(16, 185, 129, 0.4)';
    } else {
      toast.style.background = 'rgba(15, 23, 42, 0.95)';
      toast.style.border = '1px solid rgba(59, 130, 246, 0.4)';
    }
    toast.style.opacity = '1';
    toast.style.display = 'block';
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => { toast.style.display = 'none'; }, 300);
    }, 3500);
  }

  const btnToggleEyeMask = document.getElementById('btn-toggle-eyemask');
  if (btnToggleEyeMask) {
    btnToggleEyeMask.onclick = async (e) => {
      eyeMaskVisualizerActive = !eyeMaskVisualizerActive;
      btnToggleEyeMask.classList.toggle('active', eyeMaskVisualizerActive);
      await applyEyeMaskHighlight(eyeMaskVisualizerActive);
    };
  }

  // Toolbar toggles
  document.getElementById('btn-toggle-wireframe').onclick = (e) => {
    wireframeMode = !wireframeMode;
    e.currentTarget.classList.toggle('active', wireframeMode);
    if (currentMeshGroup) {
      currentMeshGroup.traverse(c => { if (c.isMesh) c.material.wireframe = wireframeMode; });
    }
  };

  document.getElementById('btn-toggle-skeleton').onclick = (e) => {
    showSkeleton = !showSkeleton;
    e.currentTarget.classList.toggle('active', showSkeleton);
    if (skeletonGroup) skeletonGroup.visible = showSkeleton;
    if (animSkeletonHelper) animSkeletonHelper.visible = showSkeleton;
  };

  document.getElementById('btn-toggle-grid').onclick = (e) => {
    showGrid = !showGrid;
    e.currentTarget.classList.toggle('active', showGrid);
    const g = scene.getObjectByName('gridHelper');
    if (g) g.visible = showGrid;
  };

  document.getElementById('btn-reset-cam').onclick = () => {
    camera.position.set(0, 1.2, 3.2);
    controls.target.set(0, 0.5, 0);
  };

  // Animation bar
  const playBtn = document.getElementById('btn-anim-play');
  playBtn.onclick = () => {
    isPlaying = !isPlaying;
    playBtn.innerHTML = isPlaying ? '<i class="fa-solid fa-pause"></i>' : '<i class="fa-solid fa-play"></i>';
  };

  document.getElementById('anim-select').onchange = (e) => {
    playAnimation(e.target.value);
  };

  const speedSlider = document.getElementById('anim-speed');
  const speedLabel = document.getElementById('speed-label');
  speedSlider.oninput = () => {
    const spd = parseFloat(speedSlider.value);
    speedLabel.textContent = `${spd}x`;
    if (currentMixer) currentMixer.timeScale = spd;
  };

  document.getElementById('btn-refresh-jobs').onclick = () => loadHistory();
  document.getElementById('btn-reanimate-fast').onclick = async () => {
    if (!currentJobData) return;
    const btn = document.getElementById('btn-reanimate-fast');
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tạo...';
    try {
      const res = await fetch(`/api/jobs/${currentJobData.id}/reanimate`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Reanimation failed');
      }
      const data = await res.json();
      console.log('Reanimated job:', data);
      await fetchAndRenderJob(currentJobData.id);
      load3DForStage(4);
    } catch (e) {
      alert(`Lỗi tạo lại motion: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-person-running"></i> Tạo Motion (0.1s)';
    }
  };

  // Facial Expression Presets click handlers
  document.querySelectorAll('.expr-btn').forEach(btn => {
    btn.onclick = () => {
      const preset = btn.getAttribute('data-preset');
      if (preset) applyExpressionPreset(preset);
    };
  });

  // Accordion collapsible categories
  document.querySelectorAll('.bs-category-header').forEach(hdr => {
    hdr.onclick = () => {
      const parent = hdr.closest('.bs-category');
      if (parent) parent.classList.toggle('collapsed');
    };
  });
});
