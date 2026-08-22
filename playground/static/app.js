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
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0, 0.5, 0);

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambient);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x444455, 0.6);
  hemi.position.set(0, 10, 0);
  scene.add(hemi);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight1.position.set(5, 10, 7);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x00d2ff, 0.6);
  dirLight2.position.set(-5, -5, -5);
  scene.add(dirLight2);

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
  if (animSkeletonHelper) {
    animSkeletonHelper.update();
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
    if (data.lan_ips && data.lan_ips.length > 0) {
      lanElem.textContent = `LAN: ${data.lan_ips[0]}:${data.port}`;
      document.getElementById('lan-pill').onclick = () => {
        const url = `http://${data.lan_ips[0]}:${data.port}`;
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
      const stClass = j.status === 'completed' ? 'completed' : (j.status === 'failed' ? 'failed' : 'running');
      const timeStr = new Date(j.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      return `
        <div class="history-item ${isSel}" data-job-id="${j.id}">
          <div class="history-header">
            <div class="history-title">${j.title}</div>
            <span class="status-badge ${stClass}">${j.status}</span>
          </div>
          <div class="history-meta">
            <span><i class="fa-regular fa-clock"></i> ${timeStr}</span>
            <span><i class="fa-solid fa-bone"></i> ${j.num_bones} bones</span>
            <span><i class="fa-solid fa-stopwatch"></i> ${j.duration_sec ? j.duration_sec + 's' : '--'}</span>
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

// Fetch single job status & render appropriate stage
async function fetchAndRenderJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) throw new Error("Job not found");
    const job = await res.json();
    currentJobData = job;

    updateUIWithJob(job);

    // If job still processing, start polling
    if (job.status.startsWith('processing') || job.status === 'queued') {
      startPolling(jobId);
    } else {
      stopPolling();
      // Load 3D model for current stage tab
      load3DForStage(currentStage);
    }
  } catch (e) {
    console.error("Error fetching job", e);
  }
}

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (res.ok) {
      const job = await res.json();
      currentJobData = job;
      updateUIWithJob(job);
      if (job.status === 'completed' || job.status === 'failed') {
        stopPolling();
        loadHistory();
        load3DForStage(currentStage);
      }
    }
  }, 1500);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

// Update UI badges, steppers, and tree
function updateUIWithJob(job) {
  const badge = document.getElementById('global-status-badge');
  badge.style.display = 'block';
  badge.textContent = job.status;
  badge.className = `status-badge ${job.status === 'completed' ? 'completed' : (job.status === 'failed' ? 'failed' : 'running')}`;

  // Update stepper cards
  for (let s = 1; s <= 4; s++) {
    const card = document.getElementById(`step-card-${s}`);
    const check = card.querySelector('.step-check');
    if (job.stage >= s) {
      card.classList.add('completed');
      if (check) check.style.display = 'inline-block';
    } else {
      card.classList.remove('completed');
      if (check) check.style.display = 'none';
    }
  }

  // Update buttons
  const isDone = job.status === 'completed';
  document.getElementById('btn-download-glb').disabled = !isDone;
  document.getElementById('btn-download-obj').disabled = job.stage < 2;

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

  if (stage === 1 || stage === 2 || stage === 3) {
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
          if (stage === 1) {
            child.material = new THREE.MeshStandardMaterial({
              color: 0xcccccc,
              roughness: 0.4,
              metalness: 0.1,
              wireframe: wireframeMode
            });
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

      if (stage === 2 || stage === 3) {
        buildSkeletonVisualizer(currentJobData.metadata?.skel);
      }
      if (stage === 3) {
        loadBoneHeatmap(selectedBoneIndex);
      }
    });
  } else if (stage === 4) {
    // Load Final Rigged and Animated GLB
    const url = `/api/jobs/${currentJobData.id}/files/rigged_glb`;
    loader.load(url, (gltf) => {
      currentMeshGroup = gltf.scene;
      currentMeshGroup.position.set(0, 0, 0);
      currentMeshGroup.updateMatrixWorld(true);

      currentMeshGroup.traverse((child) => {
        if (child.isMesh) {
          child.material.wireframe = wireframeMode;
        }
      });

      scene.add(currentMeshGroup);

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

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  init3D();
  loadSystemInfo();
  loadHistory();

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

  // Custom File Upload
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  uploadZone.onclick = () => fileInput.click();

  fileInput.onchange = async (e) => {
    if (fileInput.files.length === 0) return;
    const file = fileInput.files[0];
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/jobs/upload', { method: 'POST', body: form });
    if (res.ok) {
      const job = await res.json();
      selectJob(job.id);
      loadHistory();
    }
  };

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

  // Download buttons
  document.getElementById('btn-download-glb').onclick = () => {
    if (currentJobData) window.open(`/api/jobs/${currentJobData.id}/files/rigged_glb`, '_blank');
  };
  document.getElementById('btn-download-obj').onclick = () => {
    if (currentJobData) window.open(`/api/jobs/${currentJobData.id}/files/skeleton_obj`, '_blank');
  };

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
  document.getElementById('btn-run-pipeline').onclick = () => {
    if (currentJobData) selectJob(currentJobData.id);
  };
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
});
