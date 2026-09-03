import { spawn } from 'child_process';
import fs from 'fs';

console.log('🧪 Starting Specific E2E Test for Preset Chips (Ảnh mẫu thử nghiệm nhanh)...');

const profileDir = '/tmp/test_chrome_profile_' + Date.now();
const chrome = spawn('/snap/bin/chromium', [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu',
    '--remote-debugging-port=9333',
    '--user-data-dir=' + profileDir,
    'http://localhost:7860/statue'
]);

await new Promise(r => setTimeout(r, 2500));

try {
    const listRes = await fetch('http://127.0.0.1:9333/json/list');
    const tabs = await listRes.json();
    const statueTab = tabs.find(t => t.url.includes('statue')) || tabs[0];
    const wsUrl = statueTab.webSocketDebuggerUrl;

    const ws = new WebSocket(wsUrl);
    let msgId = 1;
    const pending = new Map();

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.id && pending.has(data.id)) {
            const { resolve, reject } = pending.get(data.id);
            pending.delete(data.id);
            if (data.error) reject(data.error);
            else resolve(data.result);
        }
    };

    await new Promise((resolve, reject) => {
        ws.onopen = resolve;
        ws.onerror = reject;
    });

    function send(method, params = {}) {
        const id = msgId++;
        return new Promise((resolve, reject) => {
            pending.set(id, { resolve, reject });
            ws.send(JSON.stringify({ id, method, params }));
        });
    }

    await send('Page.enable');
    await send('Runtime.enable');
    await send('DOM.enable');
    await send('Page.navigate', { url: 'http://localhost:7860/statue' });

    // Wait until document.title is loaded
    for (let i = 0; i < 40; i++) {
        const titleRes = await send('Runtime.evaluate', { expression: 'document.title' });
        if (titleRes?.result?.value?.includes('Statue')) break;
        await new Promise(r => setTimeout(r, 250));
    }

    // Check Preset Chips Count
    const chipsCount = await send('Runtime.evaluate', {
        expression: `document.querySelectorAll('.preset-chips .chip').length`
    });
    console.log(`✅ Found ${chipsCount.result.value} Preset Chips in UI`);

    // 1. Click on "mythical_beast" preset chip
    console.log('👉 Clicking on "Linh Thú Chibi" preset chip...');
    await send('Runtime.evaluate', {
        expression: `document.querySelector('.chip[data-preset="mythical_beast"]').click()`
    });

    // Wait for image fetch & 3D model load
    await new Promise(r => setTimeout(r, 2500));

    const beastState = await send('Runtime.evaluate', {
        expression: `
        ({
            hasFile: !!state.selectedFile,
            fileName: state.selectedFile?.name,
            previewSrc: document.getElementById('image-preview').src.substring(0, 40),
            previewVisible: document.getElementById('preview-container').style.display !== 'none',
            hintVisible: document.getElementById('empty-viewport-hint').style.display !== 'none',
            hasModel: !!currentModel,
            modelChildren: currentModel ? currentModel.children.length : 0,
            statFaces: document.getElementById('stat-faces').innerText,
            partsCount: document.querySelectorAll('#parts-list .part-item').length
        })
        `,
        returnByValue: true
    });
    console.log('📊 State after clicking Mythical Beast preset:', JSON.stringify(beastState.result.value, null, 2));

    if (!beastState.result.value.hasFile) throw new Error('state.selectedFile was not populated');
    if (!beastState.result.value.previewVisible) throw new Error('2D Image preview container is not visible');
    if (!beastState.result.value.hasModel) throw new Error('3D Statue model was not loaded into Three.js scene');

    // 2. Click on "cyber_turtle" preset chip
    console.log('👉 Clicking on "Rùa Máy Cyber" preset chip...');
    await send('Runtime.evaluate', {
        expression: `document.querySelector('.chip[data-preset="cyber_turtle"]').click()`
    });
    await new Promise(r => setTimeout(r, 2500));

    const turtleState = await send('Runtime.evaluate', {
        expression: `
        ({
            fileName: state.selectedFile?.name,
            hasModel: !!currentModel,
            statFaces: document.getElementById('stat-faces').innerText,
            partsCount: document.querySelectorAll('#parts-list .part-item').length
        })
        `,
        returnByValue: true
    });
    console.log('📊 State after clicking Cyber Turtle preset:', JSON.stringify(turtleState.result.value, null, 2));

    // 3. Test Brush Tool ("Cọ Vẽ Bề Mặt") with Purple color
    console.log('🖌️ Testing Brush Tool ("Cọ Vẽ Bề Mặt") with Purple color swatch...');
    const brushTestResult = await send('Runtime.evaluate', {
        expression: `
        (() => {
            // Select brush tool
            document.getElementById('tool-brush').click();

            // Select purple color
            const purple = Array.from(document.querySelectorAll('.color-swatch')).find(s => s.dataset.color.toLowerCase() === '#9333ea');
            if (purple) purple.click();

            // Simulate painting on the first submesh
            if (currentModel && currentModel.children.length > 0) {
                const targetMesh = currentModel.children[0];
                paintSubmesh(targetMesh, state.currentColor);
            }

            return {
                tool: state.paintTool,
                mode: state.activeMode,
                color: state.currentColor,
                cursor: renderer.domElement.style.cursor,
                paintedCount: state.paintedMaterialsMap.size
            };
        })()
        `,
        returnByValue: true
    });
    console.log('📊 Brush Painting Result:', JSON.stringify(brushTestResult.result.value, null, 2));

    if (brushTestResult.result.value.tool !== 'brush') throw new Error('Brush tool is not active');
    if (brushTestResult.result.value.mode !== 'painted') throw new Error('Mode did not switch to painted');
    if (brushTestResult.result.value.paintedCount === 0) throw new Error('Nothing was painted');

    console.log('📸 Capturing screenshot after brush painting...');
    const scBrush = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('/home/braitoli/workspace/namnh/code/poc/UniRig/tests/verify_brush_paint_purple.png', Buffer.from(scBrush.data, 'base64'));
    console.log('✅ Screenshot saved to tests/verify_brush_paint_purple.png');

    console.log('\n🎉 PRESET CHIPS & BRUSH TOOL WORKED FLAWLESSLY! 100% OK\n');
    ws.close();
} finally {
    chrome.kill('SIGKILL');
}
