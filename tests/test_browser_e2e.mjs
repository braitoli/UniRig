import { spawn } from 'child_process';
import fs from 'fs';

console.log('🚀 Starting Browser E2E Test on UniRig 3D Statue Studio...');

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

    console.log('🔌 Connected to Chrome DevTools Protocol WebSocket for tab:', statueTab.title || statueTab.url);
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
    let title = '';
    for (let i = 0; i < 40; i++) {
        const titleRes = await send('Runtime.evaluate', { expression: 'document.title' });
        title = titleRes?.result?.value || '';
        if (title.includes('Statue')) break;
        await new Promise(r => setTimeout(r, 250));
    }
    console.log('✅ Page Title Loaded:', title);
    if (!title.includes('UniRig') || !title.includes('Statue')) {
        throw new Error('Title does not match expected Statue Studio title. Got: ' + title);
    }

    // Wait for JS initialization & palette population
    for (let i = 0; i < 30; i++) {
        const swCount = await send('Runtime.evaluate', {
            expression: "document.querySelectorAll('.color-swatch').length"
        });
        if (swCount?.result?.value >= 10) break;
        await new Promise(r => setTimeout(r, 200));
    }

    // 2. Evaluate DOM elements existence
    const checkElements = await send('Runtime.evaluate', {
        expression: `
        ({
            dropZone: !!document.getElementById('drop-zone'),
            generatorCards: document.querySelectorAll('.radio-card').length,
            targetFacesSelect: !!document.getElementById('target-faces-select'),
            pedestalSelect: !!document.getElementById('pedestal-shape-select'),
            btnGenerate: !!document.getElementById('btn-generate-statue'),
            canvasContainer: !!document.getElementById('canvas-container'),
            paintTools: document.querySelectorAll('.paint-tool-btn').length,
            paletteColors: document.querySelectorAll('.color-swatch').length,
            viewModes: document.querySelectorAll('.mode-btn').length,
            automationBtn: !!document.getElementById('btn-open-auto-modal'),
            downloadPlaster: !!document.getElementById('dl-plaster')
        })
        `,
        returnByValue: true
    });

    console.log('📊 DOM UI Components Checked:', JSON.stringify(checkElements.result.value, null, 2));

    if (checkElements.result.value.paletteColors < 10) {
        throw new Error('Color palette was not populated properly');
    }
    if (checkElements.result.value.paintTools < 3) {
        throw new Error('Painting tools are missing');
    }

    // 3. Test Clicking on Brush Tool
    console.log('🖌️ Testing Interaction: Clicking Brush Tool...');
    await send('Runtime.evaluate', { expression: `document.getElementById('tool-brush').click()` });
    const brushActive = await send('Runtime.evaluate', {
        expression: `document.getElementById('tool-brush').classList.contains('active')`
    });
    console.log('✅ Brush Tool Active state:', brushActive.result.value);
    if (!brushActive.result.value) throw new Error('Brush tool did not become active');

    // 4. Test Clicking on Bucket Tool
    console.log('🪣 Testing Interaction: Clicking Bucket Tool...');
    await send('Runtime.evaluate', { expression: `document.getElementById('tool-bucket').click()` });
    const bucketActive = await send('Runtime.evaluate', {
        expression: `document.getElementById('tool-bucket').classList.contains('active')`
    });
    console.log('✅ Bucket Tool Active state:', bucketActive.result.value);

    // 5. Test Opening Automation Modal
    console.log('🤖 Testing Interaction: Opening Automation & Webhook Modal...');
    await send('Runtime.evaluate', { expression: `document.getElementById('btn-open-auto-modal').click()` });
    await new Promise(r => setTimeout(r, 600));

    const modalVisible = await send('Runtime.evaluate', {
        expression: `document.getElementById('automation-modal').style.display !== 'none'`
    });
    console.log('✅ Automation Modal Visible:', modalVisible.result.value);
    if (!modalVisible.result.value) throw new Error('Automation modal failed to open');

    // 6. Test Webhook Ping Test Button
    console.log('📡 Testing Webhook Ping Test from UI with httpbin.org ...');
    await send('Runtime.evaluate', {
        expression: `
        document.getElementById('auto-webhook-url').value = 'https://httpbin.org/post';
        document.getElementById('btn-test-webhook').click();
        `
    });
    
    // Poll for ping result
    let pingResultText = '';
    for (let i = 0; i < 30; i++) {
        await new Promise(r => setTimeout(r, 500));
        const res = await send('Runtime.evaluate', {
            expression: `document.getElementById('webhook-test-result').innerText`
        });
        pingResultText = res?.result?.value || '';
        if (pingResultText && !pingResultText.includes('Đang ping')) {
            break;
        }
    }

    console.log('✅ Webhook Ping Test UI Result:', pingResultText);
    if (!pingResultText.includes('Thành công')) {
        throw new Error('Webhook test did not show success in UI. Got: ' + pingResultText);
    }

    // 7. Close Modal
    await send('Runtime.evaluate', { expression: `document.getElementById('btn-close-auto-modal').click()` });
    await new Promise(r => setTimeout(r, 400));

    // 8. Capture screenshot
    console.log('📸 Capturing E2E Browser Screenshot...');
    const screenshot = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('/home/braitoli/workspace/namnh/code/poc/UniRig/tests/e2e_browser_test.png', Buffer.from(screenshot.data, 'base64'));
    console.log('✅ Screenshot saved to tests/e2e_browser_test.png');

    console.log('\n🎉 ALL BROWSER E2E TESTS PASSED 100% SUCCESSFULLY!\n');
    ws.close();
} finally {
    chrome.kill('SIGKILL');
}
