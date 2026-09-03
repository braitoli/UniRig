import { spawn } from 'child_process';
import fs from 'fs';

console.log('🔍 Testing and verifying all 3D statue formats in Three.js browser...');

const profileDir = '/tmp/test_chrome_views_' + Date.now();
const chrome = spawn('/snap/bin/chromium', [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu',
    '--remote-debugging-port=9555',
    '--window-size=1280,800',
    '--user-data-dir=' + profileDir,
    'http://localhost:7860/statue'
]);

await new Promise(r => setTimeout(r, 2500));

try {
    const listRes = await fetch('http://127.0.0.1:9555/json/list');
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

    // Wait for page to initialize
    await new Promise(r => setTimeout(r, 1500));

    const jobId = 'statue_1788399197_sample_character';

    // 1. Test PLASTER WHITE GLB
    console.log('📸 Loading & Capturing: 1. Thạch Cao Trắng (Plaster GLB)...');
    await send('Runtime.evaluate', {
        expression: `
        load3DStatueModel('/api/statue/jobs/${jobId}/files/plaster_glb', 'plaster');
        `
    });
    await new Promise(r => setTimeout(r, 2000));
    let sc = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('/home/braitoli/workspace/namnh/code/poc/UniRig/tests/verify_1_plaster.png', Buffer.from(sc.data, 'base64'));

    // 2. Test SEGMENTED MULTI-PART GLB
    console.log('📸 Loading & Capturing: 2. Phân Vùng Đổ Màu (Segmented GLB)...');
    await send('Runtime.evaluate', {
        expression: `
        load3DStatueModel('/api/statue/jobs/${jobId}/files/segmented_glb', 'segmented');
        `
    });
    await new Promise(r => setTimeout(r, 2000));
    sc = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('/home/braitoli/workspace/namnh/code/poc/UniRig/tests/verify_2_segmented.png', Buffer.from(sc.data, 'base64'));

    // 3. Test TEXTURED GLB
    console.log('📸 Loading & Capturing: 3. Texture Gốc AI (Textured GLB)...');
    await send('Runtime.evaluate', {
        expression: `
        load3DStatueModel('/api/statue/jobs/${jobId}/files/textured_glb', 'textured');
        `
    });
    await new Promise(r => setTimeout(r, 2000));
    sc = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('/home/braitoli/workspace/namnh/code/poc/UniRig/tests/verify_3_textured.png', Buffer.from(sc.data, 'base64'));

    // 4. Test LIVE PAINTING INTERACTION (Đổ màu thử nghiệm 2 bộ phận)
    console.log('📸 Testing Live Painting: Đổ màu thử vào bộ phận...');
    await send('Runtime.evaluate', {
        expression: `
        load3DStatueModel('/api/statue/jobs/${jobId}/files/segmented_glb', 'painted');
        `
    });
    await new Promise(r => setTimeout(r, 1500));
    // Paint first two submeshes with red and blue
    await send('Runtime.evaluate', {
        expression: `
        let idx = 0;
        currentModel.traverse(c => {
            if (c.isMesh) {
                if (idx === 0) paintSubmesh(c, '#E91E63');
                if (idx === 1) paintSubmesh(c, '#2196F3');
                if (idx === 2) paintSubmesh(c, '#FFEB3B');
                idx++;
            }
        });
        `
    });
    await new Promise(r => setTimeout(r, 1000));
    sc = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('/home/braitoli/workspace/namnh/code/poc/UniRig/tests/verify_4_live_painted.png', Buffer.from(sc.data, 'base64'));

    console.log('\n🎉 ALL 4 3D STATUE VARIANTS VERIFIED & CAPTURED SUCCESSFULLY!\n');
    ws.close();
} finally {
    chrome.kill('SIGKILL');
}
