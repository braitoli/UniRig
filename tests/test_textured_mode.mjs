import { spawn } from 'child_process';
import fs from 'fs';

const chrome = spawn('/snap/bin/chromium', [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu',
    '--remote-debugging-port=9666',
    '--window-size=1280,800',
    '--user-data-dir=/tmp/test_chrome_textured_' + Date.now(),
    'http://localhost:7860/statue'
]);

await new Promise(r => setTimeout(r, 2500));

try {
    const listRes = await fetch('http://127.0.0.1:9666/json/list');
    const tabs = await listRes.json();
    const statueTab = tabs.find(t => t.url.includes('statue')) || tabs[0];
    const ws = new WebSocket(statueTab.webSocketDebuggerUrl);

    let id = 1;
    const pending = new Map();
    ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.id && pending.has(d.id)) {
            pending.get(d.id)(d.result);
            pending.delete(d.id);
        }
    };
    await new Promise(r => ws.onopen = r);

    const send = (method, params = {}) => new Promise(res => {
        const msgId = id++;
        pending.set(msgId, res);
        ws.send(JSON.stringify({ id, method, params }));
    });

    await send('Page.enable');
    await send('Runtime.enable');
    await new Promise(r => setTimeout(r, 2000));

    console.log('1. Click preset cyber_turtle (Rùa Máy Cyber)...');
    await send('Runtime.evaluate', {
        expression: `document.querySelector('.chip[data-preset="cyber_turtle"]').click()`
    });
    await new Promise(r => setTimeout(r, 3000));

    console.log('2. Click mode-btn textured (Texture AI gốc)...');
    await send('Runtime.evaluate', {
        expression: `document.querySelector('.mode-btn[data-mode="textured"]').click()`
    });
    await new Promise(r => setTimeout(r, 3500));

    const state = await send('Runtime.evaluate', {
        expression: `({
            activeMode: state.activeMode,
            currentGlbUrl: state.currentGlbUrl,
            loaderDisplay: document.getElementById('canvas-loader').style.display,
            loaderText: document.getElementById('canvas-loader-text').innerText,
            hasModel: !!currentModel
        })`,
        returnByValue: true
    });
    console.log('State in Textured Mode:', JSON.stringify(state.result.value, null, 2));

    const sc = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('/home/braitoli/workspace/namnh/code/poc/UniRig/tests/verify_textured_mode_success.png', Buffer.from(sc.data, 'base64'));
    console.log('✅ Screenshot saved to tests/verify_textured_mode_success.png');

    console.log('3. Switch back to Segmented Mode...');
    await send('Runtime.evaluate', {
        expression: `document.querySelector('.mode-btn[data-mode="segmented"]').click()`
    });
    await new Promise(r => setTimeout(r, 3000));

    const stateSeg = await send('Runtime.evaluate', {
        expression: `({
            activeMode: state.activeMode,
            currentGlbUrl: state.currentGlbUrl,
            loaderDisplay: document.getElementById('canvas-loader').style.display,
            hasModel: !!currentModel
        })`,
        returnByValue: true
    });
    console.log('State in Segmented Mode:', JSON.stringify(stateSeg.result.value, null, 2));

    ws.close();
} finally {
    chrome.kill('SIGKILL');
}
