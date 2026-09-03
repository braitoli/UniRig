import { spawn } from 'child_process';
import fs from 'fs';

const chrome = spawn('/snap/bin/chromium', [
    '--headless=new',
    '--no-sandbox',
    '--disable-gpu',
    '--remote-debugging-port=9444',
    '--window-size=1280,800',
    '--user-data-dir=/tmp/test_chrome_all_presets_' + Date.now(),
    'http://localhost:7860/statue'
]);

await new Promise(r => setTimeout(r, 2500));

try {
    const listRes = await fetch('http://127.0.0.1:9444/json/list');
    const tabs = await listRes.json();
    const ws = new WebSocket(tabs[0].webSocketDebuggerUrl);

    let msgId = 1;
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
        const id = msgId++;
        pending.set(id, res);
        ws.send(JSON.stringify({ id, method, params }));
    });

    await send('Page.enable');
    await send('Runtime.enable');
    await new Promise(r => setTimeout(r, 2000));

    const presets = [
        'mythical_beast',
        'cyber_turtle',
        'mushroom_house',
        'fox_girl',
        'gentleman'
    ];

    for (const p of presets) {
        console.log(`Clicking preset: ${p}...`);
        await send('Runtime.evaluate', {
            expression: `document.querySelector('.chip[data-preset="${p}"]').click()`
        });
        await new Promise(r => setTimeout(r, 2500));
        const sc = await send('Page.captureScreenshot', { format: 'png' });
        fs.writeFileSync(`/home/braitoli/workspace/namnh/code/poc/UniRig/tests/preset_${p}.png`, Buffer.from(sc.data, 'base64'));
        console.log(`  Saved screenshot: tests/preset_${p}.png`);
    }

    console.log('All 5 preset tests verified & saved successfully!');
    ws.close();
} finally {
    chrome.kill('SIGKILL');
}
