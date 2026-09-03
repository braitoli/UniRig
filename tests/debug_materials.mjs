import { spawn } from 'child_process';

const chrome = spawn('/snap/bin/chromium', [
    '--headless=new', '--no-sandbox', '--disable-gpu', '--remote-debugging-port=9444',
    'http://localhost:7860/statue'
]);
await new Promise(r => setTimeout(r, 2000));
const listRes = await fetch('http://127.0.0.1:9444/json/list');
const tabs = await listRes.json();
const tab = tabs.find(t => t.url.includes('statue')) || tabs[0];
const ws = new WebSocket(tab.webSocketDebuggerUrl);

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

await send('Runtime.enable');
await new Promise(r => setTimeout(r, 1500));

// Click rabbit
await send('Runtime.evaluate', { expression: `document.querySelector('.chip[data-preset="rabbit"]').click()` });
await new Promise(r => setTimeout(r, 3000));

// Inspect
const res = await send('Runtime.evaluate', {
    expression: `
    (() => {
        const meshes = [];
        if (!currentModel) return { error: 'No currentModel' };
        currentModel.traverse(c => {
            if (c.isMesh) {
                meshes.push({
                    name: c.name,
                    hasNormals: !!c.geometry.attributes.normal,
                    matType: c.material?.type,
                    color: c.material?.color ? '#' + c.material.color.getHexString() : null,
                    roughness: c.material?.roughness,
                    metalness: c.material?.metalness,
                    map: !!c.material?.map,
                    visible: c.visible
                });
            }
        });
        return {
            meshCount: meshes.length,
            meshes,
            sceneLights: scene.children.filter(c => c.isLight).map(l => ({ type: l.type, color: l.color.getHexString(), intensity: l.intensity }))
        };
    })()
    `,
    returnByValue: true
});

console.log('Result:', JSON.stringify(res.result.value, null, 2));
chrome.kill();
