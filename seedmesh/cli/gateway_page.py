"""The gateway's single page.

Kept as a Python string rather than a data file so `seedmesh gateway` works from any install
without packaging concerns, and self-contained so it renders with no network at all -- the
whole point is a machine that may be deep behind a NAT.

The layout puts provenance beside the answer on purpose. A chat box that only shows text
would be a nicer way to use the swarm and a worse way to understand it, and understanding it
is what distinguishes this from pointing at someone else's API.
"""

PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seedmesh</title>
<style>
 :root { color-scheme: light dark; --fg:#141414; --dim:#666; --line:#ddd; --bg:#fff;
         --panel:#f7f7f8; --accent:#2b6; --bad:#d44; }
 @media (prefers-color-scheme: dark) {
   :root { --fg:#e9e9e9; --dim:#9a9a9a; --line:#333; --bg:#151515; --panel:#1d1d1d; } }
 * { box-sizing:border-box; }
 body { margin:0; font:15px/1.55 system-ui,-apple-system,sans-serif; color:var(--fg);
        background:var(--bg); display:flex; height:100vh; overflow:hidden; }
 main { flex:1; display:flex; flex-direction:column; min-width:0; }
 aside { width:22rem; border-left:1px solid var(--line); background:var(--panel);
         overflow-y:auto; padding:1rem; font-size:.85rem; }
 @media (max-width:56rem) { body { flex-direction:column; height:auto; overflow:auto; }
   aside { width:auto; border-left:0; border-top:1px solid var(--line); } }
 header { padding:.8rem 1.2rem; border-bottom:1px solid var(--line); display:flex;
          align-items:baseline; gap:.6rem; }
 header b { font-size:1rem; } header span { color:var(--dim); font-size:.8rem; }
 #log { flex:1; overflow-y:auto; padding:1.2rem; }
 .turn { margin-bottom:1.4rem; }
 .me { font-weight:600; margin-bottom:.35rem; }
 .reply { white-space:pre-wrap; }
 .meta { color:var(--dim); font-size:.78rem; margin-top:.4rem; }
 .err { color:var(--bad); }
 form { display:flex; gap:.5rem; padding:1rem 1.2rem; border-top:1px solid var(--line); }
 input,button { font:inherit; padding:.6rem .8rem; border:1px solid var(--line);
                border-radius:6px; background:var(--bg); color:var(--fg); }
 input { flex:1; } button { cursor:pointer; }
 button:disabled { opacity:.5; cursor:default; }
 h2 { font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; color:var(--dim);
      margin:1.2rem 0 .4rem; }
 h2:first-child { margin-top:0; }
 .row { display:flex; justify-content:space-between; gap:.5rem; padding:.15rem 0; }
 .pill { display:inline-block; padding:.1rem .45rem; border-radius:4px; font-size:.75rem; }
 .ok { background:#2b62; } .warn { background:#fa02; } .no { background:#d442; }
 .mono { font-family:ui-monospace,monospace; font-size:.78rem; }
 .note { color:var(--dim); font-size:.75rem; margin-top:.3rem; }
</style>

<main>
  <header><b>Seedmesh</b><span id="model"></span></header>
  <div id="log"></div>
  <form id="f"><input id="p" placeholder="Ask the swarm..." autocomplete="off" autofocus>
    <button id="send">Send</button></form>
</main>
<aside id="side"></aside>

<script>
const TOKEN = new URLSearchParams(location.search).get('token') || '';
const H = {'Content-Type':'application/json','X-Seedmesh-Token':TOKEN};
const log = document.getElementById('log'), side = document.getElementById('side');
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function turn(entry) {
  const d = document.createElement('div');
  d.className = 'turn';
  const body = entry.ok
    ? `<div class="reply">${esc(entry.reply)}</div>`
    : `<div class="reply err">${esc(entry.error)}${entry.detail ? ': ' + esc(entry.detail) : ''}</div>`;
  d.innerHTML = `<div class="me">${esc(entry.prompt)}</div>${body}` +
    `<div class="meta">${(entry.elapsed_ms/1000).toFixed(1)}s</div>`;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
}

document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const input = document.getElementById('p'), btn = document.getElementById('send');
  const prompt = input.value.trim(); if (!prompt) return;
  input.value = ''; btn.disabled = true; input.disabled = true;
  try {
    const r = await fetch('/api/chat', {method:'POST', headers:H,
      body: JSON.stringify({prompt})});
    turn(await r.json());
  } catch (err) {
    turn({ok:false, prompt, error:'could not reach the gateway', elapsed_ms:0});
  }
  btn.disabled = false; input.disabled = false; input.focus(); refresh();
};

function serverRow(s) {
  // Self-reported and observed are kept apart here for the same reason as in the CLI
  // monitor: blending them would launder a claim into a measurement.
  const observed = s.observers
    ? `<span class="pill ok">${s.reliability.toFixed(2)} / ${s.observers} obs</span>`
    : `<span class="pill warn">untested</span>`;
  return `<div class="row"><span>${esc(s.label)} <span class="mono">${esc(s.blocks)}</span></span>${observed}</div>`
       + `<div class="note mono">${esc(s.profile)}${s.throughput ? ' &middot; ' + s.throughput.toFixed(0) + ' tok/s claimed' : ''}</div>`;
}

async function refresh() {
  let s; try { s = await (await fetch('/api/status', {headers:H})).json(); } catch { return; }
  document.getElementById('model').textContent = s.model || '';
  let h = '';

  if (s.swarm) {
    const cov = s.swarm.covered
      ? `<span class="pill ok">usable &middot; ${s.swarm.min_replicas}x</span>`
      : `<span class="pill no">${s.swarm.missing.length} block(s) unhosted</span>`;
    h += `<h2>Swarm</h2><div class="row"><span>${s.swarm.blocks} blocks</span>${cov}</div>`;
    h += (s.swarm.servers || []).map(serverRow).join('');
    if (!s.swarm.covered)
      h += `<div class="note">A model needs every block hosted. More servers on blocks that are already covered do not help.</div>`;
  } else if (s.swarm_error) {
    h += `<h2>Swarm</h2><div class="note err">${esc(s.swarm_error)}</div>`;
  }

  if (s.measured) {
    h += `<h2>What this client measured</h2>`;
    h += `<div class="row"><span>requests</span><span>${s.measured.requests}</span></div>`;
    h += `<div class="row"><span>failed</span><span>${s.measured.failed}</span></div>`;
    h += `<div class="row"><span>servers seen</span><span>${s.measured.servers}</span></div>`;
  }

  if (s.verification) {
    const v = s.verification;
    h += `<h2>Verification</h2>`;
    h += `<div class="row"><span>checked</span><span>${v.checked}</span></div>`;
    for (const [k, n] of Object.entries(v.verdicts || {}))
      h += `<div class="row"><span>${esc(k)}</span><span>${n}</span></div>`;
    if (v.skipped) {
      h += `<div class="row"><span>skipped</span><span class="pill warn">${v.skipped}</span></div>`;
      // Said plainly. A swarm that cannot verify must not read as one that verified and
      // found nothing wrong.
      h += `<div class="note">${v.skipped_unlocated
        ? 'Other servers host these blocks, but none has a direct address &mdash; a relayed peer&rsquo;s address belongs to the relay, so it cannot be shown to be a different operator.'
        : 'No independent server hosts the same blocks, so nothing could be double-checked.'}</div>`;
    }
    if (!v.checked && !v.skipped)
      h += `<div class="note">Nothing sampled yet.</div>`;
  }

  if (s.excluded && s.excluded.length) {
    h += `<h2>Excluded from routing</h2>`;
    h += s.excluded.map(p => `<div class="row mono">...${esc(p.slice(-8))}</div>`).join('');
    h += `<div class="note">Corroborated evidence of incorrect work. Slow or unreliable servers are never excluded &mdash; only proven-wrong ones.</div>`;
  }
  side.innerHTML = h;
}

refresh(); setInterval(refresh, 15000);
</script>
"""
