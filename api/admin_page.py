"""Owner admin page (served at /admin). The token is kept only in page memory."""

ADMIN_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex"><title>Thyroid ML System · Admin</title>
<style>
 body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f7fb;color:#1d2433}
 main{max-width:1000px;margin:24px auto;padding:0 16px;display:grid;gap:16px}
 .card{background:#fff;border:1px solid #e3e8f0;border-radius:12px;padding:16px 20px}
 h1{font-size:22px;margin:0} h2{font-size:17px;margin:0 0 10px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
 .metric{background:#f5f7fb;border-radius:8px;padding:10px} .metric b{display:block;font-size:20px}
 table{width:100%;border-collapse:collapse;font-size:13px} th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #eef1f6}
 input[type=password],input[type=text]{padding:8px 10px;border:1px solid #cfd6e4;border-radius:8px;width:100%;box-sizing:border-box}
 button{background:#2f6fec;color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer;margin-top:8px}
 button.danger{background:#d64545} .muted{color:#6b7488} .err{color:#c62828} .hidden{display:none}
 label.t{display:block;margin:6px 0}
</style></head><body><main>
<div class="card"><h1>🛠️ Thyroid ML System · Admin</h1><p class="muted">Owner only. Uses the ADMIN_TOKEN set in Render.
 Statistics live in memory and reset when the free server restarts. <a href="/">Back to the demo</a></p></div>

<div class="card" id="login"><h2>Unlock</h2>
 <input id="tok" type="password" placeholder="Admin token" autocomplete="off">
 <button id="unlock">Unlock</button> <span id="loginErr" class="err"></span></div>

<div id="dash" class="hidden">
 <div class="card"><h2>Overview</h2><div class="grid" id="metrics"></div></div>
 <div class="card"><h2>Predictions since restart</h2><table id="counters"></table></div>
 <div class="card"><h2>Endpoints</h2><table id="endpoints"></table></div>
 <div class="card"><h2>Logs</h2>
  <label><input type="radio" name="kind" value="events" checked> Events</label>
  <label><input type="radio" name="kind" value="requests"> Requests</label>
  <table id="logs"></table></div>
 <div class="card"><h2>Maintenance</h2>
  <label class="t">Banner shown on the demo page (empty = none)<input id="banner" type="text" maxlength="300"></label>
  <label class="t"><input id="tabular" type="checkbox"> Lab-value predictions on</label>
  <label class="t"><input id="batch" type="checkbox"> Batch CSV scoring on</label>
  <label class="t"><input id="image" type="checkbox"> Image predictions on</label>
  <button id="save">Save settings</button> <span id="saveMsg" class="muted"></span>
  <hr><button id="reset" class="danger">Reset statistics</button>
  <button id="lock" style="background:#555">Lock admin</button></div>
</div>
</main>
<script>
let TOKEN = "";
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function call(method, path, body) {
  const r = await fetch("/admin/api" + path, {method, headers: {"X-Admin-Token": TOKEN, "Content-Type": "application/json"},
                                               body: body ? JSON.stringify(body) : undefined});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || ("HTTP " + r.status));
  return j;
}
function table(el, rows) {
  if (!rows.length) { el.innerHTML = '<tr><td class="muted">Nothing yet</td></tr>'; return; }
  const cols = Object.keys(rows.reduce((a, r) => Object.assign(a, r), {}));
  el.innerHTML = "<tr>" + cols.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr>" +
    rows.map((r) => "<tr>" + cols.map((c) => `<td>${esc(c === "ts" ? new Date(r[c] * 1000).toISOString().replace("T", " ").slice(0, 19) : r[c])}</td>`).join("") + "</tr>").join("");
}
async function load() {
  const s = await call("GET", "/stats");
  const up = s.uptime_seconds;
  $("#metrics").innerHTML = [["Up for", `${Math.floor(up / 3600)}h ${Math.floor(up % 3600 / 60)}m`],
    ["Model version", s.model_version], ["Tabular models", s.tabular_models.length],
    ["Image model", s.image_model_loaded ? "loaded" : "not loaded yet"]]
    .map(([k, v]) => `<div class="metric">${esc(k)}<b>${esc(v)}</b></div>`).join("");
  table($("#counters"), Object.entries(s.counters).map(([k, v]) => ({counter: k, count: v})));
  table($("#endpoints"), s.endpoints);
  const kind = document.querySelector("input[name=kind]:checked").value;
  table($("#logs"), (await call("GET", `/logs?kind=${kind}&limit=150`)).items);
  const cfg = await call("GET", "/settings");
  $("#banner").value = cfg.maintenance_banner; $("#tabular").checked = cfg.tabular_enabled;
  $("#batch").checked = cfg.batch_enabled; $("#image").checked = cfg.image_enabled;
}
$("#unlock").onclick = async () => {
  TOKEN = $("#tok").value; $("#tok").value = "";
  try { await load(); $("#login").classList.add("hidden"); $("#dash").classList.remove("hidden"); $("#loginErr").textContent = ""; }
  catch (e) { TOKEN = ""; $("#loginErr").textContent = e.message; }
};
document.querySelectorAll("input[name=kind]").forEach((r) => r.onchange = load);
$("#save").onclick = async () => {
  try { await call("PUT", "/settings", {maintenance_banner: $("#banner").value, tabular_enabled: $("#tabular").checked,
        batch_enabled: $("#batch").checked, image_enabled: $("#image").checked}); $("#saveMsg").textContent = "Saved"; load(); }
  catch (e) { $("#saveMsg").textContent = e.message; }
};
$("#reset").onclick = async () => {
  if (prompt("Type RESET to clear usage statistics") !== "RESET") return;
  await call("POST", "/reset-stats?confirm=RESET"); load();
};
$("#lock").onclick = () => { TOKEN = ""; location.reload(); };
</script></body></html>
"""
