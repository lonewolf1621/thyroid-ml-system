"""Single-page demo UI served at "/" (plain HTML + fetch, no extra dependencies)."""

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thyroid ML System · Demo</title>
<style>
  :root { --ink:#1d2433; --muted:#5b6578; --line:#dfe3ea; --accent:#2457d6; --bg:#f6f7fa; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:var(--ink); background:var(--bg); }
  main { max-width: 980px; margin: 0 auto; padding: 32px 20px 60px; }
  h1 { font-size: 26px; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 8px; }
  .note { font-size: 13px; color: #8a4b00; background:#fff4e5; border:1px solid #ffd9a8; padding:8px 12px; border-radius:8px; margin: 12px 0 24px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  section { background:#fff; border:1px solid var(--line); border-radius:12px; padding:20px; }
  h2 { font-size: 17px; margin: 0 0 14px; }
  .fields { display:grid; grid-template-columns: repeat(3, 1fr); gap: 10px 12px; }
  label { display:flex; flex-direction:column; font-size:12px; color:var(--muted); gap:3px; }
  input, select { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; color:var(--ink); }
  .checks { display:grid; grid-template-columns: repeat(2, 1fr); gap: 4px 12px; margin-top: 12px; font-size:13px; }
  .checks label { flex-direction:row; align-items:center; gap:6px; color:var(--ink); font-size:13px; }
  button { margin-top: 16px; width:100%; padding:10px; font:inherit; font-weight:600; color:#fff; background:var(--accent); border:0; border-radius:8px; cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
  .result { margin-top: 14px; padding: 12px; border-radius: 8px; background: var(--bg); min-height: 48px; font-size:14px; }
  .big { font-size: 20px; font-weight: 700; }
  .bar { height: 8px; background: var(--line); border-radius: 4px; overflow: hidden; margin-top: 6px; }
  .bar > div { height: 100%; background: var(--accent); }
  .presets { display:flex; gap:8px; margin-bottom:12px; }
  .presets button { margin:0; width:auto; padding:5px 10px; font-size:12px; background:#eef2fb; color:var(--accent); }
  footer { margin-top: 24px; font-size: 13px; color: var(--muted); }
  a { color: var(--accent); }
</style>
</head>
<body>
<main>
  <h1>Thyroid ML System</h1>
  <p class="sub">Hypothyroidism risk from lab values (scikit-learn / XGBoost pipelines) and benign vs malignant
  classification of thyroid images (EfficientNet-B0), served with FastAPI.</p>
  <div class="note">Portfolio research demo. Not a medical device and not for clinical decisions.</div>

  <div class="grid">
    <section>
      <h2>Clinical data → hypothyroidism risk</h2>
      <div class="presets">
        <button type="button" onclick="preset('normal')">Example: normal labs</button>
        <button type="button" onclick="preset('hypo')">Example: hypothyroid labs</button>
      </div>
      <form id="tab">
        <div class="fields">
          <label>Age<input name="age" type="number" step="1" min="1" max="119" value="45" required></label>
          <label>Sex<select name="sex"><option>F</option><option>M</option></select></label>
          <label>Model<select name="model_name">
            <option>RandomForest</option><option>XGBoost</option><option>LogisticRegression</option></select></label>
          <label>TSH (mU/L)<input name="TSH" type="number" step="0.01" min="0" value="1.3" required></label>
          <label>T3 (nmol/L)<input name="T3" type="number" step="0.01" min="0" value="2.0" required></label>
          <label>TT4 (nmol/L)<input name="TT4" type="number" step="0.1" min="0" value="110" required></label>
          <label>T4U<input name="T4U" type="number" step="0.01" min="0" value="1.0" required></label>
          <label>FTI<input name="FTI" type="number" step="0.1" min="0" value="110" required></label>
          <label>Referral source<select name="referral_source">
            <option>other</option><option>SVI</option><option>SVHC</option><option>SVHD</option><option>STMW</option></select></label>
        </div>
        <div class="checks">
          <label><input type="checkbox" name="on_thyroxine">On thyroxine</label>
          <label><input type="checkbox" name="on_antithyroid_medication">On antithyroid medication</label>
          <label><input type="checkbox" name="thyroid_surgery">Thyroid surgery</label>
          <label><input type="checkbox" name="I131_treatment">I131 treatment</label>
          <label><input type="checkbox" name="query_hypothyroid">Suspected hypothyroid</label>
          <label><input type="checkbox" name="query_hyperthyroid">Suspected hyperthyroid</label>
          <label><input type="checkbox" name="goitre">Goitre</label>
          <label><input type="checkbox" name="pregnant">Pregnant</label>
          <label><input type="checkbox" name="sick">Sick</label>
          <label><input type="checkbox" name="lithium">Lithium</label>
        </div>
        <button id="tabBtn" type="submit">Predict</button>
      </form>
      <div class="result" id="tabOut">Result appears here.</div>
    </section>

    <section>
      <h2>Thyroid image → benign / malignant</h2>
      <form id="img">
        <label>Image (JPG or PNG)<input name="file" type="file" accept="image/*" required></label>
        <label><input type="checkbox" name="explain" checked>Show Grad-CAM heatmap (where the model looked)</label>
        <button id="imgBtn" type="submit">Classify image</button>
      </form>
      <div class="result" id="imgOut">The first image request loads the model, so it can take ~30 seconds.</div>
    </section>
  </div>

  <section>
    <h2>Batch: score a CSV of patients</h2>
    <p>One patient per row, up to 1000 rows. Start from the <a href="/predict/batch/template">CSV template</a>.</p>
    <form id="batch">
      <label>CSV file<input name="file" type="file" accept=".csv,text/csv" required></label>
      <button id="batchBtn" type="submit">Score CSV (RandomForest)</button>
    </form>
    <div class="result" id="batchOut">Results and a download link appear here.</div>
  </section>

  <footer>
    API docs: <a href="/docs">/docs</a> ·
    Code: <a href="https://github.com/lonewolf1621/thyroid-ml-system">github.com/lonewolf1621/thyroid-ml-system</a>
  </footer>
</main>
<script>
const BOOLS = ["on_thyroxine","on_antithyroid_medication","thyroid_surgery","I131_treatment",
  "query_hypothyroid","query_hyperthyroid","goitre","pregnant","sick","lithium"];
const PRESETS = {
  normal: {age:45, TSH:1.3, T3:2.0, TT4:110, T4U:1.0, FTI:110},
  hypo:   {age:55, TSH:60,  T3:0.5, TT4:40,  T4U:1.0, FTI:40}
};
function preset(k){ const f=document.getElementById('tab'); for (const [n,v] of Object.entries(PRESETS[k])) f.elements[n].value=v; }
function pct(x){ return (x*100).toFixed(1) + '%'; }
function bar(x){ return '<div class="bar"><div style="width:' + Math.round(x*100) + '%"></div></div>'; }

document.getElementById('tab').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target, out = document.getElementById('tabOut'), btn = document.getElementById('tabBtn');
  const body = { age:+f.age.value, sex:f.sex.value, referral_source:f.referral_source.value,
    TSH:+f.TSH.value, T3:+f.T3.value, TT4:+f.TT4.value, T4U:+f.T4U.value, FTI:+f.FTI.value };
  for (const b of BOOLS) body[b] = f.elements[b].checked ? 't' : 'f';
  btn.disabled = true; out.textContent = 'Predicting…';
  try {
    const r = await fetch('/predict?model_name=' + encodeURIComponent(f.model_name.value),
      {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail));
    out.innerHTML = '<div class="big">' + (j.prediction === 'hypothyroid' ? 'Hypothyroid' : 'Negative') + '</div>'
      + 'Probability of hypothyroidism: <b>' + pct(j.probability_hypothyroid) + '</b>' + bar(j.probability_hypothyroid)
      + '<div style="margin-top:6px;color:#5b6578">' + j.model + ' · model ' + j.model_version + '</div>';
  } catch (err) { out.textContent = 'Error: ' + err.message; }
  btn.disabled = false;
});

document.getElementById('img').addEventListener('submit', async (e) => {
  e.preventDefault();
  const out = document.getElementById('imgOut'), btn = document.getElementById('imgBtn');
  const fd = new FormData(); fd.append('file', e.target.file.files[0]);
  const explain = e.target.explain.checked;
  btn.disabled = true; out.textContent = 'Classifying… (first request loads the model)';
  try {
    const r = await fetch('/predict-image' + (explain ? '?explain=true' : ''), {method:'POST', body: fd});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || 'Request failed');
    out.innerHTML = '<div class="big">' + j.prediction[0].toUpperCase() + j.prediction.slice(1) + '</div>'
      + 'Confidence: <b>' + pct(j.confidence) + '</b>' + bar(j.confidence)
      + '<div style="margin-top:6px;color:#5b6578">' + j.model + '</div>'
      + (j.gradcam_png_base64 ? '<img alt="Grad-CAM heatmap" style="max-width:100%;margin-top:10px;border-radius:6px" src="data:image/png;base64,'
         + j.gradcam_png_base64 + '"><div style="color:#5b6578;font-size:13px">' + j.gradcam_note + '</div>' : '');
  } catch (err) { out.textContent = 'Error: ' + err.message; }
  btn.disabled = false;
});

document.getElementById('batch').addEventListener('submit', async (e) => {
  e.preventDefault();
  const out = document.getElementById('batchOut'), btn = document.getElementById('batchBtn');
  const file = e.target.file.files[0];
  btn.disabled = true; out.textContent = 'Scoring…';
  try {
    const fd = new FormData(); fd.append('file', file);
    const r = await fetch('/predict/batch', {method:'POST', body: fd});
    const j = await r.json();
    if (!r.ok) throw new Error(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail));
    const fd2 = new FormData(); fd2.append('file', file);
    const csv = await (await fetch('/predict/batch?output=csv', {method:'POST', body: fd2})).blob();
    const url = URL.createObjectURL(csv);
    out.innerHTML = '<div class="big">' + j.scored + ' scored · ' + j.hypothyroid + ' flagged hypothyroid</div>'
      + (j.errors.length ? '<div>' + j.errors.length + ' row(s) had errors, e.g. row ' + j.errors[0].row + ': ' + j.errors[0].errors[0] + '</div>' : '')
      + '<a download="thyroid_predictions.csv" href="' + url + '">Download predictions (CSV)</a>';
  } catch (err) { out.textContent = 'Error: ' + err.message; }
  btn.disabled = false;
});
</script>
</body>
</html>
"""
