(function () {
  const $ = (id) => document.getElementById(id);
  const INK = '#12263a', TEAL = '#0e7c7b', RED = '#d1495b', GOLD = '#edae49', GRID = '#e1e8ee', MUTE = '#5b6f82';

  // ---- Synthetic data: 40 Pune flats (area in sq ft, price in Rs lakh) ----
  let seed = 7;
  const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const N = 40, A = [], P = [];
  for (let i = 0; i < N; i++) {
    const a = Math.round((500 + rnd() * 2000) / 10) * 10;
    A.push(a);
    P.push(Math.max(12, Math.round((0.04 * a + 10 + gauss() * 7) * 10) / 10));
  }
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const mu = mean(A), sd = Math.sqrt(mean(A.map((a) => (a - mu) ** 2)));

  const S = { scale: true, lr: 0.1, method: 'batch', w: 0, b: 0, sw: 0, sb: 0, step: 0, hist: [], path: [], play: false, max: 300, spf: 3, g: null };
  const feat = (a) => (S.scale ? (a - mu) / sd : a);
  const F = () => A.map(feat);
  const rmse = (w, b) => { const f = F(); let s = 0; for (let i = 0; i < N; i++) { const e = w * f[i] + b - P[i]; s += e * e; } return Math.sqrt(s / N); };
  const pred = (a) => S.w * feat(a) + S.b;

  function opt() {
    const f = F(), mf = mean(f), mp = mean(P);
    let c = 0, v = 0;
    for (let i = 0; i < N; i++) { c += (f[i] - mf) * (P[i] - mp); v += (f[i] - mf) ** 2; }
    const w = c / v;
    return { w, b: mp - w * mf };
  }

  // ---- Training ----
  function stepOnce() {
    const f = F();
    let idx = [];
    if (S.method === 'batch') for (let i = 0; i < N; i++) idx.push(i);
    else if (S.method === 'sgd') idx = [Math.floor(rnd() * N)];
    else for (let k = 0; k < 8; k++) idx.push(Math.floor(rnd() * N));
    let gw = 0, gb = 0;
    for (const i of idx) { const e = S.w * f[i] + S.b - P[i]; gw += e * f[i]; gb += e; }
    gw /= idx.length; gb /= idx.length;
    S.w -= S.lr * gw; S.b -= S.lr * gb; S.step++; S.g = [gw, gb];
    S.path.push([S.w, S.b]); S.hist.push(rmse(S.w, S.b));
  }

  function reset() {
    S.play = false; $('play').textContent = 'Play';
    S.w = S.sw; S.b = S.sb; S.step = 0; S.g = null;
    S.path = [[S.w, S.b]]; S.hist = [rmse(S.w, S.b)];
    render();
  }

  function status() {
    const r = S.hist[S.hist.length - 1], r0 = S.hist[0];
    if (!isFinite(r) || r > 1e6 || (S.step > 2 && r > r0 * 50)) return ['bad', 'Diverging: the learning rate is too high for this setup, so the error is growing instead of shrinking. Lower α or turn scaling on.'];
    const o = opt(), ro = rmse(o.w, o.b);
    if (r < ro * 1.02) return ['good', 'Converged: the model is within 2% of the best possible fit.'];
    if (S.step >= 100 && r > ro * 1.3) return ['warn', 'Crawling: the error is falling very slowly. Try a larger learning rate or turn feature scaling on.'];
    return ['', S.step ? 'Learning…' : 'Press Step or Play to begin.'];
  }

  // ---- Canvas helpers ----
  function fit(c) {
    const d = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
    c.width = r.width * d; c.height = r.height * d;
    const x = c.getContext('2d'); x.setTransform(d, 0, 0, d, 0, 0);
    x.font = '12px "Public Sans",sans-serif';
    return [x, r.width, r.height];
  }
  const line = (x, x1, y1, x2, y2, col, w) => { x.beginPath(); x.moveTo(x1, y1); x.lineTo(x2, y2); x.strokeStyle = col; x.lineWidth = w || 1; x.stroke(); };

  function drawScatter() {
    const [x, W, H] = fit($('c1')); const m = { l: 44, r: 12, t: 12, b: 36 };
    const X = (a) => m.l + (a - 400) / 2300 * (W - m.l - m.r), Y = (p) => H - m.b - p / 140 * (H - m.t - m.b);
    x.fillStyle = MUTE; x.textAlign = 'right';
    [0, 35, 70, 105, 140].forEach((p) => { line(x, m.l, Y(p), W - m.r, Y(p), GRID); x.fillText('₹' + p, m.l - 6, Y(p) + 4); });
    x.textAlign = 'center';
    [500, 1000, 1500, 2000, 2500].forEach((a) => { line(x, X(a), m.t, X(a), H - m.b, GRID); x.fillText(a, X(a), H - m.b + 16); });
    x.fillText('Area (sq ft)', (W + m.l) / 2, H - 4);
    x.save(); x.translate(11, H / 2); x.rotate(-Math.PI / 2); x.fillText('Price (₹ lakh)', 0, 0); x.restore();
    const ok = isFinite(S.w) && isFinite(S.b);
    x.save(); x.beginPath(); x.rect(m.l, m.t, W - m.l - m.r, H - m.t - m.b); x.clip();
    if (ok) for (let i = 0; i < N; i++) line(x, X(A[i]), Y(P[i]), X(A[i]), Y(pred(A[i])), 'rgba(209,73,91,.35)', 1);
    x.fillStyle = TEAL;
    for (let i = 0; i < N; i++) { x.beginPath(); x.arc(X(A[i]), Y(P[i]), 3.5, 0, 7); x.fill(); }
    if (ok) line(x, X(400), Y(pred(400)), X(2700), Y(pred(2700)), RED, 2.5);
    x.restore();
  }

  function drawLoss() {
    const [x, W, H] = fit($('c2')); const m = { l: 50, r: 12, t: 12, b: 34 };
    const o = opt(), lro = Math.log10(rmse(o.w, o.b));
    const h = S.hist.map((v) => Math.log10(Math.min(Math.max(v, 1e-3), 1e12)));
    let lo = Math.floor(Math.min(lro, ...h) * 2) / 2, hi = Math.ceil(Math.max(...h) * 2) / 2;
    if (hi - lo < 1) hi = lo + 1;
    const xm = Math.max(S.step, 50);
    const X = (i) => m.l + i / xm * (W - m.l - m.r), Y = (v) => H - m.b - (v - lo) / (hi - lo) * (H - m.t - m.b);
    x.fillStyle = MUTE; x.textAlign = 'right';
    for (let v = Math.ceil(lo); v <= hi; v++) { line(x, m.l, Y(v), W - m.r, Y(v), GRID); x.fillText('10^' + v, m.l - 6, Y(v) + 4); }
    x.textAlign = 'center'; x.fillText('Step', (W + m.l) / 2, H - 4);
    x.setLineDash([5, 4]); line(x, m.l, Y(lro), W - m.r, Y(lro), GOLD, 2); x.setLineDash([]);
    x.textAlign = 'left'; x.fillStyle = '#9a6b00'; x.fillText('best possible', m.l + 6, Y(lro) - 5);
    x.beginPath(); h.forEach((v, i) => (i ? x.lineTo(X(i), Y(v)) : x.moveTo(X(i), Y(v))));
    x.strokeStyle = TEAL; x.lineWidth = 2.5; x.stroke();
  }

  // contour map (cached per scaling mode)
  let G = null;
  function buildGrid() {
    const o = opt();
    const R = S.scale ? { w0: o.w - 30, w1: o.w + 30, b0: o.b - 70, b1: o.b + 70 } : { w0: o.w - 0.04, w1: o.w + 0.04, b0: o.b - 70, b1: o.b + 70 };
    const n = 140, c = document.createElement('canvas'); c.width = c.height = n;
    const g = c.getContext('2d'), im = g.createImageData(n, n), f = F();
    const val = new Float32Array(n * n); let mx = 0;
    const Lmin = Math.max(rmse(o.w, o.b), 1e-6);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const w = R.w0 + (i + .5) / n * (R.w1 - R.w0), b = R.b1 - (j + .5) / n * (R.b1 - R.b0);
      let s = 0; for (let k = 0; k < N; k++) { const e = w * f[k] + b - P[k]; s += e * e; }
      const t = Math.log10(Math.sqrt(s / N) / Lmin); val[j * n + i] = t; mx = Math.max(mx, t);
    }
    for (let p = 0; p < n * n; p++) {
      const t = val[p] / mx, band = Math.floor(t * 14), u = band / 14, k = p * 4, alt = band % 2 ? 7 : 0;
      im.data[k] = 231 + (22 - 231) * u - alt; im.data[k + 1] = 243 + (50 - 243) * u - alt; im.data[k + 2] = 241 + (79 - 241) * u - alt; im.data[k + 3] = 255;
    }
    g.putImageData(im, 0, 0);
    G = { c, R, scale: S.scale, o };
  }

  const cm = { l: 44, r: 12, t: 12, b: 36 };
  function mapXY(W, H) {
    const R = G.R;
    return [(w) => cm.l + (w - R.w0) / (R.w1 - R.w0) * (W - cm.l - cm.r), (b) => H - cm.b - (b - R.b0) / (R.b1 - R.b0) * (H - cm.t - cm.b)];
  }
  function drawMap() {
    if (!G || G.scale !== S.scale) buildGrid();
    const [x, W, H] = fit($('c3')), [X, Y] = mapXY(W, H);
    x.imageSmoothingEnabled = true;
    x.drawImage(G.c, cm.l, cm.t, W - cm.l - cm.r, H - cm.t - cm.b);
    x.save(); x.beginPath(); x.rect(cm.l, cm.t, W - cm.l - cm.r, H - cm.t - cm.b); x.clip();
    x.beginPath(); S.path.forEach((p, i) => (isFinite(p[0] + p[1]) ? (i ? x.lineTo(X(p[0]), Y(p[1])) : x.moveTo(X(p[0]), Y(p[1]))) : 0));
    x.strokeStyle = RED; x.lineWidth = 2; x.stroke();
    x.fillStyle = '#fff'; x.strokeStyle = INK; x.lineWidth = 2;
    x.beginPath(); x.arc(X(S.sw), Y(S.sb), 5, 0, 7); x.fill(); x.stroke();
    if (isFinite(S.w + S.b)) { x.fillStyle = RED; x.beginPath(); x.arc(X(S.w), Y(S.b), 5, 0, 7); x.fill(); }
    const ox = X(G.o.w), oy = Y(G.o.b);
    line(x, ox - 7, oy - 7, ox + 7, oy + 7, GOLD, 3); line(x, ox - 7, oy + 7, ox + 7, oy - 7, GOLD, 3);
    x.restore();
    x.fillStyle = MUTE; x.textAlign = 'center';
    x.fillText(S.scale ? 'w: price per standard-deviation of area' : 'w: ₹ lakh per sq ft', (W + cm.l) / 2, H - 6);
    x.save(); x.translate(11, H / 2); x.rotate(-Math.PI / 2); x.fillText('b: base price (₹ lakh)', 0, 0); x.restore();
  }

  // ---- Readouts ----
  const fmtLR = (v) => (v < 0.001 ? v.toExponential(1) : v.toFixed(3));
  function render() {
    drawScatter(); drawLoss(); drawMap();
    const wr = S.scale ? S.w / sd : S.w, br = S.scale ? S.b - S.w * mu / sd : S.b, ok = isFinite(wr + br);
    $('o1').textContent = ok ? '₹' + Math.round(wr * 1e5).toLocaleString('en-IN') : '–';
    $('o2').textContent = ok ? '₹' + br.toFixed(1) + ' L' : '–';
    const r = S.hist[S.hist.length - 1];
    $('o3').textContent = isFinite(r) && r < 1e6 ? '±₹' + r.toFixed(1) + ' L' : '–';
    $('o4').textContent = S.step;
    const a = parseFloat($('pa').value), pr = pred(a);
    $('pp').textContent = ok && isFinite(a) && Math.abs(pr) < 1e6 ? '₹' + pr.toFixed(1) + ' L' : '–';
    const st = status(); $('status').className = st[0]; $('status').textContent = st[1];
    $('gline').textContent = S.g ? 'Step ' + S.step + ':  ∂J/∂w = ' + S.g[0].toPrecision(3) + ',  ∂J/∂b = ' + S.g[1].toPrecision(3) + '   →   w ← w − α·∂J/∂w,  b ← b − α·∂J/∂b' : '';
  }

  function tick() {
    if (!S.play) return;
    for (let k = 0; k < S.spf && S.step < S.max; k++) stepOnce();
    render();
    if (S.step >= S.max || status()[0] === 'bad' || !isFinite(S.w + S.b)) { S.play = false; $('play').textContent = 'Play'; }
    else requestAnimationFrame(tick);
  }

  // ---- Controls ----
  function setLR(v) { S.lr = v; $('lr').value = Math.log10(v); $('lrv').textContent = fmtLR(v); }
  $('lr').addEventListener('input', (e) => { setLR(Math.pow(10, parseFloat(e.target.value))); });
  $('scale').addEventListener('change', (e) => { S.scale = e.target.value === '1'; S.sw = 0; S.sb = 0; setLR(S.scale ? 0.1 : 5e-7); G = null; reset(); });
  $('method').addEventListener('change', (e) => { S.method = e.target.value; reset(); });
  $('max').addEventListener('change', (e) => { S.max = Math.max(1, parseInt(e.target.value, 10) || 300); });
  $('spf').addEventListener('change', (e) => { S.spf = parseInt(e.target.value, 10); });
  $('step').addEventListener('click', () => { S.play = false; $('play').textContent = 'Play'; stepOnce(); render(); });
  $('play').addEventListener('click', () => {
    if (S.play) { S.play = false; $('play').textContent = 'Play'; return; }
    if (S.step >= S.max) reset();
    S.play = true; $('play').textContent = 'Pause'; requestAnimationFrame(tick);
  });
  $('reset').addEventListener('click', reset);
  $('pa').addEventListener('input', render);
  $('c3').addEventListener('click', (e) => {
    const r = e.target.getBoundingClientRect(), R = G.R;
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (px < cm.l || px > r.width - cm.r || py < cm.t || py > r.height - cm.b) return;
    S.sw = R.w0 + (px - cm.l) / (r.width - cm.l - cm.r) * (R.w1 - R.w0);
    S.sb = R.b0 + (r.height - cm.b - py) / (r.height - cm.t - cm.b) * (R.b1 - R.b0);
    reset();
  });
  window.addEventListener('resize', render);

  setLR(0.1); reset();
})();
