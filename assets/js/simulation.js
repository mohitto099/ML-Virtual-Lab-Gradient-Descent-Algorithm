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
    c.style.width = '100%'; c.style.height = '300px'; c.style.display = 'block';
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

  // ---- Loss surface data (cached per scaling mode) ----
  let G = null;
  const V3 = { az: -35, el: 28 };
  const col = (u) => 'hsl(' + (265 - 205 * u) + ',65%,' + (32 + 18 * u) + '%)';

  function buildGrid() {
    const o = opt();
    const R = S.scale ? { w0: o.w - 30, w1: o.w + 30, b0: o.b - 70, b1: o.b + 70 } : { w0: o.w - 0.04, w1: o.w + 0.04, b0: o.b - 70, b1: o.b + 70 };
    const n = 80, f = F(), val = new Float32Array(n * n), Lmin = Math.max(rmse(o.w, o.b), 1e-6);
    let mx = 0;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const w = R.w0 + i / (n - 1) * (R.w1 - R.w0), b = R.b1 - j / (n - 1) * (R.b1 - R.b0);
      let s = 0; for (let k = 0; k < N; k++) { const e = w * f[k] + b - P[k]; s += e * e; }
      const t = Math.log10(Math.sqrt(s / N) / Lmin); val[j * n + i] = t; if (t > mx) mx = t;
    }
    G = { n, val, R, mx, Lmin, scale: S.scale, o };
  }

  // marching squares: contour segments at level L, in grid units
  function segs(L) {
    const n = G.n, val = G.val, out = [];
    const lerp = (a, b) => (L - a) / (b - a);
    for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
      const a = val[j * n + i], b = val[j * n + i + 1], c = val[(j + 1) * n + i + 1], d = val[(j + 1) * n + i];
      const k = (a > L ? 8 : 0) + (b > L ? 4 : 0) + (c > L ? 2 : 0) + (d > L ? 1 : 0);
      if (k === 0 || k === 15) continue;
      const T = [i + lerp(a, b), j], Rt = [i + 1, j + lerp(b, c)], B = [i + lerp(d, c), j + 1], Lf = [i, j + lerp(a, d)];
      const E = { 1: [Lf, B], 2: [B, Rt], 3: [Lf, Rt], 4: [T, Rt], 5: [T, Lf, B, Rt], 6: [T, B], 7: [T, Lf], 8: [T, Lf], 9: [T, B], 10: [T, Rt, Lf, B], 11: [T, Rt], 12: [Lf, Rt], 13: [B, Rt], 14: [Lf, B] }[k];
      for (let q = 0; q < E.length; q += 2) out.push([E[q][0], E[q][1], E[q + 1][0], E[q + 1][1]]);
    }
    return out;
  }

  const uv = (p) => [(p[0] - G.R.w0) / (G.R.w1 - G.R.w0), (p[1] - G.R.b0) / (G.R.b1 - G.R.b0)];
  const zAt = (k) => { const v = S.hist[k]; const z = isFinite(v) ? Math.log10(Math.max(v, 1e-9) / G.Lmin) / G.mx : 1.05; return Math.min(Math.max(z, 0), 1.05); };

  function tracePath(x, proj) {
    let pen = false; x.beginPath();
    S.path.forEach((p, k) => {
      if (!isFinite(p[0] + p[1])) { pen = false; return; }
      const [u, v] = uv(p);
      if (u < -0.05 || u > 1.05 || v < -0.05 || v > 1.05) { pen = false; return; }
      const q = proj(u, v, zAt(k));
      if (pen) x.lineTo(q[0], q[1]); else x.moveTo(q[0], q[1]);
      pen = true;
    });
    x.strokeStyle = RED; x.lineWidth = 2.5; x.lineJoin = 'round'; x.stroke();
  }

  function markers(x, proj) {
    const o = uv([G.o.w, G.o.b]), q0 = proj(o[0], o[1], 0);
    line(x, q0[0] - 7, q0[1] - 7, q0[0] + 7, q0[1] + 7, GOLD, 3); line(x, q0[0] - 7, q0[1] + 7, q0[0] + 7, q0[1] - 7, GOLD, 3);
    const dot = (p, k, fill) => {
      if (!isFinite(p[0] + p[1])) return;
      const [u, v] = uv(p); if (u < 0 || u > 1 || v < 0 || v > 1) return;
      const q = proj(u, v, zAt(k)); x.fillStyle = fill; x.strokeStyle = INK; x.lineWidth = 2;
      x.beginPath(); x.arc(q[0], q[1], 5, 0, 7); x.fill(); x.stroke();
    };
    dot(S.path[0], 0, '#fff');
    if (S.path.length > 1) dot(S.path[S.path.length - 1], S.path.length - 1, RED);
  }

  function drawContour() {
    if (!G || G.scale !== S.scale) buildGrid();
    const [x, W, H] = fit($('c4')), m = { l: 44, r: 12, t: 12, b: 36 }, pw = W - m.l - m.r, ph = H - m.t - m.b, n = G.n;
    const px = (u) => m.l + u * pw, py = (v) => m.t + (1 - v) * ph, proj = (u, v) => [px(u), py(v)];
    x.fillStyle = '#f7fafc'; x.fillRect(m.l, m.t, pw, ph);
    for (let k = 1; k <= 14; k++) {
      x.strokeStyle = col(k / 15); x.lineWidth = 1.4; x.beginPath();
      for (const s of segs(G.mx * k / 15)) { x.moveTo(px(s[0] / (n - 1)), py(1 - s[1] / (n - 1))); x.lineTo(px(s[2] / (n - 1)), py(1 - s[3] / (n - 1))); }
      x.stroke();
    }
    x.save(); x.beginPath(); x.rect(m.l, m.t, pw, ph); x.clip();
    tracePath(x, proj); markers(x, proj);
    x.restore();
    x.strokeStyle = GRID; x.strokeRect(m.l, m.t, pw, ph);
    x.fillStyle = MUTE; x.textAlign = 'center';
    x.fillText(S.scale ? 'w: price per standard-deviation of area' : 'w: ₹ lakh per sq ft', m.l + pw / 2, H - 8);
    x.save(); x.translate(11, m.t + ph / 2); x.rotate(-Math.PI / 2); x.fillText('b: base price (₹ lakh)', 0, 0); x.restore();
  }

  function drawSurface() {
    if (!G || G.scale !== S.scale) buildGrid();
    const [x, W, H] = fit($('c3')), n = G.n, az = V3.az * Math.PI / 180, el = V3.el * Math.PI / 180;
    const sc = Math.min(W, H * 1.25) * 0.62, cx = W / 2, cy = H * 0.56;
    const proj = (u, v, z) => {
      const X = u - 0.5, Y = v - 0.5, Z = (z - 0.5) * 0.9;
      const a = X * Math.cos(az) - Y * Math.sin(az), d = X * Math.sin(az) + Y * Math.cos(az);
      return [cx + a * sc, cy - (Z * Math.cos(el) + d * Math.sin(el)) * sc];
    };
    const seg = (a, b, c, t, w) => { const p = proj(...a), q = proj(...b); line(x, p[0], p[1], q[0], q[1], c, w); if (t) { x.fillStyle = MUTE; x.textAlign = 'center'; x.fillText(t, q[0], q[1] - 6); } };
    const fl = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]];
    for (let i = 0; i < 4; i++) seg(fl[i], fl[i + 1], '#c5d2dd', '', 1);
    seg([0, 0, 0], [1, 0, 0], MUTE, 'w', 1.5); seg([0, 0, 0], [0, 1, 0], MUTE, 'b', 1.5); seg([0, 0, 0], [0, 0, 1], MUTE, 'error', 1.5);
    x.strokeStyle = 'rgba(18,38,58,.2)'; x.lineWidth = 1;
    for (let s = 0; s < n; s += 5) {
      x.beginPath();
      for (let i = 0; i < n; i++) { const q = proj(i / (n - 1), 1 - s / (n - 1), Math.min(G.val[s * n + i] / G.mx, 1)); if (i) x.lineTo(q[0], q[1]); else x.moveTo(q[0], q[1]); }
      x.stroke(); x.beginPath();
      for (let j = 0; j < n; j++) { const q = proj(s / (n - 1), 1 - j / (n - 1), Math.min(G.val[j * n + s] / G.mx, 1)); if (j) x.lineTo(q[0], q[1]); else x.moveTo(q[0], q[1]); }
      x.stroke();
    }
    for (let k = 1; k <= 14; k++) {
      x.strokeStyle = col(k / 15); x.lineWidth = 1.6; x.beginPath();
      for (const s of segs(G.mx * k / 15)) {
        const p = proj(s[0] / (n - 1), 1 - s[1] / (n - 1), k / 15), q = proj(s[2] / (n - 1), 1 - s[3] / (n - 1), k / 15);
        x.moveTo(p[0], p[1]); x.lineTo(q[0], q[1]);
      }
      x.stroke();
    }
    tracePath(x, proj); markers(x, proj);
  }

  // ---- Readouts ----
  const fmtLR = (v) => (v < 0.001 ? v.toExponential(1) : v.toFixed(3));
  function render() {
    drawScatter(); drawLoss(); drawSurface(); drawContour();
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
  $('c4').addEventListener('click', (e) => {
    const r = e.target.getBoundingClientRect(), R = G.R, l = 44, rt = 12, t = 12, b = 36;
    const px = e.clientX - r.left, py = e.clientY - r.top;
    if (px < l || px > r.width - rt || py < t || py > r.height - b) return;
    S.sw = R.w0 + (px - l) / (r.width - l - rt) * (R.w1 - R.w0);
    S.sb = R.b0 + (r.height - b - py) / (r.height - t - b) * (R.b1 - R.b0);
    reset();
  });
  let drag = null;
  $('c3').addEventListener('pointerdown', (e) => { drag = [e.clientX, e.clientY, V3.az, V3.el]; e.target.setPointerCapture(e.pointerId); });
  $('c3').addEventListener('pointermove', (e) => {
    if (!drag) return;
    V3.az = drag[2] + (e.clientX - drag[0]) * 0.5;
    V3.el = Math.max(5, Math.min(80, drag[3] + (e.clientY - drag[1]) * 0.3));
    render();
  });
  $('c3').addEventListener('pointerup', () => { drag = null; });
  window.addEventListener('resize', render);

  setLR(0.1); reset();
})();
