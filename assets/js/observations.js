(function () {
  const KEY = 'gdlab_runs', M = { batch: 'Batch', sgd: 'Stochastic', mini: 'Mini-batch' };
  const fmtLR = (v) => (v < 0.001 ? v.toExponential(1) : String(+v.toFixed(3)));
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { return []; } };
  function remark(r) {
    const noisy = r.method === 'sgd' ? ' The path is noisy because each step uses one house.' : '';
    if (r.res === 'Diverged') return 'Error exploded: α is too large for this setup.';
    if (r.res === 'Converged') return (r.steps <= 5 ? 'Jumped almost straight to the best fit.' : r.steps <= 80 ? 'Smooth, steady descent.' : 'Reached the best fit, but slowly.') + noisy;
    if (!r.scale) return 'Raw areas make b crawl. Turn feature scaling on.' + noisy;
    return (r.lr < 0.01 ? 'α is too small: the error falls very slowly.' : 'Needs more steps to settle.') + noisy;
  }
  function render() {
    const runs = load(), tb = document.querySelector('#tbl tbody');
    tb.innerHTML = '';
    runs.forEach((r, i) => {
      const tr = document.createElement('tr');
      const cells = [i + 1, r.scale ? 'On' : 'Off', M[r.method] || r.method, fmtLR(r.lr), r.steps, r.err === null ? 'exploded' : r.err.toFixed(1), r.res, remark(r)];
      cells.forEach((c, k) => { const td = document.createElement('td'); td.textContent = c; if (k === 6) td.style.fontWeight = 600, td.style.color = r.res === 'Converged' ? '#14633f' : r.res === 'Diverged' ? '#8e1f30' : '#7a5200'; tr.appendChild(td); });
      tb.appendChild(tr);
    });
    document.getElementById('empty').style.display = runs.length ? 'none' : 'block';
    document.getElementById('tbl').style.display = runs.length ? 'table' : 'none';
    document.getElementById('best').textContent = runs.length ? 'For this dataset, the best possible typical error is about ₹' + runs[0].best.toFixed(1) + ' lakh. A run is "Converged" when it gets within 2% of that.' : 'The best possible error for this dataset is about ₹6.7 lakh.';
  }
  document.getElementById('clear').addEventListener('click', () => { try { localStorage.removeItem(KEY); } catch (e) {} render(); });
  document.getElementById('csv').addEventListener('click', () => {
    const rows = [['Run', 'Scaling', 'Method', 'Alpha', 'Steps', 'Final error (Rs lakh)', 'Result', 'Remark']];
    load().forEach((r, i) => rows.push([i + 1, r.scale ? 'On' : 'Off', M[r.method], r.lr, r.steps, r.err === null ? 'exploded' : r.err.toFixed(2), r.res, remark(r)]));
    const csv = rows.map((r) => r.map((c) => '"' + String(c).replace(/"/g, '""') + '"').join(',')).join('\n');
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); a.download = 'gradient-descent-observations.csv'; a.click();
  });
  window.addEventListener('storage', render);
  render();
})();
