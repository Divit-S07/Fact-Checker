/* ─── FactGuard AI — Main Application Logic ─── */

const API_BASE = 'http://localhost:8000';

/* ══════════════════════════════════════════
   State
══════════════════════════════════════════ */
const state = {
  currentPanel: 'fact-check',
  selectedTopic: 'Technology',
  lastDigestMarkdown: '',
  lastDigestFilename: '',
  isLoading: false,
};

/* ══════════════════════════════════════════
   Utilities
══════════════════════════════════════════ */
function toast(msg, type = 'info', duration = 3500) {
  const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${icons[type] || 'ℹ️'}</span> ${msg}`;
  document.getElementById('toast-container').appendChild(t);
  setTimeout(() => {
    t.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function verdictClass(verdict) {
  const map = { True: 'verdict-true', False: 'verdict-false', Misleading: 'verdict-misleading', Unverifiable: 'verdict-unverifiable' };
  return map[verdict] || 'verdict-unverifiable';
}

function verdictEmoji(verdict) {
  const map = { True: '✅', False: '❌', Misleading: '⚠️', Unverifiable: '❓' };
  return map[verdict] || '❓';
}

function credClass(cred) {
  if (cred >= 8) return 'cred-high';
  if (cred >= 6) return 'cred-mid';
  return 'cred-low';
}

function animateBars() {
  // Animate confidence fills
  setTimeout(() => {
    document.querySelectorAll('.confidence-fill[data-width]').forEach(el => {
      el.style.width = el.dataset.width;
    });
    document.querySelectorAll('.cred-fill[data-width]').forEach(el => {
      el.style.width = el.dataset.width;
    });
    document.querySelectorAll('.dist-fill[data-width]').forEach(el => {
      el.style.width = el.dataset.width;
    });
  }, 100);
}

/* ══════════════════════════════════════════
   Navigation
══════════════════════════════════════════ */
function switchPanel(panelId) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`panel-${panelId}`)?.classList.add('active');
  document.getElementById(`nav-${panelId}`)?.classList.add('active');
  state.currentPanel = panelId;

  if (panelId === 'history') loadHistory();
  if (panelId === 'sources') loadSources();
  if (panelId === 'stats') loadStats();
}

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => switchPanel(btn.dataset.panel));
});

/* ══════════════════════════════════════════
   API Health Check
══════════════════════════════════════════ */
async function checkHealth() {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  try {
    await apiFetch('/health');
    dot.className = 'status-dot online';
    txt.textContent = 'API Online';
  } catch {
    dot.className = 'status-dot offline';
    txt.textContent = 'API Offline';
  }
}

checkHealth();
setInterval(checkHealth, 30000);

/* ══════════════════════════════════════════
   Claim Textarea
══════════════════════════════════════════ */
const claimInput = document.getElementById('claim-input');
const charCount = document.getElementById('char-count');

claimInput.addEventListener('input', () => {
  const len = claimInput.value.length;
  charCount.textContent = `${len} / 2000`;
  if (len > 1900) charCount.style.color = 'var(--amber)';
  else if (len > 2000) charCount.style.color = 'var(--red)';
  else charCount.style.color = 'var(--text-muted)';
  claimInput.value = claimInput.value.slice(0, 2000);
});

document.querySelectorAll('.example-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    claimInput.value = btn.dataset.text;
    claimInput.dispatchEvent(new Event('input'));
    claimInput.focus();
  });
});

/* ══════════════════════════════════════════
   Pipeline Progress Animation
══════════════════════════════════════════ */
const NODES = ['extract', 'search', 'judge', 'critic', 'digest'];
const LABELS = [
  'Extracting claims from text...',
  'Searching Tavily + RAG archive for evidence...',
  'Weighing evidence and forming verdict...',
  'Critic reviewing verdict for bias...',
  'Compiling digest and saving...'
];

let progTimer = null;

function startProgress(progressId, labelId) {
  const progress = document.getElementById(progressId);
  progress.style.display = 'block';
  let step = 0;

  NODES.forEach(n => {
    const node = document.getElementById(`prog-${n}`);
    if (node) node.className = 'prog-node';
  });

  function advance() {
    if (step < NODES.length) {
      const node = document.getElementById(`prog-${NODES[step]}`);
      if (node) node.classList.add('active');
      const lbl = document.getElementById(labelId);
      if (lbl) lbl.textContent = LABELS[step];
      step++;
      progTimer = setTimeout(advance, 2800);
    }
  }

  advance();
}

function stopProgress(progressId) {
  if (progTimer) clearTimeout(progTimer);
  // Mark all done
  NODES.forEach(n => {
    const node = document.getElementById(`prog-${n}`);
    if (node) { node.className = 'prog-node done'; }
  });
  setTimeout(() => {
    const progress = document.getElementById(progressId);
    if (progress) progress.style.display = 'none';
  }, 600);
}

/* ══════════════════════════════════════════
   Claim Card Builder
══════════════════════════════════════════ */
function buildClaimCard(result, index) {
  const verdict = result.verdict || 'Unverifiable';
  const confidence = result.confidence || 0;
  const pct = Math.round(confidence * 100);
  const vc = verdictClass(verdict);
  const ve = verdictEmoji(verdict);
  const claim = result.claim || '';
  const summary = result.summary || 'No summary available.';
  const evFor = result.evidence_for || [];
  const evAgainst = result.evidence_against || [];
  const sources = result.sources || [];

  const sourcesHTML = sources.slice(0, 6).map(s => {
    const cred = s.credibility || 5;
    return `
      <div class="source-item">
        <span class="source-cred ${credClass(cred)}">${cred}/10</span>
        <a href="${s.url || '#'}" class="source-link" target="_blank" rel="noopener">${s.title || s.url || 'Source'}</a>
      </div>`;
  }).join('');

  const evForHTML = evFor.slice(0, 3).map(e => `<div class="evidence-item for">${String(e).substring(0, 250)}</div>`).join('');
  const evAgainstHTML = evAgainst.slice(0, 3).map(e => `<div class="evidence-item against">${String(e).substring(0, 250)}</div>`).join('');

  const card = document.createElement('div');
  card.className = 'claim-card';
  card.dataset.verdict = verdict;

  card.innerHTML = `
    <div class="claim-card-header" onclick="toggleCard(this.closest('.claim-card'))">
      <div class="verdict-badge ${vc}">${ve} ${verdict.toUpperCase()}</div>
      <div class="claim-text">${claim.substring(0, 150)}${claim.length > 150 ? '…' : ''}</div>
      <div class="confidence-ring">
        <span class="confidence-value">${pct}%</span>
        <span class="confidence-label">confidence</span>
      </div>
      <svg class="expand-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    </div>
    <div class="claim-card-body">
      <div class="card-section">
        <div class="section-title">Summary</div>
        <div class="summary-text">${summary}</div>
      </div>
      <div class="card-section">
        <div class="section-title">Confidence</div>
        <div class="confidence-bar">
          <div class="confidence-fill" data-width="${pct}%" style="width:0%"></div>
        </div>
      </div>
      ${evFor.length ? `
        <div class="card-section">
          <div class="section-title">✅ Supporting Evidence</div>
          <div class="evidence-list">${evForHTML}</div>
        </div>` : ''}
      ${evAgainst.length ? `
        <div class="card-section">
          <div class="section-title">❌ Counter Evidence</div>
          <div class="evidence-list">${evAgainstHTML}</div>
        </div>` : ''}
      ${sources.length ? `
        <div class="card-section">
          <div class="section-title">📚 Sources</div>
          <div class="sources-list">${sourcesHTML}</div>
        </div>` : ''}
      ${result.critique ? `
        <div class="card-section">
          <div class="section-title">🎯 Editor's Note</div>
          <div class="summary-text" style="border-left-color: var(--purple);">${result.critique}</div>
        </div>` : ''}
    </div>`;

  return card;
}

function toggleCard(card) {
  card.classList.toggle('expanded');
  if (card.classList.contains('expanded')) {
    animateBars();
  }
}

/* ══════════════════════════════════════════
   Fact-Check
══════════════════════════════════════════ */
document.getElementById('fact-check-btn').addEventListener('click', async () => {
  const text = claimInput.value.trim();
  if (!text) { toast('Please enter a claim or text to fact-check.', 'warning'); return; }
  if (state.isLoading) return;

  state.isLoading = true;
  const btn = document.getElementById('fact-check-btn');
  btn.disabled = true;
  btn.textContent = 'Processing...';

  document.getElementById('fact-check-results').innerHTML = '';
  startProgress('pipeline-progress', 'progress-label');

  try {
    const data = await apiFetch('/api/fact-check', {
      method: 'POST',
      body: JSON.stringify({ text, save_result: true })
    });

    stopProgress('pipeline-progress');

    const container = document.getElementById('fact-check-results');
    container.innerHTML = '';

    if (!data.claim_results || data.claim_results.length === 0) {
      container.innerHTML = '<div class="empty-state"><div class="empty-icon">🤔</div><p>No verifiable claims found in the input.</p></div>';
      return;
    }

    // Header summary
    const summary = document.createElement('div');
    summary.className = 'digest-result-card';
    const counts = {};
    data.claim_results.forEach(r => { const v = r.verdict || 'Unverifiable'; counts[v] = (counts[v] || 0) + 1; });
    summary.innerHTML = `
      <div class="digest-header">
        <div>
          <div class="digest-title">🔍 Fact-Check Results</div>
          <div class="digest-meta">${data.claim_results.length} claim(s) analyzed • ${new Date().toLocaleString()}</div>
        </div>
      </div>
      <div class="verdict-summary">
        ${Object.entries(counts).map(([v, c]) => `
          <div class="verdict-pill vp-${v.toLowerCase()}">
            <span class="count">${c}</span> ${verdictEmoji(v)} ${v}
          </div>`).join('')}
      </div>
      ${data.export_path ? `
        <div class="export-bar">
          📄 Report exported → <code style="font-family:monospace;font-size:0.8em">${data.export_path}</code>
          <button class="btn-secondary" style="margin-left:auto" onclick="showMarkdownModal()">Preview</button>
        </div>` : ''}`;
    container.appendChild(summary);

    // Store markdown for preview
    state.lastDigestMarkdown = data.digest_markdown || '';
    state.lastDigestFilename = (data.export_path || '').split('\\').pop().split('/').pop();

    // Individual claim cards
    data.claim_results.forEach((result, i) => {
      const card = buildClaimCard(result, i);
      card.style.animationDelay = `${i * 0.08}s`;
      container.appendChild(card);
    });

    animateBars();
    toast(`Fact-check complete! ${data.claim_results.length} claim(s) verified.`, 'success');

  } catch (err) {
    stopProgress('pipeline-progress');
    document.getElementById('fact-check-results').innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">💥</div>
        <p>Pipeline error: ${err.message}</p>
      </div>`;
    toast(`Error: ${err.message}`, 'error', 5000);
  } finally {
    state.isLoading = false;
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg> Fact-Check Now`;
  }
});

/* ══════════════════════════════════════════
   Daily Digest
══════════════════════════════════════════ */
document.querySelectorAll('.topic-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.topic-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if (btn.dataset.topic === 'custom') {
      document.getElementById('custom-topic-input').style.display = 'block';
      state.selectedTopic = '';
    } else {
      document.getElementById('custom-topic-input').style.display = 'none';
      state.selectedTopic = btn.dataset.topic;
    }
  });
});

document.getElementById('custom-topic-field').addEventListener('input', (e) => {
  state.selectedTopic = e.target.value.trim();
});

document.getElementById('digest-btn').addEventListener('click', async () => {
  const topic = state.selectedTopic || document.getElementById('custom-topic-field').value.trim() || 'Technology';
  if (!topic) { toast('Please enter or select a topic.', 'warning'); return; }
  if (state.isLoading) return;

  state.isLoading = true;
  const btn = document.getElementById('digest-btn');
  btn.disabled = true;
  btn.textContent = 'Generating...';

  const exportMd = document.getElementById('export-md-toggle').checked;
  document.getElementById('digest-results').innerHTML = '';
  document.getElementById('digest-progress').style.display = 'block';
  document.getElementById('digest-progress-label').textContent = `Generating ${topic} digest…`;

  try {
    const data = await apiFetch('/api/daily-digest', {
      method: 'POST',
      body: JSON.stringify({ topic, export_md: exportMd })
    });

    document.getElementById('digest-progress').style.display = 'none';

    const container = document.getElementById('digest-results');
    container.innerHTML = '';

    const counts = {};
    (data.claim_results || []).forEach(r => { const v = r.verdict || 'Unverifiable'; counts[v] = (counts[v] || 0) + 1; });

    const wrapper = document.createElement('div');
    wrapper.className = 'digest-result-card';
    wrapper.innerHTML = `
      <div class="digest-header">
        <div>
          <div class="digest-title">📰 ${topic} Daily Digest</div>
          <div class="digest-meta">${new Date().toLocaleDateString('en-US', {weekday:'long', year:'numeric', month:'long', day:'numeric'})}</div>
        </div>
      </div>
      <div class="verdict-summary">
        ${Object.entries(counts).map(([v, c]) => `
          <div class="verdict-pill vp-${v.toLowerCase()}">
            <span class="count">${c}</span> ${verdictEmoji(v)} ${v}
          </div>`).join('')}
      </div>
      ${data.export_path ? `
        <div class="export-bar">
          📄 Markdown exported → <code style="font-family:monospace;font-size:0.8em">${data.export_path.split(/[\\\/]/).pop()}</code>
          <button class="btn-secondary" style="margin-left:auto" onclick="showMarkdownModal()">Preview MD</button>
        </div>` : ''}`;
    container.appendChild(wrapper);

    state.lastDigestMarkdown = data.digest_markdown || '';
    state.lastDigestFilename = (data.export_path || '').split(/[\\\/]/).pop();

    (data.claim_results || []).forEach((result, i) => {
      const card = buildClaimCard(result, i);
      card.style.animationDelay = `${i * 0.07}s`;
      container.appendChild(card);
    });

    animateBars();
    toast(`${topic} digest generated with ${data.claim_results?.length || 0} stories!`, 'success');

  } catch (err) {
    document.getElementById('digest-progress').style.display = 'none';
    document.getElementById('digest-results').innerHTML = `
      <div class="empty-state"><div class="empty-icon">💥</div><p>Error: ${err.message}</p></div>`;
    toast(`Error: ${err.message}`, 'error', 5000);
  } finally {
    state.isLoading = false;
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10l4 4v12a2 2 0 01-2 2z"/></svg> Generate Digest`;
  }
});

/* ══════════════════════════════════════════
   History Panel
══════════════════════════════════════════ */
async function loadHistory(params = {}) {
  const container = document.getElementById('history-results');
  container.innerHTML = '<div class="empty-state"><div class="spinner" style="margin:0 auto 1rem"></div><p>Loading history...</p></div>';

  try {
    const qs = new URLSearchParams();
    if (params.verdict) qs.set('verdict', params.verdict);
    if (params.topic) qs.set('topic', params.topic);
    if (params.start_date) qs.set('start_date', params.start_date);
    if (params.end_date) qs.set('end_date', params.end_date);
    qs.set('limit', '50');

    const data = await apiFetch(`/api/saved-claims?${qs}`);
    const claims = data.claims || [];

    if (!claims.length) {
      container.innerHTML = '<div class="empty-state"><div class="empty-icon">📋</div><p>No claims match your filters. Start fact-checking!</p></div>';
      return;
    }

    container.innerHTML = `<p style="font-size:0.8rem;color:var(--text-muted);margin-bottom:1rem">${data.total} claim(s) found</p>`;
    claims.slice().reverse().forEach((result, i) => {
      const card = buildClaimCard(result, i);
      card.style.animationDelay = `${i * 0.04}s`;
      container.appendChild(card);
    });
    animateBars();

  } catch (err) {
    container.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>Failed to load: ${err.message}</p></div>`;
  }
}

document.getElementById('filter-apply').addEventListener('click', () => {
  loadHistory({
    verdict: document.getElementById('filter-verdict').value,
    topic: document.getElementById('filter-topic').value,
    start_date: document.getElementById('filter-start').value,
    end_date: document.getElementById('filter-end').value,
  });
});

document.getElementById('filter-reset').addEventListener('click', () => {
  document.getElementById('filter-verdict').value = '';
  document.getElementById('filter-topic').value = '';
  document.getElementById('filter-start').value = '';
  document.getElementById('filter-end').value = '';
  loadHistory();
});

document.getElementById('filter-week').addEventListener('click', () => {
  const end = new Date();
  const start = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  const iso = d => d.toISOString().slice(0, 10);
  document.getElementById('filter-start').value = iso(start);
  document.getElementById('filter-end').value = iso(end);
  document.getElementById('filter-verdict').value = '';
  document.getElementById('filter-topic').value = '';
  loadHistory({ start_date: iso(start), end_date: iso(end) });
});

/* ══════════════════════════════════════════
   Trusted Sources Panel
══════════════════════════════════════════ */
async function loadSources() {
  const grid = document.getElementById('sources-grid');
  grid.innerHTML = '<div class="empty-state"><div class="spinner" style="margin:0 auto 1rem"></div><p>Loading sources...</p></div>';

  try {
    const data = await apiFetch('/api/trusted-sources');
    const sources = data.sources || {};
    grid.innerHTML = '';

    const typeColors = {
      'fact-checker': '#4f8ef7',
      'news': '#22d3ee',
      'health': '#4ade80',
      'science': '#a855f7',
    };

    Object.entries(sources).sort((a, b) => b[1].credibility - a[1].credibility).forEach(([domain, info], i) => {
      const card = document.createElement('div');
      card.className = 'source-card';
      card.style.animationDelay = `${i * 0.04}s`;
      const score = info.credibility || 5;
      const pct = (score / 10 * 100);
      const typeColor = typeColors[info.type] || '#6b7280';

      card.innerHTML = `
        <div class="source-card-header">
          <span class="source-domain">${domain}</span>
          <div class="credibility-score">
            <span class="cred-score-val" style="color:${score >= 8 ? 'var(--green)' : score >= 6 ? 'var(--amber)' : 'var(--red)'}">${score}</span>
            <span class="cred-max">/10</span>
          </div>
        </div>
        <span class="source-type-badge" style="background:${typeColor}20;color:${typeColor}">${info.type}</span>
        <div class="source-notes">${info.notes}</div>
        <div class="cred-bar"><div class="cred-fill" data-width="${pct}%" style="width:0%"></div></div>`;
      grid.appendChild(card);
    });

    animateBars();
  } catch (err) {
    grid.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>Failed to load: ${err.message}</p></div>`;
  }
}

/* ══════════════════════════════════════════
   Stats Panel
══════════════════════════════════════════ */
async function loadStats() {
  const area = document.getElementById('stats-area');
  area.innerHTML = '<div class="empty-state"><div class="spinner" style="margin:0 auto 1rem"></div><p>Loading stats...</p></div>';

  try {
    const data = await apiFetch('/api/stats');
    area.innerHTML = '';

    const statCards = [
      { icon: '🔍', value: data.total_claims_checked || 0, label: 'Claims Checked' },
      { icon: '📰', value: data.total_digests || 0, label: 'Digests Generated' },
      { icon: '📄', value: data.exports_count || 0, label: 'Reports Exported' },
    ];

    statCards.forEach((s, i) => {
      const card = document.createElement('div');
      card.className = 'stat-card';
      card.style.animationDelay = `${i * 0.1}s`;
      card.innerHTML = `
        <div class="stat-icon">${s.icon}</div>
        <div class="stat-value">${s.value}</div>
        <div class="stat-label">${s.label}</div>`;
      area.appendChild(card);
    });

    // Verdict distribution
    const dist = data.verdict_distribution || {};
    if (Object.keys(dist).length) {
      const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
      const distCard = document.createElement('div');
      distCard.className = 'verdict-dist';
      const distColors = {
        True: 'var(--verdict-true)',
        False: 'var(--verdict-false)',
        Misleading: 'var(--verdict-misleading)',
        Unverifiable: 'var(--verdict-unverifiable)'
      };
      const rows = Object.entries(dist).map(([v, c]) => `
        <div class="dist-bar-row">
          <div class="dist-label">${verdictEmoji(v)} ${v}</div>
          <div class="dist-bar">
            <div class="dist-fill" data-width="${(c/total*100).toFixed(1)}%" style="width:0%;background:${distColors[v] || 'var(--blue)'}"></div>
          </div>
          <div class="dist-count">${c}</div>
        </div>`).join('');
      distCard.innerHTML = `<div class="dist-title">📊 Verdict Distribution</div>${rows}`;
      area.appendChild(distCard);
    }

    // Topic distribution
    const topics = data.topic_distribution || {};
    if (Object.keys(topics).length) {
      const topicCard = document.createElement('div');
      topicCard.className = 'verdict-dist';
      const total = Object.values(topics).reduce((a, b) => a + b, 0) || 1;
      const rows = Object.entries(topics).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([t, c]) => `
        <div class="dist-bar-row">
          <div class="dist-label">${t}</div>
          <div class="dist-bar">
            <div class="dist-fill" data-width="${(c/total*100).toFixed(1)}%" style="width:0%;background:var(--blue)"></div>
          </div>
          <div class="dist-count">${c}</div>
        </div>`).join('');
      topicCard.innerHTML = `<div class="dist-title">🗂️ Topic Distribution</div>${rows}`;
      area.appendChild(topicCard);
    }

    animateBars();
  } catch (err) {
    area.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>Failed to load: ${err.message}</p></div>`;
  }
}

/* ══════════════════════════════════════════
   Markdown Modal
══════════════════════════════════════════ */
function showMarkdownModal() {
  if (!state.lastDigestMarkdown) { toast('No markdown available.', 'warning'); return; }
  document.getElementById('md-preview').textContent = state.lastDigestMarkdown;
  document.getElementById('md-modal').style.display = 'flex';
}

document.getElementById('modal-close').addEventListener('click', () => {
  document.getElementById('md-modal').style.display = 'none';
});

document.getElementById('md-modal').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) e.currentTarget.style.display = 'none';
});

document.getElementById('copy-md').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(state.lastDigestMarkdown);
    toast('Markdown copied to clipboard!', 'success');
  } catch {
    toast('Copy failed — use Ctrl+A in the preview.', 'warning');
  }
});

document.getElementById('download-md').addEventListener('click', () => {
  const blob = new Blob([state.lastDigestMarkdown], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = state.lastDigestFilename || 'factguard_report.md';
  a.click();
  URL.revokeObjectURL(url);
  toast('Downloading...', 'success');
});
