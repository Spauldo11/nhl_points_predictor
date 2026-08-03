/**
 * NHL Points Inference Engine - Client Application Script
 * Communicates with Python Web API and renders interactive model predictions & telemetry visualizations.
 */

// Dynamic API Base URL detection for local development vs hosted cloud backend (Render / Hugging Face)
const API_BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:')
  ? ''
  : 'https://nhlpointspredictor.onrender.com'; // Replace with your live backend server URL when deployed

document.addEventListener('DOMContentLoaded', () => {
  // DOM Element References
  const form = document.getElementById('prediction-form');
  const playerInput = document.getElementById('player-input');
  const clearSearchBtn = document.getElementById('clear-search-btn');
  const seasonSelect = document.getElementById('season-select');
  const predictBtn = document.getElementById('predict-btn');
  const btnText = predictBtn.querySelector('.btn-text');
  const btnSpinner = predictBtn.querySelector('.btn-spinner');
  const autocompleteList = document.getElementById('autocomplete-list');
  const quickChips = document.querySelectorAll('.chip-btn');

  // Alert Elements
  const alertBanner = document.getElementById('alert-banner');
  const alertMessage = document.getElementById('alert-message');
  const closeAlertBtn = document.getElementById('close-alert-btn');

  // Results Container Elements
  const resultsWrapper = document.getElementById('results-wrapper');
  const displayPlayerName = document.getElementById('display-player-name');
  const displayTeamName = document.getElementById('display-team-name');
  const displaySeasonUsed = document.getElementById('display-season-used');
  const displayAge = document.getElementById('display-age');
  const playerPosBadge = document.getElementById('player-pos-badge');

  // Primary Metric Elements
  const primaryPtsVal = document.getElementById('primary-pts-val');
  const modelBadge = document.getElementById('model-badge');
  const heroPtsSubtitle = document.getElementById('hero-pts-subtitle');
  const actualPtsVal = document.getElementById('actual-pts-val');
  const actualGpVal = document.getElementById('actual-gp-val');
  const actualPpgVal = document.getElementById('actual-ppg-val');

  // Model Variants Elements
  const m1Val = document.getElementById('m1-val');
  const m2Val = document.getElementById('m2-val');
  const m3Val = document.getElementById('m3-val');
  const m4Val = document.getElementById('m4-val');
  const mAvgVal = document.getElementById('mavg-val');
  const variantItems = document.querySelectorAll('.variant-item');

  // Stat Matrix Elements
  const statG = document.getElementById('stat-g');
  const statA = document.getElementById('stat-a');
  const statSog = document.getElementById('stat-sog');
  const statPpg = document.getElementById('stat-ppg');
  const statToi = document.getElementById('stat-toi');
  const statAtoi = document.getElementById('stat-atoi');
  const statPlusminus = document.getElementById('stat-plusminus');
  const statPim = document.getElementById('stat-pim');

  // Table Elements
  const featuresTableBody = document.getElementById('features-table-body');
  const tableSearchInput = document.getElementById('table-search-input');

  // Global State
  let radarChartInstance = null;
  let debounceTimer = null;
  let selectedAutocompleteIndex = -1;
  let currentPredictionData = null;
  let activeModelKey = 'm1';

  // Initialize App
  init();

  function init() {
    setupEventListeners();
    fetchSeasons();
    // Default prediction load for initial view
    executePrediction("Connor McDavid", 2026);
  }

  function setupEventListeners() {
    // Model Variants Selection
    variantItems.forEach(item => {
      item.addEventListener('click', () => {
        const modelKey = item.getAttribute('data-model');
        if (!modelKey || !currentPredictionData) return;

        // Set active UI state
        variantItems.forEach(v => v.classList.remove('active'));
        item.classList.add('active');
        activeModelKey = modelKey;

        // Update Hero Primary Projection Card
        selectModelProjection(modelKey);
      });
    });

    // Search input typing for autocomplete
    playerInput.addEventListener('input', (e) => {
      const query = e.target.value.trim();
      clearSearchBtn.style.display = query ? 'block' : 'none';

      clearTimeout(debounceTimer);
      if (query.length < 2) {
        hideAutocomplete();
        return;
      }
      debounceTimer = setTimeout(() => fetchAutocomplete(query), 200);
    });

    // Keyboard navigation in autocomplete
    playerInput.addEventListener('keydown', (e) => {
      const items = autocompleteList.querySelectorAll('.autocomplete-item');
      if (items.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedAutocompleteIndex = Math.min(selectedAutocompleteIndex + 1, items.length - 1);
        updateAutocompleteSelection(items);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedAutocompleteIndex = Math.max(selectedAutocompleteIndex - 1, 0);
        updateAutocompleteSelection(items);
      } else if (e.key === 'Enter') {
        if (selectedAutocompleteIndex >= 0 && items[selectedAutocompleteIndex]) {
          e.preventDefault();
          selectAutocompleteItem(items[selectedAutocompleteIndex].dataset.name);
        }
      } else if (e.key === 'Escape') {
        hideAutocomplete();
      }
    });

    // Clear search button
    clearSearchBtn.addEventListener('click', () => {
      playerInput.value = '';
      clearSearchBtn.style.display = 'none';
      playerInput.focus();
      hideAutocomplete();
    });

    // Close alert button
    closeAlertBtn.addEventListener('click', hideAlert);

    // Quick pick chips
    quickChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const playerName = chip.getAttribute('data-name');
        playerInput.value = playerName;
        clearSearchBtn.style.display = 'block';
        hideAutocomplete();
        executePrediction(playerName, seasonSelect.value);
      });
    });

    // Form submit
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const playerName = playerInput.value.trim();
      if (!playerName) return;
      hideAutocomplete();
      executePrediction(playerName, seasonSelect.value);
    });

    // Hide autocomplete on click outside
    document.addEventListener('click', (e) => {
      if (!form.contains(e.target)) {
        hideAutocomplete();
      }
    });

    // Table filter search
    tableSearchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const rows = featuresTableBody.querySelectorAll('tr');
      rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
      });
    });
  }

  // Fetch Available Seasons from API
  async function fetchSeasons() {
    try {
      const res = await fetch(`${API_BASE_URL}/api/seasons`);
      if (res.ok) {
        const data = await res.json();
        if (data.seasons && data.seasons.length > 0) {
          seasonSelect.innerHTML = data.seasons.map(s => `
            <option value="${s}" ${s === 2026 ? 'selected' : ''}>
              ${s} ${s === 2026 ? '(Next Season Projection)' : ''}
            </option>
          `).join('');
        }
      }
    } catch (err) {
      console.warn("Using default season dropdown options.");
    }
  }

  // Fetch Player Autocomplete Suggestions
  async function fetchAutocomplete(query) {
    try {
      const res = await fetch(`${API_BASE_URL}/api/players?query=${encodeURIComponent(query)}&limit=8`);
      if (!res.ok) return;
      const data = await res.json();
      renderAutocomplete(data.players || []);
    } catch (err) {
      hideAutocomplete();
    }
  }

  function renderAutocomplete(players) {
    if (players.length === 0) {
      hideAutocomplete();
      return;
    }
    selectedAutocompleteIndex = -1;
    autocompleteList.innerHTML = players.map(p => `
      <div class="autocomplete-item" data-name="${escapeHtml(p)}">
        <span>${escapeHtml(p)}</span>
        <i class="fa-solid fa-angle-right" style="opacity:0.4;"></i>
      </div>
    `).join('');

    autocompleteList.classList.remove('hidden');

    autocompleteList.querySelectorAll('.autocomplete-item').forEach(item => {
      item.addEventListener('click', () => {
        selectAutocompleteItem(item.dataset.name);
      });
    });
  }

  function updateAutocompleteSelection(items) {
    items.forEach((item, idx) => {
      if (idx === selectedAutocompleteIndex) {
        item.classList.add('selected');
        item.scrollIntoView({ block: 'nearest' });
      } else {
        item.classList.remove('selected');
      }
    });
  }

  function selectAutocompleteItem(name) {
    playerInput.value = name;
    clearSearchBtn.style.display = 'block';
    hideAutocomplete();
    executePrediction(name, seasonSelect.value);
  }

  function hideAutocomplete() {
    autocompleteList.classList.add('hidden');
    autocompleteList.innerHTML = '';
    selectedAutocompleteIndex = -1;
  }

  // Execute Main Prediction API Call
  async function executePrediction(player, season) {
    setLoadingState(true);
    hideAlert();

    try {
      const url = `${API_BASE_URL}/api/predict?player=${encodeURIComponent(player)}&season=${encodeURIComponent(season)}`;
      const res = await fetch(url);
      const data = await res.json();

      if (!res.ok || data.error) {
        showAlert(data.error || `Could not find stats for '${player}'.`);
        resultsWrapper.classList.add('hidden');
        return;
      }

      // Check if fallback season warning exists
      if (data.warning) {
        showAlert(data.warning, 'warning');
      }

      currentPredictionData = data;
      renderPredictionResults(data);
      resultsWrapper.classList.remove('hidden');

    } catch (err) {
      showAlert("Server communication error. Ensure Python server is running.");
      resultsWrapper.classList.add('hidden');
    } finally {
      setLoadingState(false);
    }
  }

  // Render All Prediction Data in UI
  function renderPredictionResults(data) {
    const raw = data.raw_stats || {};
    const preds = data.model_predictions || {};
    const features = data.features || {};

    // Header identity
    displayPlayerName.textContent = data.player_name;
    displayTeamName.innerHTML = `<i class="fa-solid fa-shield-halved"></i> ${raw.Team || 'NHL'}`;
    displaySeasonUsed.innerHTML = `<i class="fa-solid fa-database"></i> ${data.used_season} STATS`;
    displayAge.innerHTML = `<i class="fa-solid fa-user"></i> AGE ${raw.Age || '--'}`;
    playerPosBadge.textContent = raw.Pos || 'F';

    // Model Variants Score Displays
    m1Val.textContent = preds.m1 ? `${preds.m1.pts} PTS` : '--';
    m2Val.textContent = preds.m2 ? `${preds.m2.pts} PTS` : '--';
    m3Val.textContent = preds.m3 ? `${preds.m3.pts} PTS` : '--';
    m4Val.textContent = preds.m4 ? `${preds.m4.pts} PTS` : '--';
    mAvgVal.textContent = preds.avg ? `${preds.avg.pts} PTS` : '--';

    // Update Hero Card Projection based on currently active model key
    selectModelProjection(activeModelKey);

    // Actual Prev Season Metrics
    const prevPts = raw.PTS !== undefined && raw.PTS !== null ? raw.PTS : '--';
    const prevGp = raw.GP !== undefined && raw.GP !== null ? raw.GP : '--';
    actualPtsVal.textContent = prevPts;
    actualGpVal.textContent = prevGp;

    if (prevPts !== '--' && prevGp !== '--' && Number(prevGp) > 0) {
      actualPpgVal.textContent = (Number(prevPts) / Number(prevGp)).toFixed(2);
    } else {
      actualPpgVal.textContent = '--';
    }

    // Mini Stats Matrix
    statG.textContent = raw.G !== undefined ? raw.G : '--';
    statA.textContent = raw.A !== undefined ? raw.A : '--';
    statSog.textContent = raw.SOG !== undefined ? raw.SOG : '--';
    statPpg.textContent = raw.PPG !== undefined ? raw.PPG : '--';
    statToi.textContent = formatTOI(raw.TOI);
    statAtoi.textContent = formatTOI(raw.ATOI);
    statPlusminus.textContent = raw['+/-'] !== undefined ? raw['+/-'] : '--';
    statPim.textContent = raw.PIM !== undefined ? raw.PIM : '--';

    // Render Feature Tensor Table
    renderFeatureTable(features, data.imputed_features || []);

    // Render Radar Analytics Chart
    renderRadarChart(raw);
  }

  // Update Hero Primary Projection Card for selected model key
  function selectModelProjection(key) {
    if (!currentPredictionData || !currentPredictionData.model_predictions) return;
    const preds = currentPredictionData.model_predictions;
    const modelObj = preds[key] || preds['m1'];

    if (!modelObj) return;

    // Animate points number
    animateNumber(primaryPtsVal, modelObj.pts);

    // Update badge and subtitle
    modelBadge.textContent = modelObj.label || 'Model Projection';
    if (key === 'avg') {
      heroPtsSubtitle.textContent = 'Ensemble Weighted Average across 4 Neural Networks';
    } else {
      heroPtsSubtitle.textContent = 'Projected Regular Season Total Points';
    }
  }

  // Feature Tensor Table Renderer
  function renderFeatureTable(features, imputedList) {
    featuresTableBody.innerHTML = '';
    const entries = Object.entries(features);

    entries.forEach(([key, val]) => {
      const isImputed = imputedList.includes(key);
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-weight: 700; color: var(--color-text-primary);">
          ${escapeHtml(key)}
          ${isImputed ? '<span class="imputed-flag">(Imputed)</span>' : ''}
        </td>
        <td style="font-family: var(--font-mono); font-weight: 700; color: var(--color-accent-cyan);">${val}</td>
        <td style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--color-text-muted);">TENSOR_SCALED_NUMERICAL</td>
      `;
      featuresTableBody.appendChild(tr);
    });
  }

  // Chart.js Performance Profile Radar Chart
  function renderRadarChart(raw) {
    const canvasEl = document.getElementById('stat-radar-chart');
    if (!canvasEl) return;
    const ctx = canvasEl.getContext('2d');

    if (radarChartInstance) {
      radarChartInstance.destroy();
    }

    // Extract metrics normalized relative to high-tier NHL baselines
    const goals = Math.min(100, ((raw.G || 0) / 50) * 100);
    const assists = Math.min(100, ((raw.A || 0) / 70) * 100);
    const shots = Math.min(100, ((raw.SOG || 0) / 300) * 100);
    const powerplay = Math.min(100, ((raw.PPG || 0) / 20) * 100);
    const toi = Math.min(100, (parseTOISeconds(raw.TOI) / 1800) * 100);
    const plusMinus = Math.min(100, Math.max(0, (((raw['+/-'] || 0) + 30) / 60) * 100));

    radarChartInstance = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Goals (G)', 'Assists (A)', 'Shots (SOG)', 'Powerplay (PPG)', 'TOI Volume', '+/- Index'],
        datasets: [{
          label: 'Performance Index',
          data: [goals, assists, shots, powerplay, toi, plusMinus],
          backgroundColor: 'rgba(0, 229, 255, 0.15)',
          borderColor: '#00e5ff',
          borderWidth: 2,
          pointBackgroundColor: '#00e5ff',
          pointBorderColor: '#ffffff',
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.12)' },
            grid: { color: 'rgba(255, 255, 255, 0.08)' },
            pointLabels: {
              color: '#94a3b8',
              font: { family: 'JetBrains Mono', size: 10, weight: '600' }
            },
            ticks: { display: false, max: 100 }
          }
        }
      }
    });
  }

  // Utilities
  function setLoadingState(loading) {
    if (loading) {
      btnText.classList.add('hidden');
      btnSpinner.classList.remove('hidden');
      predictBtn.disabled = true;
    } else {
      btnText.classList.remove('hidden');
      btnSpinner.classList.add('hidden');
      predictBtn.disabled = false;
    }
  }

  function showAlert(msg, type = 'danger') {
    alertMessage.textContent = msg;
    alertBanner.classList.remove('hidden');
  }

  function hideAlert() {
    alertBanner.classList.add('hidden');
  }

  function animateNumber(element, target) {
    const start = 0;
    const duration = 700;
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = (start + (target - start) * eased).toFixed(1);
      element.textContent = current;

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.textContent = target.toFixed(1);
      }
    }

    requestAnimationFrame(update);
  }

  function formatTOI(val) {
    if (!val && val !== 0) return '--';
    if (typeof val === 'number') {
      const mins = Math.floor(val / 60);
      const secs = Math.round(val % 60);
      return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }
    return String(val);
  }

  function parseTOISeconds(val) {
    if (typeof val === 'number') return val;
    if (typeof val === 'string' && val.includes(':')) {
      const p = val.split(':');
      return (parseInt(p[0]) * 60) + parseInt(p[1]);
    }
    return parseFloat(val) || 0;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
});
