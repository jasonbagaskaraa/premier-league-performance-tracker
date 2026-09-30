/* ═══════════════════════════════════════════════════════════════
   Premier League Dashboard — Application JavaScript
   Handles navigation, data fetching, Chart.js rendering, and
   the interactive match predictor.
   ═══════════════════════════════════════════════════════════════ */

const API = '';  // Same origin

// ─── Chart.js Global Defaults ───────────────────────────────
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 10;

// Palette
const COLORS = {
    indigo: '#6366f1', purple: '#a855f7', emerald: '#10b981',
    amber: '#f59e0b', rose: '#f43f5e', sky: '#38bdf8',
    cyan: '#22d3ee', pink: '#ec4899', lime: '#84cc16',
    orange: '#f97316', teal: '#14b8a6', blue: '#3b82f6'
};

const TEAM_COLORS = [
    COLORS.indigo, COLORS.emerald, COLORS.amber, COLORS.rose,
    COLORS.sky, COLORS.purple, COLORS.cyan, COLORS.pink,
    COLORS.lime, COLORS.orange, COLORS.teal, COLORS.blue,
    '#e879f9', '#fbbf24', '#34d399', '#fb923c',
    '#a78bfa', '#f472b6', '#2dd4bf', '#fcd34d'
];

// ─── Navigation ─────────────────────────────────────────────
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        const sectionId = link.dataset.section;
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById(`section-${sectionId}`).classList.add('active');
    });
});

// ─── Data Cache ─────────────────────────────────────────────
const cache = {};

async function fetchData(endpoint) {
    if (cache[endpoint]) return cache[endpoint];
    const res = await fetch(`${API}/api/${endpoint}`);
    const data = await res.json();
    cache[endpoint] = data;
    return data;
}

// ─── Initialize All Sections ────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initOverview();
    initPlayers();
    initTeams();
    initPredictions();
    initMatchPredictor();
});

// ═══ SECTION: OVERVIEW ══════════════════════════════════════
async function initOverview() {
    const teams = await fetchData('teams');
    const form = await fetchData('team-form');

    // KPIs
    const totalGoals = teams.reduce((s, t) => s + (t.Gls || 0), 0);
    const totalAssists = teams.reduce((s, t) => s + (t.Ast || 0), 0);
    const topScorer = teams.reduce((a, b) => (a.Gls || 0) > (b.Gls || 0) ? a : b);
    const avgPoss = (teams.reduce((s, t) => s + (t.Poss || 0), 0) / teams.length).toFixed(1);
    const hotTeams = form.filter(t => t.form_status === 'Hot Form').length;

    document.getElementById('kpi-grid').innerHTML = `
        <div class="kpi-card"><div class="kpi-label">Total Goals</div>
            <div class="kpi-value">${totalGoals}</div>
            <div class="kpi-detail">${teams.length} teams • 380 matches</div></div>
        <div class="kpi-card"><div class="kpi-label">Total Assists</div>
            <div class="kpi-value">${totalAssists}</div>
            <div class="kpi-detail">Season-wide playmaking</div></div>
        <div class="kpi-card"><div class="kpi-label">Top Scoring Club</div>
            <div class="kpi-value">${topScorer.Gls}</div>
            <div class="kpi-detail">${topScorer.Squad}</div></div>
        <div class="kpi-card"><div class="kpi-label">Avg Possession</div>
            <div class="kpi-value">${avgPoss}%</div>
            <div class="kpi-detail">League average</div></div>
        <div class="kpi-card"><div class="kpi-label">Teams in Hot Form</div>
            <div class="kpi-value">${hotTeams}</div>
            <div class="kpi-detail">11+ pts in last 5</div></div>
    `;

    // Team Goals Bar Chart
    const sorted = [...teams].sort((a, b) => b.Gls - a.Gls);
    new Chart(document.getElementById('chart-team-goals'), {
        type: 'bar',
        data: {
            labels: sorted.map(t => t.Squad),
            datasets: [{
                label: 'Goals Scored',
                data: sorted.map(t => t.Gls),
                backgroundColor: sorted.map((_, i) => TEAM_COLORS[i % TEAM_COLORS.length]),
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { maxRotation: 45, minRotation: 30, font: { size: 10 } } },
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });

    // Team Form Horizontal Bar
    const formSorted = [...form].sort((a, b) => b.pts_last_5 - a.pts_last_5);
    const formColors = formSorted.map(t =>
        t.form_status === 'Hot Form' ? COLORS.emerald :
        t.form_status === 'Cold Form' ? COLORS.rose : COLORS.indigo
    );
    new Chart(document.getElementById('chart-team-form'), {
        type: 'bar',
        data: {
            labels: formSorted.map(t => t.Squad),
            datasets: [{
                label: 'Points (Last 5)',
                data: formSorted.map(t => t.pts_last_5),
                backgroundColor: formColors,
                borderRadius: 5,
                borderSkipped: false
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { beginAtZero: true, max: 15, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { ticks: { font: { size: 10 } } }
            }
        }
    });

    // Possession Doughnut
    const possData = [...teams].sort((a, b) => b.Poss - a.Poss).slice(0, 8);
    new Chart(document.getElementById('chart-possession'), {
        type: 'doughnut',
        data: {
            labels: possData.map(t => t.Squad),
            datasets: [{
                data: possData.map(t => t.Poss),
                backgroundColor: possData.map((_, i) => TEAM_COLORS[i]),
                borderWidth: 0,
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            cutout: '55%',
            plugins: {
                legend: { position: 'right', labels: { font: { size: 11 }, padding: 12 } }
            }
        }
    });
}

// ═══ SECTION: PLAYER RANKINGS ═══════════════════════════════
async function initPlayers() {
    const data = await fetchData('composite-scores');
    const players = data.players;
    const positions = data.positions;
    const squads = data.squads;

    // Populate filters
    const posSel = document.getElementById('filter-position');
    positions.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p; opt.textContent = p;
        posSel.appendChild(opt);
    });

    const sqSel = document.getElementById('filter-squad');
    squads.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        sqSel.appendChild(opt);
    });

    function renderTable(filtered) {
        const tbody = document.getElementById('player-table-body');
        tbody.innerHTML = filtered.map((p, i) => {
            const score = p.composite_score ?? '-';
            const barW = p.composite_score ? p.composite_score : 0;
            const barColor = barW >= 80 ? COLORS.emerald : barW >= 50 ? COLORS.indigo : COLORS.amber;
            return `<tr>
                <td>${i + 1}</td>
                <td style="color:${COLORS.sky};font-weight:600;">${p.Player || '-'}</td>
                <td>${p.Squad || '-'}</td>
                <td><span class="status-badge stable">${p.pos_group || p.Pos || '-'}</span></td>
                <td>${(p['90s'] ?? '-')}</td>
                <td>${p.Gls ?? '-'}</td>
                <td>${p.Ast ?? '-'}</td>
                <td style="font-weight:700;color:${COLORS.amber}">${p['G+A'] ?? '-'}</td>
                <td><span class="score-bar" style="width:${barW}px;background:${barColor}"></span>${score}</td>
            </tr>`;
        }).join('');
    }

    renderTable(players);

    function applyFilters() {
        const pos = posSel.value;
        const sq = sqSel.value;
        let filtered = players;
        if (pos) filtered = filtered.filter(p => p.pos_group === pos);
        if (sq) filtered = filtered.filter(p => p.Squad === sq);
        renderTable(filtered);
    }

    posSel.addEventListener('change', applyFilters);
    sqSel.addEventListener('change', applyFilters);
}

// ═══ SECTION: TEAM ANALYTICS ════════════════════════════════
async function initTeams() {
    const form = await fetchData('team-form');
    const gk = await fetchData('gk-metrics');
    const teams = await fetchData('teams');

    // Form Table
    const ftb = document.getElementById('form-table-body');
    ftb.innerHTML = form.map(t => {
        const statusClass = t.form_status === 'Hot Form' ? 'hot' : t.form_status === 'Cold Form' ? 'cold' : 'stable';
        return `<tr>
            <td>${t.rank}</td>
            <td style="font-weight:600;color:${COLORS.sky}">${t.Squad}</td>
            <td style="font-weight:700">${t.pts_last_5}</td>
            <td>${t.gf_last_5}</td>
            <td>${t.ga_last_5}</td>
            <td style="color:${t.gd_last_5 >= 0 ? COLORS.emerald : COLORS.rose};font-weight:600">${t.gd_last_5 > 0 ? '+' : ''}${t.gd_last_5}</td>
            <td><span class="status-badge ${statusClass}">${t.form_status}</span></td>
        </tr>`;
    }).join('');

    // GF vs GA scatter
    const teamsSorted = [...teams].sort((a, b) => b.Gls - a.Gls);
    new Chart(document.getElementById('chart-gf-ga'), {
        type: 'bar',
        data: {
            labels: teamsSorted.map(t => t.Squad),
            datasets: [
                {
                    label: 'Goals For',
                    data: teamsSorted.map(t => t.Gls),
                    backgroundColor: `${COLORS.emerald}cc`,
                    borderRadius: 4, borderSkipped: false
                },
                {
                    label: 'Assists',
                    data: teamsSorted.map(t => t.Ast),
                    backgroundColor: `${COLORS.indigo}cc`,
                    borderRadius: 4, borderSkipped: false
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { ticks: { maxRotation: 45, font: { size: 10 } } },
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });

    // Goalkeeper Chart
    const gkSorted = [...gk].sort((a, b) => (b.ga_per90 ?? 0) - (a.ga_per90 ?? 0));
    new Chart(document.getElementById('chart-gk'), {
        type: 'bar',
        data: {
            labels: gkSorted.map(g => g.Player ? g.Player.split(' ').pop() : '?'),
            datasets: [{
                label: 'GA per 90',
                data: gkSorted.map(g => g.ga_per90),
                backgroundColor: gkSorted.map(g =>
                    (g.ga_per90 ?? 2) <= 1.0 ? COLORS.emerald : (g.ga_per90 ?? 2) <= 1.5 ? COLORS.amber : COLORS.rose
                ),
                borderRadius: 4, borderSkipped: false
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { ticks: { font: { size: 10 } } }
            }
        }
    });
}

// ═══ SECTION: ML PREDICTIONS ════════════════════════════════
async function initPredictions() {
    const mlPlayers = await fetchData('player-predictions');

    // ML Stats
    const overperf = mlPlayers.filter(p => p.efficiency_status === 'Clinical Overperformer').length;
    const expected = mlPlayers.filter(p => p.efficiency_status === 'Performing As Expected').length;
    const underperf = mlPlayers.filter(p => p.efficiency_status === 'High Potential / Underconverting').length;

    document.getElementById('ml-stats-grid').innerHTML = `
        <div class="kpi-card"><div class="kpi-label">Players Analyzed</div>
            <div class="kpi-value">${mlPlayers.length}</div>
            <div class="kpi-detail">Min 360+ minutes played</div></div>
        <div class="kpi-card"><div class="kpi-label">Overperformers</div>
            <div class="kpi-value" style="color:${COLORS.emerald}">${overperf}</div>
            <div class="kpi-detail">Clinical finishers</div></div>
        <div class="kpi-card"><div class="kpi-label">As Expected</div>
            <div class="kpi-value" style="color:${COLORS.indigo}">${expected}</div>
            <div class="kpi-detail">Performing to model</div></div>
        <div class="kpi-card"><div class="kpi-label">High Potential</div>
            <div class="kpi-value" style="color:${COLORS.amber}">${underperf}</div>
            <div class="kpi-detail">Underconverting chances</div></div>
    `;

    // Scatter Plot: Predicted vs Actual
    const statusColorMap = {
        'Clinical Overperformer': COLORS.emerald,
        'Performing As Expected': COLORS.indigo,
        'High Potential / Underconverting': COLORS.amber
    };

    const datasets = {};
    mlPlayers.forEach(p => {
        const s = p.efficiency_status;
        if (!datasets[s]) datasets[s] = { label: s, data: [], backgroundColor: statusColorMap[s] + 'cc', pointRadius: 5, pointHoverRadius: 8 };
        datasets[s].data.push({ x: p.predicted_ga_per90, y: p.actual_ga_per90, player: p.Player });
    });

    const maxVal = Math.max(
        ...mlPlayers.map(p => Math.max(p.predicted_ga_per90 || 0, p.actual_ga_per90 || 0))
    ) + 0.1;

    new Chart(document.getElementById('chart-ml-scatter'), {
        type: 'scatter',
        data: {
            datasets: [
                ...Object.values(datasets),
                {
                    label: 'Par Expected Rate',
                    data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
                    type: 'line',
                    borderColor: '#4b5563',
                    borderDash: [6, 4],
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const d = ctx.raw;
                            if (d.player) return `${d.player}: Pred ${d.x.toFixed(2)}, Actual ${d.y.toFixed(2)}`;
                            return '';
                        }
                    }
                }
            },
            scales: {
                x: { title: { display: true, text: 'Predicted G+A/90' }, beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { title: { display: true, text: 'Actual G+A/90' }, beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });

    // ML Player Table
    const top30 = mlPlayers.slice(0, 30);
    const mlTbody = document.getElementById('ml-player-table-body');
    mlTbody.innerHTML = top30.map(p => {
        const diff = p.performance_differential ?? p.ga_differential ?? 0;
        const diffColor = diff > 0 ? COLORS.emerald : diff < -0.15 ? COLORS.amber : COLORS.indigo;
        const statusClass = (p.efficiency_status || '').includes('Overperformer') || (p.efficiency_status || '').includes('Rock') || (p.efficiency_status || '').includes('Elite') ? 'overperformer' :
            (p.efficiency_status || '').includes('Under') || (p.efficiency_status || '').includes('Vulnerable') ? 'underperformer' : 'expected';
        return `<tr>
            <td style="color:${COLORS.sky};font-weight:600;">${p.Player}</td>
            <td>${p.Squad}</td>
            <td><span class="status-badge stable">${p.pos_group}</span></td>
            <td>${p['90s']}</td>
            <td style="font-size:12px;color:#94a3b8;">${p.target_metric_name || 'G+A/90'}</td>
            <td style="font-weight:700;color:${COLORS.amber}">${p.actual_target_metric ?? p.actual_ga_per90 ?? '-'}</td>
            <td>${p.predicted_target_metric ?? p.predicted_ga_per90 ?? '-'}</td>
            <td style="color:${diffColor};font-weight:700">${diff > 0 ? '+' : ''}${diff}</td>
            <td><span class="status-badge ${statusClass}">${p.efficiency_status ?? '-'}</span></td>
        </tr>`;
    }).join('');
}

// ═══ SECTION: MATCH PREDICTOR ═══════════════════════════════
async function initMatchPredictor() {
    const teams = await fetchData('teams');
    const squads = teams.map(t => t.Squad).sort();

    const homeSel = document.getElementById('select-home');
    const awaySel = document.getElementById('select-away');

    squads.forEach((s, i) => {
        homeSel.appendChild(new Option(s, s));
        awaySel.appendChild(new Option(s, s));
    });

    // Default selections
    if (squads.length >= 2) {
        homeSel.value = squads.find(s => s === 'Arsenal') || squads[0];
        awaySel.value = squads.find(s => s === 'Chelsea') || squads[1];
    }

    document.getElementById('btn-predict').addEventListener('click', async () => {
        const home = homeSel.value;
        const away = awaySel.value;
        if (home === away) { alert('Please select two different teams.'); return; }

        const btn = document.getElementById('btn-predict');
        btn.textContent = '⏳ Predicting...';
        btn.disabled = true;

        try {
            const res = await fetch(`${API}/api/predict-match`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ home_team: home, away_team: away })
            });
            const data = await res.json();

            document.getElementById('result-home').textContent = data.home_team;
            document.getElementById('result-away').textContent = data.away_team;

            // Animate bars
            document.getElementById('bar-home').style.width = `${data.home_win_prob}%`;
            document.getElementById('bar-draw').style.width = `${data.draw_prob}%`;
            document.getElementById('bar-away').style.width = `${data.away_win_prob}%`;

            document.getElementById('val-home').textContent = `${data.home_win_prob}%`;
            document.getElementById('val-draw').textContent = `${data.draw_prob}%`;
            document.getElementById('val-away').textContent = `${data.away_win_prob}%`;

            document.getElementById('result-verdict').innerHTML =
                `🏆 <strong>Model Prediction:</strong> ${data.predicted_outcome}`;

            document.getElementById('prediction-result').classList.remove('hidden');
        } catch (err) {
            alert('Prediction failed. Check server is running.');
            console.error(err);
        } finally {
            btn.innerHTML = '<span class="btn-icon">⚡</span> Predict Match';
            btn.disabled = false;
        }
    });
}
