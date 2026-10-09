const API_BASE = 'http://127.0.0.1:8000/api';
let currentProjectId = 'p-101';
let pollInterval = null;




document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize API and Data Loading
  checkHealth();
  loadProjectList();
  loadProjectData(currentProjectId);

  // 2. Initialize UI Interactions (Auto-expanding textarea)
  const queryInput = document.getElementById('queryInput');
  if (queryInput) {
    queryInput.addEventListener('input', function() {
      this.style.height = 'auto'; // Reset height
      this.style.height = this.scrollHeight + 'px'; // Expand to fit
    });
  }
});


// Single Page Application (SPA) Tab Switching without URL refreshes or 404 errors
function switchTab(tabName) {
  // Update nav active states
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`nav-${tabName}`);
  if (activeBtn) activeBtn.classList.add('active');

  // Hide all views
  const views = ['workspace', 'trials', 'protocol'];
  views.forEach(v => {
    const el = document.getElementById(`view-${v}`);
    if (el) el.style.display = 'none';
  });

  // Display target view
  const target = document.getElementById(`view-${tabName}`);
  if (target) target.style.display = 'block';

  // Update breadcrumb text
  const breadcrumbMap = {
    'workspace': 'Research assistant',
    'trials': 'Clinical trials',
    'protocol': 'Trial Protocol'
  };
  const bEl = document.getElementById('currentViewBreadcrumb');
  if (bEl) bEl.textContent = breadcrumbMap[tabName] || 'Workspace';
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      document.getElementById('serverDot')?.classList.add('online');
      const textEl = document.getElementById('serverStatusText');
      if (textEl) textEl.textContent = 'FastAPI Connected';
    }
  } catch (err) {
    document.getElementById('serverDot')?.classList.remove('online');
    const textEl = document.getElementById('serverStatusText');
    if (textEl) textEl.textContent = 'FastAPI Disconnected';
  }
}

async function loadProjectList() {
  try {
    const res = await fetch(`${API_BASE}/research`);
    const projects = await res.json();
    const listEl = document.getElementById('recentList');
    if (!listEl) return;

    listEl.innerHTML = projects.map(p => `
      <div class="recent-item ${p.id === currentProjectId ? 'active' : ''}" onclick="selectProject('${p.id}')">
        ${p.disease || p.query.substring(0, 24) + '...'}
      </div>
    `).join('');
  } catch (err) {
    console.error('Failed to load project list:', err);
  }
}

function selectProject(projectId) {
  currentProjectId = projectId;
  loadProjectList();
  loadProjectData(projectId);
}

async function loadProjectData(projectId) {
  try {
    const res = await fetch(`${API_BASE}/research/${projectId}`);
    if (!res.ok) return;
    const data = await res.json();

    // Fill search input & synthesis box
    const queryInput = document.getElementById('queryInput');
    if (queryInput) queryInput.value = data.query || '';

    const synthesisEl = document.getElementById('synthesisText');
    if (synthesisEl) {
    // Converts raw markdown into clean, styled HTML
        synthesisEl.innerHTML = typeof marked !== 'undefined'
        ? marked.parse(data.literature_review)
        : data.literature_review;
    }
    const cohortVal = document.getElementById('cohortVal');
    if (cohortVal) cohortVal.textContent = `${data.cohort_matches || 0} adults`;

    fetchEvidence(projectId);
    fetchProtocol(projectId);
    fetchEvents(projectId);

    if (data.status === 'processing') {
      startPolling(projectId);
    }
  } catch (err) {
    console.error('Error loading project details:', err);
  }
}

async function fetchEvidence(projectId) {
  try {
    const res = await fetch(`${API_BASE}/research/${projectId}/evidence`);
    const data = await res.json();

    const sourcesList = document.getElementById('sourcesList');
    if (sourcesList) {
      sourcesList.innerHTML = `
        <div class="source-item">
          <div class="source-title">ClinicalTrials.gov Data</div>
          <div class="source-authors">${(data.clinical_trials || 'No data').replace(/\n/g, '<br>')}</div>
        </div>
      `;
    }

    const trialsGrid = document.getElementById('trialsGrid');
    if (trialsGrid) {
      trialsGrid.innerHTML = `
        <div class="trial-card">
          <h4>Live Evidence Stream</h4>
          <p style="font-size:12px; color:#64748b; margin-top:6px; white-space:pre-wrap;">${data.clinical_trials || 'Fetching records...'}</p>
        </div>
      `;
    }
  } catch (err) {
    console.error('Error loading evidence:', err);
  }
}

async function fetchProtocol(projectId) {
  try {
    const res = await fetch(`${API_BASE}/research/${projectId}/protocol`);
    const proto = await res.json();

    const protoContainer = document.getElementById('protocolContainer');
    if (protoContainer) {
      if (!proto.title) {
        protoContainer.innerHTML = '<p style="color:#64748b;">Protocol generation in progress...</p>';
        return;
      }
      protoContainer.innerHTML = `
        <h3 style="margin-bottom:10px;">${proto.title}</h3>
        <p style="margin-bottom:8px;"><strong>Hypothesis:</strong> ${proto.hypothesis}</p>
        <p style="margin-bottom:12px;"><strong>Dosage Regimen:</strong> ${proto.dosage_regimen}</p>
        <h4 style="margin-top:12px;">Inclusion Criteria</h4>
        <ul style="margin-left:20px; font-size:13px; margin-bottom:12px;">${(proto.inclusion_criteria || []).map(i => `<li>${i}</li>`).join('')}</ul>
        <h4>Exclusion Criteria</h4>
        <ul style="margin-left:20px; font-size:13px;">${(proto.exclusion_criteria || []).map(e => `<li>${e}</li>`).join('')}</ul>
      `;
    }
  } catch (err) {
    console.error('Error loading protocol:', err);
  }
}



async function fetchEvents(projectId) {
  try {
    const res = await fetch(`${API_BASE}/research/${projectId}/events`);
    const data = await res.json();

    const timeline = document.getElementById('eventsTimeline');
    if (!timeline) return;

    if (!data.events || data.events.length === 0) {
      timeline.innerHTML = '<p style="font-size:12px; color:#64748b;">Initializing multi-agent pipeline...</p>';
      return;
    }

    // Map events by agent to track current step and completion status
    const agentMap = {};
    data.events.forEach(e => {
      agentMap[e.agent] = {
        message: e.message,
        status: e.status // 'in_progress' or 'completed'
      };
    });

    // Render agents sequentially
    timeline.innerHTML = Object.keys(agentMap).map(agentName => {
      const info = agentMap[agentName];
      const isCompleted = info.status === 'completed';
      const dotClass = isCompleted ? 'completed-dot' : 'active-dot';

      return `
        <div class="event-step">
          <span class="step-dot ${dotClass}"></span>
          <div class="step-content">
            <strong style="color: ${isCompleted ? '#0f172a' : '#0284c7'};">${agentName}</strong>
            <p style="color: #64748b; font-size: 12px; margin-top: 2px;">${info.message}</p>
          </div>
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error('Error loading events:', err);
  }
}


async function submitResearchQuery() {
  const queryInput = document.getElementById('queryInput');
  const searchBtn = document.getElementById('searchBtn');
  if (!queryInput || !queryInput.value.trim()) return;

  searchBtn.disabled = true;
  searchBtn.textContent = 'Agents Running...';

  try {
    const res = await fetch(`${API_BASE}/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: queryInput.value })
    });

    const data = await res.json();
    currentProjectId = data.project_id;

    loadProjectList();
    startPolling(currentProjectId);
  } catch (err) {
    console.error('Error initiating research:', err);
    searchBtn.disabled = false;
    searchBtn.textContent = 'Research';
  }
}

function startPolling(projectId) {
  if (pollInterval) clearInterval(pollInterval);

  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/research/${projectId}/status`);
      const data = await res.json();

      fetchEvents(projectId);

      if (data.status === 'completed' || data.status === 'failed') {
        clearInterval(pollInterval);
        const searchBtn = document.getElementById('searchBtn');
        if (searchBtn) {
          searchBtn.disabled = false;
          searchBtn.textContent = 'Research';
        }
        loadProjectData(projectId);
      }
    } catch (err) {
      clearInterval(pollInterval);
    }
  }, 2000);
}

function openNewResearchModal() {
  const query = prompt("Enter your clinical research question:");
  if (query) {
    document.getElementById('queryInput').value = query;
    submitResearchQuery();
  }
}