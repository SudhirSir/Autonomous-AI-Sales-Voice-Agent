/* Fortivex Autonomous AI Sales Voice Agent — Frontend Controller */
(() => {
  const state = {
    sessionId: `call-${Math.random().toString(36).slice(2, 10)}`,
    ws: null,
    wsReady: false,
    isListening: false,
    isSpeaking: false,
    ttsRate: 1.05,
    sentimentHistory: [0],
    recognition: null,
    audioCtx: null,
    analyser: null,
    micStream: null,
    animFrame: null,
  };

  const LOCALES = { en: 'en-US', es: 'es-ES', fr: 'fr-FR', de: 'de-DE', hi: 'hi-IN' };

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');

  // ---------------- Navigation ----------------
  document.querySelectorAll('.nav-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.nav-tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.view').forEach((v) => v.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.view;
      $(target).classList.add('active');
      if (target === 'view-crm') loadCRM();
      if (target === 'view-kb') loadKnowledge();
      if (target === 'view-insights') loadInsights();
    });
  });

  // ---------------- WebSocket Connection ----------------
  function connectWebSocket() {
    $('sessionIdPill').textContent = `Session: ${state.sessionId}`;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/call/${encodeURIComponent(state.sessionId)}`;
    const ws = new WebSocket(url);
    state.ws = ws;

    ws.onopen = () => {
      state.wsReady = true;
      $('wsDot').className = 'dot';
      $('wsLabel').textContent = 'Live WS';
    };

    ws.onclose = () => {
      state.wsReady = false;
      $('wsDot').className = 'dot warn';
      $('wsLabel').textContent = 'HTTP Fallback';
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'session_ready') {
        $('engineBadge').textContent = `Engine: ${msg.ai_engine}`;
        if (msg.memory) renderMemory(msg.memory);
        if (msg.qualification) renderQualification(msg.qualification);
      } else if (msg.type === 'agent_thinking') {
        $('voiceStatus').textContent = 'AI Agent reasoning (RAG + BANT + CRM)...';
      } else if (msg.type === 'turn_result') {
        handleTurnResponse(msg.data);
      } else if (msg.type === 'call_summary') {
        showSummaryToastAndSwitch(msg.summary);
      } else if (msg.type === 'tts_stop') {
        stopSpeaking();
      }
    };
  }

  // ---------------- Turn Execution ----------------
  async function sendTurn(text, inputMode = 'text') {
    const clean = (text || '').trim();
    if (!clean) return;

    // Barge-in: stop AI speaking immediately when user submits new turn
    stopSpeaking();
    appendMessage('user', clean, { language: $('langSelect').value || 'auto' });

    const langOverride = $('langSelect').value || null;
    if (state.ws && state.wsReady && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(
        JSON.stringify({
          type: 'turn',
          text: clean,
          language_override: langOverride,
          input_mode: inputMode,
        })
      );
      return;
    }

    // Graceful HTTP fallback
    $('voiceStatus').textContent = 'Processing turn via REST API...';
    const res = await fetch('/api/turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        text: clean,
        language_override: langOverride,
        input_mode: inputMode,
      }),
    });
    if (res.ok) {
      const data = await res.json();
      handleTurnResponse(data);
    }
  }

  function handleTurnResponse(data) {
    $('engineBadge').textContent = `Engine: ${data.ai_engine}`;
    $('voiceStatus').textContent = `Responded in ${data.latency_ms}ms (${data.language.toUpperCase()})`;
    $('crmSyncPill').textContent = data.crm_synced ? 'CRM Sync: Live ✓' : 'CRM Sync: Pending';

    appendMessage('assistant', data.reply_text, {
      language: data.language,
      sentiment: data.sentiment,
      latency: data.latency_ms,
    });

    // Update sentiment history & sparkline
    if (data.sentiment) {
      state.sentimentHistory.push(data.sentiment.polarity);
      renderSentiment(data.sentiment);
    }
    renderMemory(data.memory);
    renderQualification(data.qualification);
    renderBookingAndRag(data);

    if ($('autoSpeakToggle').checked) {
      speakText(data.reply_text, data.tts_locale);
    }
  }

  function appendMessage(role, text, meta = {}) {
    const box = $('transcriptBox');
    const div = document.createElement('div');
    div.className = `msg ${role}`;

    let metaBits = [`<span>${role === 'user' ? 'Buyer' : 'Nova AI'}</span>`];
    if (meta.language) metaBits.push(`<span>${esc(meta.language.toUpperCase())}</span>`);
    if (meta.sentiment) metaBits.push(`<span>Sentiment: ${esc(meta.sentiment.label)} (${meta.sentiment.polarity})</span>`);
    if (meta.latency !== undefined) metaBits.push(`<span>${meta.latency}ms</span>`);

    div.innerHTML = `<div>${esc(text)}</div><div class="msg-meta">${metaBits.join(' · ')}</div>`;
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
  }

  // ---------------- Intelligence Panels ----------------
  function renderQualification(q) {
    $('leadScoreVal').textContent = `${q.score} / 100`;
    $('leadStageVal').textContent = q.stage;
    $('leadAcvVal').textContent = `$${Number(q.estimated_acv_usd || 0).toLocaleString()}`;

    const badge = $('tierBadge');
    badge.textContent = q.tier;
    badge.className = `tag ${q.tier.toLowerCase()}`;

    const b = q.bant || {};
    $('bantBudgetText').textContent = b.budget || 'Not disclosed';
    $('bantBudgetBar').style.width = `${(b.budget_score / 25) * 100}%`;
    $('bantAuthText').textContent = b.authority || 'Not disclosed';
    $('bantAuthBar').style.width = `${(b.authority_score / 25) * 100}%`;
    $('bantNeedText').textContent = b.need || 'Exploring';
    $('bantNeedBar').style.width = `${(b.need_score / 25) * 100}%`;
    $('bantTimeText').textContent = b.timeline || 'Not disclosed';
    $('bantTimeBar').style.width = `${(b.timeline_score / 25) * 100}%`;
  }

  function renderMemory(m) {
    $('memName').textContent = m.name || 'Unknown';
    $('memCompanyRole').textContent = [m.company, m.role].filter(Boolean).join(' · ') || '—';
    $('memEmailSize').textContent = [m.email, m.team_size ? `${m.team_size} seats` : null].filter(Boolean).join(' · ') || '—';
    $('memLang').textContent = (m.preferred_language || 'en').toUpperCase() + (m.returning_lead ? ' (Returning Lead ✓)' : '');

    const tags = [];
    (m.pain_points || []).forEach((p) => tags.push(`<span class="tag">${esc(p)}</span>`));
    (m.objections || []).forEach((o) => tags.push(`<span class="tag neg">${esc(o)}</span>`));
    (m.competitors_mentioned || []).forEach((c) => tags.push(`<span class="tag warm">vs ${esc(c)}</span>`));
    $('memSignalsRow').innerHTML = tags.join('');
  }

  function renderSentiment(s) {
    const badge = $('sentimentLabelBadge');
    badge.textContent = `${s.label} (${s.polarity > 0 ? '+' : ''}${s.polarity})`;
    badge.className = `tag ${s.polarity >= 0.2 ? 'hot' : s.polarity < 0 ? 'neg' : 'cold'}`;

    const pts = state.sentimentHistory.slice(-14);
    const stepX = pts.length > 1 ? 300 / (pts.length - 1) : 300;
    const coords = pts
      .map((v, idx) => {
        const x = 10 + idx * stepX;
        const y = 22 - v * 16;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
    $('sentimentPath').setAttribute('points', coords);
  }

  function renderBookingAndRag(data) {
    const bookBox = $('bookingStatusBox');
    if (data.booked_appointment) {
      const a = data.booked_appointment;
      bookBox.innerHTML = `
        <div style="padding:10px; border-radius:9px; background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35); color:#a7f3d0;">
          <strong>Demo Confirmed:</strong> ${esc(a.slot_iso)}<br/>
          <span>Meeting Link: <code>${esc(a.meeting_link)}</code></span>
        </div>`;
    } else if (data.proposed_slots && data.proposed_slots.length) {
      bookBox.innerHTML =
        `<div><strong>Proposed Calendar Slots</strong> (click to confirm):</div>` +
        data.proposed_slots
          .map(
            (s, i) =>
              `<button class="slot-btn" data-slot="${i + 1}">${esc(new Date(s).toUTCString().replace(':00 GMT', ' UTC'))}</button>`
          )
          .join('');
      bookBox.querySelectorAll('.slot-btn').forEach((btn) => {
        btn.addEventListener('click', () => sendTurn(`Option ${btn.dataset.slot} works great, please book it.`));
      });
    }

    const ragBox = $('ragChunksBox');
    const chunks = data.retrieved_chunks || [];
    if (!chunks.length) {
      ragBox.innerHTML = `<div style="font-size:12px; color:var(--muted);">No specific KB chunks needed for this turn.</div>`;
      return;
    }
    ragBox.innerHTML = chunks
      .map(
        (c) => `
        <div class="rag-item">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <strong style="color:#c7d2fe;">${esc(c.title)}</strong>
            <span class="tag">${Math.round(c.score * 100)}% match</span>
          </div>
          <div style="color:var(--muted);">${esc(c.content.slice(0, 170))}${c.content.length > 170 ? '…' : ''}</div>
        </div>`
      )
      .join('');
  }

  // ---------------- Real-Time STT & TTS + Waveform ----------------
  function speakText(text, locale = 'en-US') {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = locale;
    utter.rate = state.ttsRate;
    const voices = window.speechSynthesis.getVoices();
    const match = voices.find((v) => v.lang.toLowerCase().startsWith(locale.slice(0, 2).toLowerCase()));
    if (match) utter.voice = match;

    utter.onstart = () => {
      state.isSpeaking = true;
      $('voiceStatus').textContent = `Speaking (${locale}) — click Stop or speak to barge in`;
    };
    utter.onend = () => {
      state.isSpeaking = false;
      $('voiceStatus').textContent = state.isListening ? 'Listening...' : 'Ready';
      if ($('continuousVoiceToggle').checked && !state.isListening) {
        startListening();
      }
    };
    window.speechSynthesis.speak(utter);
  }

  function stopSpeaking() {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    state.isSpeaking = false;
  }

  async function startWaveform() {
    const canvas = $('waveformCanvas');
    const ctx = canvas.getContext('2d');
    try {
      state.micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = state.audioCtx.createMediaStreamSource(state.micStream);
      state.analyser = state.audioCtx.createAnalyser();
      state.analyser.fftSize = 128;
      src.connect(state.analyser);
    } catch {
      state.analyser = null;
    }

    const buf = new Uint8Array(state.analyser ? state.analyser.frequencyBinCount : 64);
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const mid = canvas.height / 2;
      if (state.analyser && state.isListening) {
        state.analyser.getByteFrequencyData(buf);
      } else {
        for (let i = 0; i < buf.length; i++) {
          buf[i] = state.isSpeaking ? 45 + Math.sin(Date.now() / 120 + i) * 35 : 8 + Math.sin(Date.now() / 400 + i * 0.4) * 5;
        }
      }
      const barW = canvas.width / buf.length;
      for (let i = 0; i < buf.length; i++) {
        const h = Math.max(4, (buf[i] / 255) * (canvas.height - 14));
        ctx.fillStyle = state.isListening ? '#f43f5e' : state.isSpeaking ? '#10b981' : '#6366f1';
        ctx.fillRect(i * barW + 2, mid - h / 2, Math.max(2, barW - 4), h);
      }
      state.animFrame = requestAnimationFrame(draw);
    };
    if (!state.animFrame) draw();
  }

  function startListening() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      $('voiceStatus').textContent = 'Web Speech API unavailable in this browser; use Chrome/Edge or text input.';
      return;
    }
    stopSpeaking();
    const rec = new SpeechRec();
    const selectedLang = $('langSelect').value;
    rec.lang = selectedLang ? LOCALES[selectedLang] : 'en-US';
    rec.interimResults = true;
    rec.continuous = false;

    rec.onstart = () => {
      state.isListening = true;
      $('micToggleBtn').classList.add('listening');
      $('micLabel').textContent = 'Listening... (Click to Stop)';
      $('voiceStatus').textContent = `Listening (${rec.lang})...`;
      startWaveform();
    };

    rec.onresult = (event) => {
      let interim = '';
      let finalTxt = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalTxt += t;
        else interim += t;
      }
      $('interimText').textContent = interim;
      if (finalTxt.trim()) {
        $('interimText').textContent = '';
        sendTurn(finalTxt.trim(), 'voice');
      }
    };

    rec.onerror = (e) => {
      $('voiceStatus').textContent = `Mic note: ${e.error}`;
      stopListening();
    };

    rec.onend = () => {
      stopListening();
    };

    state.recognition = rec;
    rec.start();
  }

  function stopListening() {
    state.isListening = false;
    $('micToggleBtn').classList.remove('listening');
    $('micLabel').textContent = 'Start Voice Call (STT)';
    if (state.recognition) {
      try { state.recognition.stop(); } catch {}
      state.recognition = null;
    }
  }

  $('micToggleBtn').addEventListener('click', () => {
    if (state.isListening) stopListening();
    else startListening();
  });

  $('stopSpeakBtn').addEventListener('click', () => {
    stopSpeaking();
    if (state.ws && state.wsReady) {
      state.ws.send(JSON.stringify({ type: 'barge_in' }));
    }
  });

  $('textTurnForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const val = $('textInput').value;
    $('textInput').value = '';
    sendTurn(val, 'text');
  });

  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => sendTurn(chip.dataset.utterance, 'voice'));
  });

  $('newCallBtn').addEventListener('click', () => {
    stopSpeaking();
    stopListening();
    if (state.ws) state.ws.close();
    state.sessionId = `call-${Math.random().toString(36).slice(2, 10)}`;
    state.sentimentHistory = [0];
    $('transcriptBox').innerHTML = '';
    $('ragChunksBox').innerHTML = '';
    $('bookingStatusBox').textContent = 'No appointment booked yet.';
    connectWebSocket();
  });

  $('endSummarizeBtn').addEventListener('click', async () => {
    const res = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/summarize`, { method: 'POST' });
    if (res.ok) {
      const summary = await res.json();
      showSummaryToastAndSwitch(summary);
    }
  });

  function showSummaryToastAndSwitch() {
    document.querySelector('[data-view="view-insights"]').click();
  }

  // ---------------- CRM, KB & Insights Views ----------------
  async function loadCRM() {
    const [leadsRes, apptsRes, whRes] = await Promise.all([
      fetch('/api/leads'),
      fetch('/api/appointments'),
      fetch('/api/webhooks'),
    ]);
    const leads = await leadsRes.json();
    const appts = await apptsRes.json();
    const hooks = await whRes.json();

    $('crmLeadsBody').innerHTML = leads
      .map(
        (l) => `
      <tr>
        <td><strong>${esc(l.name)}</strong><br/><small style="color:var(--muted);">${esc(l.email || '—')}</small></td>
        <td>${esc(l.company || '—')}<br/><small style="color:var(--muted);">${esc(l.role || '—')}</small></td>
        <td><span class="tag">${esc(l.preferred_language.toUpperCase())}</span></td>
        <td><strong>${l.score}</strong>/100</td>
        <td><span class="tag ${l.tier.toLowerCase()}">${esc(l.tier)}</span> <small>${esc(l.stage)}</small></td>
        <td>$${Number(l.estimated_acv_usd || 0).toLocaleString()}</td>
        <td>${(l.pain_points || []).map((p) => `<span class="tag">${esc(p)}</span>`).join('')}
            ${(l.objections || []).map((o) => `<span class="tag neg">${esc(o)}</span>`).join('')}</td>
      </tr>`
      )
      .join('');

    $('appointmentsList').innerHTML = appts.length
      ? appts
          .map(
            (a) => `
        <div class="rag-item" style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
          <div>
            <strong>${esc(a.lead_name)}</strong> (${esc(a.company || 'Prospect')}) · <span class="tag ${a.status === 'Confirmed' ? 'hot' : 'neg'}">${esc(a.status)}</span><br/>
            <small style="color:var(--muted);">${esc(a.slot_iso)} · ${esc(a.meeting_link)}</small>
          </div>
          ${a.status === 'Confirmed' ? `<button class="btn btn-danger cancel-appt" data-id="${esc(a.id)}" style="padding:4px 9px; font-size:11px;">Cancel</button>` : ''}
        </div>`
          )
          .join('')
      : '<div style="color:var(--muted); font-size:13px;">No appointments scheduled.</div>';

    document.querySelectorAll('.cancel-appt').forEach((b) => {
      b.addEventListener('click', async () => {
        await fetch(`/api/appointments/${encodeURIComponent(b.dataset.id)}`, { method: 'DELETE' });
        loadCRM();
      });
    });

    $('webhooksList').innerHTML = hooks.length
      ? hooks
          .slice(0, 6)
          .map(
            (h) => `
        <div class="rag-item">
          <div style="display:flex; justify-content:space-between;">
            <strong>${esc(h.event_type)}</strong>
            <span class="tag hot">Delivered</span>
          </div>
          <small style="color:var(--muted);">${esc(h.target)} · ${esc(h.created_at)}</small>
          <pre class="json-box">${esc(JSON.stringify(h.payload, null, 2))}</pre>
        </div>`
          )
          .join('')
      : '<div style="color:var(--muted); font-size:13px;">No webhook events yet.</div>';
  }

  $('refreshCrmBtn').addEventListener('click', loadCRM);

  async function loadKnowledge() {
    const res = await fetch('/api/knowledge');
    const data = await res.json();
    const chunks = data.chunks || [];
    $('kbChunkCount').textContent = `${chunks.length} chunks`;
    $('kbChunksList').innerHTML = chunks
      .map(
        (c) => `
      <div class="rag-item">
        <div style="display:flex; justify-content:space-between;">
          <strong>${esc(c.title)}</strong>
          <span class="tag">${esc(c.category)}</span>
        </div>
        <div style="color:var(--muted); margin-top:4px;">${esc(c.preview)}</div>
      </div>`
      )
      .join('');
  }

  $('kbSearchForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = $('kbSearchInput').value.trim();
    if (!q) return;
    const res = await fetch(`/api/knowledge/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    $('kbSearchResults').innerHTML = (data.results || [])
      .map(
        (r) => `
      <div class="rag-item">
        <div style="display:flex; justify-content:space-between;">
          <strong>${esc(r.title)}</strong>
          <span class="tag hot">${Math.round(r.score * 100)}% similarity</span>
        </div>
        <div style="margin-top:4px;">${esc(r.content)}</div>
      </div>`
      )
      .join('') || '<div style="color:var(--muted);">No matching chunks above threshold.</div>';
  });

  $('kbAddForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      title: $('kbDocTitle').value,
      category: $('kbDocCategory').value,
      content: $('kbDocContent').value,
    };
    const res = await fetch('/api/knowledge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      $('kbDocTitle').value = '';
      $('kbDocContent').value = '';
      loadKnowledge();
    }
  });

  async function loadInsights() {
    const [insRes, sumRes] = await Promise.all([fetch('/api/insights'), fetch('/api/summaries')]);
    const ins = await insRes.json();
    const sums = await sumRes.json();

    $('insightsKpiGrid').innerHTML = [
      ['Total Calls', ins.total_calls],
      ['Qualified Hot Leads', ins.hot_leads],
      ['Booked Demos', ins.booked_demos],
      ['Demo Conversion', `${ins.conversion_rate_pct}%`],
      ['Avg Lead Score', `${ins.average_lead_score}/100`],
      ['Pipeline Value', `$${Number(ins.total_pipeline_usd || 0).toLocaleString()}`],
    ]
      .map(([k, v]) => `<div class="stat-box"><small>${esc(k)}</small><strong>${esc(v)}</strong></div>`)
      .join('');

    const pains = (ins.top_pain_points || []).map((x) => `<span class="tag">${esc(x.label)} (${x.count})</span>`).join('') || '—';
    const objs = (ins.top_objections || []).map((x) => `<span class="tag neg">${esc(x.label)} (${x.count})</span>`).join('') || '—';
    $('insightsTrendsBox').innerHTML = `
      <div style="margin-bottom:10px;"><small style="color:var(--muted); display:block; margin-bottom:4px;">Pain Points</small>${pains}</div>
      <div><small style="color:var(--muted); display:block; margin-bottom:4px;">Common Objections</small>${objs}</div>`;

    const langs = Object.entries(ins.language_distribution || {})
      .map(([k, v]) => `<span class="tag">${esc(k.toUpperCase())}: ${v}</span>`)
      .join('');
    const stages = Object.entries(ins.stage_funnel || {})
      .map(([k, v]) => `<span class="tag warm">${esc(k)}: ${v}</span>`)
      .join('');
    $('insightsFunnelBox').innerHTML = `
      <div style="margin-bottom:10px;"><small style="color:var(--muted); display:block; margin-bottom:4px;">Pipeline Stages</small>${stages}</div>
      <div><small style="color:var(--muted); display:block; margin-bottom:4px;">Languages Served</small>${langs}</div>`;

    $('summariesList').innerHTML = sums
      .map(
        (s) => `
      <div class="rag-item">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <strong>${esc(s.lead_name)}${s.company ? ` · ${esc(s.company)}` : ''}</strong>
          <div>
            <span class="tag">${esc(s.language.toUpperCase())}</span>
            <span class="tag ${s.qualification.tier.toLowerCase()}">${s.qualification.score}/100 (${esc(s.qualification.tier)})</span>
            <span class="tag">Sentiment: ${s.average_sentiment} (${esc(s.sentiment_trend)})</span>
          </div>
        </div>
        <p style="margin:6px 0; line-height:1.5;">${esc(s.executive_summary)}</p>
        <div style="font-size:12px; color:#a7f3d0;"><strong>Next Steps:</strong> ${(s.next_steps || []).map(esc).join(' · ')}</div>
      </div>`
      )
      .join('');
  }

  $('refreshInsightsBtn').addEventListener('click', loadInsights);

  // ---------------- Settings Modal ----------------
  $('openSettingsBtn').addEventListener('click', () => $('settingsModal').classList.add('open'));
  $('closeSettingsBtn').addEventListener('click', () => $('settingsModal').classList.remove('open'));
  $('ttsRateInput').addEventListener('input', (e) => {
    state.ttsRate = parseFloat(e.target.value);
    $('ttsRateVal').textContent = state.ttsRate.toFixed(2);
  });
  $('settingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const key = $('geminiKeyInput').value.trim();
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gemini_api_key: key }),
    });
    if (res.ok) {
      const d = await res.json();
      $('engineBadge').textContent = `Engine: ${d.ai_engine}`;
      $('settingsModal').classList.remove('open');
    }
  });

  // Initialize
  connectWebSocket();
  startWaveform();
  appendMessage(
    'assistant',
    "Hi! I'm Nova, Fortivex's autonomous AI sales voice specialist. Ask me about pricing, SOC2 security, or CRM integrations in English, Spanish, French, German, or Hindi—or click a quick scenario above to test.",
    { language: 'en' }
  );
})();
