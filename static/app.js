/* Fortivex Voice — Full-Stack SaaS Frontend Controller */
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
    mediaRecorder: null,
    recordedChunks: [],
    activeAudio: null,
    audioCtx: null,
    analyser: null,
    micStream: null,
    animFrame: null,
    streamingBubble: null,
  };

  const LOCALES = {
    en: 'en-US',
    hi: 'hi-IN',
    hinglish: 'en-IN',
    es: 'es-ES',
    fr: 'fr-FR',
    de: 'de-DE',
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) =>
    String(s ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');

  function showToast(message, kind = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.textContent = message;
    $('toastContainer').appendChild(el);
    setTimeout(() => el.remove(), 4200);
  }

  // ---------------- Primary Navigation ----------------
  function switchMainView(viewId) {
    document.querySelectorAll('.nav-tab').forEach((t) => t.classList.toggle('active', t.dataset.view === viewId));
    document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === viewId));
    if (viewId === 'view-crm') loadCRM();
    if (viewId === 'view-kb') loadKnowledge();
    if (viewId === 'view-history') loadCallHistory();
    if (viewId === 'view-insights') loadInsights();
  }

  document.querySelectorAll('.nav-tab').forEach((tab) => {
    tab.addEventListener('click', () => switchMainView(tab.dataset.view));
  });

  function switchSubPane(subId) {
    document.querySelectorAll('.sub-tab').forEach((t) => t.classList.toggle('active', t.dataset.sub === subId));
    document.querySelectorAll('.sub-pane').forEach((p) => p.classList.toggle('active', p.id === subId));
  }

  document.querySelectorAll('.sub-tab').forEach((tab) => {
    tab.addEventListener('click', () => switchSubPane(tab.dataset.sub));
  });

  // ---------------- WebSocket Connection ----------------
  function connectWebSocket() {
    $('sessionIdPill').textContent = state.sessionId;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/call/${encodeURIComponent(state.sessionId)}`;
    const ws = new WebSocket(url);
    state.ws = ws;

    ws.onopen = () => {
      state.wsReady = true;
      $('wsDot').className = 'dot';
      $('wsLabel').textContent = 'Connected';
    };

    ws.onclose = () => {
      state.wsReady = false;
      $('wsDot').className = 'dot warn';
      $('wsLabel').textContent = 'REST Mode';
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'session_ready') {
        $('engineBadge').textContent = msg.ai_engine;
        $('voiceProviderBadge').textContent = `STT: ${msg.stt_provider || '—'} · TTS: ${msg.tts_provider || '—'}`;
        if (msg.memory) renderMemory(msg.memory);
        if (msg.qualification) renderQualification(msg.qualification);
      } else if (msg.type === 'transcript_stt') {
        if (msg.transcript) {
          appendMessage('user', msg.transcript, { language: 'voice' });
        } else if (msg.error) {
          showToast(msg.error, 'error');
        }
      } else if (msg.type === 'agent_thinking') {
        $('voiceStatus').textContent = 'Thinking...';
        $('bargeInBadge').style.display = 'none';
        state.streamingBubble = null;
      } else if (msg.type === 'turn_delta') {
        updateStreamingBubble(msg.delta);
      } else if (msg.type === 'turn_result') {
        if (state.streamingBubble) {
          state.streamingBubble.remove();
          state.streamingBubble = null;
        }
        handleTurnResponse(msg.data);
      } else if (msg.type === 'call_summary') {
        openSummaryModal(msg.summary);
      } else if (msg.type === 'tts_stop') {
        stopSpeaking();
        $('bargeInBadge').style.display = 'inline-flex';
        $('voiceStatus').textContent = 'Interrupted (Barge-In)';
      }
    };
  }

  function updateStreamingBubble(delta) {
    if (!state.streamingBubble) {
      const box = $('transcriptBox');
      const row = document.createElement('div');
      row.className = 'msg-row assistant';
      row.innerHTML = `<div class="msg-bubble"></div><div class="msg-meta">Nova (AI) · streaming...</div>`;
      box.appendChild(row);
      state.streamingBubble = row;
    }
    const bubble = state.streamingBubble.querySelector('.msg-bubble');
    bubble.textContent = (bubble.textContent ? bubble.textContent + ' ' : '') + delta;
    $('transcriptBox').scrollTop = $('transcriptBox').scrollHeight;
  }

  // ---------------- Turn Execution ----------------
  async function sendTurn(text, inputMode = 'text') {
    const clean = (text || '').trim();
    if (!clean) return;

    stopSpeaking();
    const chosenLang = $('langSelect').value;
    appendMessage('user', clean, { language: chosenLang || 'auto' });

    const langOverride = chosenLang || null;
    const tz = $('timezoneSelect') ? $('timezoneSelect').value : 'UTC';
    if (state.ws && state.wsReady && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(
        JSON.stringify({
          type: 'turn',
          text: clean,
          language_override: langOverride,
          input_mode: inputMode,
          timezone: tz,
        })
      );
      return;
    }

    $('voiceStatus').textContent = 'Processing...';
    const res = await fetch('/api/turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        text: clean,
        language_override: langOverride,
        input_mode: inputMode,
        timezone: tz,
      }),
    });
    if (res.ok) {
      handleTurnResponse(await res.json());
    } else {
      showToast('Failed to process turn', 'error');
      $('voiceStatus').textContent = 'Error';
    }
  }

  function handleTurnResponse(data) {
    $('engineBadge').textContent = data.ai_engine;
    $('voiceProviderBadge').textContent = `STT: ${data.stt_provider || '—'} · TTS: ${data.tts_provider || '—'}`;
    $('voiceStatus').textContent = `${data.latency_ms}ms · ${data.language.toUpperCase()}${data.intent ? ' · ' + data.intent.intent : ''}`;

    const syncPill = $('crmSyncPill');
    const extStat = data.external_crm_status || 'not_configured';
    if (extStat === 'synced') {
      syncPill.textContent = `Saved locally · CRM Synced (${data.crm_mode})`;
      syncPill.className = 'badge hot';
    } else if (extStat === 'failed') {
      syncPill.textContent = 'Saved locally · External CRM Failed';
      syncPill.className = 'badge neg';
    } else if (extStat === 'mock') {
      syncPill.textContent = 'Saved locally · Mock CRM Mode';
      syncPill.className = 'badge warm';
    } else {
      syncPill.textContent = 'Saved locally (SQLite)';
      syncPill.className = 'badge blue';
    }

    appendMessage('assistant', data.reply_text, {
      language: data.language,
      sentiment: data.sentiment,
      latency: data.latency_ms,
    });

    if (data.sentiment) {
      state.sentimentHistory.push(data.sentiment.polarity);
      renderSentiment(data.sentiment);
    }
    renderMemory(data.memory, data.qualification.stage);
    renderQualification(data.qualification);
    renderBookingAndRag(data);
    renderToolsAndLatency(data);

    if (data.booking_error) {
      showToast(data.booking_error, 'error');
      switchSubPane('sub-rag');
    } else if (data.booked_appointment || (data.proposed_slots && data.proposed_slots.length)) {
      switchSubPane('sub-rag');
    }

    if ($('autoSpeakToggle').checked) {
      speakReply(data.reply_text, data.language, data.tts_locale, data.tts_provider);
    }
  }

  function appendMessage(role, text, meta = {}) {
    const box = $('transcriptBox');
    const row = document.createElement('div');
    row.className = `msg-row ${role}`;

    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const parts = [role === 'user' ? 'Prospect' : 'Nova (AI)', timeStr];
    if (meta.language && meta.language !== 'auto') parts.push(meta.language.toUpperCase());
    if (meta.sentiment) parts.push(`${meta.sentiment.label} | Intent ${meta.sentiment.intent_score}%`);
    if (meta.latency !== undefined) parts.push(`${meta.latency}ms`);

    row.innerHTML = `
      <div class="msg-bubble">${esc(text)}</div>
      <div class="msg-meta">${parts.map(esc).join(' · ')}</div>`;
    box.appendChild(row);
    box.scrollTop = box.scrollHeight;
  }

  // ---------------- Deal Intelligence Rendering ----------------
  function renderQualification(q) {
    $('leadScoreVal').textContent = `${q.score} / 100`;
    $('leadAcvVal').textContent = `$${Number(q.estimated_acv_usd || 0).toLocaleString()}`;

    const badge = $('tierBadge');
    badge.textContent = `${q.tier} · ${q.stage}`;
    const cls = q.tier === 'Disqualified' ? 'neg' : q.tier.toLowerCase();
    badge.className = `badge ${cls}`;

    const b = q.bant || {};
    $('bantBudgetText').textContent = b.budget || 'Not disclosed';
    $('bantBudgetBar').style.width = `${(b.budget_score / 25) * 100}%`;
    $('bantAuthText').textContent = b.authority || 'Not disclosed';
    $('bantAuthBar').style.width = `${(b.authority_score / 25) * 100}%`;
    $('bantNeedText').textContent = b.need || 'Exploring';
    $('bantNeedBar').style.width = `${(b.need_score / 25) * 100}%`;
    $('bantTimeText').textContent = b.timeline || 'Not disclosed';
    $('bantTimeBar').style.width = `${(b.timeline_score / 25) * 100}%`;

    const missing = q.missing_dimensions || [];
    const filled = 4 - missing.length;
    $('bantCompleteness').textContent = `BANT Completeness: ${filled}/4 · Priority: ${q.lead_priority || 'P3'}`;
    $('bantMissingText').textContent = missing.length ? `Missing: ${missing.join(', ')}` : 'All 4 BANT dimensions captured';
  }

  function renderMemory(m, stage = 'Discovery') {
    $('memName').textContent = [m.name, m.phone].filter(Boolean).join(' · ') || '—';
    $('memCompanyRole').textContent = [m.company, m.role].filter(Boolean).join(' · ') || '—';
    $('memEmailSize').textContent = [m.email, m.team_size ? `${m.team_size} seats` : null].filter(Boolean).join(' · ') || '—';
    $('memLang').textContent = `${(m.preferred_language || 'en').toUpperCase()} · ${stage}${m.returning_lead ? ' (Returning)' : ''}`;

    const tags = [];
    (m.pain_points || []).forEach((p) => tags.push(`<span class="badge blue">${esc(p)}</span>`));
    (m.objections || []).forEach((o) => tags.push(`<span class="badge neg">${esc(o)}</span>`));
    (m.competitors_mentioned || []).forEach((c) => tags.push(`<span class="badge warm">vs ${esc(c)}</span>`));
    $('memSignalsRow').innerHTML = tags.join('');
  }

  function renderSentiment(s) {
    $('intentScoreVal').textContent = `${s.intent_score} / 100`;
    const badge = $('sentimentLabelBadge');
    badge.textContent = `${s.label} (${s.polarity > 0 ? '+' : ''}${s.polarity}) · Readiness: ${s.purchase_readiness}`;
    badge.className = `badge ${s.polarity >= 0.2 ? 'hot' : s.polarity < 0 ? 'neg' : ''}`;

    const pts = state.sentimentHistory.slice(-14);
    const stepX = pts.length > 1 ? 300 / (pts.length - 1) : 300;
    const coords = pts.map((v, idx) => `${(10 + idx * stepX).toFixed(1)},${(21 - v * 15).toFixed(1)}`).join(' ');
    $('sentimentPath').setAttribute('points', coords);
  }

  function renderBookingAndRag(data) {
    const bookBox = $('bookingStatusBox');
    if (data.booked_appointment) {
      const a = data.booked_appointment;
      bookBox.className = 'list-item';
      bookBox.style = 'background:var(--green-soft); border-color:var(--green-border); color:#065f46; margin-bottom:0;';
      bookBox.innerHTML = `
        <strong>${esc(a.status)} (${esc(a.id)})</strong> — ${esc(new Date(a.slot_iso).toUTCString().replace(':00 GMT', ' UTC'))}<br/>
        <span style="font-size:12px;">Link: <code class="mono">${esc(a.meeting_link)}</code></span>`;
    } else if (data.proposed_slots && data.proposed_slots.length) {
      bookBox.className = 'list-item';
      bookBox.style = 'margin-bottom:0;';
      bookBox.innerHTML =
        `<div style="font-weight:600; font-size:12.5px;">Available business-hour slots (click to book):</div>` +
        data.proposed_slots
          .map((s, i) => `<button class="slot-btn" data-slot="${i + 1}">${esc(new Date(s).toUTCString().replace(':00 GMT', ' UTC'))}</button>`)
          .join('');
      bookBox.querySelectorAll('.slot-btn').forEach((btn) => {
        btn.addEventListener('click', () => sendTurn(`Option ${btn.dataset.slot} works great, please book it.`));
      });
    }

    const gBadge = $('kbGroundedBadge');
    if (data.contradiction_detected) {
      gBadge.textContent = 'Contradiction Warning';
      gBadge.className = 'badge neg';
    } else {
      gBadge.textContent = data.kb_grounded ? 'Grounded' : 'Out-of-Scope Guardrail';
      gBadge.className = `badge ${data.kb_grounded ? 'hot' : 'warm'}`;
    }

    const ragBox = $('ragChunksBox');
    const chunks = data.retrieved_chunks || [];
    if (!chunks.length) {
      ragBox.innerHTML = `<div class="empty-state" style="padding:16px;">No matching KB chunks (refusal guardrail active for out-of-scope queries).</div>`;
      return;
    }
    ragBox.innerHTML =
      (data.contradiction_note ? `<div class="list-item" style="border-color:var(--red-border); color:#991b1b; font-size:12px;">${esc(data.contradiction_note)}</div>` : '') +
      chunks
        .map(
          (c) => `
        <div class="list-item">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:4px;">
            <strong style="font-size:12.5px;">#${c.rank || 1} ${esc(c.title)}</strong>
            <span class="badge blue">${Math.round((c.final_confidence || c.score) * 100)}% conf</span>
          </div>
          <div style="font-size:11px; color:var(--ink-muted); margin-bottom:4px;">
            Doc: ${esc(c.source_file || c.doc_id)} · Sparse: ${c.sparse_score} · Dense: ${c.dense_score} · RRF: ${c.rrf_score ?? 0} · Rerank: ${c.reranker_score ?? 0}
          </div>
          <div style="font-size:12.5px; color:var(--ink-secondary);">${esc(c.content.slice(0, 165))}${c.content.length > 165 ? '…' : ''}</div>
        </div>`
        )
        .join('');
  }

  function renderToolsAndLatency(data) {
    const lb = data.latency_breakdown || {};
    $('latencyBox').innerHTML = `
      <strong>Turn Latency Breakdown:</strong>
      RAG ${lb.rag_ms ?? 0}ms · Generation ${lb.llm_ms ?? 0}ms · Total <strong>${data.latency_ms}ms</strong> · TTS: <code>${esc(data.tts_provider)}</code>`;

    const logs = data.tool_logs || [];
    $('toolLogsBox').innerHTML = logs.length
      ? logs
          .map(
            (t) => `
        <div class="list-item">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <code class="mono" style="font-size:12px; font-weight:600;">${esc(t.tool_name)}</code>
            <span class="badge ${t.status === 'success' ? 'hot' : 'neg'}">${esc(t.status)} · ${t.duration_ms}ms</span>
          </div>
          <div style="font-size:12px; color:var(--ink-secondary); margin-top:4px;">${esc(t.summary)}</div>
        </div>`
          )
          .join('')
      : '<div class="empty-state">No tools executed.</div>';
  }

  // ---------------- Real-Time STT, TTS, VAD & Barge-In ----------------
  async function speakReply(text, lang, locale = 'en-US', provider = 'browser') {
    stopSpeaking();
    // Use server TTS whenever a genuine server speech synthesizer is active
    if (provider && !provider.startsWith('browser')) {
      try {
        const res = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, language: lang, rate: state.ttsRate }),
        });
        const ct = res.headers.get('content-type') || '';
        if (res.ok && ct.startsWith('audio/')) {
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          state.activeAudio = audio;
          state.isSpeaking = true;
          $('voiceStatus').textContent = `Speaking (${provider})`;
          audio.onended = () => {
            state.isSpeaking = false;
            URL.revokeObjectURL(url);
            $('voiceStatus').textContent = 'Ready';
            if ($('continuousVoiceToggle').checked && !state.isListening) startListening();
          };
          await audio.play();
          return;
        }
      } catch {}
    }

    if (!('speechSynthesis' in window)) return;
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = locale;
    utter.rate = state.ttsRate;
    const voices = window.speechSynthesis.getVoices();
    const match = voices.find((v) => v.lang.toLowerCase().startsWith(locale.slice(0, 2).toLowerCase()));
    if (match) utter.voice = match;

    utter.onstart = () => {
      state.isSpeaking = true;
      $('voiceStatus').textContent = `Speaking (${locale})`;
    };
    utter.onend = () => {
      state.isSpeaking = false;
      $('voiceStatus').textContent = state.isListening ? 'Listening...' : 'Ready';
      if ($('continuousVoiceToggle').checked && !state.isListening) startListening();
    };
    window.speechSynthesis.speak(utter);
  }

  function stopSpeaking() {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    if (state.activeAudio) {
      state.activeAudio.pause();
      state.activeAudio = null;
    }
    state.isSpeaking = false;
  }

  async function ensureAudioAnalyser() {
    if (state.analyser) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
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
  }

  async function startWaveform() {
    const canvas = $('waveformCanvas');
    const ctx = canvas.getContext('2d');
    await ensureAudioAnalyser();
    const buf = new Uint8Array(state.analyser ? state.analyser.frequencyBinCount : 48);

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const mid = canvas.height / 2;
      if (state.analyser && state.isListening) {
        state.analyser.getByteFrequencyData(buf);
      } else {
        for (let i = 0; i < buf.length; i++) {
          buf[i] = state.isSpeaking
            ? 38 + Math.sin(Date.now() / 110 + i * 0.6) * 28
            : 4 + Math.sin(Date.now() / 500 + i * 0.4) * 3;
        }
      }
      const step = canvas.width / buf.length;
      for (let i = 0; i < buf.length; i++) {
        const h = Math.max(3, (buf[i] / 255) * (canvas.height - 8));
        ctx.fillStyle = state.isListening ? '#dc2626' : state.isSpeaking ? '#059669' : '#cbd5e1';
        ctx.fillRect(i * step + 1, mid - h / 2, Math.max(2, step - 3), h);
      }
      state.animFrame = requestAnimationFrame(draw);
    };
    if (!state.animFrame) draw();
  }

  async function startServerAudioRecording() {
    await ensureAudioAnalyser();
    if (!state.micStream || typeof MediaRecorder === 'undefined') {
      showToast('Microphone unavailable for server recording.', 'error');
      return;
    }
    stopSpeaking();
    state.recordedChunks = [];
    const rec = new MediaRecorder(state.micStream);
    state.mediaRecorder = rec;
    state.isListening = true;
    $('micToggleBtn').classList.add('listening');
    $('micLabel').textContent = 'Stop & Send';
    $('voiceStatus').textContent = 'Recording audio...';

    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) state.recordedChunks.push(e.data);
    };
    rec.onstop = async () => {
      state.isListening = false;
      $('micToggleBtn').classList.remove('listening');
      $('micLabel').textContent = 'Speak';
      const blob = new Blob(state.recordedChunks, { type: rec.mimeType || 'audio/webm' });
      const buf = await blob.arrayBuffer();
      const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
      if (state.ws && state.wsReady) {
        state.ws.send(
          JSON.stringify({
            type: 'audio_chunk',
            audio_b64: b64,
            mime_type: blob.type || 'audio/webm',
            language_override: $('langSelect').value || null,
          })
        );
      } else {
        const fd = new FormData();
        fd.append('audio', blob, 'speech.webm');
        if ($('langSelect').value) fd.append('language', $('langSelect').value);
        const r = await fetch('/api/stt', { method: 'POST', body: fd });
        const d = await r.json();
        if (d.transcript) sendTurn(d.transcript, 'voice');
        else showToast(d.error || 'No transcript returned from server STT.', 'error');
      }
    };
    rec.start();
  }

  function startListening() {
    if ($('serverAudioToggle').checked) {
      startServerAudioRecording();
      return;
    }
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      showToast('Browser STT unavailable; trying server audio recording or type below.', 'info');
      startServerAudioRecording();
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
      $('micLabel').textContent = 'Listening...';
      $('voiceStatus').textContent = `Listening (${rec.lang})`;
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
      $('voiceStatus').textContent = `Mic: ${e.error}`;
      stopListening();
    };
    rec.onend = () => stopListening();

    state.recognition = rec;
    rec.start();
  }

  function stopListening() {
    state.isListening = false;
    $('micToggleBtn').classList.remove('listening');
    $('micLabel').textContent = 'Speak';
    if (state.recognition) {
      try { state.recognition.stop(); } catch {}
      state.recognition = null;
    }
    if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
      state.mediaRecorder.stop();
    }
  }

  $('micToggleBtn').addEventListener('click', () => {
    if (state.isListening) stopListening();
    else startListening();
  });

  $('stopSpeakBtn').addEventListener('click', () => {
    stopSpeaking();
    if (state.ws && state.wsReady) state.ws.send(JSON.stringify({ type: 'barge_in' }));
  });

  $('textTurnForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const val = $('textInput').value;
    $('textInput').value = '';
    sendTurn(val, 'text');
  });

  $('runScenarioBtn').addEventListener('click', () => {
    const val = $('scenarioSelect').value;
    if (val) sendTurn(val, 'voice');
  });

  $('newCallBtn').addEventListener('click', () => {
    stopSpeaking();
    stopListening();
    if (state.ws) state.ws.close();
    state.sessionId = `call-${Math.random().toString(36).slice(2, 10)}`;
    state.sentimentHistory = [0];
    $('transcriptBox').innerHTML = '';
    $('ragChunksBox').innerHTML = '<div class="empty-state" style="padding:18px;">Retrieved RAG knowledge chunks will appear here on each buyer turn.</div>';
    $('toolLogsBox').innerHTML = '<div class="empty-state" style="padding:16px;">No tools executed yet.</div>';
    $('bookingStatusBox').className = 'list-item';
    $('bookingStatusBox').style = 'margin-bottom:0; color:var(--ink-muted); font-size:12.5px;';
    $('bookingStatusBox').textContent = 'No meeting scheduled yet. Ask to book a demo to receive available business-hour slots.';
    $('crmSyncPill').textContent = 'CRM: Standby';
    $('crmSyncPill').className = 'badge';
    connectWebSocket();
    appendMessage(
      'assistant',
      "Hello! You've reached Fortivex Sales. I can answer questions about pricing, security compliance, or CRM integrations in English, Hindi, Hinglish, Spanish, French, or German—and book a live demo when you're ready.",
      { language: 'en' }
    );
  });

  // ---------------- Call Summary Modal ----------------
  $('endSummarizeBtn').addEventListener('click', async () => {
    const res = await fetch(`/api/sessions/${encodeURIComponent(state.sessionId)}/summarize`, { method: 'POST' });
    if (res.ok) openSummaryModal(await res.json());
  });

  function openSummaryModal(s) {
    $('summaryModalBody').innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
        <div>
          <strong style="font-size:15px;">${esc(s.lead_name)}</strong>
          ${s.company ? `<span style="color:var(--ink-muted);"> · ${esc(s.company)}</span>` : ''}
        </div>
        <div style="display:flex; gap:6px;">
          <span class="badge">${esc(s.language.toUpperCase())}</span>
          <span class="badge ${s.qualification.tier === 'Disqualified' ? 'neg' : s.qualification.tier.toLowerCase()}">${s.qualification.score}/100 (${esc(s.qualification.tier)})</span>
        </div>
      </div>
      <p style="margin:0 0 12px; color:var(--ink-secondary); line-height:1.6;">${esc(s.executive_summary)}</p>
      <div class="kv-grid" style="margin-bottom:12px;">
        <div class="kv-cell"><label>Sentiment Trend</label><div>${s.average_sentiment} (${esc(s.sentiment_trend)})</div></div>
        <div class="kv-cell"><label>Appointment Outcome</label><div>${esc(s.appointment_outcome)}</div></div>
        <div class="kv-cell"><label>Sources Cited</label><div>${(s.knowledge_sources_used || []).map(esc).join(', ') || 'None'}</div></div>
        <div class="kv-cell"><label>Follow-Up Date</label><div>${esc(s.follow_up_date || '—')}</div></div>
      </div>
      <div style="font-size:12.5px; font-weight:600; margin-bottom:6px;">Next Recommended Actions</div>
      <ul style="margin:0 0 16px; padding-left:18px; color:var(--ink-secondary);">
        ${(s.next_steps || []).map((step) => `<li style="margin-bottom:4px;">${esc(step)}</li>`).join('')}
      </ul>
      <div style="display:flex; justify-content:flex-end; gap:8px;">
        <button class="btn btn-primary" id="viewAllInsightsBtn">Open Insights &amp; Recaps</button>
      </div>`;
    $('summaryModal').classList.add('open');
    $('viewAllInsightsBtn').addEventListener('click', () => {
      $('summaryModal').classList.remove('open');
      switchMainView('view-insights');
    });
  }
  $('closeSummaryBtn').addEventListener('click', () => $('summaryModal').classList.remove('open'));

  // ---------------- CRM & Calendar View ----------------
  async function loadCRM() {
    const [leadsRes, apptsRes, whRes, statusRes] = await Promise.all([
      fetch('/api/leads'),
      fetch('/api/appointments'),
      fetch('/api/webhooks'),
      fetch('/api/crm/status'),
    ]);
    const leads = await leadsRes.json();
    const appts = await apptsRes.json();
    const hooks = await whRes.json();
    const crmStatus = await statusRes.json();

    $('crmModeBadge').textContent = crmStatus.mode || 'local-sqlite';

    const jobs = crmStatus.all_jobs || [];
    $('crmSyncJobsBox').innerHTML = jobs.length
      ? jobs
          .slice(0, 6)
          .map(
            (j) => `
        <div class="list-item" style="margin-bottom:8px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
            <strong style="font-size:12.5px;">${esc(j.adapter)} · ${esc(j.operation)}</strong>
            <div style="display:flex; gap:6px; align-items:center;">
              <span class="badge ${j.status === 'synced' ? 'hot' : j.status === 'mock' ? 'warm' : 'neg'}">${esc(j.status)}${j.http_status ? ' (' + j.http_status + ')' : ''}</span>
              ${j.status === 'failed' ? `<button class="btn btn-sm retry-crm-btn" data-job="${esc(j.id)}">Retry</button>` : ''}
            </div>
          </div>
          <div style="font-size:11.5px; color:var(--ink-muted); margin-top:3px;">
            Record ID: <code class="mono">${esc(j.external_record_id || 'none')}</code> · Retries: ${j.retry_count ?? 0}${j.error_code ? ' · Code: ' + esc(j.error_code) : ''}
          </div>
          ${j.failure_reason ? `<div style="font-size:11.5px; color:var(--ink-secondary); margin-top:3px;">${esc(j.failure_reason)}</div>` : ''}
        </div>`
          )
          .join('')
      : '<div class="list-item" style="font-size:12px; color:var(--ink-muted);">Saved locally (SQLite). No external CRM adapter configured.</div>';

    document.querySelectorAll('.retry-crm-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const r = await fetch(`/api/crm/retry/${encodeURIComponent(btn.dataset.job)}`, { method: 'POST' });
        if (r.ok) {
          const d = await r.json();
          showToast(`Retry ${d.provider}: ${d.external_sync_status}`, d.external_sync_status === 'synced' ? 'success' : 'error');
          loadCRM();
        }
      });
    });

    $('crmLeadsBody').innerHTML = leads.length
      ? leads
          .map(
            (l) => `
      <tr>
        <td><strong>${esc(l.name)}</strong><br/><span style="font-size:12px; color:var(--ink-muted);">${esc([l.email, l.phone].filter(Boolean).join(' · ') || '—')}</span></td>
        <td>${esc(l.company || '—')}<br/><span style="font-size:12px; color:var(--ink-muted);">${esc(l.role || '—')}</span></td>
        <td><span class="badge">${esc(l.preferred_language.toUpperCase())}</span></td>
        <td><strong>${l.score}</strong>/100</td>
        <td><span class="badge ${l.tier === 'Disqualified' ? 'neg' : l.tier.toLowerCase()}">${esc(l.tier)}</span> <span style="font-size:12px; color:var(--ink-secondary);">${esc(l.stage)}</span></td>
        <td style="font-variant-numeric:tabular-nums;">$${Number(l.estimated_acv_usd || 0).toLocaleString()}</td>
        <td><div class="tags-wrap" style="margin:0;">
          ${(l.pain_points || []).map((p) => `<span class="badge blue">${esc(p)}</span>`).join('')}
          ${(l.objections || []).map((o) => `<span class="badge neg">${esc(o)}</span>`).join('')}
        </div></td>
      </tr>`
          )
          .join('')
      : `<tr><td colspan="7" class="empty-state">No CRM leads recorded yet.</td></tr>`;

    $('appointmentsList').innerHTML = appts.length
      ? appts
          .map(
            (a) => `
        <div class="list-item" style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
          <div>
            <strong>${esc(a.lead_name)}</strong>${a.company ? ` (${esc(a.company)})` : ''}
            <span class="badge ${a.status === 'Cancelled' ? 'neg' : 'hot'}" style="margin-left:6px;">${esc(a.status)}</span><br/>
            <span style="font-size:12px; color:var(--ink-muted);">${esc(new Date(a.slot_iso).toUTCString().replace(':00 GMT', ' UTC'))} · <code class="mono">${esc(a.id)}</code></span>
          </div>
          ${
            a.status !== 'Cancelled'
              ? `<div style="display:flex; gap:6px;">
                  <button class="btn btn-sm resched-appt" data-id="${esc(a.id)}" data-name="${esc(a.lead_name)}" data-email="${esc(a.lead_email || '')}" data-company="${esc(a.company || '')}">Reschedule</button>
                  <button class="btn btn-sm btn-danger-outline cancel-appt" data-id="${esc(a.id)}">Cancel</button>
                </div>`
              : ''
          }
        </div>`
          )
          .join('')
      : '<div class="empty-state">No appointments scheduled.</div>';

    document.querySelectorAll('.cancel-appt').forEach((b) => {
      b.addEventListener('click', async () => {
        await fetch(`/api/appointments/${encodeURIComponent(b.dataset.id)}`, { method: 'DELETE' });
        showToast('Appointment cancelled', 'info');
        loadCRM();
      });
    });

    document.querySelectorAll('.resched-appt').forEach((b) => {
      b.addEventListener('click', () => openBookingModal(b.dataset));
    });

    $('webhooksList').innerHTML = hooks.length
      ? hooks
          .slice(0, 5)
          .map(
            (h) => `
        <div class="list-item">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="font-size:12.5px;">${esc(h.event_type)}</strong>
            <span class="badge ${h.delivered ? 'hot' : 'neg'}">${h.delivered ? 'Delivered' : 'Failed'}</span>
          </div>
          <div style="font-size:11.5px; color:var(--ink-muted); margin-top:2px;">${esc(h.target)} · ${esc(h.created_at)}</div>
          <pre class="json-box">${esc(JSON.stringify(h.payload, null, 2))}</pre>
        </div>`
          )
          .join('')
      : '<div class="empty-state">No CRM delivery logs yet.</div>';
  }

  $('refreshCrmBtn').addEventListener('click', loadCRM);
  $('backupDbBtn').addEventListener('click', async () => {
    const res = await fetch('/api/admin/backup', { method: 'POST' });
    if (res.ok) {
      const d = await res.json();
      showToast(`Backup created: ${d.backup_file}`, 'success');
    } else {
      showToast('Backup failed', 'error');
    }
  });

  async function openBookingModal(reschedData = null) {
    const tz = $('timezoneSelect') ? $('timezoneSelect').value : 'UTC';
    const res = await fetch(`/api/appointments/slots?timezone=${encodeURIComponent(tz)}`);
    const data = await res.json();
    const localSlots = data.local_slots || (data.slots || []).map((s) => ({ slot_iso: s, local_display: s }));
    $('bookSlotSelect').innerHTML = localSlots
      .map((item) => `<option value="${esc(item.slot_iso)}">${esc(item.local_display)} (${esc(item.slot_iso)})</option>`)
      .join('');

    if (reschedData && reschedData.id) {
      $('bookModalTitle').textContent = `Reschedule Appointment (${reschedData.id})`;
      $('rescheduleApptId').value = reschedData.id;
      $('bookLeadName').value = reschedData.name || '';
      $('bookLeadEmail').value = reschedData.email || '';
      $('bookCompany').value = reschedData.company || '';
      $('bookLeadName').disabled = true;
      $('bookSubmitBtn').textContent = 'Save New Slot';
    } else {
      $('bookModalTitle').textContent = 'Schedule Demo Appointment';
      $('rescheduleApptId').value = '';
      $('manualBookForm').reset();
      $('bookLeadName').disabled = false;
      $('bookSubmitBtn').textContent = 'Confirm Booking';
    }
    $('bookModal').classList.add('open');
  }

  $('openBookModalBtn').addEventListener('click', () => openBookingModal(null));
  $('closeBookModalBtn').addEventListener('click', () => $('bookModal').classList.remove('open'));
  $('manualBookForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const reschedId = $('rescheduleApptId').value;
    let res;
    if (reschedId) {
      res = await fetch(`/api/appointments/${encodeURIComponent(reschedId)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot_iso: $('bookSlotSelect').value }),
      });
    } else {
      res = await fetch('/api/appointments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lead_name: $('bookLeadName').value.trim(),
          lead_email: $('bookLeadEmail').value.trim() || null,
          company: $('bookCompany').value.trim() || null,
          slot_iso: $('bookSlotSelect').value,
        }),
      });
    }
    if (res.ok) {
      showToast(reschedId ? 'Appointment rescheduled' : 'Appointment confirmed', 'success');
      $('bookModal').classList.remove('open');
      loadCRM();
    } else {
      const err = await res.json();
      showToast(err.detail || 'Booking error', 'error');
    }
  });

  // ---------------- Knowledge Base View ----------------
  async function loadKnowledge() {
    const res = await fetch('/api/knowledge');
    const data = await res.json();
    const chunks = data.chunks || [];
    $('kbChunkCount').textContent = `${chunks.length} chunks`;
    $('kbChunksList').innerHTML = chunks
      .map(
        (c) => `
      <div class="list-item">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
          <strong style="font-size:13px;">${esc(c.title)}</strong>
          <div style="display:flex; gap:6px; align-items:center;">
            <span class="badge">${esc(c.category)}</span>
            ${
              c.parent_id && c.parent_id.startsWith('doc-')
                ? `<button class="btn btn-sm btn-danger-outline del-doc-btn" data-id="${esc(c.parent_id)}">Delete</button>`
                : ''
            }
          </div>
        </div>
        <div style="font-size:11px; color:var(--ink-muted); margin-top:2px;">Source: ${esc(c.source_file || c.doc_id)}</div>
        <div style="color:var(--ink-secondary); font-size:12.5px; margin-top:4px;">${esc(c.preview)}</div>
      </div>`
      )
      .join('');

    document.querySelectorAll('.del-doc-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const r = await fetch(`/api/knowledge/${encodeURIComponent(btn.dataset.id)}`, { method: 'DELETE' });
        if (r.ok) {
          showToast('Document removed from vector index', 'info');
          loadKnowledge();
        }
      });
    });
  }

  $('kbSearchForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = $('kbSearchInput').value.trim();
    if (!q) return;
    const res = await fetch(`/api/knowledge/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    $('kbSearchResults').innerHTML =
      (data.results || [])
        .map(
          (r) => `
      <div class="list-item">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong style="font-size:12.5px;">${esc(r.title)}</strong>
          <span class="badge hot">${Math.round(r.score * 100)}% hybrid</span>
        </div>
        <div style="font-size:11px; color:var(--ink-muted);">Sparse: ${r.sparse_score} · Dense: ${r.dense_score}</div>
        <div style="font-size:12.5px; color:var(--ink-secondary); margin-top:4px;">${esc(r.content)}</div>
      </div>`
        )
        .join('') || '<div class="empty-state">No matching knowledge chunks found.</div>';
  });

  $('kbFileForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = $('kbFileInput').files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    fd.append('category', $('kbFileCategory').value || 'Uploaded');
    const res = await fetch('/api/knowledge/upload', { method: 'POST', body: fd });
    if (res.ok) {
      const d = await res.json();
      showToast(`Indexed ${d.added_chunks} chunks from ${d.filename}`, 'success');
      $('kbFileForm').reset();
      loadKnowledge();
    } else {
      const err = await res.json();
      showToast(err.detail || 'Upload failed', 'error');
    }
  });

  $('kbAddForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const res = await fetch('/api/knowledge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: $('kbDocTitle').value,
        category: $('kbDocCategory').value,
        content: $('kbDocContent').value,
      }),
    });
    if (res.ok) {
      showToast('Knowledge document indexed', 'success');
      $('kbDocTitle').value = '';
      $('kbDocContent').value = '';
      loadKnowledge();
    }
  });

  // ---------------- Call History View ----------------
  async function loadCallHistory() {
    const res = await fetch('/api/sessions');
    const sessions = await res.json();
    $('historySessionsList').innerHTML = sessions.length
      ? sessions
          .map(
            (s) => `
        <div class="session-row" data-sid="${esc(s.session_id)}">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="font-size:13px;">${esc(s.lead_name)}</strong>
            <span class="badge ${s.tier === 'Disqualified' ? 'neg' : s.tier.toLowerCase()}">${s.score}/100</span>
          </div>
          <div style="font-size:11.5px; color:var(--ink-muted); margin-top:3px;">
            ${esc(s.company || 'No company')} · ${esc(s.language.toUpperCase())} · ${s.message_count} msgs
          </div>
        </div>`
          )
          .join('')
      : '<div class="empty-state">No recorded calls yet.</div>';

    document.querySelectorAll('.session-row').forEach((row) => {
      row.addEventListener('click', async () => {
        document.querySelectorAll('.session-row').forEach((r) => r.classList.remove('active'));
        row.classList.add('active');
        const detailRes = await fetch(`/api/sessions/${encodeURIComponent(row.dataset.sid)}`);
        if (!detailRes.ok) return;
        const d = await detailRes.json();
        $('historyDetailBox').innerHTML = `
          <div class="kv-grid" style="margin-bottom:12px;">
            <div class="kv-cell"><label>Session ID</label><div class="mono">${esc(d.session_id)}</div></div>
            <div class="kv-cell"><label>Lead Tier &amp; Score</label><div>${esc(d.qualification.tier)} (${d.qualification.score}/100)</div></div>
            <div class="kv-cell"><label>Long-Term Returning</label><div>${d.memory.returning_lead ? 'Yes' : 'No'} (Prior calls: ${d.memory.prior_calls_count})</div></div>
            <div class="kv-cell"><label>Tools Executed</label><div>${(d.tool_logs || []).length} logged actions</div></div>
          </div>
          <div style="font-size:12.5px; font-weight:600; margin-bottom:6px;">Transcript</div>
          <div style="max-height:280px; overflow-y:auto; margin-bottom:12px;">
            ${(d.messages || [])
              .map(
                (m) => `<div class="list-item" style="margin-bottom:6px;">
                  <strong>${m.role === 'user' ? 'Prospect' : 'Nova AI'}</strong> <span class="badge">${esc(m.language.toUpperCase())}</span>
                  <div style="margin-top:3px; color:var(--ink-secondary);">${esc(m.text)}</div>
                </div>`
              )
              .join('')}
          </div>
          <div style="font-size:12.5px; font-weight:600; margin-bottom:6px;">Tool Execution Audit</div>
          <pre class="json-box">${esc(JSON.stringify(d.tool_logs || [], null, 2))}</pre>`;
      });
    });
  }
  $('refreshHistoryBtn').addEventListener('click', loadCallHistory);

  // ---------------- Insights & Recaps View ----------------
  async function loadInsights() {
    const params = new URLSearchParams();
    if ($('filterDateFrom').value) params.set('date_from', $('filterDateFrom').value);
    if ($('filterDateTo').value) params.set('date_to', $('filterDateTo').value);
    if ($('filterTier').value) params.set('tier', $('filterTier').value);
    if ($('filterOutcome').value) params.set('outcome', $('filterOutcome').value);
    const qs = params.toString() ? `?${params.toString()}` : '';

    const [insRes, sumRes] = await Promise.all([fetch(`/api/insights${qs}`), fetch(`/api/summaries${qs}`)]);
    const ins = await insRes.json();
    const sums = await sumRes.json();

    $('insightsKpiGrid').innerHTML = [
      ['Total Calls', ins.total_calls],
      ['Unique Leads', ins.unique_leads],
      ['Qualified (Hot+Warm)', ins.qualified_leads],
      ['Booked Demos', ins.booked_demos],
      ['Booking Rate', `${ins.booking_rate_pct}%`],
      ['Avg Real Duration', `${ins.average_call_duration_sec ?? 0}s`],
      ['Qualified Pipeline', `$${Number(ins.total_pipeline_usd || 0).toLocaleString()}`],
    ]
      .map(([k, v]) => `<div class="kpi-card"><small>${esc(k)}</small><strong>${esc(v)}</strong></div>`)
      .join('');

    const pains = (ins.top_pain_points || []).map((x) => `<span class="badge blue">${esc(x.label)} (${x.count})</span>`).join('') || '—';
    const objs = (ins.top_objections || []).map((x) => `<span class="badge neg">${esc(x.label)} (${x.count})</span>`).join('') || '—';
    const faqs = (ins.frequently_asked_questions || []).map((x) => `<span class="badge">${esc(x.topic)} (${x.count})</span>`).join('') || '—';
    $('insightsTrendsBox').innerHTML = `
      <div style="margin-bottom:10px;"><div style="font-size:12px; color:var(--ink-muted); margin-bottom:4px;">Pain Points</div><div class="tags-wrap">${pains}</div></div>
      <div style="margin-bottom:10px;"><div style="font-size:12px; color:var(--ink-muted); margin-bottom:4px;">Objections</div><div class="tags-wrap">${objs}</div></div>
      <div><div style="font-size:12px; color:var(--ink-muted); margin-bottom:4px;">Frequently Asked Topics</div><div class="tags-wrap">${faqs}</div></div>`;

    const langs = Object.entries(ins.language_distribution || {}).map(([k, v]) => `<span class="badge">${esc(k.toUpperCase())}: ${v}</span>`).join('');
    const stages = Object.entries(ins.stage_funnel || {}).map(([k, v]) => `<span class="badge warm">${esc(k)}: ${v}</span>`).join('');
    const perf = ins.agent_performance || {};
    $('insightsFunnelBox').innerHTML = `
      <div style="margin-bottom:10px;"><div style="font-size:12px; color:var(--ink-muted); margin-bottom:4px;">Pipeline Stages</div><div class="tags-wrap">${stages}</div></div>
      <div style="margin-bottom:10px;"><div style="font-size:12px; color:var(--ink-muted); margin-bottom:4px;">Languages Served</div><div class="tags-wrap">${langs}</div></div>
      <div style="font-size:12.5px; color:var(--ink-secondary);">
        Avg Turns/Call: <strong>${perf.avg_turns_per_call ?? 0}</strong> ·
        Avg Latency: <strong>${perf.avg_response_latency_ms ?? 0}ms</strong> ·
        Interruptions: <strong>${perf.total_interruptions ?? 0}</strong> ·
        Qualification Rate: <strong>${perf.qualification_rate_pct ?? 0}%</strong>
      </div>`;

    $('summariesList').innerHTML = sums.length
      ? sums
          .map(
            (s) => `
      <div class="list-item">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:6px;">
          <strong>${esc(s.lead_name)}${s.company ? ` · ${esc(s.company)}` : ''}</strong>
          <div style="display:flex; gap:6px;">
            <span class="badge">${esc(s.language.toUpperCase())}</span>
            <span class="badge ${s.qualification.tier === 'Disqualified' ? 'neg' : s.qualification.tier.toLowerCase()}">${s.qualification.score}/100 (${esc(s.qualification.tier)})</span>
            <span class="badge">Sentiment: ${s.average_sentiment} (${esc(s.sentiment_trend)})</span>
          </div>
        </div>
        <p style="margin:6px 0; color:var(--ink-secondary); line-height:1.55;">${esc(s.executive_summary)}</p>
        <div style="font-size:12px; color:var(--ink-muted); margin-bottom:4px;">
          Sources: ${(s.knowledge_sources_used || []).map(esc).join(', ') || 'None'} · Follow-up: ${esc(s.follow_up_date || '—')}
        </div>
        <div style="font-size:12px; color:var(--green);"><strong>Next Steps:</strong> ${(s.next_steps || []).map(esc).join(' · ')}</div>
      </div>`
          )
          .join('')
      : '<div class="empty-state">No call summary reports match the current filters.</div>';
  }

  $('applyFiltersBtn').addEventListener('click', loadInsights);
  $('resetFiltersBtn').addEventListener('click', () => {
    $('filterDateFrom').value = '';
    $('filterDateTo').value = '';
    $('filterTier').value = '';
    $('filterOutcome').value = '';
    loadInsights();
  });

  // ---------------- Settings Modal ----------------
  $('openSettingsBtn').addEventListener('click', () => $('settingsModal').classList.add('open'));
  $('closeSettingsBtn').addEventListener('click', () => $('settingsModal').classList.remove('open'));
  $('ttsRateInput').addEventListener('input', (e) => {
    state.ttsRate = parseFloat(e.target.value);
    $('ttsRateVal').textContent = state.ttsRate.toFixed(2);
  });
  $('settingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      stt_provider: $('sttProviderSelect').value,
      tts_provider: $('ttsProviderSelect').value,
      crm_mock_mode: $('crmMockToggle').checked,
    };
    if ($('geminiKeyInput').value.trim()) payload.gemini_api_key = $('geminiKeyInput').value.trim();
    if ($('crmWebhookInput').value.trim()) payload.crm_webhook_url = $('crmWebhookInput').value.trim();
    if ($('hubspotTokenInput').value.trim()) payload.hubspot_access_token = $('hubspotTokenInput').value.trim();

    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      const d = await res.json();
      $('engineBadge').textContent = d.ai_engine;
      $('settingsModal').classList.remove('open');
      showToast(`Settings saved (${d.ai_engine} · STT: ${d.stt_provider})`, 'success');
    } else {
      showToast('Failed to save configuration', 'error');
    }
  });

  // ---------------- Auth Session Modal ----------------
  async function checkAuthStatus() {
    try {
      const r = await fetch('/api/auth/status');
      if (!r.ok) return;
      const d = await r.json();
      if (d.token_required && !d.authenticated) {
        $('authBtn').style.display = 'inline-flex';
      } else {
        $('authBtn').style.display = 'none';
      }
    } catch {}
  }

  if ($('authBtn')) {
    $('authBtn').addEventListener('click', () => $('authModal').classList.add('open'));
    $('closeAuthModalBtn').addEventListener('click', () => $('authModal').classList.remove('open'));
    $('authForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const token = $('authTokenInput').value;
      const r = await fetch('/api/auth/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      });
      if (r.ok) {
        $('authTokenInput').value = '';
        $('authModal').classList.remove('open');
        $('authBtn').style.display = 'none';
        showToast('Authenticated session active', 'success');
      } else {
        showToast('Invalid API token', 'error');
      }
    });
  }

  // Boot
  checkAuthStatus();
  connectWebSocket();
  startWaveform();
  appendMessage(
    'assistant',
    "Hello! You've reached Fortivex Sales. I can answer questions about pricing, security compliance, or CRM integrations in English, Hindi, Hinglish, Spanish, French, or German—and book a live demo when you're ready.",
    { language: 'en' }
  );
})();
