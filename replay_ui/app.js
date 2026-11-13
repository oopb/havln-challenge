/* HA-VLN Replay Studio: no build dependency, no external network calls. */
'use strict';

const $ = (id) => document.getElementById(id);
const state = { episode: null, data: null, filtered: [], selected: -1,
  detail: null, tab: 'input', requestSeq: 0 };
const MAX_TEXT = 5_000_000;

function node(tag, className, value) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (value !== undefined && value !== null) el.textContent = String(value);
  return el;
}
function add(parent, ...children) { for (const child of children) parent.append(child); return parent; }
function fmt(value) {
  if (value === undefined) return '(未记录)';
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value, null, 2); } catch { return String(value); }
}
function pre(value) {
  const el = node('pre', 'json-pre', fmt(value));
  if (el.textContent.length > MAX_TEXT) el.textContent = el.textContent.slice(0, MAX_TEXT) + '\n[前端展示已截断；原始 JSON 仍可从磁盘读取]';
  return el;
}
function section(title, content, open = false) {
  const wrap = node('details', 'data-section'); wrap.open = open;
  add(wrap, node('summary', '', title), content instanceof Node ? content : pre(content));
  return wrap;
}
function message(text, bad = false) {
  const toast = $('toast'); toast.textContent = text; toast.className = 'toast' + (bad ? ' bad' : '');
  clearTimeout(message.timer);
  message.timer = setTimeout(() => { toast.className = 'toast hidden'; }, 4500);
}
async function json(path) {
  const response = await fetch(path, {cache: 'no-store'});
  if (!response.ok) {
    let text = String(response.status);
    try { text = (await response.json()).error || text; } catch {}
    throw new Error(text);
  }
  return response.json();
}
function pathFor(episode, suffix) {
  return '/api/episode/' + encodeURIComponent(episode) + '/' + suffix;
}
function frameUrl(ident, channel = 'rgb', maxEdge = 0) {
  if (!ident || !/^obs_[0-9]+$/.test(ident)) return null;
  return pathFor(state.episode, 'frame/' + ident) + '?channel=' + channel +
    (maxEdge ? '&max_edge=' + maxEdge : '');
}
function mediaUrl(ref) {
  if (!ref) return null;
  if (ref.kind === 'observation') return frameUrl(ref.observation_id, ref.sensor || 'rgb', ref.max_edge || 0);
  if (ref.kind === 'stored' && /^[a-f0-9]{64}$/.test(ref.sha256 || '')) {
    return pathFor(state.episode, 'media/' + ref.sha256);
  }
  return null;
}
function setImage(el, placeholder, url, alt) {
  el.onload = () => { placeholder.classList.add('hidden'); el.classList.remove('hidden'); };
  el.onerror = () => { el.classList.add('hidden'); placeholder.classList.remove('hidden'); };
  el.alt = alt;
  if (!url) { el.removeAttribute('src'); el.classList.add('hidden'); placeholder.classList.remove('hidden'); }
  else { el.classList.remove('hidden'); el.src = url; }
}
function label(ev) {
  if (ev.kind === 'api') return (ev.role === 'visual_perception' ? 'ObservationAgent' :
    ev.role === 'navigation_agent' ? 'NavigationAgent' :
    ev.role === 'local_control' ? 'LocalController' : (ev.role || 'Model')) +
    ' · ' + (ev.purpose || 'API');
  if (ev.kind === 'action') return '#'+String((ev.step ?? 0)+1)+' · '+(ev.action || 'ACTION');
  if (ev.kind === 'tool') return (ev.tool || 'Tool') + (ev.status ? ' · ' + ev.status : '');
  if (ev.kind === 'observation') return '传感器 · ' + (ev.observation_id || '?');
  if (ev.kind === 'runtime_event') return ev.name || '运行事件';
  return ev.kind || '事件';
}
function badgeText(ev) {
  if (ev.kind === 'api') return ev.cache_hit ? '缓存' :
    ev.success === false ? '失败' : ev.role === 'navigation_agent' ? 'NAV' : 'VLM';
  if (ev.kind === 'action') return 'ACTION';
  if (ev.kind === 'tool') return 'TOOL';
  return 'EVENT';
}
function isIssue(ev) {
  return (ev.kind === 'api' && ev.success === false) ||
    (ev.kind === 'tool' && /(BLOCKED|PARTIAL|ERROR|ATTENTION|INVALID|UNAVAILABLE)/.test(ev.status || '')) ||
    (ev.kind === 'runtime_event' && /error|budget|failure|liveness/i.test(ev.name || ''));
}
function filterItems() {
  if (!state.data) return;
  const mode = $('modeSelect').value;
  const q = $('searchBox').value.toLowerCase().trim();
  const events = state.data.events;
  state.filtered = events.filter(ev => {
    if (mode === 'api' && ev.kind !== 'api') return false;
    if (mode === 'action' && ev.kind !== 'action') return false;
    if (mode === 'issues' && !isIssue(ev)) return false;
    if (q && !(label(ev)+' '+JSON.stringify(ev)).toLowerCase().includes(q)) return false;
    return true;
  }).map(ev => ev.sequence);
  $('eventCounter').textContent = state.filtered.length + ' / ' + events.length;
  $('scrubber').max = String(Math.max(0, state.filtered.length-1));
  $('jumpInput').max = String(Math.max(1, state.filtered.length));
  renderList();
  if (state.filtered.length) {
    let position = state.filtered.indexOf(state.selected);
    if (position < 0) {
      position = state.filtered.findIndex(i => i >= state.selected);
      if (position < 0) position = state.filtered.length-1;
    }
    selectIndex(state.filtered[position]);
  } else {
    state.selected = -1;
    $('positionText').textContent = '没有匹配的事件';
    $('eventTitle').textContent = '';
    $('detailsBody').replaceChildren(node('p','muted','当前筛选没有事件。'));
    setImage($('rgbImg'), $('rgbMissing'), null, '');
    setImage($('depthImg'), $('depthMissing'), null, '');
    setImage($('afterImg'), $('afterMissing'), null, '');
  }
}
function renderList() {
  const list = $('eventList'); list.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (let order = 0; order < state.filtered.length; order++) {
    const index = state.filtered[order], ev = state.data.events[index];
    const btn = node('button', 'timeline-entry' +
      (index === state.selected ? ' selected' : '') + (isIssue(ev) ? ' issue' : ''));
    btn.type = 'button';
    const head = node('div','timeline-head');
    add(head, node('span', 'event-index', String(order + 1).padStart(3,'0')),
      node('span', 'kind-badge ' + (isIssue(ev) ? 'warn' : ''), badgeText(ev)));
    add(btn, head, node('div','timeline-title', label(ev)));
    const sub = ev.observation_id || ev.after_id ||
      ev.decision_id || ev.tool_call_id || '';
    if (sub) btn.append(node('div', 'timeline-sub', sub));
    btn.addEventListener('click', () => selectIndex(index));
    fragment.append(btn);
  }
  list.append(fragment);
}
async function selectIndex(index) {
  if (!state.data || index < 0 || index >= state.data.events.length) return;
  state.selected = index;
  const pos = state.filtered.indexOf(index);
  $('scrubber').value = String(Math.max(0, pos));
  $('jumpInput').value = String(pos + 1);
  $('positionText').textContent = (pos + 1) + ' / ' + state.filtered.length;
  const ev = state.data.events[index];
  $('eventTitle').textContent = label(ev);
  $('visualLabel').textContent = ev.observation_id || ev.after_id || '—';
  $('detailTitle').textContent = label(ev);
  $('detailType').textContent = badgeText(ev);
  document.querySelectorAll('.timeline-entry').forEach((el, order) => {
    el.classList.toggle('selected', state.filtered[order] === index);
  });
  const active = document.querySelector('.timeline-entry.selected');
  if (active) active.scrollIntoView({block:'nearest'});
  let frame = ev.observation_id;
  if (!frame) {
    for (let i=index-1; i>=0; i--) {
      if (state.data.events[i].observation_id) {
        frame = state.data.events[i].observation_id; break;
      }
    }
  }
  $('rgbSource').textContent = frame || '';
  $('afterSource').textContent = ev.after_id || '';
  setImage($('rgbImg'), $('rgbMissing'), frameUrl(frame), '当前真实 RGB ' + frame);
  setImage($('depthImg'), $('depthMissing'), frameUrl(frame,'depth'), '当前 Depth ' + frame);
  setImage($('afterImg'), $('afterMissing'), frameUrl(ev.kind === 'action' ? ev.after_id : null),
    '动作执行后的 RGB');
  $('mediaGallery').replaceChildren();
  $('mediaNote').textContent = '';
  $('executionSummary').textContent = '读取中…';
  $('poseStrip').textContent = ev.observation_id ? 'Observation: ' + ev.observation_id : '当前帧来自最近已记录的观察';
  const seq = ++state.requestSeq;
  try {
    const detail = await json(pathFor(state.episode, 'event/' + index));
    if (seq !== state.requestSeq) return;
    state.detail = detail;
    renderDetail(detail);
  } catch (e) {
    if (seq !== state.requestSeq) return;
    $('detailsBody').replaceChildren(node('p','error-text','载入失败：'+e.message));
    $('executionSummary').textContent = e.message;
  }
}
function move(delta) {
  if (!state.filtered.length) return;
  const current = state.filtered.indexOf(state.selected);
  const next = Math.max(0, Math.min(state.filtered.length-1, current + delta));
  selectIndex(state.filtered[next]);
}
function mediaCard(ref, i) {
  const card = node('figure','media-card');
  const url = mediaUrl(ref);
  if (url) {
    const img = node('img'); img.src = url; img.loading = 'lazy';
    img.alt = '模型输入图片 ' + (i+1);
    img.addEventListener('click', () => window.open(url, '_blank', 'noopener'));
    card.append(img);
  } else card.append(node('div','media-placeholder','图片未记录或无法还原'));
  const suffix = ref.kind === 'observation' ?
    (ref.observation_id + ' · ' + (ref.sensor || 'rgb').toUpperCase()) :
    ref.kind === 'stored' ? '原始发送图片' : '不可用';
  card.append(node('figcaption','', '#'+(i+1)+' · '+suffix));
  return card;
}
function prettyContext(value) {
  try { return JSON.parse(value); } catch { return value; }
}
function inputView(row) {
  const wrap = node('div','detail-stack');
  const messages = row.messages || [];
  const replay = row.replay;
  if (!replay || replay.incomplete) {
    wrap.append(node('p','notice',
      '旧运行或采集不完整：部分历史对话、发送图片与响应 Schema 无法从旧日志中恢复。不会补造模型输入。'));
  }
  const system = messages.find(m => m.role === 'system');
  const user = messages.find(m => m.role === 'user');
  if (system) wrap.append(section('System Prompt · 原文',
                                   system.content, true));
  if (user) wrap.append(section('User Context · 实际发送的 JSON 文本',
                                 prettyContext(user.content), true));
  if (replay) {
    if (replay.request_parameters)
      wrap.append(section('请求参数 / Response Schema', replay.request_parameters));
    const history = replay.prior_messages || [];
    if (history.length) {
      const group = node('div', 'history-stack');
      history.forEach((turn,i) => {
        const messageBox = node('div','history-message');
        const content = turn.content;
        if (typeof content === 'string') messageBox.append(pre(prettyContext(content)));
        else if (Array.isArray(content)) {
          for (const part of content) {
            if (part.type === 'text') messageBox.append(pre(prettyContext(part.text)));
            if (part.type === 'image_url' && part.image_url?.replay_media) {
              messageBox.append(mediaCard(part.image_url.replay_media, i));
            }
          }
        }
        group.append(section('历史消息 ' + (i+1) + ' · ' + (turn.role || '?'),
                             messageBox, false));
      });
      wrap.append(section('完整历史对话 · ' + history.length + ' 条', group));
    } else wrap.append(node('p','muted','本次请求无先前对话消息。'));
    const media = replay.current_media || [];
    if (media.length) {
      const gallery = node('div','media-gallery');
      media.forEach((ref, i) => gallery.append(mediaCard(ref,i)));
      wrap.append(section('当前 User 消息附带图片 · '+media.length+' 张', gallery, true));
    }
  }
  return wrap;
}
function outputView(row) {
  const wrap = node('div','detail-stack');
  const status = node('div', 'output-status ' + (row.success === false ? 'error-text' : ''));
  status.textContent = 'API '+(row.success ? '成功' : '失败') +
    ' · ' + (row.model || '未知模型') +
    ' · ' + (row.seconds == null ? '延迟未知' : Number(row.seconds).toFixed(2) + 's');
  wrap.append(status);
  wrap.append(section('原始模型响应（未加工）', row.raw_response, true));
  wrap.append(section('解析并校验的结果', row.result, true));
  if (row.error || row.validation_error_type)
    wrap.append(section('错误 / Schema 校验信息', {
      error:row.error, detail:row.error_detail,
      validation_error_type:row.validation_error_type,
      field:row.validation_field, normalization:row.response_normalization,
      repair:row.repair_result, retry:row.retry_scheduled,
      fallback:row.fallback_reason
    }, true));
  wrap.append(section('模型 / Tokens / 传输信息', {
    role:row.role, purpose:row.purpose, model:row.model,
    response_model:row.response_model, usage:row.usage,
    finish_reason:row.finish_reason, endpoint_id:row.endpoint_id,
    attempt_number:row.attempt_number, completion_request_id:row.completion_request_id,
    cache_hit:row.cache_hit, request_body_sha256:row.request_body_sha256
  }));
  return wrap;
}
function renderDetail(detail) {
  const ev = detail.event, row = detail.record || {};
  const body = $('detailsBody'); body.replaceChildren();
  const media = row.replay?.current_media || [];
  const gallery = $('mediaGallery');
  media.forEach((ref, i) => gallery.append(mediaCard(ref,i)));
  $('mediaNote').textContent = ev.kind === 'api' ?
    (row.replay ? media.length + ' 张当前图片，历史图片见输入' :
    '历史日志未记录完整发送图片') : '仅 API 请求会显示实际模型图片';
  const output = $('executionSummary');
  if (ev.kind === 'api') {
    output.textContent = (row.success ? '模型请求成功' : '模型请求失败') +
      ' · ' + (row.model || '') +
      (row.error ? ' · '+row.error : '') +
      (row.cache_hit ? ' · 命中缓存（未发送 HTTP 请求）' : '');
    const id = ev.observation_id || '';
    if (id) $('poseStrip').textContent = '推理绑定帧：' + id +
      ' · decision_id: ' + (row.decision_id || '—') +
      ' · request #' + (row.request_number ?? 'cache');
    if (state.tab === 'input') body.append(inputView(row));
    else if (state.tab === 'output') body.append(outputView(row));
    else body.append(section('api.jsonl · 完整原始记录', row, true));
  } else {
    const pose = row.pose_after || row.pose || {};
    $('poseStrip').textContent = 'Action #'+((row.step ?? -1)+1) +
      (pose.stance_id ? ' · Stance '+pose.stance_id : '') +
      (typeof pose.pitch_deg === 'number' ? ' · Pitch '+pose.pitch_deg+'°' : '') +
      (typeof pose.heading === 'number' ? ' · Yaw '+(pose.heading*180/Math.PI).toFixed(1)+'°' : '');
    if (ev.kind === 'action') {
      output.textContent = (row.action || '动作') +
        ' · Input '+(ev.observation_id || '—')+
        ' → Output '+(ev.after_id || '—')+
        ' · '+(row.safety?.status || '未记录检查');
    } else if (ev.kind === 'tool') {
      output.textContent = (row.call?.tool || '工具') +
        ' · ' + (row.result?.status || '无返回状态');
    } else output.textContent = label(ev);
    body.append(section('事件 / 工具 / 动作记录', row, true));
    if (state.tab === 'input' && ev.kind === 'tool') {
      body.prepend(section('工具调用参数', row.call, true));
    }
    if (state.tab === 'output' && ev.kind === 'tool') {
      body.prepend(section('工具执行结果', row.result, true));
    }
  }
}
function scoreNumber(value, decimals = 4) {
  return typeof value === 'number' && Number.isFinite(value) ?
    value.toFixed(decimals) : '—';
}
function scoreRate(value) {
  return typeof value === 'number' && Number.isFinite(value) ?
    (value * 100).toFixed(2) + '%' : '—';
}
function renderScoring(scoring) {
  const scored = scoring && scoring.status === 'scored';
  const metrics = scored ? scoring.metrics : {};
  $('officialScore').textContent = scored ? scoreNumber(scoring.score, 5) : '—';
  $('scoreStatus').textContent = scored ? '离线复算 · 非官方成绩' :
    scoring?.status === 'invalid_metrics' ? '评测数据无效' : '缺少评测数据';
  $('scoreStatus').classList.toggle('warning', !scored);
  $('scoreSR').textContent = scoreRate(metrics?.SR);
  $('scoreNE').textContent = scoreNumber(metrics?.NE, 3);
  $('scoreCR').textContent = scoreRate(metrics?.CR);
  $('scoreTCR').textContent = scoreNumber(metrics?.TCR, 3);
  $('scoreNavigation').textContent = scored ? scoreNumber(scoring.navigation, 6) : '—';
  $('scoreSocial').textContent = scored ? scoreNumber(scoring.social, 6) : '—';
  $('scoreFactor').textContent = scored ? scoreNumber(scoring.social_factor, 6) : '—';
  $('scoreProvenance').textContent = scored ?
    '评分来源：' + (scoring.imported_from || '导入评测结果') +
      ' · ' + (scoring.scope === 'imported_split_metrics' ? '聚合指标' : '逐 Episode 评测') +
      ' · 不是 CodaBench 官方排行榜成绩' :
    '没有独立评测的目标距离、成功状态和人体碰撞记录，不会将模型预测当作真实评分。';
  $('scoreCoverage').textContent =
    (scoring?.split || state.data?.scoring_split || '') +
    (scoring?.evaluated_episodes ?
      ' · 已评测 ' + scoring.evaluated_episodes + ' / ' +
        scoring.expected_split_episodes : ' · 无已验证样本数');
  $('scoringLimitations').textContent = (scoring?.limitations || [
    '仅保存动作与 RGB-D 的运行轨迹无法直接计算 SR、NE、CR、TCR。'
  ]).join('；');
  const evidence = $('episodeMetricEvidence');
  evidence.replaceChildren();
  const one = scoring?.per_episode;
  if (one && typeof one === 'object') {
    const header = node('strong','','本 Episode 的独立评测依据');
    const table = node('div','single-metric-row');
    [['是否无碰撞成功', one.SR ? '是' : '否'],
      ['终点误差', scoreNumber(one.NE,3)+' m'],
      ['调整后人体碰撞', scoreNumber(one.TCR,2)],
      ['是否发生额外碰撞', one.CR ? '是' : '否']].forEach(([key,value]) => {
        const item = node('div');
        item.append(node('small','',key), node('strong','',value));
        table.append(item);
      });
    evidence.append(header,table);
  } else {
    evidence.append(node('p','muted',
      '本 Episode 没有独立的真实评测记录；不能从 STOP 或图像变化推断是否成功。'));
  }
}
async function calculateScorePreview() {
  const values = ['SR','NE','CR','TCR'].map(key => $('calc'+key).value.trim());
  if (values.some(v => v === '')) {
    $('calcResult').textContent = '请填写全部四项指标，空值不能当作 0。';
    return;
  }
  const params = new URLSearchParams();
  ['sr','ne','cr','tcr'].forEach((key,i) => params.set(key, values[i]));
  $('calcResult').textContent = '计算中…';
  try {
    const response = await json('/api/score/calculate?' + params.toString());
    const c = response.calculation;
    $('calcResult').textContent = '试算 Score = ' + scoreNumber(c.score, 6) +
      ' · Navigation ' + scoreNumber(c.navigation, 5) +
      ' · Social ' + scoreNumber(c.social, 5) + '（非官方）';
  } catch(e) {
    $('calcResult').textContent = '计算失败：' + e.message;
  }
}
async function loadEpisode(id) {
  state.episode = id; state.selected = -1; state.detail = null;
  state.requestSeq++;
  $('eventList').replaceChildren(node('p','muted pad','正在读取回放索引…'));
  try {
    state.data = await json(pathFor(id,'timeline'));
    const meta = state.data.metadata || {}, sum = state.data.summary || {};
    renderScoring(state.data.scoring);
    $('instruction').textContent = meta.instruction || '该 episode 未记录任务文本';
    $('totalActions').textContent = '动作 '+state.data.counts.actions;
    $('totalApis').textContent = 'API '+state.data.counts.api;
    $('totalTools').textContent = '工具 '+state.data.counts.tools;
    $('stopReason').textContent = '结束 '+(sum.finish_reason || sum.stop_reason || state.data.failure?.error_type || '未知');
    $('orderBadge').textContent = state.data.precise_order ? '精确因果时间线' : '旧日志 · 近似顺序';
    $('orderBadge').classList.toggle('warning', !state.data.precise_order);
    filterItems();
  } catch(e) { message('无法读取 Episode：'+e.message, true); }
}
async function init() {
  try {
    const found = await json('/api/episodes');
    const select = $('episodeSelect');
    found.episodes.forEach(id => select.append(node('option','', id)));
    if (!found.episodes.length) throw new Error('没有找到 episode');
    select.addEventListener('change', () => loadEpisode(select.value));
    $('modeSelect').addEventListener('change', filterItems);
    $('calcButton').addEventListener('click', calculateScorePreview);
    $('searchBox').addEventListener('input', filterItems);
    $('scrubber').addEventListener('input', () => {
      if (state.filtered.length) selectIndex(state.filtered[Number($('scrubber').value)]);
    });
    $('firstBtn').onclick = () => move(-1e9);
    $('lastBtn').onclick = () => move(1e9);
    $('prevBtn').onclick = () => move(-1);
    $('nextBtn').onclick = () => move(1);
    $('jumpBtn').onclick = () => {
      const value = Number($('jumpInput').value);
      if (Number.isInteger(value) && value >= 1 && value <= state.filtered.length)
        selectIndex(state.filtered[value-1]);
      else message('输入范围内的事件编号', true);
    };
    $('jumpInput').addEventListener('keydown', e => { if (e.key === 'Enter') $('jumpBtn').click(); });
    document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
      state.tab = tab.dataset.tab;
      document.querySelectorAll('.tab').forEach(el => el.classList.toggle('active', el === tab));
      if (state.detail) renderDetail(state.detail);
    }));
    document.addEventListener('keydown', e => {
      if (e.altKey || e.ctrlKey || e.metaKey ||
          ['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName)) return;
      if (e.key === 'ArrowLeft') { e.preventDefault(); move(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); move(1); }
    });
    await loadEpisode(found.episodes[0]);
  } catch(e) { message('回放器初始化失败：'+e.message, true); }
}
init();
