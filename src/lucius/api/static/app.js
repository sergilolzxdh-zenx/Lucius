/* Lucius control center. Plain DOM, no build step. All text is inserted with textContent. */
"use strict";

const TOKEN = document.querySelector('meta[name="lucius-token"]').content;
const $view = document.getElementById("view");
const S = { status: null, feed: [], page: null, taxonomy: {} };

// -- helpers ------------------------------------------------------------------------------------------
async function api(method, path, body) {
  const init = { method, headers: {} };
  if (method !== "GET") init.headers["X-Lucius-Token"] = TOKEN;
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  const res = await fetch("/api" + path, init);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const msg = (data && data.message) || (data && data.detail && JSON.stringify(data.detail)) || res.statusText;
    throw new Error(msg);
  }
  return data;
}
const GET = (p) => api("GET", p);
const POST = (p, b = {}) => api("POST", p, b);
const PUT = (p, b) => api("PUT", p, b);
const PATCH = (p, b) => api("PATCH", p, b);
const DEL = (p) => api("DELETE", p);

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (k === "checked" || k === "disabled" || k === "selected" || k === "hidden") el[k] = !!v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  const add = (c) => {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) c.forEach(add);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  };
  children.forEach(add);
  return el;
}
const fmtTime = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : "—");
const fmtClock = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString() : "");
function ago(ts) {
  if (!ts) return "—";
  const s = Date.now() / 1000 - ts;
  if (s < 60) return `${Math.max(0, Math.round(s))}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}
const dur = (s) => (s == null ? "—" : s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));
const chip = (text, cls) => h("span", { class: `chip ${cls || String(text || "").replace(/\s+/g, "_")}` }, text ?? "—");
const bar = (v, cls = "") => h("div", { class: `bar ${cls}`, title: pct(v) }, h("i", { style: `width:${Math.round((v || 0) * 100)}%` }));
const short = (id) => (id ? String(id).slice(-8) : "—");
const link = (href, text) => h("a", { href }, text);
const json = (v) => h("pre", {}, JSON.stringify(v, null, 2));
const empty = (text) => h("div", { class: "empty" }, text);
const notice = (text, cls = "") => h("div", { class: `notice ${cls}` }, text);
const field = (label, input) => h("label", { class: "field" }, label, input);

let toastTimer = null;
function toast(msg, bad = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = bad ? "bad" : "";
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 7000 : 3500);
}
async function act(fn, ok) {
  try {
    const out = await fn();
    if (ok) toast(typeof ok === "function" ? ok(out) : ok);
    await rerender();
    return out;
  } catch (e) {
    toast(e.message || String(e), true);
    return null;
  }
}
function table(headers, rows, onClick) {
  if (!rows.length) return empty("Nothing here yet.");
  return h("table", {},
    h("thead", {}, h("tr", {}, headers.map((x) => h("th", {}, x)))),
    h("tbody", {}, rows.map((r, i) => h("tr", onClick ? { class: "click", onclick: () => onClick(i) } : {},
      r.map((c) => h("td", {}, c))))));
}
function select(options, value, attrs = {}) {
  return h("select", attrs, options.map((o) => {
    const [v, label] = Array.isArray(o) ? o : [o, o];
    return h("option", { value: v, selected: v === value }, label);
  }));
}
async function taxonomy(kind) {
  if (!S.taxonomy[kind]) S.taxonomy[kind] = await GET(`/taxonomy/${kind}`);
  return S.taxonomy[kind];
}
const LABEL_COLORS = ["#5b8cff", "#3ecf8e", "#f5a524", "#c77dff", "#ff7a59", "#7cc4ff", "#e5e36b", "#f0506e", "#56d4c9", "#a0a8b8"];
const labelColor = (label) => {
  let hsh = 0;
  for (const ch of String(label)) hsh = (hsh * 31 + ch.charCodeAt(0)) >>> 0;
  return LABEL_COLORS[hsh % LABEL_COLORS.length];
};

// -- routing ------------------------------------------------------------------------------------------
const ROUTES = {
  agent: pageAgent, runs: pageRun, watch: pageWatch, sessions: pageSession, import: pageImport,
  skills: pageSkills, memory: pageMemory, failures: pageFailures, practice: pagePractice,
  datasets: pageDatasets, benchmarks: pageBenchmarks, settings: pageSettings,
};
function route() {
  const path = location.hash.replace(/^#\/?/, "").split("?")[0];
  const parts = (path || "agent").split("/").map(decodeURIComponent);
  return { name: ROUTES[parts[0]] ? parts[0] : "agent", args: parts.slice(1) };
}
let renderSeq = 0;
async function render(keepScroll = false) {
  const { name, args } = route();
  document.querySelectorAll("#sidebar a").forEach((a) => {
    const r = a.dataset.route;
    a.classList.toggle("active", r === name || (name === "runs" && r === "agent") || (name === "sessions" && r === "watch"));
  });
  const seq = ++renderSeq;
  const scroll = window.scrollY;
  if (!keepScroll) $view.replaceChildren(h("div", { class: "muted" }, "Loading…"));
  try {
    const page = await ROUTES[name](...args);
    if (seq !== renderSeq) return;
    S.page = { name, args, live: page.live || [] };
    $view.replaceChildren(page.el || page);
    if (keepScroll) window.scrollTo(0, scroll);
  } catch (e) {
    if (seq !== renderSeq) return;
    $view.replaceChildren(notice(`Could not load this view: ${e.message}`, "bad"));
  }
}
function userIsEditing() {
  const a = document.activeElement;
  return a && $view.contains(a) && ["INPUT", "TEXTAREA", "SELECT"].includes(a.tagName);
}
async function rerender() { if (!userIsEditing()) await render(true); }
let liveTimer = null;
function scheduleLive() {
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => { if (!userIsEditing()) render(true); }, 700);
}
window.addEventListener("hashchange", () => render());

// -- status bar, takeover banner, events ----------------------------------------------------------------------
async function refreshStatus() {
  try {
    S.status = await GET("/status");
  } catch (e) {
    document.getElementById("conn-state").textContent = "API unreachable";
    return;
  }
  const st = S.status;
  document.getElementById("conn-state").textContent = st.live_blender ? "Blender connected" : "Blender: not connected";
  const rec = st.recording || {};
  const ind = document.getElementById("rec-indicator");
  if (rec.recording) {
    ind.className = rec.paused ? "rec-on rec-paused" : "rec-on";
    ind.textContent = rec.paused ? "PAUSED" : `REC ${rec.task_text ? "· " + rec.task_text : ""}`;
    ind.title = rec.capture_allowed === false ? `capture paused: ${rec.privacy_reason || "outside Blender"}` : "recording";
  } else {
    ind.className = "rec-off";
    ind.textContent = "not recording";
  }
  const c = st.counts;
  document.getElementById("topbar-status").replaceChildren(...[
    h("span", {}, `${c.demonstrations} demos`), h("span", {}, `${c.learned_skills} learned skills`),
    h("span", {}, `${c.failures} failure records`),
    c.needs_review ? h("a", { href: "#/agent" }, `${c.needs_review} awaiting review`) : null,
    st.jobs.length ? h("span", {}, `${st.jobs.length} job(s) running`) : null].filter(Boolean));
  renderTakeover(st.takeover);
}
function renderTakeover(t) {
  const el = document.getElementById("takeover-banner");
  if (!t) { el.hidden = true; el.replaceChildren(); return; }
  if (!el.hidden && el.dataset.id === t.id) return;
  el.dataset.id = t.id;
  el.hidden = false;
  const reason = h("input", { placeholder: "Why? (optional — stored only if you write it)", style: "flex:1;min-width:240px" });
  el.replaceChildren(h("div", { class: "stack" },
    h("div", { class: "row" }, h("strong", {}, "Human takeover requested"), chip(t.reason_code, "warn"),
      h("span", { class: "muted" }, t.message)),
    t.failed_checkpoints.length ? h("div", { class: "small muted" }, "Failed checks: " + t.failed_checkpoints.join(", ")) : null,
    h("div", { class: "small muted" }, "Correct the scene in Blender, then resume. Your correction is recorded and learned from."),
    h("div", { class: "row" }, reason,
      h("button", { class: "primary", onclick: () => act(() => POST("/takeover/resume", { reason: reason.value || null }), "Resumed") }, "Resume agent"),
      h("button", { class: "danger", onclick: () => act(() => POST("/takeover/abort", {}), "Run aborted") }, "Abort run"),
      link(`#/runs/${t.run_id}`, "open run"))));
}
function connectEvents() {
  const es = new EventSource("/api/events/stream");
  es.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch { return; }
    S.feed.unshift(ev);
    S.feed.length = Math.min(S.feed.length, 200);
    if (!["FRAME_CAPTURED", "ACTION_CAPTURED", "BLENDER_STATE"].includes(ev.type)) statusSoon();
    if (S.page && (S.page.live.includes(ev.type) || S.page.live.includes("*"))) scheduleLive();
    const feed = document.getElementById("live-feed");
    if (feed) feed.prepend(feedLine(ev));
  };
  es.onerror = () => { document.getElementById("conn-state").textContent = "event stream reconnecting…"; };
}
let statusTimer = null;
function statusSoon() { clearTimeout(statusTimer); statusTimer = setTimeout(refreshStatus, 300); }
function feedLine(ev) {
  const p = ev.payload || {};
  const detail = p.status || p.verdict || p.to || p.name || p.title || p.stage || p.reason_code || "";
  return h("div", {}, h("span", { class: "t" }, fmtClock(ev.ts)), h("strong", {}, ev.type), " ",
    h("span", { class: "muted" }, `${detail} ${ev.subject_id ? short(ev.subject_id) : ""}`));
}

// -- Agent (dashboard) --------------------------------------------------------------------------------------
async function pageAgent() {
  const [st, runs, review] = await Promise.all([GET("/status"), GET("/runs?limit=15"), GET("/runs?status=needs_human&limit=20")]);
  S.status = st;
  const c = st.counts;
  const ok = c.runs.success || 0;
  const totalRuns = Object.values(c.runs).reduce((a, b) => a + b, 0);
  const task = h("textarea", { placeholder: "Describe the task, e.g. “Make another simple sword blockout with a longer blade”", rows: 3 });
  const backend = select([["live", "Live Blender (add-on)"], ["headless", "Headless Blender"]], "live");
  const takeover = h("input", { type: "checkbox", checked: true });
  const refs = h("input", { placeholder: "reference media ids (optional, comma separated)" });
  const run = () => act(async () => {
    if (!task.value.trim()) throw new Error("Describe a task first");
    return POST("/runs", {
      task_text: task.value.trim(), backend: backend.value, allow_takeover: takeover.checked,
      reference_ids: refs.value.split(",").map((s) => s.trim()).filter(Boolean),
    });
  }, (job) => `Run queued (${short(job.id)})`);
  const el = h("div", {},
    h("h1", {}, "Agent"),
    h("p", { class: "sub" }, "Teach by demonstration, then ask the agent to reuse what it learned. Every run is verified; unverifiable results are sent to you for review."),
    h("div", { class: "grid g4" },
      stat("Demonstrations", c.demonstrations), stat("Learned skills", `${c.learned_skills}`, `${c.validated_skills} validated or high-confidence; built-in capabilities not counted`),
      stat("Verified runs", `${ok} / ${totalRuns}`), stat("Failure records", c.failures)),
    h("div", { class: "grid g2", style: "margin-top:14px" },
      h("div", { class: "card stack" }, h("h2", { style: "margin-top:0" }, "Run a task"), task,
        h("div", { class: "row" }, backend, h("label", { class: "check" }, takeover, "allow human takeover"),
          h("span", { class: "spacer" }), h("button", { class: "primary", onclick: run }, "Run")),
        refs,
        st.jobs.length ? h("div", {}, h("h3", {}, "Active jobs"), st.jobs.map((j) =>
          h("div", { class: "row small" }, chip(j.status), h("span", {}, j.title), h("span", { class: "faint" }, j.backend || "")))) : null),
      h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "Live events"),
        h("div", { class: "feed", id: "live-feed" }, S.feed.slice(0, 60).map(feedLine)))),
    review.length ? h("div", {}, h("h2", {}, "Awaiting your review"),
      h("p", { class: "sub small" }, "These runs executed without errors but include checks only a person can judge. Nothing is counted as learned until you decide."),
      h("div", { class: "card" }, table(["Task", "When", "Checks", "Decision"], review.map((r) => [
        link(`#/runs/${r.id}`, r.task_text), ago(r.started_at),
        `${r.metrics.checkpoints_passed ?? 0}/${r.metrics.checkpoints_total ?? 0} passed`, reviewButtons(r.id)])))) : null,
    h("h2", {}, "Recent runs"),
    h("div", { class: "card" }, table(["Task", "Mode", "Verdict", "Checks", "Takeovers", "Recoveries", "When"],
      runs.map((r) => [r.task_text, r.mode, chip(r.status), `${r.metrics.checkpoints_passed ?? 0}/${r.metrics.checkpoints_total ?? 0}`,
        r.metrics.takeovers ?? 0, r.metrics.recoveries ?? 0, ago(r.started_at)]), (i) => { location.hash = `#/runs/${runs[i].id}`; })),
    h("h2", {}, "Model providers"),
    h("div", { class: "card small" }, providerSummary(st.providers)));
  return { el, live: ["TASK_COMPLETED", "JOB_STATUS", "HUMAN_TAKEOVER", "HUMAN_RESUME", "SKILL_PROMOTED"] };
}
function stat(k, v, title) {
  return h("div", { class: "card stat", title: title || "" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v ?? "—"));
}
function providerSummary(p) {
  const rows = Object.entries(p).filter(([k]) => k !== "notes").map(([k, v]) => [k, v ? chip(String(v), "ok") : chip("not configured", "seed")]);
  const notes = Object.entries(p.notes || {}).map(([k, v]) => h("div", { class: "faint" }, `${k}: ${v}`));
  return h("div", {}, table(["Capability", "Provider"], rows), notes,
    h("div", { class: "muted", style: "margin-top:6px" }, "Without a vision/language provider Lucius still records, segments, extracts skills, retrieves, executes and measures; checks that need judgement go to you."));
}
function reviewButtons(runId) {
  const rating = select([["", "rating"], "1", "2", "3", "4", "5"], "", { class: "small" });
  const send = (passed) => act(() => POST(`/runs/${runId}/review`, { passed, rating: rating.value ? Number(rating.value) : null }),
    (r) => `Recorded: ${r.status}`);
  return h("div", { class: "row" }, rating, h("button", { class: "small ok", onclick: () => send(true) }, "Looks right"),
    h("button", { class: "small danger", onclick: () => send(false) }, "Not right"));
}

// -- Run detail ---------------------------------------------------------------------------------------------
async function pageRun(runId) {
  const d = await GET(`/runs/${runId}`);
  const r = d.run;
  const plan = r.plan || { steps: [], unresolved: [] };
  const flow = h("div", { class: "state-flow" });
  d.transitions.forEach((t, i) => {
    if (i === 0) flow.append(h("span", { class: "st" }, t.from_state));
    flow.append(h("span", { class: "arrow", title: t.reason_code }, "→"), h("span", { class: "st", title: `${t.reason_code}\n${(t.evidence || []).join("\n")}` }, t.to_state));
  });
  const el = h("div", {},
    h("div", { class: "row between" }, h("h1", {}, r.task_text), chip(r.status)),
    h("p", { class: "sub" }, `${r.mode} · ${r.environment} · arm ${r.arm || "default"} · started ${fmtTime(r.started_at)}`,
      r.session_id ? [" · ", link(`#/sessions/${r.session_id}`, "session timeline")] : null),
    r.metrics.human_review ? notice(`Reviewed by ${r.metrics.human_review.by}: ${r.metrics.human_review.passed ? "looks right" : "not right"}${r.metrics.human_review.rating ? ` (rating ${r.metrics.human_review.rating}/5)` : ""}.`) : null,
    r.status === "needs_human" ? h("div", { class: "card", style: "margin-bottom:14px" },
      h("strong", {}, "This run needs your judgement. "), h("span", { class: "muted" }, "Look at the result in Blender, then decide."),
      h("div", { style: "margin-top:8px" }, reviewButtons(r.id))) : null,
    h("h2", {}, "State machine"), h("div", { class: "card" }, d.transitions.length ? [flow,
      h("div", { class: "small muted", style: "margin-top:8px" }, `ended: ${d.transitions[d.transitions.length - 1].reason_code}`)] : empty("No transitions recorded.")),
    h("h2", {}, "Plan"),
    h("div", { class: "card" },
      plan.unresolved && plan.unresolved.length ? notice("Unresolved: " + plan.unresolved.map((u) => u.detail || u.reason).join("; "), "warn") : null,
      table(["Phase", "Skill", "Actions", "Checks", "Guards", "Why"], plan.steps.map((s) => [
        s.phase, s.skill_id ? link(`#/skills/${s.skill_id}`, s.skill_name || s.skill_id) : (s.skill_name || "—"),
        s.actions.length, (s.checkpoints || []).map((c) => c.id).join(", "), (s.guards || []).length,
        h("span", { class: "small faint" }, (s.reason_codes || []).slice(0, 3).join(", "))]))),
    h("h2", {}, "Evaluation"),
    h("div", { class: "card" }, table(["Level", "Check", "Result", "Method", "Evidence"], d.evaluations.map((e) => [
      `L${e.level}`, e.checkpoint_id || "run", e.passed === 1 ? chip("pass", "ok") : e.passed === 0 ? chip("fail", "bad") : chip("not evaluated", "warn"),
      `${e.method}${e.subjective ? " (subjective)" : ""}`, h("details", {}, h("summary", {}, e.evidence.reason_code || "details"), json(e.evidence))]))),
    d.corrections.length ? h("div", {}, h("h2", {}, "Human corrections"), h("div", { class: "card" }, table(["Kind", "Reason", "Outcome", "Steps", "When"],
      d.corrections.map((c) => [c.kind, c.reason || h("span", { class: "faint" }, "no reason given"), c.outcome || "—", c.correction_steps.length, fmtTime(c.start_time)])))) : null);
  return { el, live: ["TASK_COMPLETED", "STATE_TRANSITION"] };
}

// -- Watch Me ------------------------------------------------------------------------------------------------
async function pageWatch() {
  const [w, sessions] = await Promise.all([GET("/watch"), GET("/sessions?kind=live_demo&limit=30")]);
  const rec = w.recording;
  const task = h("input", { placeholder: "What are you about to demonstrate? e.g. “simple sword blockout”", style: "flex:1" });
  const consent = h("input", { type: "checkbox" });
  const src = w.sources;
  const sourcesCard = h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "Capture sources"),
    src ? h("div", { class: "kv" },
      h("span", {}, "Screen frames"), sourceState(src.screen, src.unavailable.screen),
      h("span", {}, "Window context"), sourceState(!!src.window, src.unavailable.window, src.window),
      h("span", {}, "Mouse & keyboard"), sourceState(src.input, src.unavailable.input),
      h("span", {}, "Blender add-on"), sourceState(src.blender_bridge, src.unavailable.blender_bridge))
      : h("p", { class: "muted" }, "Sources are probed when you start or press Detect."),
    h("div", { class: "row", style: "margin-top:10px" },
      h("button", { disabled: rec, onclick: () => act(() => POST("/watch/sources"), "Capture sources detected") }, "Detect")));
  let control;
  if (!rec) {
    control = h("div", { class: "card stack" },
      h("div", { class: "row" }, task, h("button", {
        class: "record", onclick: () => act(() => POST("/watch/start", { task_text: task.value || null, training_consent: consent.checked }),
          "Recording started — work in Blender as usual"),
      }, "● Start WATCH ME")),
      h("label", { class: "check small" }, consent, "Also allow this demonstration in training datasets (off by default; skills and memory learn from it either way)"),
      notice("Only Blender windows are captured. Keystrokes typed in other applications and windows whose titles look sensitive (password managers, keychains) are dropped. Stop at any time; you can delete a recording afterwards."));
  } else {
    const note = h("input", { placeholder: "Add a note, e.g. “bevel too early — tip too broad”", style: "flex:1" });
    const outcome = select([["success", "it worked"], ["failure", "it failed"], ["partial", "partially"], ["unknown", "not sure"]], "success");
    const s = w.stats || {};
    control = h("div", { class: "card stack" },
      h("div", { class: "row" }, h("span", { class: "rec-on" }, w.paused ? "PAUSED" : "RECORDING"),
        h("strong", {}, w.task_text || "untitled demonstration"), h("span", { class: "muted" }, `started ${ago(w.started_at)}`)),
      w.capture_allowed === false ? notice(`Capture is paused: ${w.privacy_reason || "Blender is not focused"}`, "warn") : null,
      h("div", { class: "grid g4" }, stat("Frames", s.frames), stat("Events", s.events), stat("Dropped (privacy)", (s.frames_skipped_privacy || 0) + (s.input_dropped_privacy || 0)),
        stat("Blender events", s.bridge_events)),
      h("div", { class: "row" }, note, h("button", { onclick: () => act(() => POST("/watch/annotate", { text: note.value }), "Note added") }, "Add note")),
      h("div", { class: "row" },
        w.paused ? h("button", { onclick: () => act(() => POST("/watch/resume"), "Resumed") }, "Resume")
          : h("button", { onclick: () => act(() => POST("/watch/pause"), "Paused") }, "Pause"),
        h("span", { class: "spacer" }), h("span", { class: "muted small" }, "Outcome"), outcome,
        h("button", { class: "primary", onclick: () => act(() => POST("/watch/stop", { outcome: outcome.value }), "Saved — processing in the background") }, "■ Stop & save")),
      (s.errors || []).length ? notice("Recorder warnings: " + s.errors.join("; "), "warn") : null);
  }
  const el = h("div", {},
    h("h1", {}, "Watch Me"), h("p", { class: "sub" }, "Demonstrate in Blender; Lucius records frames, input and Blender state, then learns reusable skills."),
    h("div", { class: "grid", style: "grid-template-columns: minmax(0,2fr) minmax(0,1fr)" }, control, sourcesCard),
    h("h2", {}, "Your demonstrations"),
    h("div", { class: "card" }, sessionTable(sessions)));
  return { el, live: ["SESSION_STARTED", "SESSION_ENDED", "PROCESSING_STAGE", ...(rec ? ["*"] : [])] };
}
function sourceState(ok, why, name) {
  return ok ? h("span", {}, chip("available", "ok"), name ? h("span", { class: "faint" }, ` ${name}`) : null)
    : h("span", {}, chip("unavailable", "bad"), why ? h("span", { class: "faint small" }, ` ${why}`) : null);
}
function sessionTable(sessions) {
  return table(["Task", "Recorded", "Events", "Frames", "Status", "Training", ""], sessions.map((s) => [
    link(`#/sessions/${s.id}`, s.task_text || short(s.id)), fmtTime(s.start_time), s.event_count, s.frame_count,
    chip(s.status), s.policy.training_allowed ? chip("included", "ok") : chip("excluded", "seed"),
    h("div", { class: "row" },
      h("a", { class: "btn small", href: `/api/sessions/${s.id}/export` }, "Export"),
      h("button", { class: "small danger", onclick: () => { if (confirm("Delete this recording and its frames?")) act(() => DEL(`/sessions/${s.id}`), "Deleted"); } }, "Delete"))]));
}

// -- Session timeline ------------------------------------------------------------------------------------------
async function pageSession(sessionId, segParam) {
  const [info, tl] = await Promise.all([GET(`/sessions/${sessionId}`), GET(`/sessions/${sessionId}/timeline`)]);
  const s = info.session;
  const segs = tl.segments;
  const t0 = s.start_time;
  const t1 = Math.max(s.end_time || 0, ...segs.map((x) => x.t_end), ...tl.steps.map((x) => x.t_end), t0 + 1);
  const span = t1 - t0;
  let selected = segs.find((x) => x.id === segParam) || segs[0];
  const detail = h("div", {});
  const track = h("div", { class: "timeline" });
  const drawTrack = () => {
    track.replaceChildren(
      ...segs.map((seg) => h("div", {
        class: `tl-seg ${selected && seg.id === selected.id ? "sel" : ""}`,
        style: `left:${((seg.t_start - t0) / span) * 100}%;width:${Math.max(0.4, ((seg.t_end - seg.t_start) / span) * 100)}%;background:${labelColor(seg.label)}`,
        title: `${seg.label} ${seg.title || ""} (${dur(seg.t_end - seg.t_start)})`,
        onclick: () => { selected = seg; drawTrack(); drawDetail(); },
      }, seg.title || seg.label)),
      ...tl.markers.map((m) => h("div", { class: `tl-mark ${m.kind.startsWith("takeover") ? "takeover" : m.kind}`, style: `left:${((m.ts - t0) / span) * 100}%`, title: `${m.kind} ${JSON.stringify(m.payload).slice(0, 120)}` })));
  };
  const drawDetail = async () => {
    if (!selected) { detail.replaceChildren(empty("No segments yet — the session may still be processing.")); return; }
    const labels = Object.keys(await taxonomy("segment_label"));
    const cats = Object.keys(await taxonomy("intent_category"));
    const targets = Object.keys(await taxonomy("intent_target"));
    const seg = selected;
    const labelSel = select(labels.includes(seg.label) ? labels : [seg.label, ...labels], seg.label);
    const title = h("input", { value: seg.title || "", placeholder: "title" });
    const it = seg.intent || {};
    const cat = select(cats.includes(it.category) || !it.category ? cats : [it.category, ...cats], it.category || cats[0]);
    const tgt = select(["", ...targets], it.target || "");
    const outcome = select(["unknown", "success", "failure", "corrected"], seg.outcome);
    const segSteps = tl.steps.filter((st) => st.idx >= seg.step_start && st.idx <= seg.step_end);
    const splitAt = select(segSteps.slice(1).map((st) => [String(st.idx), `before step ${st.idx} (${st.action_type})`]), "");
    const next = segs[segs.indexOf(seg) + 1];
    detail.replaceChildren(h("div", { class: "card stack" },
      h("div", { class: "row between" }, h("strong", {}, `${seg.title || seg.label}`),
        h("span", {}, chip(seg.origin, seg.origin === "human" ? "human_confirmed" : seg.origin === "model" ? "model_inferred" : "observed"),
          seg.locked ? chip("locked", "ok") : null)),
      h("div", { class: "kv" }, h("span", {}, "Time"), h("span", {}, `${dur(seg.t_start - t0)} – ${dur(seg.t_end - t0)} (${dur(seg.t_end - seg.t_start)})`),
        h("span", {}, "Steps"), h("span", {}, `${seg.step_start}–${seg.step_end}`),
        h("span", {}, "Label confidence"), h("span", {}, pct(seg.label_confidence)),
        h("span", {}, "Intent"), h("span", {}, it.category ? `${it.category}${it.target ? " → " + it.target : ""} (${pct(it.confidence)}, ${it.source})` : "—"),
        h("span", {}, "Boundary"), h("span", { class: "small" }, (seg.boundary_reasons || []).map((b) => b.code || b.reason || JSON.stringify(b)).join(", ") || "—")),
      seg.summary ? h("div", { class: "muted small" }, seg.summary) : null,
      h("div", { class: "row" }, labelSel, title, h("button", {
        onclick: () => act(() => POST(`/segments/${seg.id}/relabel`, { label: labelSel.value, title: title.value || null }), "Segment relabelled"),
      }, "Relabel")),
      h("div", { class: "row" }, cat, tgt, h("button", {
        onclick: () => act(() => POST(`/segments/${seg.id}/intent`, { category: cat.value, target: tgt.value || null }), "Intent set"),
      }, "Set intent")),
      h("div", { class: "row" }, outcome, h("button", { onclick: () => act(() => POST(`/segments/${seg.id}/outcome`, { outcome: outcome.value }), "Outcome set") }, "Set outcome"),
        segSteps.length > 1 ? [splitAt, h("button", { onclick: () => act(() => POST(`/segments/${seg.id}/split`, { at_step: Number(splitAt.value) }), "Segment split") }, "Split")] : null,
        next ? h("button", { onclick: () => act(() => POST("/segments/merge", { segment_ids: [seg.id, next.id] }), "Merged") }, "Merge with next") : null),
      h("div", { class: "faint small" }, "Edits lock the segment: re-processing never overwrites what you changed.")));
  };
  drawTrack();
  await drawDetail();
  const frames = tl.frames;
  const stride = Math.max(1, Math.ceil(frames.length / 40));
  const sampled = frames.filter((_, i) => i % stride === 0);
  const policy = s.policy;
  const labels = [...new Set(segs.map((x) => x.label))];
  const el = h("div", {},
    h("div", { class: "row between" }, h("h1", {}, s.task_text || "Session"), h("span", {}, chip(s.kind), " ", chip(s.status))),
    h("p", { class: "sub" }, `${fmtTime(s.start_time)} · ${tl.steps.length} steps · ${segs.length} segments · ${frames.length} frames · source ${s.source}`),
    h("div", { class: "grid g3" },
      h("div", { class: "card small" }, h("h3", { style: "margin-top:0" }, "Processing"),
        Object.entries(info.processing.stages || info.processing || {}).map(([k, v]) => h("div", { class: "row" }, h("span", { style: "width:90px" }, k), chip(v.status || v)))),
      h("div", { class: "card small stack" }, h("h3", { style: "margin-top:0" }, "Data use"),
        h("div", {}, "Training datasets: ", policy.training_allowed ? chip("included", "ok") : chip("excluded", "seed")),
        h("div", {}, "Consent: ", chip(policy.consent_status)),
        h("div", { class: "row" },
          h("button", { class: "small", onclick: () => act(() => PATCH(`/sessions/${s.id}/policy`, { training_consent: !policy.training_allowed }), "Policy updated") },
            policy.training_allowed ? "Opt out of training" : "Allow in training"),
          h("button", { class: "small", onclick: () => act(() => POST(`/sessions/${s.id}/process?force=true`), "Re-processing queued") }, "Re-process"))),
      h("div", { class: "card small" }, h("h3", { style: "margin-top:0" }, "Episode"),
        info.episode ? h("div", {}, info.episode.summary) : h("span", { class: "faint" }, "not built yet"))),
    h("h2", {}, "Timeline"), track,
    h("div", { class: "legend", style: "margin:8px 0" }, labels.map((l) => h("span", {}, h("i", { style: `background:${labelColor(l)}` }), l)),
      h("span", {}, h("i", { style: "background:var(--warn)" }), "takeover"), h("span", {}, h("i", { style: "background:var(--bad)" }), "undo"),
      h("span", {}, h("i", { style: "background:var(--info)" }), "note")),
    detail,
    sampled.length ? h("div", {}, h("h2", {}, "Frames"), h("div", { class: "frames" }, sampled.map((f) => h("figure", {},
      h("a", { href: `/api/frames/${f.id}/image`, target: "_blank", rel: "noopener" }, h("img", { src: `/api/frames/${f.id}/image`, loading: "lazy", alt: `frame ${f.seq}` })),
      h("figcaption", {}, dur(f.ts - t0)))))) : null,
    h("h2", {}, "Steps"),
    h("div", { class: "card steps" }, stepsTable(tl.steps, t0)));
  return { el, live: ["PROCESSING_STAGE", "SEGMENT_EDITED", "TRAJECTORY_BUILT"] };
}
function stepsTable(steps, t0) {
  return table(["#", "t", "Action", "Target", "Details", "Known by", "Conf."], steps.map((st) => {
    const params = st.action_payload.params || {};
    const conf = st.action_confidence;
    const cands = st.candidate_actions || [];
    const known = h("div", {}, chip(st.action_source), h("div", { class: "faint small" }, st.evidence_kind));
    let action = h("span", {}, st.action_type, st.undo_redo_flag ? chip(st.undo_redo_flag, "bad") : null, st.actor !== "human" ? chip(st.actor, "info") : null);
    if (cands.length && st.action_source !== "human_confirmed") {
      const pick = select(cands.map((c) => [c.action_type, `${c.action_type} (${pct(c.confidence)})`]), st.action_type, { class: "small" });
      action = h("div", { class: "stack" }, action, h("div", { class: "row" }, pick,
        h("button", { class: "small", onclick: () => act(() => POST(`/steps/${st.id}/confirm`, { action_type: pick.value }), "Confirmed") }, "Confirm")));
    }
    return [st.idx, dur(st.t_start - t0), action, st.action_payload.object || st.selection_hint || "—",
      h("span", { class: "mono small" }, JSON.stringify(params).slice(0, 90)), known, num(conf)];
  }));
}

// -- Import (external demonstrations) ---------------------------------------------------------------------------
async function pageImport(demoId) {
  if (demoId) return pageImportDetail(demoId);
  const demos = await GET("/demonstrations");
  const files = h("input", { type: "file", multiple: true });
  const rows = h("div", { class: "stack" });
  const roleInputs = [];
  files.addEventListener("change", () => {
    roleInputs.length = 0;
    rows.replaceChildren(...[...files.files].map((f) => {
      const isImg = /\.(png|jpe?g|webp|bmp)$/i.test(f.name);
      const role = select(["demonstration", "reference", "before", "after", "intermediate", "target", "project", "instruction"],
        /\.blend$/i.test(f.name) ? "project" : /\.(txt|md)$/i.test(f.name) ? "instruction" : isImg ? "reference" : "demonstration");
      const view = select(["front", "side", "top", "back", "left", "right"], "front");
      const target = h("input", { placeholder: "depicts (e.g. guard; empty = whole)" });
      roleInputs.push({ role, view, target });
      return h("div", { class: "row small" }, h("span", { style: "min-width:220px" }, f.name), role, view, target);
    }));
  });
  const title = h("input", { placeholder: "Title" });
  const taskText = h("input", { placeholder: "What does it show? e.g. “sword blockout”" });
  const instr = h("textarea", { placeholder: "Written instructions, one step per line (optional)" });
  const license = h("input", { value: "unknown", placeholder: "licence" });
  const refOnly = h("input", { type: "checkbox" });
  const consent = h("input", { type: "checkbox" });
  const submit = () => act(async () => {
    const fd = new FormData();
    [...files.files].forEach((f) => fd.append("files", f));
    fd.append("roles", JSON.stringify(roleInputs.map((r) => ({ role: r.role.value, view: r.view.value, target: r.target.value || null }))));
    fd.append("title", title.value || "Imported demonstration");
    if (taskText.value) fd.append("task_text", taskText.value);
    if (instr.value) fd.append("instructions", instr.value);
    fd.append("license", license.value || "unknown");
    fd.append("reference_only", refOnly.checked);
    fd.append("training_consent", consent.checked);
    const d = await api("POST", "/demonstrations", fd);
    location.hash = `#/import/${d.id}`;
    return d;
  }, "Uploaded — analysis running");
  const el = h("div", {},
    h("h1", {}, "Import"), h("p", { class: "sub" }, "Teach from videos, screenshots, before/after pairs, reference images, .blend files or written steps. Everything converges on the same learning pipeline, marked by how it is known."),
    h("div", { class: "card stack" },
      h("div", { class: "grid g2" }, field("Title", title), field("Task", taskText)),
      field("Files", files), rows, field("Instructions", instr),
      h("div", { class: "row" }, field("Licence", license), h("label", { class: "check" }, refOnly, "reference only (never a procedure)"),
        h("label", { class: "check" }, consent, "allow in training datasets")),
      notice("External media never enter training datasets unless you allow it here, and the licence you state is kept with every sample."),
      h("div", { class: "row" }, h("span", { class: "spacer" }), h("button", { class: "primary", onclick: submit }, "Upload & analyse"))),
    h("h2", {}, "Imported demonstrations"),
    h("div", { class: "card" }, table(["Title", "Sources", "Status", "Training", "Updated"], demos.map((d) => [
      d.title, d.source_types.join(", "), chip(d.status), d.policy.training_allowed ? chip("included", "ok") : chip("excluded", "seed"), ago(d.updated_at)]),
      (i) => { location.hash = `#/import/${demos[i].id}`; })));
  return { el, live: ["MEDIA_STATUS"] };
}
async function pageImportDetail(demoId) {
  const d = await GET(`/demonstrations/${demoId}`);
  const demo = d.demonstration;
  const el = h("div", {},
    h("div", { class: "row between" }, h("h1", {}, demo.title), chip(demo.status)),
    h("p", { class: "sub" }, `${demo.source_types.join(", ")} · source class ${demo.source_class} · licence ${demo.policy.license}`),
    demo.error ? notice(`Error: ${demo.error.message || JSON.stringify(demo.error)}`, "bad") : null,
    h("div", { class: "grid g2" },
      h("div", { class: "card small" }, h("h3", { style: "margin-top:0" }, "Progress"),
        demo.status_history.map((x) => h("div", { class: "row" }, chip(x.status), h("span", { class: "faint" }, fmtClock(x.ts)), x.detail ? h("span", { class: "muted" }, x.detail) : null))),
      h("div", { class: "card small stack" }, h("h3", { style: "margin-top:0" }, "Data use"),
        h("div", {}, "Training: ", demo.policy.training_allowed ? chip("allowed", "ok") : chip("not allowed", "seed"),
          " · reference only: ", demo.policy.reference_only ? "yes" : "no"),
        h("div", { class: "row" },
          h("button", { class: "small", onclick: () => act(() => POST(`/demonstrations/${demo.id}/policy`, { consent: !demo.policy.training_allowed }), "Updated") },
            demo.policy.training_allowed ? "Exclude from training" : "Allow in training"),
          h("button", { class: "small", onclick: () => act(() => POST(`/demonstrations/${demo.id}/reprocess`), "Re-analysis started") }, "Re-analyse"),
          demo.session_id ? link(`#/sessions/${demo.session_id}`, "full timeline") : null))),
    d.constraints.length ? h("div", {}, h("h2", {}, "Measured from references"),
      h("div", { class: "card" }, table(["Constraint", "Target", "Value", "Known by", "Conf."], d.constraints.map((c) => [
        c.constraint_type, c.target, h("span", { class: "mono small" }, JSON.stringify(c.value).slice(0, 120)), chip(c.source === "measured" ? "observed" : c.source), num(c.confidence)])))) : null,
    d.timeline ? h("div", {}, h("h2", {}, "Reconstructed steps"),
      h("p", { class: "sub small" }, "Steps inferred from media are candidates: confirm the right action to raise its confidence."),
      h("div", { class: "card steps" }, stepsTable(d.timeline.steps, d.timeline.session.start_time))) : null);
  return { el, live: ["MEDIA_STATUS", "PROCESSING_STAGE"] };
}

// -- Skills ----------------------------------------------------------------------------------------------------
async function pageSkills(skillId) {
  if (skillId) return pageSkill(skillId);
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const status = params.get("status") || "";
  const q = params.get("q") || "";
  const skills = await GET(`/skills?include_inactive=true${status ? "&status=" + status : ""}${q ? "&search=" + encodeURIComponent(q) : ""}`);
  const statusSel = select([["", "all statuses"], "candidate_pattern", "candidate_skill", "validated", "high_confidence", "disabled", "merged"], status);
  const search = h("input", { placeholder: "search", value: q });
  const go = () => { location.hash = `#/skills?status=${statusSel.value}&q=${encodeURIComponent(search.value)}`; };
  statusSel.addEventListener("change", go);
  search.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  const learned = skills.filter((s) => s.source_class !== "system_seeded");
  const seeded = skills.filter((s) => s.source_class === "system_seeded");
  const rowsOf = (list) => list.map((s) => [h("div", {}, h("strong", {}, s.name), h("div", { class: "faint small" }, s.purpose)), chip(s.status),
    h("div", { class: "row" }, bar(s.confidence), h("span", { class: "small" }, num(s.confidence))), `v${s.version}`,
    `${s.success_count}/${s.usage_count}`, h("span", { class: "small" }, s.origin_sources.join(", ") || "—")]);
  const el = h("div", {},
    h("h1", {}, "Skills"), h("p", { class: "sub" }, "Reusable procedures learned from demonstrations. Status is earned by evidence: repeated demonstrations, verified runs and your review."),
    h("div", { class: "row", style: "margin-bottom:12px" }, statusSel, search, h("button", { onclick: go }, "Filter"),
      h("span", { class: "spacer" }), skillImport()),
    h("h2", {}, "Learned"), h("div", { class: "card" }, table(["Skill", "Status", "Confidence", "Version", "Uses (ok/all)", "Sources"], rowsOf(learned),
      (i) => { location.hash = `#/skills/${learned[i].id}`; })),
    h("h2", {}, "Built-in capabilities"), h("p", { class: "sub small" }, "System-seeded building blocks. They are not presented as something you taught."),
    h("div", { class: "card" }, table(["Skill", "Status", "Confidence", "Version", "Uses (ok/all)", "Sources"], rowsOf(seeded),
      (i) => { location.hash = `#/skills/${seeded[i].id}`; })));
  return { el, live: ["SKILL_CANDIDATE_CREATED", "SKILL_PROMOTED", "SKILL_DEMOTED", "SKILL_VERSIONED"] };
}
function skillImport() {
  const f = h("input", { type: "file", accept: ".json", style: "max-width:220px" });
  const asNew = h("input", { type: "checkbox" });
  return h("span", { class: "row small" }, f, h("label", { class: "check" }, asNew, "as new"),
    h("button", { class: "small", onclick: () => act(async () => {
      if (!f.files[0]) throw new Error("choose a skill .json file");
      const fd = new FormData(); fd.append("file", f.files[0]); fd.append("as_new", asNew.checked);
      return api("POST", "/skills/import", fd);
    }, (r) => `Imported ${r.skill_id} as a candidate`) }, "Import skill"));
}
async function pageSkill(skillId) {
  const d = await GET(`/skills/${skillId}`);
  const s = d.skill;
  const def = s.definition;
  const sum = d.summary;
  const tabs = ["Definition", "Evidence", "Versions", "Failures", "Edit"];
  let active = sessionStorage.getItem("skillTab") || "Definition";
  const body = h("div", {});
  const tabBar = h("div", { class: "tabs" });
  const drawTabs = () => {
    tabBar.replaceChildren(...tabs.map((t) => h("button", { class: t === active ? "active" : "", onclick: () => { active = t; sessionStorage.setItem("skillTab", t); drawTabs(); } }, t)));
    body.replaceChildren(skillTab(active, d));
  };
  drawTabs();
  const disabled = s.status === "disabled";
  const el = h("div", {},
    h("div", { class: "row between" }, h("h1", {}, def.name), h("span", {}, chip(s.status), " ", chip(s.source_class))),
    h("p", { class: "sub" }, def.purpose),
    h("div", { class: "grid g4" }, stat("Confidence", num(s.confidence)), stat("Version", `v${s.current_version}`),
      stat("Verified uses", `${s.success_count}/${s.usage_count}`), stat("Your reviews", `${s.human_confirmations}✓ ${s.human_rejections}✗`)),
    h("div", { class: "row", style: "margin:14px 0" },
      h("button", { class: "ok", onclick: () => act(() => POST(`/skills/${s.id}/review`, { accept: true }), "Confirmed") }, "Confirm skill"),
      h("button", { class: "danger", onclick: () => act(() => POST(`/skills/${s.id}/review`, { accept: false }), "Rejected") }, "Reject"),
      h("button", { onclick: () => act(() => POST(`/skills/${s.id}/disable`, { disabled: !disabled }), disabled ? "Enabled" : "Disabled") }, disabled ? "Enable" : "Disable"),
      h("a", { class: "btn", href: `/api/skills/${s.id}/export` }, "Export"),
      h("span", { class: "spacer" }), h("span", { class: "small muted" }, `updated ${ago(s.updated_at)}`)),
    tabBar, body);
  return { el, live: ["SKILL_VERSIONED", "SKILL_PROMOTED", "SKILL_DEMOTED", "SKILL_EVIDENCE_ADDED"] };
}
function skillTab(tab, d) {
  const s = d.skill;
  const def = s.definition;
  if (tab === "Definition") {
    return h("div", { class: "stack" },
      h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, "Parameters"),
        table(["Name", "Kind", "Default", "Range / choices", "Observed", "Source"], def.parameters.map((p) => [
          h("strong", {}, p.name), p.kind, JSON.stringify(p.default), p.choices ? p.choices.join(", ") : p.range ? p.range.join(" … ") : "—",
          h("span", { class: "mono small" }, JSON.stringify(p.observed_values || []).slice(0, 60)), p.source || "—"]))),
      ...def.phases.map((ph) => h("div", { class: "card" }, h("div", { class: "row between" }, h("strong", {}, ph.name),
        h("span", { class: "small muted" }, (ph.checkpoints || []).length ? `checks: ${ph.checkpoints.join(", ")}` : "")),
        table(["Action", "Arguments", "Mode", "Optional"], ph.actions.map((a) => [a.action_type,
          h("span", { class: "mono small" }, JSON.stringify(a.args).slice(0, 140)), a.requires_mode || "—", a.optional ? "yes" : ""])))),
      h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, "Checkpoints"),
        table(["Id", "Description", "Level", "Method", "Required"], def.checkpoints.map((c) => [c.id, c.description, `L${c.level}`, c.method, c.required ? "yes" : "no"]))),
      def.failure_conditions && def.failure_conditions.length ? h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, "Known failure conditions"),
        table(["Condition", "Phase", "Premature action", "Guard check"], def.failure_conditions.map((f) => [f.description || f.id, f.phase || "—", f.trigger_action || "—", f.guard_checkpoint || "—"]))) : null,
      def.recovery_actions && def.recovery_actions.length ? h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, "Recovery"),
        table(["When", "Actions", "Source", "Worked"], def.recovery_actions.map((r) => [(r.when_checkpoints || []).join(", ") || r.when || "—",
          (r.actions || []).map((a) => a.action_type).join(" → "), r.source || "—", `${r.successes}/${r.attempts}`]))) : null,
      def.notes && def.notes.length ? h("div", { class: "card small muted" }, def.notes.map((n) => h("div", {}, n))) : null);
  }
  if (tab === "Evidence") {
    const b = s.confidence_breakdown || {};
    return h("div", { class: "stack" },
      h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, "Why this confidence"),
        h("div", { class: "kv" }, Object.entries(b).flatMap(([k, v]) => [h("span", {}, k), h("span", { class: "mono small" }, typeof v === "object" ? JSON.stringify(v) : String(v))]))),
      h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, "Provenance"), json(d.provenance)),
      h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, "Examples"),
        table(["Role", "Source", "Outcome", "Weight", "Session / run"], d.examples.map((e) => [e.role, chip(e.source_class), e.outcome || "—", num(e.evidence_weight),
          e.session_id ? link(`#/sessions/${e.session_id}`, short(e.session_id)) : e.run_id ? link(`#/runs/${e.run_id}`, short(e.run_id)) : "—"]))));
  }
  if (tab === "Versions") {
    return h("div", { class: "card" }, table(["Version", "Change", "By", "When", ""], d.versions.map((v) => [`v${v.version}`, v.change_note, v.created_by, fmtTime(v.created_at),
      v.version !== s.current_version ? h("button", { class: "small", onclick: () => { if (confirm(`Roll back to v${v.version}? (creates a new version)`)) act(() => POST(`/skills/${s.id}/rollback`, { version: v.version }), "Rolled back"); } }, "Roll back") : chip("current", "ok")])));
  }
  if (tab === "Failures") {
    return h("div", { class: "card" }, table(["Problem", "Phase", "Rule", "Status", "Seen"], d.failures.map((f) => [link(`#/failures/${f.id}`, f.observed_problem), f.phase || "—", f.future_rule || "—", chip(f.rule_status), f.occurrence_count])));
  }
  const edit = h("textarea", { rows: 10 }, JSON.stringify({ name: def.name, purpose: def.purpose, triggers: def.triggers, notes: def.notes }, null, 2));
  const mergeInto = h("input", { placeholder: "target skill id" });
  const phaseBoxes = def.phases.map((p) => ({ name: p.name, box: h("input", { type: "checkbox" }) }));
  const newName = h("input", { placeholder: "name of the new skill" });
  return h("div", { class: "stack" },
    h("div", { class: "card stack" }, h("h3", { style: "margin-top:0" }, "Edit (creates a new version)"), edit,
      h("button", { class: "primary", onclick: () => act(() => PATCH(`/skills/${s.id}`, { changes: JSON.parse(edit.value) }), "Saved as a new version") }, "Save")),
    h("div", { class: "card stack" }, h("h3", { style: "margin-top:0" }, "Split phases into a new skill"),
      h("div", { class: "row" }, phaseBoxes.map((p) => h("label", { class: "check" }, p.box, p.name))), newName,
      h("button", { onclick: () => act(() => POST(`/skills/${s.id}/split`, { phases: phaseBoxes.filter((p) => p.box.checked).map((p) => p.name), new_name: newName.value }), "Split") }, "Split")),
    h("div", { class: "card stack" }, h("h3", { style: "margin-top:0" }, "Merge this skill into another"), mergeInto,
      h("button", { class: "danger", onclick: () => { if (confirm("Merge? This skill will be marked merged.")) act(() => POST(`/skills/${s.id}/merge`, { into: mergeInto.value }), "Merged"); } }, "Merge")));
}

// -- Memory ------------------------------------------------------------------------------------------------------
async function pageMemory(tabArg) {
  const tabs = ["Episodes", "Semantic", "Preferences", "Learning graph", "Retrieval debug"];
  const active = tabArg && tabs.includes(tabArg) ? tabArg : "Episodes";
  let body;
  if (active === "Episodes") {
    const eps = await GET("/memory/episodes");
    body = h("div", { class: "card" }, table(["Task", "Class", "Outcome", "Summary", "Confidence", "Session"], eps.map((e) => [
      e.task_text || "—", e.task_class || "—", chip(e.outcome), h("span", { class: "small" }, e.summary.slice(0, 220)), num(e.confidence), link(`#/sessions/${e.session_id}`, short(e.session_id))])));
  } else if (active === "Semantic") {
    const st = await GET("/memory/semantic?include_rejected=true");
    body = h("div", { class: "card" }, table(["Statement", "Status", "Support", "Confidence", ""], st.map((x) => [x.statement, chip(x.status), x.support_count ?? "—", num(x.confidence),
      h("div", { class: "row" }, h("button", { class: "small ok", onclick: () => act(() => POST(`/memory/semantic/${x.id}/review`, { accept: true }), "Accepted") }, "Accept"),
        h("button", { class: "small danger", onclick: () => act(() => POST(`/memory/semantic/${x.id}/review`, { accept: false }), "Rejected") }, "Reject"))])));
  } else if (active === "Preferences") {
    const prefs = await GET("/memory/preferences");
    body = h("div", { class: "card" }, h("p", { class: "sub small" }, "Workflow habits learned from your demonstrations. They bias retrieval; they never override the task."),
      table(["Preference", "Value", "Confidence", "Source", ""], Object.entries(prefs).map(([k, p]) => {
        const v = h("input", { value: JSON.stringify(p.value), style: "width:120px" });
        return [k, v, num(p.confidence), p.source || "—", h("div", { class: "row" },
          h("button", { class: "small", onclick: () => act(() => PUT(`/memory/preferences/${k}`, { value: JSON.parse(v.value) }), "Saved") }, "Set"),
          h("button", { class: "small danger", onclick: () => act(() => DEL(`/memory/preferences/${k}`), "Cleared") }, "Clear"))];
      })));
  } else if (active === "Learning graph") {
    body = await learningGraph();
  } else {
    body = retrievalDebug();
  }
  const el = h("div", {}, h("h1", {}, "Memory"),
    h("p", { class: "sub" }, "Episodic (what happened), semantic (general statements), procedural (skills) and failure memory, with provenance for every item."),
    h("div", { class: "tabs" }, tabs.map((t) => h("button", { class: t === active ? "active" : "", onclick: () => { location.hash = `#/memory/${t}`; } }, t))), body);
  return { el, live: active === "Retrieval debug" ? [] : ["MEMORY_CREATED"] };
}
async function learningGraph() {
  const g = await GET("/graph?limit=500");
  if (!g.nodes.length) return empty("The graph fills as Lucius learns: demonstrations, skills, failures, corrections and runs, linked by evidence.");
  const W = 1000, H = 540;
  // Force simulation (d3-style): short-range repulsion, spring links, gentle centring; the view is
  // fitted to the result afterwards, so nothing is clamped against the frame.
  let seed = 7;
  const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  const nodes = g.nodes.map((n) => ({ ...n, key: `${n.kind}:${n.id}`, x: rand() * W, y: rand() * H, vx: 0, vy: 0 }));
  const idx = Object.fromEntries(nodes.map((n) => [n.key, n]));
  const edges = g.edges.map((e) => ({ ...e, s: idx[`${e.src_kind}:${e.src_id}`], t: idx[`${e.dst_kind}:${e.dst_id}`] })).filter((e) => e.s && e.t);
  const degree = {};
  for (const e of edges) { degree[e.s.key] = (degree[e.s.key] || 0) + 1; degree[e.t.key] = (degree[e.t.key] || 0) + 1; }
  let alpha = 1;
  for (let it = 0; it < 320; it++, alpha *= 0.978) {
    for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const l2 = Math.max(1, dx * dx + dy * dy);
      if (l2 > 300 * 300) continue;
      const f = (-220 * alpha) / l2;
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const e of edges) {
      const dx = e.t.x - e.s.x, dy = e.t.y - e.s.y;
      const d = Math.max(1, Math.hypot(dx, dy));
      const strength = 1 / Math.min(degree[e.s.key], degree[e.t.key]);
      const l = ((d - 60) / d) * alpha * strength * 0.5;
      e.t.vx -= dx * l; e.t.vy -= dy * l; e.s.vx += dx * l; e.s.vy += dy * l;
    }
    for (const n of nodes) {
      n.vx += (W / 2 - n.x) * 0.04 * alpha; n.vy += (H / 2 - n.y) * 0.06 * alpha;
      n.x += n.vx; n.y += n.vy; n.vx *= 0.6; n.vy *= 0.6;
    }
  }
  const xs = nodes.map((n) => n.x), ys = nodes.map((n) => n.y);
  const pad = 40;
  const box = [Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) - Math.min(...xs) + 2 * pad + 120, Math.max(...ys) - Math.min(...ys) + 2 * pad];
  const KIND = { skill: "#5b8cff", session: "#3ecf8e", failure: "#f0506e", correction: "#f5a524", run: "#7cc4ff", segment: "#a0a8b8", media: "#c77dff", episode: "#56d4c9" };
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", box.join(" "));
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  for (const e of edges) {
    const l = document.createElementNS(NS, "line");
    Object.entries({ x1: e.s.x, y1: e.s.y, x2: e.t.x, y2: e.t.y, stroke: "#3a4353", "stroke-width": 1 }).forEach(([k, v]) => l.setAttribute(k, v));
    const t = document.createElementNS(NS, "title"); t.textContent = e.rel; l.appendChild(t);
    svg.appendChild(l);
  }
  const targets = { skill: "skills", session: "sessions", failure: "failures", run: "runs" };
  for (const n of nodes) {
    const c = document.createElementNS(NS, "circle");
    Object.entries({ cx: n.x, cy: n.y, r: n.kind === "skill" ? 7 : 5, fill: KIND[n.kind] || "#8b94a5" }).forEach(([k, v]) => c.setAttribute(k, v));
    const t = document.createElementNS(NS, "title"); t.textContent = `${n.kind} ${n.id}`; c.appendChild(t);
    if (targets[n.kind]) { c.style.cursor = "pointer"; c.addEventListener("click", () => { location.hash = `#/${targets[n.kind]}/${n.id}`; }); }
    svg.appendChild(c);
    if (n.kind === "skill") {
      const label = document.createElementNS(NS, "text");
      label.setAttribute("x", n.x + 9); label.setAttribute("y", n.y + 3); label.textContent = n.id.replace(/^seed_/, "");
      svg.appendChild(label);
    }
  }
  const rels = [...new Set(edges.map((e) => e.rel))];
  return h("div", {}, h("div", { class: "legend", style: "margin-bottom:8px" }, Object.entries(KIND).map(([k, c]) => h("span", {}, h("i", { style: `background:${c}` }), k)),
    h("span", { class: "faint" }, `relations: ${rels.join(", ")}`)), h("div", { class: "graph-wrap" }, svg));
}
function retrievalDebug() {
  const q = h("input", { placeholder: "task text", style: "flex:1" });
  const strategy = select(["hybrid", "vector", "lexical", "episodes_only", "none"], "hybrid");
  const out = h("div", {});
  const run = async () => {
    try {
      const r = await POST("/retrieval/debug", { text: q.value, strategy: strategy.value });
      const items = (title, list) => h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, title), table(["Item", "Score", "Components", "Why"], list.map((i) => [
        i.title, num(i.score, 3), h("span", { class: "mono small" }, JSON.stringify(i.components)), h("span", { class: "small faint" }, i.reason_codes.join(", "))])));
      out.replaceChildren(h("div", { class: "stack" }, items("Skills", r.skills), items("Failures", r.failures), items("Semantic", r.semantic), items("Episodes", r.episodes)));
    } catch (e) { toast(e.message, true); }
  };
  q.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  return h("div", { class: "stack" }, h("div", { class: "row" }, q, strategy, h("button", { class: "primary", onclick: run }, "Retrieve")),
    h("p", { class: "sub small" }, "Shows exactly what the planner would see, with score components and reason codes. Queries are recorded like real retrievals."), out);
}

// -- Failures ------------------------------------------------------------------------------------------------------
async function pageFailures(failureId) {
  if (failureId) return pageFailure(failureId);
  const list = await GET("/failures");
  const el = h("div", {}, h("h1", {}, "Failures"),
    h("p", { class: "sub" }, "What went wrong, why, how it was corrected and the rule learned for next time. Rules are proposed from evidence and promoted when corrections keep working."),
    h("div", { class: "card" }, table(["Problem", "Phase", "Future rule", "Rule", "Seen", "Corrections (ok/tried)", "Confidence"], list.map((f) => [
      f.observed_problem, f.phase || "—", h("span", { class: "small" }, f.future_rule || "—"), chip(f.rule_status), f.occurrence_count,
      `${f.correction_successes}/${f.correction_attempts}`, num(f.confidence)]), (i) => { location.hash = `#/failures/${list[i].id}`; })));
  return { el, live: ["FAILURE_RECORDED", "FAILURE_PROMOTED"] };
}
async function pageFailure(failureId) {
  const d = await GET(`/failures/${failureId}`);
  const f = d.failure;
  const inputs = {};
  const editable = ["observed_problem", "likely_cause", "correction", "future_rule", "phase"];
  const form = h("div", { class: "grid g2" }, editable.map((k) => field(k.replace(/_/g, " "), inputs[k] = h("input", { value: f[k] || "" }))),
    field("symptoms (one per line)", inputs.symptoms = h("textarea", { rows: 3 }, (f.symptoms || []).join("\n"))));
  const save = () => act(() => PATCH(`/failures/${f.id}`, {
    changes: { ...Object.fromEntries(editable.map((k) => [k, inputs[k].value || null])), symptoms: inputs.symptoms.value.split("\n").map((x) => x.trim()).filter(Boolean) },
  }), "Saved");
  const note = h("input", { placeholder: "note (optional)" });
  const el = h("div", {},
    h("div", { class: "row between" }, h("h1", {}, f.observed_problem), chip(f.rule_status)),
    h("p", { class: "sub" }, `${f.task_class || "any task"} · phase ${f.phase || "—"} · seen ${f.occurrence_count}× · priority ${num(f.retrieval_priority)}`),
    h("div", { class: "card stack" }, h("strong", {}, "Future rule: ", f.future_rule || "—"),
      h("div", { class: "row" }, note, h("button", { class: "ok", onclick: () => act(() => POST(`/failures/${f.id}/review`, { accept: true, note: note.value }), "Rule accepted") }, "Accept rule"),
        h("button", { class: "danger", onclick: () => act(() => POST(`/failures/${f.id}/review`, { accept: false, note: note.value }), "Rule rejected") }, "Reject rule"))),
    h("h2", {}, "Edit"), h("div", { class: "card stack" }, form, h("button", { class: "primary", onclick: save }, "Save")),
    h("h2", {}, "Evidence"), h("div", { class: "card" }, table(["Kind", "Source", "Detail", "Session / run", "When"], f.evidence.map((e) => [e.kind, chip(e.source_class), e.detail || "—",
      e.session_id ? link(`#/sessions/${e.session_id}`, short(e.session_id)) : e.run_id ? link(`#/runs/${e.run_id}`, short(e.run_id)) : "—", fmtTime(e.ts)]))),
    h("h2", {}, "Corrections"), h("div", { class: "card" }, table(["Kind", "Reason", "Outcome", "Before", "After"], d.corrections.map((c) => [c.kind, c.reason || h("span", { class: "faint" }, "no reason given"), c.outcome || "—",
      c.before_frame_id ? h("img", { src: `/api/frames/${c.before_frame_id}/image`, style: "height:60px" }) : "—",
      c.after_frame_id ? h("img", { src: `/api/frames/${c.after_frame_id}/image`, style: "height:60px" }) : "—"]))));
  return { el, live: ["FAILURE_PROMOTED", "CORRECTION_RECORDED"] };
}

// -- Practice ------------------------------------------------------------------------------------------------------
async function pagePractice(name) {
  if (name) return pageCurriculum(name);
  const list = await GET("/practice");
  const el = h("div", {}, h("h1", {}, "Practice"),
    h("p", { class: "sub" }, "Curricula of sampled tasks with measured mastery gates. Stages that need something the agent was never shown report it instead of pretending."),
    h("div", { class: "grid g2" }, list.map((c) => h("div", { class: "card stack" },
      h("div", { class: "row between" }, h("strong", {}, c.title), link(`#/practice/${c.name}`, "open")),
      h("div", { class: "muted small" }, c.description), ladder(c.overview)))));
  return { el, live: ["MASTERY_UPDATED"] };
}
function ladder(o) {
  return h("div", { class: "ladder" }, o.stages.map((s) => h("div", { class: `rung ${s.status} ${s.stage === o.current_stage ? "current" : ""}` },
    h("div", {}, `${s.stage}. ${s.name}`), chip(s.status), s.metrics.attempts ? h("div", { class: "faint small" }, `${s.metrics.attempts} attempts · ${pct(s.metrics.completion_rate)}`) : null)));
}
async function pageCurriculum(name) {
  const d = await GET(`/practice/${name}`);
  const o = d.overview;
  const stageSel = select(o.stages.map((s) => [String(s.stage), `${s.stage}. ${s.name}`]), String(o.current_stage || 1));
  const attempts = h("input", { type: "number", value: 5, min: 1, max: 50, style: "width:80px" });
  const seed = h("input", { type: "number", placeholder: "seed", style: "width:90px" });
  const backend = select([["headless", "Headless Blender"], ["live", "Live Blender"]], "headless");
  const takeover = h("input", { type: "checkbox" });
  const train = () => act(() => POST(`/practice/${name}/train`, {
    stage: Number(stageSel.value), attempts: Number(attempts.value), seed: seed.value ? Number(seed.value) : null, backend: backend.value, allow_takeover: takeover.checked,
  }), "Practice started");
  const el = h("div", {}, h("h1", {}, o.title), h("p", { class: "sub" }, o.description), ladder(o),
    h("div", { class: "card row", style: "margin-top:14px" }, stageSel, h("span", { class: "muted small" }, "attempts"), attempts, seed, backend,
      h("label", { class: "check small" }, takeover, "ask me when stuck"), h("span", { class: "spacer" }), h("button", { class: "primary", onclick: train }, "Practise")),
    h("h2", {}, "Stages"),
    h("div", { class: "card" }, table(["Stage", "Status", "Attempts", "Completion", "Checks", "Takeovers", "False success", "Generalisation", "Unmet gates", "Override"], o.stages.map((s) => {
      const ov = select([["", "—"], "mastered", "locked", "practicing"], (s.human_override && s.human_override.status) || "", { class: "small" });
      ov.addEventListener("change", () => act(() => POST(`/practice/${name}/stages/${s.stage}/override`, { status: ov.value || null }), "Override saved"));
      const m = s.metrics;
      return [`${s.stage}. ${s.name}`, chip(s.status), m.attempts, pct(m.completion_rate), pct(m.checkpoint_pass_rate), pct(m.takeover_rate), pct(m.false_success_rate), pct(m.generalization),
        h("span", { class: "small faint" }, (s.unmet || []).join(", ") || "—"), ov];
    }))));
  return { el, live: ["MASTERY_UPDATED", "JOB_STATUS"] };
}

// -- Datasets --------------------------------------------------------------------------------------------------------
async function pageDatasets() {
  const [list, training] = await Promise.all([GET("/datasets"), GET("/training")]);
  const name = h("input", { placeholder: "dataset name" });
  const purpose = select([["training", "training (consented only)"], ["export", "export"]], "training");
  const minQ = h("input", { type: "number", step: "0.05", value: 0.5, min: 0, max: 1, style: "width:80px" });
  const includeMedia = h("input", { type: "checkbox" });
  const report = h("div", {});
  const el = h("div", {}, h("h1", {}, "Datasets"),
    h("p", { class: "sub" }, "Build versioned datasets with provenance on every sample. Sessions enter training sets only with consent; nothing is included silently."),
    h("div", { class: "card row" }, name, purpose, h("span", { class: "muted small" }, "min quality"), minQ, h("label", { class: "check small" }, includeMedia, "include permitted media"),
      h("span", { class: "spacer" }), h("button", { onclick: async () => {
        try {
          const r = await GET("/datasets/validate");
          report.replaceChildren(h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, `Quality: mean ${num(r.summary.mean_score)}`),
            table(["Issue", "Count"], Object.entries(r.summary.issue_counts).map(([k, v]) => [k, v])),
            r.summary.dataset_issues.map((i) => notice(i.detail, "warn"))));
        } catch (e) { toast(e.message, true); }
      } }, "Check quality"),
      h("button", { class: "primary", onclick: () => act(() => POST("/datasets", { name: name.value || "dataset", purpose: purpose.value, min_quality: Number(minQ.value), include_media: includeMedia.checked }),
        (r) => `Built ${r.samples} sample(s); excluded ${Object.values(r.excluded).reduce((a, b) => a + b.length, 0)}`) }, "Build")),
    report,
    h("h2", {}, "Built datasets"),
    h("div", { class: "card" }, table(["Name", "Samples", "Excluded", "Mean quality", "Created", ""], list.map((d) => [d.name, d.stats.samples,
      h("span", { class: "small" }, Object.entries(d.stats.excluded || {}).map(([k, v]) => `${k}: ${v}`).join(", ") || "—"), num(d.quality_report.mean_score), fmtTime(d.created_at),
      h("div", { class: "row" }, h("a", { class: "btn small", href: `/api/datasets/${d.id}/download` }, "Download"),
        h("button", { class: "small", onclick: () => act(() => POST(`/datasets/${d.id}/training-files`), (r) => `Wrote ${Object.entries(r.files).map(([k, v]) => `${k}: ${v}`).join(", ")}`) }, "Training files"))]))),
    h("h2", {}, "Training"),
    h("div", { class: "card stack" }, notice(training.note, "warn"),
      h("div", { class: "kv" }, h("span", {}, "Recommendation"), h("span", {}, chip(training.advisor.recommendation)),
        h("span", {}, "Why"), h("span", { class: "small" }, (training.advisor.reasons || []).join("; ") || training.advisor.note || "—"),
        h("span", {}, "Registered backends"), h("span", {}, Object.keys(training.backends).length ? JSON.stringify(training.backends) : "none"))));
  return { el, live: ["DATASET_BUILT"] };
}

// -- Benchmarks ------------------------------------------------------------------------------------------------------
async function pageBenchmarks() {
  const [b, exps] = await Promise.all([GET("/benchmarks"), GET("/experiments")]);
  const armBoxes = [...b.arms, ...Object.keys(b.unavailable_arms)].map((a) => ({ a, box: h("input", { type: "checkbox", checked: a === "baseline" || a === "memory_enhanced" }) }));
  const benchBoxes = b.benchmarks.map((x) => ({ n: x.name, box: h("input", { type: "checkbox", checked: x.difficulty <= 3 }) }));
  const name = h("input", { placeholder: "experiment name" });
  const repeats = h("input", { type: "number", value: 1, min: 1, max: 20, style: "width:70px" });
  const backend = select([["headless", "Headless Blender"], ["live", "Live Blender"]], "headless");
  const start = () => act(() => POST("/experiments", { name: name.value || "experiment", arms: armBoxes.filter((x) => x.box.checked).map((x) => x.a),
    benchmarks: benchBoxes.filter((x) => x.box.checked).map((x) => x.n), repeats: Number(repeats.value), backend: backend.value }), "Experiment started");
  const el = h("div", {}, h("h1", {}, "Benchmarks"),
    h("p", { class: "sub" }, "Controlled comparisons: does memory help? Every arm runs the same tasks on a reset scene; results carry bootstrap confidence intervals."),
    h("div", { class: "card stack" },
      h("div", { class: "row" }, h("strong", {}, "Arms"), armBoxes.map((x) => h("label", { class: "check", title: b.unavailable_arms[x.a] || "" }, x.box, x.a,
        b.unavailable_arms[x.a] ? h("span", { class: "faint small" }, " (unavailable)") : null))),
      h("div", { class: "row" }, h("strong", {}, "Benchmarks"), benchBoxes.map((x) => h("label", { class: "check" }, x.box, x.n))),
      h("div", { class: "row" }, name, h("span", { class: "muted small" }, "repeats"), repeats, backend, h("span", { class: "spacer" }), h("button", { class: "primary", onclick: start }, "Run experiment"))),
    h("h2", {}, "Benchmarks"),
    h("div", { class: "card" }, table(["Name", "Description", "Difficulty", "Variants", "Evaluation"], b.benchmarks.map((x) => [x.name, x.description, x.difficulty, x.variants.length, x.evaluation_method]))),
    h("h2", {}, "Experiments"),
    exps.length ? exps.map((e) => h("div", { class: "card", style: "margin-bottom:10px" },
      h("div", { class: "row between" }, h("strong", {}, e.name), h("span", {}, chip(e.status), " ", h("span", { class: "faint small" }, fmtTime(e.created_at)))),
      armTable(e.summary.arms || {}),
      Object.entries(e.summary.unavailable_arms || {}).map(([a, why]) => h("div", { class: "faint small" }, `${a}: ${why}`)))) : empty("No experiments yet."));
  return { el, live: ["JOB_STATUS", "TASK_COMPLETED"] };
}
function armTable(arms) {
  return table(["Arm", "Runs", "Success", "95% CI", "Checks passed", "Takeovers", "Δ vs baseline"], Object.entries(arms).map(([a, s]) => [a, s.runs,
    h("div", { class: "row" }, bar(s.success_rate, "ok"), pct(s.success_rate)), s.success_ci95 ? `${pct(s.success_ci95[0])}–${pct(s.success_ci95[1])}` : "—",
    pct(s.checkpoint_pass_rate), num(s.mean_takeovers), s.delta_vs_baseline != null ? `${s.delta_vs_baseline >= 0 ? "+" : ""}${pct(s.delta_vs_baseline)}` : "—"]));
}

// -- Settings --------------------------------------------------------------------------------------------------------
async function pageSettings() {
  const d = await GET("/settings");
  const cfg = d.config;
  const inputs = {};
  const sections = ["recording", "privacy", "providers", "processing", "safety", "blender"];
  const sectionCard = (sec) => h("div", { class: "card stack" }, h("h3", { style: "margin-top:0" }, sec),
    h("div", { class: "grid g2" }, Object.entries(cfg[sec]).map(([k, v]) => {
      let input;
      if (sec === "blender" && k === "bridge_token") input = h("input", { value: v ? "(set in config file)" : "(from discovery file)", disabled: true });
      else if (typeof v === "boolean") input = h("input", { type: "checkbox", checked: v });
      else if (typeof v === "number") input = h("input", { type: "number", step: "any", value: v });
      else if (Array.isArray(v)) input = h("textarea", { rows: 3 }, v.join("\n"));
      else input = h("input", { value: v ?? "" });
      if (!(sec === "blender" && k === "bridge_token")) inputs[`${sec}.${k}`] = { input, orig: v };
      return field(k.replace(/_/g, " "), input);
    })));
  const save = () => act(() => {
    const body = {};
    for (const [key, { input, orig }] of Object.entries(inputs)) {
      const [sec, k] = key.split(".");
      let val;
      if (typeof orig === "boolean") val = input.checked;
      else if (typeof orig === "number") val = Number(input.value);
      else if (Array.isArray(orig)) val = input.value.split("\n").map((x) => x.trim()).filter(Boolean);
      else val = input.value === "" && orig === null ? null : input.value;
      if (JSON.stringify(val) !== JSON.stringify(orig)) (body[sec] = body[sec] || {})[k] = val;
    }
    return PUT("/settings", body);
  }, (r) => (r.restart_required.length ? `Saved. Restart Lucius to apply: ${r.restart_required.join(", ")}` : "Saved"));
  const el = h("div", {}, h("h1", {}, "Settings"), h("p", { class: "sub" }, `Data directory: ${d.data_dir}`),
    h("div", { class: "card stack", style: "margin-bottom:14px" }, h("strong", {}, "Credentials"),
      Object.entries(d.credentials).map(([name, present]) => h("div", {}, `${name}: `, present ? chip("present in environment", "ok") : chip("not set", "seed"))),
      h("div", { class: "muted small" }, "Credentials are read from the environment by the provider SDK and are never stored by Lucius. Set providers.llm / vlm / evaluation to “gemini” (and providers.gemini_model — see `lucius models`) or “anthropic” to enable model-assisted refinement, inference and judgement."),
      Object.entries(d.provider_notes || {}).map(([k, v]) => h("div", { class: "faint small" }, `${k}: ${v}`))),
    h("div", { class: "stack" }, sections.map(sectionCard)),
    h("div", { class: "row", style: "margin-top:14px" }, h("span", { class: "spacer" }), h("button", { class: "primary", onclick: save }, "Save settings")));
  return { el, live: [] };
}

// -- boot -----------------------------------------------------------------------------------------------------------
GET("/events/recent?limit=60").then((list) => {
  const seen = new Set(S.feed.map((e) => `${e.type}${e.ts}`));
  S.feed.push(...list.filter((e) => !seen.has(`${e.type}${e.ts}`)));
  S.feed.sort((a, b) => b.ts - a.ts);
  const feed = document.getElementById("live-feed");
  if (feed) feed.replaceChildren(...S.feed.slice(0, 60).map(feedLine));
}).catch(() => {});
refreshStatus();
setInterval(refreshStatus, 5000);
connectEvents();
render();
