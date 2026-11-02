// Review dashboard - plain JavaScript, no build step.
const $ = (id) => document.getElementById(id);
let token = localStorage.getItem("token");
let watchingJob = null;     // job shown in the live preview
let selectedFlag = null;

// ---------- helpers ----------
async function api(path, opts = {}) {
  opts.headers = Object.assign({ Authorization: "Bearer " + token }, opts.headers || {});
  const res = await fetch(path, opts);
  if (res.status === 401) { logout(); throw new Error("Please log in"); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}
const withToken = (url) => url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
const fmt = (s) => (s == null ? "-" : s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function showPage(name) {
  document.querySelectorAll(".page").forEach((p) => p.classList.add("hidden"));
  $("page-" + name).classList.remove("hidden");
  document.querySelectorAll("nav button[data-page]").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
  if (name === "review") loadQueue();
  if (name === "stats") loadStats();
  if (name === "live") loadJobs();
}

// ---------- login ----------
$("loginForm").onsubmit = async (e) => {
  e.preventDefault();
  try {
    const res = await fetch("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: $("username").value, password: $("password").value }) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    token = data.token;
    localStorage.setItem("token", token);
    localStorage.setItem("who", data.username + " (" + data.role + ")");
    start();
  } catch (err) { $("loginError").textContent = err.message; }
};
function logout() {
  localStorage.removeItem("token"); token = null;
  $("nav").classList.add("hidden");
  document.querySelectorAll(".page").forEach((p) => p.classList.add("hidden"));
  $("page-login").classList.remove("hidden");
}
$("logout").onclick = logout;
document.querySelectorAll("nav button[data-page]").forEach((b) => (b.onclick = () => showPage(b.dataset.page)));

// ---------- live ----------
$("startStream").onclick = async () => {
  const url = $("streamUrl").value.trim();
  if (!url) return ($("liveMsg").textContent = "Paste the phone camera URL first (or 0 for the webcam).");
  try {
    const r = await api("/streams", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, name: $("streamName").value || "Live session" }) });
    $("liveMsg").textContent = "Session " + r.job_id + " started. Loading the detector can take a few seconds...";
    watch(r.job_id); loadJobs();
  } catch (err) { $("liveMsg").textContent = "Error: " + err.message; }
};
$("uploadVideo").onclick = async () => {
  const f = $("videoFile").files[0];
  if (!f) return ($("liveMsg").textContent = "Choose a video file first.");
  const fd = new FormData(); fd.append("file", f);
  $("liveMsg").textContent = "Uploading...";
  try {
    const r = await api("/videos", { method: "POST", body: fd });
    $("liveMsg").textContent = "Video uploaded. Job " + r.job_id + " is processing.";
    watch(r.job_id); loadJobs();
  } catch (err) { $("liveMsg").textContent = "Error: " + err.message; }
};
function watch(jobId) {
  watchingJob = jobId;
  $("watchingLabel").textContent = "(session " + jobId + ")";
  $("preview").src = withToken("/jobs/" + jobId + "/preview.mjpg") + "&t=" + Date.now();
}
async function stopJob(jobId) { await api("/streams/" + jobId + "/stop", { method: "POST" }); loadJobs(); }

async function loadJobs() {
  const jobs = await api("/jobs");
  $("jobsTable").querySelector("tbody").innerHTML = jobs.slice(0, 10).map((j) => `
    <tr><td>${j.id}</td><td>${esc(j.name)}</td><td>${j.status}${j.error ? " ⚠" : ""}</td><td>${j.avg_fps}</td>
    <td>${j.live ? `<button class="secondary" onclick="watch(${j.id})">Watch</button>
      <button class="secondary" onclick="stopJob(${j.id})">Stop</button>` : ""}</td></tr>`).join("");
  const err = jobs.find((j) => j.id === watchingJob && j.error);
  if (err) $("liveMsg").textContent = "Session " + err.id + ": " + err.error;
}

function flagItem(f) {
  const thumb = f.has_thumb ? `<img src="${withToken("/flags/" + f.id + "/thumbnail")}">` : "<img>";
  return `<li data-id="${f.id}" onclick="openFlag(${f.id})">${thumb}
    <div>Student #${f.track_id} &middot; zone ${f.zone}<br>
    <span class="score">${(f.peak_score * 100).toFixed(0)}%</span> ${esc(f.top_class)}<br>
    <span class="hint">t=${f.start_s}s &middot; session ${f.job_id}</span>
    <span class="status-${f.status}">${f.status}</span></div></li>`;
}

// ---------- websocket: live flag push ----------
function connectWs() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws?token=" + token);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "flag_new") {
      $("liveFlags").insertAdjacentHTML("afterbegin", flagItem(msg.flag));
      refreshCount();
      if (!$("page-review").classList.contains("hidden")) loadQueue();
    } else if (msg.type === "flag_update") {
      const li = $("liveFlags").querySelector(`li[data-id="${msg.flag.id}"]`);
      if (li) li.outerHTML = flagItem(msg.flag);
      if (selectedFlag === msg.flag.id && msg.flag.has_clip && !$("detail").querySelector("video")) openFlag(msg.flag.id);
      refreshCount();
    } else if (msg.type === "job") loadJobs();
  };
  ws.onclose = () => token && setTimeout(connectWs, 3000);     // auto-reconnect
  setInterval(() => ws.readyState === 1 && ws.send("ping"), 25000);
}

// ---------- review ----------
$("statusFilter").onchange = loadQueue;
async function loadQueue() {
  const flags = await api("/flags?status=" + $("statusFilter").value);
  $("queue").innerHTML = flags.map(flagItem).join("") || "<li class='hint'>Nothing to review.</li>";
}
async function refreshCount() {
  const pending = await api("/flags?status=pending&limit=1000");
  $("pendingCount").textContent = pending.length;
}
async function openFlag(id) {
  showPage("review");
  selectedFlag = id;
  const f = await api("/flags/" + id);
  document.querySelectorAll("#queue li").forEach((li) => li.classList.toggle("sel", +li.dataset.id === id));
  const clip = f.has_clip
    ? `<video controls autoplay muted src="${withToken("/flags/" + id + "/clip")}"></video>`
    : `<p class="hint">Clip is still being recorded (a few seconds after the flag)...</p>
       ${f.has_thumb ? `<img style="max-width:100%" src="${withToken("/flags/" + id + "/thumbnail")}">` : ""}`;
  const history = f.reviews.map((r) => `<li>${esc(r.user)}: <b>${r.decision}</b> ${esc(r.note)} <span class="hint">${new Date(r.decided_at).toLocaleString()}</span></li>`).join("");
  $("detail").innerHTML = `
    <h2>Flag ${f.id} - flagged for review <span class="status-${f.status}">(${f.status})</span></h2>
    ${clip}
    <p>Student #${f.track_id} (tracker ID, not identity) &middot; zone <b>${f.zone}</b> &middot;
       ${f.start_s}s - ${f.end_s}s &middot; peak <b>${(f.peak_score * 100).toFixed(0)}%</b>,
       mean ${(f.mean_score * 100).toFixed(0)}% &middot; model's top class: ${esc(f.top_class)}</p>
    <p class="hint">Watch the clip. You decide whether there is a real concern - the score is only a hint.</p>
    <textarea id="note" placeholder="Note (optional): what you saw"></textarea>
    <div class="inline">
      <button class="confirm" onclick="review(${f.id}, 'confirmed')">Confirm concern</button>
      <button class="dismiss" onclick="review(${f.id}, 'dismissed')">Dismiss (no issue)</button>
    </div>
    ${history ? "<h2>Decisions</h2><ul>" + history + "</ul>" : ""}`;
}
async function review(id, decision) {
  await api("/flags/" + id + "/review", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, note: $("note").value }) });
  $("detail").innerHTML = `<p>Saved: flag ${id} ${decision}. Select the next flag.</p>`;
  loadQueue(); refreshCount();
}

// ---------- stats ----------
async function loadStats() {
  const s = await api("/stats");
  const items = [["Total flags", s.total_flags], ["Pending", s.pending], ["Confirmed", s.confirmed],
    ["Dismissed", s.dismissed], ["Precision (reviewed)", s.precision_reviewed == null ? "-" : (s.precision_reviewed * 100).toFixed(0) + "%"],
    ["Flags per hour", fmt(s.flags_per_hour)], ["Avg review time (s)", fmt(s.avg_review_seconds)],
    ["Avg processing fps", fmt(s.avg_processing_fps)], ["Avg latency (ms)", fmt(s.avg_latency_ms)],
    ["Footage analysed (h)", s.footage_hours], ["Model", s.model.stub ? "STUB" : s.model.model_path]];
  $("statsGrid").innerHTML = items.map(([k, v]) => `<div class="stat"><span class="hint">${k}</span><b>${v}</b></div>`).join("");
  const jobs = await api("/jobs");
  $("statsJobs").querySelector("tbody").innerHTML = jobs.map((j) => `<tr><td>${j.id}</td><td>${esc(j.name)}</td>
    <td>${j.kind}</td><td>${j.status}</td><td>${j.frames_processed}</td><td>${j.video_seconds}</td>
    <td>${j.avg_fps}</td><td>${j.avg_latency_ms}</td></tr>`).join("");
}

// ---------- start ----------
async function start() {
  $("page-login").classList.add("hidden");
  $("nav").classList.remove("hidden");
  $("who").textContent = localStorage.getItem("who") || "";
  try {
    const cfg = await api("/config");
    $("cap").textContent = cfg.alert_cap;
    const s = await api("/stats");
    $("stubBanner").classList.toggle("hidden", !s.model.stub);
  } catch (e) { return; }
  showPage("live");
  connectWs(); refreshCount();
  setInterval(() => !$("page-live").classList.contains("hidden") && loadJobs(), 5000);
}
if (token) start(); else logout();
