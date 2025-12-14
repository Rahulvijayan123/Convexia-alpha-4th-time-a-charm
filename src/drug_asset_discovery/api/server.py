from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

from drug_asset_discovery.api.run_manager import RunManager, RunState
from drug_asset_discovery.api.runtime_secrets import apply_runtime_secrets, save_runtime_secrets
from drug_asset_discovery.config import EnvSettings
from drug_asset_discovery.logging import configure_logging
from drug_asset_discovery.orchestrator.orchestrator import run_discovery
from drug_asset_discovery.storage.replay import replay_run
from drug_asset_discovery.storage.supabase_store import SupabaseStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Drug Asset Discovery", version="0.1.0")
run_manager = RunManager()


UI_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Drug Asset Discovery — Dev UI</title>
    <style>
      :root { color-scheme: dark; }
      body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: #0b1020; color: #e6e8f0; }
      .wrap { max-width: 1100px; margin: 0 auto; padding: 18px; }
      h1 { font-size: 18px; margin: 0 0 12px; font-weight: 650; letter-spacing: .2px; }
      .card { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.10); border-radius: 12px; padding: 14px; }
      .row { display: flex; gap: 10px; align-items: center; }
      input[type=text] { flex: 1; padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,.18); background: rgba(0,0,0,.25); color: #e6e8f0; outline: none; }
      button { padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.10); color: #e6e8f0; cursor: pointer; }
      button:disabled { opacity: .5; cursor: not-allowed; }
      .meta { margin-top: 10px; font-size: 12px; color: rgba(230,232,240,.72); display: flex; gap: 14px; flex-wrap: wrap; }
      .meta code { color: rgba(230,232,240,.9); }
      .tools { margin-top: 10px; display:flex; gap: 10px; align-items: center; flex-wrap: wrap; }
      .tools label { font-size: 12px; color: rgba(230,232,240,.72); display:flex; gap: 6px; align-items:center; }
      .tools input[type=text], .tools select { padding: 7px 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,.18); background: rgba(0,0,0,.25); color: #e6e8f0; outline: none; font-size: 12px; }
      .tools input[type=checkbox] { transform: translateY(1px); }
      .logs { margin-top: 14px; }
      .logbox { height: 520px; overflow: auto; background: rgba(0,0,0,.35); border: 1px solid rgba(255,255,255,.10); border-radius: 12px; padding: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New"; font-size: 12px; line-height: 1.35; }
      details.logline { border: 1px solid rgba(255,255,255,.10); background: rgba(255,255,255,.03); border-radius: 10px; margin: 0 0 8px 0; }
      details.logline[open] { background: rgba(255,255,255,.05); }
      summary.logsum { list-style: none; cursor: pointer; padding: 8px 10px; display: grid; grid-template-columns: 86px 86px 72px 120px 1fr; gap: 10px; align-items: center; }
      summary.logsum::-webkit-details-marker { display: none; }
      .cell { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .cell.msg { color: rgba(230,232,240,.92); }
      .cell.kind { color: rgba(230,232,240,.72); }
      .cell.time, .cell.delta { color: rgba(230,232,240,.60); }
      .cell.level { font-weight: 700; letter-spacing: .25px; }
      details.logline pre { margin: 0; padding: 10px; border-top: 1px solid rgba(255,255,255,.10); white-space: pre-wrap; color: rgba(230,232,240,.86); }
      .lvl-debug .cell.level { color: #9aa6ff; }
      .lvl-info .cell.level { color: #5ee38b; }
      .lvl-warning .cell.level { color: #ffcd57; }
      .lvl-error .cell.level { color: #ff6b6b; }
      .pill { display:inline-block; padding:2px 8px; border-radius: 999px; border: 1px solid rgba(255,255,255,.15); background: rgba(255,255,255,.08); }
      .muted { color: rgba(230,232,240,.60); }
      .ok { color: #5ee38b; }
      .bad { color: #ff6b6b; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <h1>Drug Asset Discovery — Dev UI (live detailed logs)</h1>
      <div class="card">
        <div class="row">
          <input id="q" type="text" placeholder="Type a search query… (e.g., 'KRAS inhibitor biotech pipeline code names')" />
          <button id="start">Search & Start</button>
          <button id="stop" disabled>Stop logs</button>
          <button id="clear">Clear</button>
        </div>
        <div class="meta">
          <span class="pill">UI run: <code id="ui_run_id" class="muted">—</code></span>
          <span class="pill">Discovery run: <code id="discovery_run_id" class="muted">—</code></span>
          <span class="pill">Status: <code id="status" class="muted">idle</code></span>
          <span class="pill">Elapsed: <code id="elapsed" class="muted">0.000s</code></span>
          <span class="pill">Events: <code id="events" class="muted">0</code></span>
          <span class="pill">Keys: <code id="keys" class="muted">checking…</code></span>
          <span class="pill">Config: <code id="cfg" class="muted">—</code></span>
        </div>
        <div class="tools">
          <label>Filter <input id="filter" type="text" placeholder="type to filter (message/kind/logger)…" /></label>
          <label>Level
            <select id="level">
              <option value="">all</option>
              <option value="DEBUG">DEBUG</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
            </select>
          </label>
          <label><input id="autoscroll" type="checkbox" checked /> autoscroll</label>
          <span class="muted">Tip: click a row to expand full details.</span>
        </div>
        <div class="logs">
          <div id="logbox" class="logbox" role="log" aria-live="polite"></div>
        </div>
      </div>
    </div>
    <script>
      const el = (id) => document.getElementById(id);
      const logbox = el("logbox");
      const q = el("q");
      const startBtn = el("start");
      const stopBtn = el("stop");
      const clearBtn = el("clear");
      const uiRunEl = el("ui_run_id");
      const discRunEl = el("discovery_run_id");
      const statusEl = el("status");
      const elapsedEl = el("elapsed");
      const eventsEl = el("events");
      const keysEl = el("keys");
      const cfgEl = el("cfg");
      const filterEl = el("filter");
      const levelEl = el("level");
      const autoscrollEl = el("autoscroll");
      let es = null;
      let lastSeq = 0;
      let eventsCount = 0;
      let lastElapsedMs = 0;
      let lastElapsedPerf = 0;
      let ticker = null;
      let streamEndExpected = false;
      let currentUiRunId = null;

      function fmtMs(ms) {
        const s = (ms / 1000).toFixed(3);
        return `${s}s`;
      }

      function updateElapsedDisplay() {
        if (!lastElapsedPerf) {
          elapsedEl.textContent = fmtMs(lastElapsedMs);
          elapsedEl.classList.remove("muted");
          return;
        }
        const drift = Math.max(0, performance.now() - lastElapsedPerf);
        elapsedEl.textContent = fmtMs(lastElapsedMs + drift);
        elapsedEl.classList.remove("muted");
      }

      function applyFilters() {
        const needle = (filterEl.value || "").trim().toLowerCase();
        const lvlNeedle = (levelEl.value || "").trim().toUpperCase();
        const kids = Array.from(logbox.children);
        for (const node of kids) {
          const lvl = (node.dataset.level || "").toUpperCase();
          const hay = (node.dataset.hay || "").toLowerCase();
          const okLvl = !lvlNeedle || lvl === lvlNeedle;
          const okNeedle = !needle || hay.includes(needle);
          node.style.display = (okLvl && okNeedle) ? "" : "none";
        }
      }

      function addLine(evt) {
        if (!evt || !evt.seq) return;
        if (evt.seq <= lastSeq) return; // avoid duplicates on buffer replay
        lastSeq = evt.seq;
        const lvl = (evt.level || "INFO").toUpperCase();
        const kind = evt.kind || "event";
        const msg = evt.message || "";
        const data = evt.data || {};

        // keep elapsed counter live and smooth
        if (typeof evt.elapsed_ms === "number") {
          lastElapsedMs = evt.elapsed_ms;
          lastElapsedPerf = performance.now();
          updateElapsedDisplay();
        }

        eventsCount += 1;
        eventsEl.textContent = String(eventsCount);
        eventsEl.classList.remove("muted");

        if (data.discovery_run_id) {
          discRunEl.textContent = data.discovery_run_id;
          discRunEl.classList.remove("muted");
        }
        if (kind === "run" && data && data.status) {
          statusEl.textContent = data.status;
          statusEl.classList.remove("muted");
        }
        // If we get a terminal event, freeze the timer shortly after.
        const isTerminal = (kind === "run") && (
          /run failed|run completed|terminal state/i.test(String(msg || "")) ||
          ["completed", "failed"].includes(String(data.status || "").toLowerCase())
        );
        if (isTerminal) {
          streamEndExpected = true;
          // Prefer explicit status if we have it.
          const st = String(data.status || "").toLowerCase();
          if (st === "failed" || st === "completed") {
            statusEl.textContent = st;
          } else if (/failed/i.test(msg)) {
            statusEl.textContent = "failed";
          } else if (/completed/i.test(msg)) {
            statusEl.textContent = "completed";
          }
          statusEl.classList.remove("muted");
          if (ticker) { clearInterval(ticker); ticker = null; }
        }

        const detailObj = {
          seq: evt.seq,
          ts: evt.ts,
          elapsed_ms: evt.elapsed_ms,
          delta_ms: evt.delta_ms,
          level: lvl,
          kind,
          message: msg,
          data,
        };

        const details = document.createElement("details");
        details.className = `logline lvl-${lvl.toLowerCase()}`;
        details.dataset.level = lvl;
        const hay = [lvl, kind, msg, (data.logger || ""), (data.file || ""), (data.func || "")].join(" ");
        details.dataset.hay = hay;

        const summary = document.createElement("summary");
        summary.className = "logsum";

        const cTime = document.createElement("div");
        cTime.className = "cell time";
        cTime.textContent = `+${fmtMs(evt.elapsed_ms || 0)}`;

        const cDelta = document.createElement("div");
        cDelta.className = "cell delta";
        cDelta.textContent = `Δ${fmtMs(evt.delta_ms || 0)}`;

        const cLvl = document.createElement("div");
        cLvl.className = "cell level";
        cLvl.textContent = lvl;

        const cKind = document.createElement("div");
        cKind.className = "cell kind";
        cKind.textContent = `<${kind}>`;

        const cMsg = document.createElement("div");
        cMsg.className = "cell msg";
        const msgOneLine = String(msg || "").replace(/\\s+/g, " ").trim();
        cMsg.textContent = msgOneLine.length > 200 ? (msgOneLine.slice(0, 200) + "…") : msgOneLine;

        summary.appendChild(cTime);
        summary.appendChild(cDelta);
        summary.appendChild(cLvl);
        summary.appendChild(cKind);
        summary.appendChild(cMsg);

        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(detailObj, null, 2);

        details.appendChild(summary);
        details.appendChild(pre);

        logbox.appendChild(details);
        applyFilters();

        if (autoscrollEl.checked) {
          logbox.scrollTop = logbox.scrollHeight;
        }
      }

      async function refreshHealth() {
        try {
          const r = await fetch("/api/ui/health");
          const j = await r.json();
          cfgEl.textContent = `${j.default_config_version}/${j.default_prompt_version} | openai_base=${j.openai_base_url}`;
          cfgEl.classList.remove("muted");
          const ok = j.openai_api_key_present && j.supabase_url_present && j.supabase_service_role_key_present;
          keysEl.textContent = ok ? "connected" : "missing env (OPENAI_API_KEY / SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)";
          keysEl.classList.remove("muted");
          keysEl.classList.toggle("ok", ok);
          keysEl.classList.toggle("bad", !ok);
        } catch {
          keysEl.textContent = "health check failed";
          keysEl.classList.remove("muted");
          keysEl.classList.add("bad");
        }
      }

      function stopLogs() {
        if (es) { es.close(); es = null; }
        stopBtn.disabled = true;
        startBtn.disabled = false;
        if (ticker) { clearInterval(ticker); ticker = null; }
      }

      async function startRun() {
        await refreshHealth();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusEl.textContent = "starting…";
        statusEl.classList.remove("muted");
        uiRunEl.textContent = "—";
        uiRunEl.classList.add("muted");
        discRunEl.textContent = "—";
        discRunEl.classList.add("muted");
        lastSeq = 0;
        eventsCount = 0;
        eventsEl.textContent = "0";
        eventsEl.classList.remove("muted");
        lastElapsedMs = 0;
        lastElapsedPerf = performance.now();
        updateElapsedDisplay();
        if (ticker) clearInterval(ticker);
        ticker = setInterval(updateElapsedDisplay, 100);
        logbox.innerHTML = "";
        streamEndExpected = false;
        currentUiRunId = null;

        const query = (q.value || "").trim();
        if (query.length < 3) {
          addLine({ seq: 1, elapsed_ms: 0, delta_ms: 0, level: "WARNING", kind: "client", message: "Please enter at least 3 characters.", data: {} });
          startBtn.disabled = false;
          stopBtn.disabled = true;
          return;
        }

        const r = await fetch("/api/ui/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query })
        });
        const j = await r.json();
        if (!r.ok) {
          addLine({ seq: 1, elapsed_ms: 0, delta_ms: 0, level: "ERROR", kind: "client", message: "Failed to start", data: { response: j } });
          startBtn.disabled = false;
          stopBtn.disabled = true;
          return;
        }

        uiRunEl.textContent = j.ui_run_id;
        uiRunEl.classList.remove("muted");
        statusEl.textContent = "running";
        currentUiRunId = j.ui_run_id;

        es = new EventSource(`/api/ui/runs/${encodeURIComponent(j.ui_run_id)}/events`);
        es.addEventListener("message", (e) => {
          try { addLine(JSON.parse(e.data)); } catch {}
        });
        es.addEventListener("ping", () => {});
        es.onerror = async () => {
          // EventSource fires `onerror` when the server closes the stream too.
          if (streamEndExpected) {
            stopLogs();
            return;
          }
          // Try to fetch the run result to show *why* it ended.
          try {
            if (currentUiRunId) {
              const rr = await fetch(`/api/ui/runs/${encodeURIComponent(currentUiRunId)}/result`);
              const rj = await rr.json();
              addLine({
                seq: lastSeq + 1,
                elapsed_ms: lastElapsedMs,
                delta_ms: 0,
                level: "ERROR",
                kind: "client",
                message: "event stream disconnected (fetched last known run state)",
                data: { run: rj }
              });
            } else {
              addLine({ seq: lastSeq + 1, elapsed_ms: lastElapsedMs, delta_ms: 0, level: "WARNING", kind: "client", message: "event stream disconnected (stopping)", data: {} });
            }
          } catch {
            addLine({ seq: lastSeq + 1, elapsed_ms: lastElapsedMs, delta_ms: 0, level: "WARNING", kind: "client", message: "event stream disconnected (stopping)", data: {} });
          }
          stopLogs();
        };
      }

      startBtn.addEventListener("click", startRun);
      stopBtn.addEventListener("click", stopLogs);
      clearBtn.addEventListener("click", () => { logbox.innerHTML = ""; lastSeq = 0; eventsCount = 0; eventsEl.textContent = "0"; });
      q.addEventListener("keydown", (e) => { if (e.key === "Enter") startRun(); });
      filterEl.addEventListener("input", applyFilters);
      levelEl.addEventListener("change", applyFilters);

      refreshHealth();
    </script>
  </body>
</html>
"""


class RunRequest(BaseModel):
    query: str = Field(min_length=3)
    config_version: str | None = None
    prompt_version: str | None = None
    replay: bool = False
    replay_run_id: str | None = None


class RunResponse(BaseModel):
    run_id: str
    status: str
    manifest: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    # Back-compat: `assets` is the validated assets list.
    assets: list[dict[str, Any]] = Field(default_factory=list)
    # v1.6+ product semantics: show both lanes to users.
    found_assets: list[dict[str, Any]] = Field(default_factory=list)
    validated_assets: list[dict[str, Any]] = Field(default_factory=list)


@app.on_event("startup")
async def _startup() -> None:
    apply_runtime_secrets()
    env = EnvSettings()
    configure_logging(env.log_level)


@app.post("/run", response_model=RunResponse)
async def post_run(
    body: RunRequest,
    x_idempotency_key: str | None = Header(default=None, alias="X-Idempotency-Key"),
) -> RunResponse:
    env = EnvSettings()
    config_version = body.config_version or env.default_config_version
    prompt_version = body.prompt_version or env.default_prompt_version

    if body.replay:
        if not body.replay_run_id:
            raise HTTPException(status_code=400, detail="replay_run_id is required when replay=true")
        if not env.supabase_url or not env.supabase_service_role_key:
            raise HTTPException(status_code=500, detail="Supabase env not configured")
        store = SupabaseStore(url=env.supabase_url, service_role_key=env.supabase_service_role_key)
        payload = await replay_run(store=store, run_id=body.replay_run_id)
        validated = payload.get("validated_assets") or payload.get("assets") or []
        found = payload.get("found_assets") or []
        return RunResponse(
            run_id=payload["run_id"],
            status=payload["status"],
            summary=None,
            assets=list(validated),
            found_assets=list(found),
            validated_assets=list(validated),
        )

    try:
        result = await run_discovery(
            user_query=body.query,
            config_version=config_version,
            prompt_version=prompt_version,
            idempotency=x_idempotency_key,
        )
        return RunResponse(
            run_id=result["run_id"],
            status=result["status"],
            manifest=result.get("manifest"),
            summary=result.get("summary"),
            assets=result.get("validated_assets") or result.get("assets", []),
            found_assets=result.get("found_assets", []),
            validated_assets=result.get("validated_assets") or result.get("assets", []),
        )
    except Exception as e:
        logger.exception("run failed")
        raise HTTPException(status_code=500, detail=str(e))


class UIRunRequest(BaseModel):
    query: str = Field(min_length=3)
    config_version: str | None = None
    prompt_version: str | None = None


class UISecretsRequest(BaseModel):
    openai_api_key: str = Field(min_length=10)
    supabase_url: str = Field(min_length=8)
    supabase_service_role_key: str = Field(min_length=20)


@app.get("/ui", response_class=HTMLResponse)
async def get_ui() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


@app.get("/api/ui/health")
async def get_ui_health() -> JSONResponse:
    env = EnvSettings()
    return JSONResponse(
        {
            "openai_base_url": env.openai_base_url,
            "openai_api_key_present": bool(env.openai_api_key and env.openai_api_key.strip()),
            "supabase_url_present": bool(env.supabase_url and env.supabase_url.strip()),
            "supabase_service_role_key_present": bool(env.supabase_service_role_key and env.supabase_service_role_key.strip()),
            "openai_api_key_in_process_env": bool(os.environ.get("OPENAI_API_KEY")),
            "supabase_url_in_process_env": bool(os.environ.get("SUPABASE_URL")),
            "supabase_service_role_key_in_process_env": bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY")),
            "default_config_version": env.default_config_version,
            "default_prompt_version": env.default_prompt_version,
            "log_level": env.log_level,
        }
    )


@app.post("/api/ui/secrets")
async def post_ui_secrets(req: Request, body: UISecretsRequest) -> JSONResponse:
    # Local dev only: refuse non-local callers.
    host = (req.client.host if req.client else "") or ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(status_code=403, detail="forbidden")

    # Set process env so existing code paths (EnvSettings) pick them up immediately.
    os.environ["OPENAI_API_KEY"] = body.openai_api_key.strip()
    os.environ["SUPABASE_URL"] = body.supabase_url.strip()
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = body.supabase_service_role_key.strip()
    save_runtime_secrets(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_service_role_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

    # Never echo secrets back.
    return JSONResponse({"ok": True})


@app.post("/api/ui/run")
async def post_ui_run(body: UIRunRequest) -> JSONResponse:
    env = EnvSettings()
    st = await run_manager.create_run()
    cfg_version = body.config_version or env.default_config_version
    prompt_version = body.prompt_version or env.default_prompt_version

    async def _do_run(state: RunState) -> dict[str, Any]:
        state.publish(
            level="INFO",
            kind="preflight",
            message="Loaded env settings for run",
            data={
                "openai_base_url": env.openai_base_url,
                "openai_api_key_present": bool(env.openai_api_key and env.openai_api_key.strip()),
                "supabase_url_present": bool(env.supabase_url and env.supabase_url.strip()),
                "supabase_service_role_key_present": bool(env.supabase_service_role_key and env.supabase_service_role_key.strip()),
                "config_version": cfg_version,
                "prompt_version": prompt_version,
            },
        )

        def _on_run_id(run_id: str) -> None:
            state.discovery_run_id = run_id
            state.publish(
                level="INFO",
                kind="run",
                message="Supabase run_id created (pipeline has begun persisting to Supabase)",
                data={"discovery_run_id": run_id},
            )

        return await run_discovery(
            user_query=body.query,
            config_version=cfg_version,
            prompt_version=prompt_version,
            idempotency=None,
            on_run_id=_on_run_id,
        )

    await run_manager.start(ui_run_id=st.ui_run_id, coro_factory=_do_run)
    return JSONResponse({"ui_run_id": st.ui_run_id})


@app.get("/api/ui/runs/{ui_run_id}")
async def get_ui_run(ui_run_id: str) -> JSONResponse:
    st = await run_manager.get(ui_run_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown ui_run_id")
    return JSONResponse(st.snapshot())


@app.get("/api/ui/runs/{ui_run_id}/result")
async def get_ui_run_result(ui_run_id: str) -> JSONResponse:
    st = await run_manager.get(ui_run_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown ui_run_id")
    return JSONResponse(
        {"status": st.status, "discovery_run_id": st.discovery_run_id, "error": st.error, "result": st.result}
    )


@app.get("/api/ui/runs/{ui_run_id}/events")
async def get_ui_run_events(ui_run_id: str) -> StreamingResponse:
    st = await run_manager.get(ui_run_id)
    if not st:
        raise HTTPException(status_code=404, detail="unknown ui_run_id")
    return StreamingResponse(run_manager.sse_stream(ui_run_id), media_type="text/event-stream")


