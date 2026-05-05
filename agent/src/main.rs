use axum::{extract::State, routing::post, Json, Router};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// HeartbeatPayload mirrors reconciler.HeartbeatMsg in Go.
#[derive(Serialize, Clone)]
struct HeartbeatPayload {
    node_id: String,
    gpu_id: String,
    used_vram_mib: u64,
    total_vram_mib: u64,
    loaded_models: Vec<String>,
    nvlink_enabled: bool,
    model_weight_affinity: std::collections::HashMap<String, u64>,
}

/// Command received from the Go scheduler.
#[derive(Deserialize, Debug)]
struct AgentCommand {
    action: String,               // "PREWARM" | "EVICT" | "CHECKPOINT"
    model_name: String,
    quantization: Option<String>, // "int8" | "int4" | null
}

/// AgentState is shared between the heartbeat loop and the HTTP handlers.
struct AgentState {
    payload: Mutex<HeartbeatPayload>,
    scheduler_url: String, // "http://scheduler:8080/heartbeat"
    loader_url: String,    // "http://localhost:8001" — Python loader sidecar
    http_client: reqwest::Client,
}

// ---------------------------------------------------------------------------
// Loader API types — mirror loader/main.py Pydantic schemas
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LoadRequest {
    model_name: String,
    repo_id: Option<String>,
    quantization: Option<String>,
}

#[derive(Deserialize, Debug)]
struct LoadResponse {
    model_name: String,
    vram_used_mib: u64,
    load_duration_sec: f64,
    affinity_cache_hit: bool,
}

#[derive(Serialize)]
struct EvictRequest {
    model_name: String,
}

#[derive(Serialize)]
struct CheckpointRequest {
    model_name: String,
    checkpoint_dir: Option<String>,
}

#[derive(Deserialize, Debug)]
struct LoaderStatus {
    loaded: Vec<String>,
    vram_used_mib: u64,
    vram_total_mib: u64,
    vram_free_mib: u64,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let node_id = std::env::var("NODE_ID").unwrap_or_else(|_| "node-001".to_string());
    let gpu_id  = std::env::var("GPU_ID").unwrap_or_else(|_| "gpu-0".to_string());
    let scheduler_url = std::env::var("SCHEDULER_URL")
        .unwrap_or_else(|_| "http://localhost:8080/heartbeat".to_string());
    let loader_url = std::env::var("LOADER_URL")
        .unwrap_or_else(|_| "http://localhost:8001".to_string());
    let listen_addr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9090".to_string());

    let initial_payload = HeartbeatPayload {
        node_id: node_id.clone(),
        gpu_id: gpu_id.clone(),
        used_vram_mib: 0,
        total_vram_mib: gpu_total_vram_mib(),
        loaded_models: vec![],
        nvlink_enabled: false,
        model_weight_affinity: std::collections::HashMap::new(),
    };

    let state = Arc::new(AgentState {
        payload: Mutex::new(initial_payload),
        scheduler_url,
        loader_url,
        http_client: reqwest::Client::new(),
    });

    let hb_state = Arc::clone(&state);
    tokio::spawn(async move { heartbeat_loop(hb_state).await; });

    let app = Router::new()
        .route("/command", post(handle_command))
        .with_state(Arc::clone(&state));

    println!("[agent] node={} gpu={} listening on {}", node_id, gpu_id, listen_addr);
    let listener = tokio::net::TcpListener::bind(&listen_addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ---------------------------------------------------------------------------
// Heartbeat loop
// ---------------------------------------------------------------------------

async fn heartbeat_loop(state: Arc<AgentState>) {
    let mut interval = time::interval(Duration::from_millis(500));
    loop {
        interval.tick().await;

        let hw = sample_gpu_state();
        let loader_status = fetch_loader_status(&state).await;

        {
            let mut p = state.payload.lock().unwrap();
            p.used_vram_mib = loader_status.as_ref().map(|s| s.vram_used_mib).unwrap_or(hw.used_vram_mib);
            p.nvlink_enabled = hw.nvlink_enabled;
            if let Some(status) = &loader_status {
                p.loaded_models = status.loaded.clone();
                for model in &status.loaded {
                    p.model_weight_affinity.insert(model.clone(), status.vram_used_mib);
                }
            }
        }

        let payload = state.payload.lock().unwrap().clone();
        if let Err(e) = state.http_client.post(&state.scheduler_url).json(&payload).send().await {
            eprintln!("[agent] heartbeat POST failed: {e}");
        }
    }
}

async fn fetch_loader_status(state: &Arc<AgentState>) -> Option<LoaderStatus> {
    let url = format!("{}/status", state.loader_url);
    state.http_client.get(&url).send().await.ok()?.json::<LoaderStatus>().await.ok()
}

// ---------------------------------------------------------------------------
// GPU sampling — conditional compilation gate
// ---------------------------------------------------------------------------

struct GpuSample { used_vram_mib: u64, nvlink_enabled: bool }

fn gpu_total_vram_mib() -> u64 {
    #[cfg(feature = "mock")] { 81_920 }
    #[cfg(not(feature = "mock"))] { query_nvidia_smi_total_vram() }
}

fn sample_gpu_state() -> GpuSample {
    #[cfg(feature = "mock")] { sample_mock() }
    #[cfg(not(feature = "mock"))] { sample_real() }
}

#[cfg(feature = "mock")]
fn sample_mock() -> GpuSample {
    let mut rng = rand::thread_rng();
    GpuSample { used_vram_mib: rng.gen_range(8_192..73_728u64), nvlink_enabled: false }
}

#[cfg(not(feature = "mock"))]
fn sample_real() -> GpuSample {
    let used_vram_mib = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok())
        .unwrap_or(0);

    let nvlink_enabled = std::process::Command::new("nvidia-smi")
        .args(["nvlink", "--status"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    GpuSample { used_vram_mib, nvlink_enabled }
}

#[cfg(not(feature = "mock"))]
fn query_nvidia_smi_total_vram() -> u64 {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse::<u64>().ok())
        .unwrap_or(81_920)
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

async fn handle_command(
    State(state): State<Arc<AgentState>>,
    Json(cmd): Json<AgentCommand>,
) -> Json<serde_json::Value> {
    println!("[agent] command: action={} model={}", cmd.action, cmd.model_name);

    match cmd.action.as_str() {
        "PREWARM" => match call_loader_load(&state, &cmd.model_name, cmd.quantization.as_deref()).await {
            Ok(resp) => Json(serde_json::json!({
                "status": "ok", "action": "PREWARM",
                "model": resp.model_name,
                "vram_used_mib": resp.vram_used_mib,
                "load_duration_sec": resp.load_duration_sec,
                "affinity_cache_hit": resp.affinity_cache_hit,
            })),
            Err(e) => Json(serde_json::json!({ "status": "error", "detail": e })),
        },
        "EVICT" => match call_loader_evict(&state, &cmd.model_name).await {
            Ok(_) => Json(serde_json::json!({ "status": "ok", "action": "EVICT", "model": cmd.model_name })),
            Err(e) => Json(serde_json::json!({ "status": "error", "detail": e })),
        },
        "CHECKPOINT" => match call_loader_checkpoint(&state, &cmd.model_name).await {
            Ok(path) => Json(serde_json::json!({ "status": "ok", "action": "CHECKPOINT", "model": cmd.model_name, "path": path })),
            Err(e)   => Json(serde_json::json!({ "status": "error", "detail": e })),
        },
        unknown => Json(serde_json::json!({ "status": "error", "message": format!("unknown action: {unknown}") })),
    }
}

// ---------------------------------------------------------------------------
// Loader HTTP calls
// ---------------------------------------------------------------------------

async fn call_loader_load(state: &Arc<AgentState>, model_name: &str, quantization: Option<&str>) -> Result<LoadResponse, String> {
    let resp = state.http_client
        .post(format!("{}/load", state.loader_url))
        .json(&LoadRequest { model_name: model_name.to_string(), repo_id: None, quantization: quantization.map(|s| s.to_string()) })
        .timeout(Duration::from_secs(300)) // large models can take minutes
        .send().await.map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("loader /load {}: {}", resp.status(), resp.text().await.unwrap_or_default()));
    }
    resp.json::<LoadResponse>().await.map_err(|e| e.to_string())
}

async fn call_loader_evict(state: &Arc<AgentState>, model_name: &str) -> Result<(), String> {
    let resp = state.http_client
        .post(format!("{}/evict", state.loader_url))
        .json(&EvictRequest { model_name: model_name.to_string() })
        .timeout(Duration::from_secs(30))
        .send().await.map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("loader /evict: {}", resp.text().await.unwrap_or_default()));
    }
    Ok(())
}

async fn call_loader_checkpoint(state: &Arc<AgentState>, model_name: &str) -> Result<String, String> {
    let resp = state.http_client
        .post(format!("{}/checkpoint", state.loader_url))
        .json(&CheckpointRequest { model_name: model_name.to_string(), checkpoint_dir: None })
        .timeout(Duration::from_secs(120))
        .send().await.map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("loader /checkpoint: {}", resp.text().await.unwrap_or_default()));
    }
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json["path"].as_str().unwrap_or("unknown").to_string())
}
