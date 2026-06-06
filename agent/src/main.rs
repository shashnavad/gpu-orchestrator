use axum::{extract::State, routing::post, Json, Router};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time;

// ---------------------------------------------------------------------------
// MIG types
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug)]
struct MIGSlice {
    slice_id: String,
    profile: String,
    total_vram_mib: u64,
    used_vram_mib: u64,
    loaded_models: Vec<String>,
    healthy: bool,
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
struct HeartbeatPayload {
    node_id: String,
    gpu_id: String,
    used_vram_mib: u64,
    total_vram_mib: u64,
    loaded_models: Vec<String>,
    nvlink_enabled: bool,
    mig_enabled: bool,
    mig_slices: Vec<MIGSlice>,
    model_weight_affinity: std::collections::HashMap<String, u64>,
}

#[derive(Deserialize, Debug)]
struct AgentCommand {
    action: String,
    model_name: String,
    quantization: Option<String>,
    slice_id: Option<String>,
}

struct AgentState {
    payload: Mutex<HeartbeatPayload>,
    scheduler_url: String,
    loader_url: String,
    http_client: reqwest::Client,
}

// ---------------------------------------------------------------------------
// Loader API types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct LoadRequest {
    model_name: String,
    repo_id: Option<String>,
    quantization: Option<String>,
    slice_id: Option<String>,
    slice_vram_cap_mib: Option<u64>,
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
// Mock model definitions
//
// Each MockModel has a stable base VRAM footprint matching real HuggingFace
// model sizes. VRAM drifts ±DRIFT_PCT of the base each tick to simulate
// realistic KV-cache and activation memory pressure during inference.
// ---------------------------------------------------------------------------

struct MockModel {
    name: &'static str,
    base_vram_mib: u64,
}

/// Drift fraction applied each tick: ±5% of base VRAM.
const DRIFT_PCT: f64 = 0.05;

/// Models resident on node-001 (MIG H100, 3x 2g.20gb slices).
/// Each model is pinned to one slice. Footprints fit within 20,480 MiB.
const MIG_MODELS: [(&str, MockModel); 3] = [
    ("0/0/0", MockModel { name: "phi-3-mini",   base_vram_mib: 3_800 }),
    ("1/0/0", MockModel { name: "llama-3-8b",   base_vram_mib: 8_192 }),
    ("2/0/0", MockModel { name: "mistral-7b",   base_vram_mib: 7_168 }),
];

/// Models resident on node-002 (non-MIG H100, 80 GiB).
/// Two large models share the full GPU; combined footprint ~49 GiB.
const PLAIN_MODELS: [MockModel; 2] = [
    MockModel { name: "llama-3-70b",    base_vram_mib: 35_840 },
    MockModel { name: "codellama-13b",  base_vram_mib: 13_312 },
];

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

    // MIG_NODE=true boots the agent as the MIG-partitioned H100 (node-001).
    // Any other value (or absent) boots as the plain H100 (node-002).
    let mig_node = std::env::var("MIG_NODE").map(|v| v == "true").unwrap_or(false);

    let (mig_enabled, initial_slices, initial_models, total_vram) =
        build_initial_state(mig_node);

    // Pre-populate model_weight_affinity so the scheduler can see that this
    // node already has these weights on NVMe and prefer it for future placements.
    let mut affinity = std::collections::HashMap::new();
    for m in &initial_models {
        affinity.insert(m.to_string(), 0u64); // size filled in on first heartbeat
    }

    let initial_payload = HeartbeatPayload {
        node_id: node_id.clone(),
        gpu_id: gpu_id.clone(),
        used_vram_mib: 0,
        total_vram_mib: total_vram,
        loaded_models: initial_models,
        nvlink_enabled: false,
        mig_enabled,
        mig_slices: initial_slices,
        model_weight_affinity: affinity,
    };

    let state = Arc::new(AgentState {
        payload: Mutex::new(initial_payload),
        scheduler_url,
        loader_url,
        http_client: reqwest::Client::new(),
    });

    let hb_state = Arc::clone(&state);
    tokio::spawn(async move { heartbeat_loop(hb_state, mig_node).await; });

    let app = Router::new()
        .route("/command", post(handle_command))
        .with_state(Arc::clone(&state));

    println!(
        "[agent] node={} gpu={} mig={} listening on {}",
        node_id, gpu_id, mig_enabled, listen_addr
    );
    let listener = tokio::net::TcpListener::bind(&listen_addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Returns (mig_enabled, initial_slices, initial_loaded_models, total_vram_mib).
fn build_initial_state(mig_node: bool) -> (bool, Vec<MIGSlice>, Vec<String>, u64) {
    if mig_node {
        let slices = MIG_MODELS.iter().map(|(sid, m)| MIGSlice {
            slice_id: sid.to_string(),
            profile: "2g.20gb".to_string(),
            total_vram_mib: 20_480,
            used_vram_mib: m.base_vram_mib,
            loaded_models: vec![m.name.to_string()],
            healthy: true,
        }).collect();
        let models = MIG_MODELS.iter().map(|(_, m)| m.name.to_string()).collect();
        (true, slices, models, 61_440) // 3 x 20,480 MiB
    } else {
        let models = PLAIN_MODELS.iter().map(|m| m.name.to_string()).collect();
        (false, vec![], models, 81_920) // full H100 80 GiB
    }
}

// ---------------------------------------------------------------------------
// Heartbeat loop
// ---------------------------------------------------------------------------

async fn heartbeat_loop(state: Arc<AgentState>, mig_node: bool) {
    let mut interval = time::interval(Duration::from_millis(500));
    loop {
        interval.tick().await;
// In mock mode we never call the Python loader — synthesize state directly.
        #[cfg(feature = "mock")]
        {
            let mut p = state.payload.lock().unwrap();
            if mig_node {
                p.mig_slices = drift_mig_slices(&p.mig_slices);
                p.used_vram_mib = p.mig_slices.iter().map(|s| s.used_vram_mib).sum();
                p.loaded_models = p.mig_slices.iter()
                    .flat_map(|s| s.loaded_models.clone())
                    .collect();

                // FIX: Collect the updates into a temporary vector first 
                // to free up the immutable borrow on `p.mig_slices`
                let mut affinity_updates = Vec::new();
                for s in &p.mig_slices {
                    for m in &s.loaded_models {
                        affinity_updates.push((m.clone(), s.used_vram_mib));
                    }
                }

                // Now it's perfectly safe to mutate `p`!
                for (model, vram) in affinity_updates {
                    p.model_weight_affinity.insert(model, vram);
                }
            } else {
                let (used, models) = drift_plain_models();
                p.used_vram_mib = used;
                p.loaded_models = models.iter().map(|(n, _)| n.to_string()).collect();
                for (name, vram) in &models {
                    p.model_weight_affinity.insert(name.to_string(), *vram);
                }
            }
        }

        // In production mode, read from the Python loader.
        #[cfg(not(feature = "mock"))]
        {
            let hw = sample_real();
            let loader_status = fetch_loader_status(&state).await;
            let mut p = state.payload.lock().unwrap();
            p.used_vram_mib = loader_status
                .as_ref()
                .map(|s| s.vram_used_mib)
                .unwrap_or(hw.used_vram_mib);
            p.nvlink_enabled = hw.nvlink_enabled;
            if let Some(status) = &loader_status {
                p.loaded_models = status.loaded.clone();
                for model in &status.loaded {
                    p.model_weight_affinity.insert(model.clone(), status.vram_used_mib);
                }
            }
        }

        let payload = state.payload.lock().unwrap().clone();
        if let Err(e) = state.http_client
            .post(&state.scheduler_url)
            .json(&payload)
            .send()
            .await
        {
            eprintln!("[agent] heartbeat POST failed: {e}");
        }
    }
}

async fn fetch_loader_status(state: &Arc<AgentState>) -> Option<LoaderStatus> {
    let url = format!("{}/status", state.loader_url);
    state.http_client.get(&url).send().await.ok()?.json::<LoaderStatus>().await.ok()
}

// ---------------------------------------------------------------------------
// Mock VRAM drift
//
// Instead of random noise across the full range, each model drifts ±DRIFT_PCT
// of its known base footprint. This keeps VRAM readings stable and meaningful:
// phi-3-mini stays around 3,800 MiB, llama-3-70b stays around 35,840 MiB.
// The ±5% swing represents KV-cache growth under active inference load.
// ---------------------------------------------------------------------------

#[cfg(feature = "mock")]
fn drift_vram(base: u64) -> u64 {
    let mut rng = rand::thread_rng();
    let delta = (base as f64 * DRIFT_PCT) as u64;
    let low  = base.saturating_sub(delta);
    let high = base + delta;
    rng.gen_range(low..=high)
}

/// Drift each MIG slice's VRAM usage independently.
/// loaded_models is preserved — models don't appear and disappear randomly.
#[cfg(feature = "mock")]
fn drift_mig_slices(current: &[MIGSlice]) -> Vec<MIGSlice> {
    MIG_MODELS.iter().zip(current.iter()).map(|((_, model_def), slice)| MIGSlice {
        used_vram_mib: drift_vram(model_def.base_vram_mib),
        ..slice.clone()
    }).collect()
}

/// Drift the plain-node models and return (total_used, [(name, vram)]).
#[cfg(feature = "mock")]
fn drift_plain_models() -> (u64, Vec<(&'static str, u64)>) {
    let models: Vec<(&str, u64)> = PLAIN_MODELS.iter()
        .map(|m| (m.name, drift_vram(m.base_vram_mib)))
        .collect();
    let total: u64 = models.iter().map(|(_, v)| v).sum();
    (total, models)
}

// ---------------------------------------------------------------------------
// Production GPU sampling (non-mock path)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "mock"))]
struct GpuSample { used_vram_mib: u64, nvlink_enabled: bool }

fn gpu_total_vram_mib() -> u64 {
    #[cfg(feature = "mock")] { 81_920 }
    #[cfg(not(feature = "mock"))] { query_nvidia_smi_total_vram() }
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
    let slice_info = cmd.slice_id.as_deref().unwrap_or("full");
    println!("[agent] command: action={} model={} slice={}", cmd.action, cmd.model_name, slice_info);

    let slice_vram_cap = cmd.slice_id.as_deref()
        .and_then(|sid| resolve_slice_vram_cap(&state, sid));

    match cmd.action.as_str() {
        "PREWARM" => match call_loader_load(
            &state,
            &cmd.model_name,
            cmd.quantization.as_deref(),
            cmd.slice_id.as_deref(),
            slice_vram_cap,
        ).await {
            Ok(resp) => Json(serde_json::json!({
                "status": "ok", "action": "PREWARM",
                "model": resp.model_name,
                "vram_used_mib": resp.vram_used_mib,
                "load_duration_sec": resp.load_duration_sec,
                "affinity_cache_hit": resp.affinity_cache_hit,
                "slice_id": slice_info,
            })),
            Err(e) => Json(serde_json::json!({ "status": "error", "detail": e })),
        },
        "EVICT" => match call_loader_evict(&state, &cmd.model_name).await {
            Ok(_) => Json(serde_json::json!({
                "status": "ok", "action": "EVICT",
                "model": cmd.model_name, "slice_id": slice_info,
            })),
            Err(e) => Json(serde_json::json!({ "status": "error", "detail": e })),
        },
        "CHECKPOINT" => match call_loader_checkpoint(&state, &cmd.model_name).await {
            Ok(path) => Json(serde_json::json!({
                "status": "ok", "action": "CHECKPOINT",
                "model": cmd.model_name, "path": path, "slice_id": slice_info,
            })),
            Err(e) => Json(serde_json::json!({ "status": "error", "detail": e })),
        },
        unknown => Json(serde_json::json!({
            "status": "error",
            "message": format!("unknown action: {unknown}")
        })),
    }
}

fn resolve_slice_vram_cap(state: &Arc<AgentState>, slice_id: &str) -> Option<u64> {
    let p = state.payload.lock().unwrap();
    p.mig_slices.iter()
        .find(|s| s.slice_id == slice_id)
        .map(|s| s.total_vram_mib)
}

// ---------------------------------------------------------------------------
// Loader HTTP calls (production path only; mock skips these)
// ---------------------------------------------------------------------------

async fn call_loader_load(
    state: &Arc<AgentState>,
    model_name: &str,
    quantization: Option<&str>,
    slice_id: Option<&str>,
    slice_vram_cap_mib: Option<u64>,
) -> Result<LoadResponse, String> {
    let resp = state.http_client
        .post(format!("{}/load", state.loader_url))
        .json(&LoadRequest {
            model_name: model_name.to_string(),
            repo_id: None,
            quantization: quantization.map(|s| s.to_string()),
            slice_id: slice_id.map(|s| s.to_string()),
            slice_vram_cap_mib,
        })
        .timeout(Duration::from_secs(300))
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
