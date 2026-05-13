use axum::{extract::State, Json};

use crate::dashboard::telemetry::{SharedTelemetry, TelemetryState};

pub async fn get_telemetry(State(telemetry): State<SharedTelemetry>) -> Json<TelemetryState> {
    let snapshot = { telemetry.read().clone() };
    Json(snapshot)
}
