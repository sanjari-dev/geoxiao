use axum::{routing::get, Router};
use tower_http::{cors::CorsLayer, trace::TraceLayer};

use crate::dashboard::{api, telemetry::SharedTelemetry};

pub async fn run_dashboard(telemetry: SharedTelemetry, port: u16) {
    let app = Router::new()
        .route("/api/system/telemetry", get(api::get_telemetry))
        .with_state(telemetry)
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let bind_address = format!("0.0.0.0:{port}");
    let listener = match tokio::net::TcpListener::bind(&bind_address).await {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, %port, "failed to bind dashboard listener");
            return;
        }
    };

    tracing::info!(%port, "dashboard server listening");

    if let Err(error) = axum::serve(listener, app).await {
        tracing::error!(%error, %port, "dashboard server stopped");
    }
}
