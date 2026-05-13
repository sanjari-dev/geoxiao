use std::sync::Arc;

use parking_lot::RwLock;

use esse::{
    config::Config,
    dashboard::{
        server,
        telemetry::{SharedTelemetry, TelemetryState},
    },
    error::EsseError,
    evolution::orchestrator,
};

#[tokio::main]
async fn main() -> Result<(), EsseError> {
    tracing_subscriber::fmt()
        .with_env_filter("esse=debug,tower_http=info")
        .init();

    let config = Config::from_default_file()?;
    let telemetry: SharedTelemetry = Arc::new(RwLock::new(TelemetryState::default()));

    let telemetry_dashboard = telemetry.clone();
    let dashboard_port = config.dashboard_port;

    tokio::spawn(async move {
        server::run_dashboard(telemetry_dashboard, dashboard_port).await;
    });

    orchestrator::run_evolution_loop(telemetry).await?;

    Ok(())
}
