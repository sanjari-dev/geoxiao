use serde::Deserialize;

use crate::error::EsseResult;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub clickhouse_url: String,
    pub postgres_url: String,
    pub dashboard_port: u16,
}

impl Config {
    pub fn from_file(path: impl AsRef<str>) -> EsseResult<Self> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path.as_ref()))
            .build()?;

        Ok(settings.try_deserialize()?)
    }

    pub fn from_default_file() -> EsseResult<Self> {
        Self::from_file("config/default")
    }
}
