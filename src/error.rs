use thiserror::Error;

pub type EsseResult<T> = Result<T, EsseError>;

#[derive(Debug, Error)]
pub enum EsseError {
    #[error("database error: {0}")]
    DatabaseError(#[source] DatabaseError),

    #[error("configuration error: {0}")]
    ConfigError(#[from] config::ConfigError),

    #[error("data parse error: {0}")]
    DataParseError(String),

    #[error("python ffi error: {0}")]
    PythonError(String),

    #[error("runtime error: {0}")]
    RuntimeError(String),

    #[error("gpu error: {0}")]
    GpuError(String),
}

#[derive(Debug, Error)]
pub enum DatabaseError {
    #[error("clickhouse error: {0}")]
    ClickHouse(#[from] clickhouse::error::Error),

    #[error("sqlx error: {0}")]
    Sqlx(#[from] sqlx::Error),

    #[error("sqlx migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),
}

impl From<DatabaseError> for EsseError {
    fn from(error: DatabaseError) -> Self {
        Self::DatabaseError(error)
    }
}

impl From<clickhouse::error::Error> for EsseError {
    fn from(error: clickhouse::error::Error) -> Self {
        Self::DatabaseError(DatabaseError::ClickHouse(error))
    }
}

impl From<sqlx::Error> for EsseError {
    fn from(error: sqlx::Error) -> Self {
        Self::DatabaseError(DatabaseError::Sqlx(error))
    }
}

impl From<sqlx::migrate::MigrateError> for EsseError {
    fn from(error: sqlx::migrate::MigrateError) -> Self {
        Self::DatabaseError(DatabaseError::Migration(error))
    }
}

impl From<pyo3::PyErr> for EsseError {
    fn from(error: pyo3::PyErr) -> Self {
        Self::PythonError(error.to_string())
    }
}

impl From<tokio::task::JoinError> for EsseError {
    fn from(error: tokio::task::JoinError) -> Self {
        Self::RuntimeError(error.to_string())
    }
}

#[cfg(feature = "gpu-cuda")]
impl From<cudarc::driver::DriverError> for EsseError {
    fn from(error: cudarc::driver::DriverError) -> Self {
        Self::GpuError(error.to_string())
    }
}
