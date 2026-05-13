use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::{postgres::PgPoolOptions, PgPool};
use uuid::Uuid;

use crate::error::EsseError;

pub struct PostgresStore {
    pool: PgPool,
}

impl PostgresStore {
    pub async fn new(db_url: &str) -> Result<Self, EsseError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(db_url)
            .await?;

        Ok(Self { pool })
    }

    pub async fn run_migrations(&self) -> Result<(), EsseError> {
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        Ok(())
    }

    #[cfg(feature = "sqlx-checked")]
    pub async fn insert_strategy_dna(
        &self,
        ast: &DummyAst,
        params: &DummyParams,
        fitness_score: Option<f64>,
        generation_born: Option<i32>,
    ) -> Result<Uuid, EsseError> {
        let ast = serialize_jsonb(ast)?;
        let params = serialize_jsonb(params)?;

        let row = sqlx::query!(
            r#"
            INSERT INTO strategy_dna (ast, params, fitness_score, generation_born)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            "#,
            ast,
            params,
            fitness_score,
            generation_born
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(row.id)
    }

    #[cfg(not(feature = "sqlx-checked"))]
    pub async fn insert_strategy_dna(
        &self,
        ast: &DummyAst,
        params: &DummyParams,
        fitness_score: Option<f64>,
        generation_born: Option<i32>,
    ) -> Result<Uuid, EsseError> {
        let ast = serialize_jsonb(ast)?;
        let params = serialize_jsonb(params)?;

        let id = sqlx::query_scalar::<_, Uuid>(
            r#"
            INSERT INTO strategy_dna (ast, params, fitness_score, generation_born)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            "#,
        )
        .bind(ast)
        .bind(params)
        .bind(fitness_score)
        .bind(generation_born)
        .fetch_one(&self.pool)
        .await?;

        Ok(id)
    }

    pub async fn insert_trade_logs_batch(
        &self,
        strategy_id: Uuid,
        trades: &[TradeLogInput],
    ) -> Result<(), EsseError> {
        let _ = (&self.pool, strategy_id, trades);

        todo!("batch trade logging must use UNNEST or bulk INSERT per RULE-PERF-05")
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DummyAst;

#[derive(Debug, Clone, Serialize)]
pub struct DummyParams;

#[derive(Debug, Clone)]
pub struct TradeLogInput {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub direction: String,
    pub gross_pips: f64,
    pub net_pips: f64,
    pub window_type: String,
}

fn serialize_jsonb<T: Serialize>(value: &T) -> Result<serde_json::Value, EsseError> {
    serde_json::to_value(value).map_err(|error| {
        EsseError::DataParseError(format!("failed to serialize JSONB payload: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postgres_store_compiles_as_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PostgresStore>();
    }

    #[test]
    fn dummy_strategy_payloads_serialize_to_json() -> Result<(), EsseError> {
        let ast = serialize_jsonb(&DummyAst)?;
        let params = serialize_jsonb(&DummyParams)?;

        assert_eq!(ast, serde_json::Value::Null);
        assert_eq!(params, serde_json::Value::Null);

        Ok(())
    }
}
