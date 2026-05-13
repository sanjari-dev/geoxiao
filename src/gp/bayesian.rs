use std::env;

use pyo3::{
    prelude::*,
    types::{PyDict, PyModule},
};

use crate::{error::EsseError, gp::population::StrategyParams};

pub struct OptunaOptimizer {
    study_name: String,
}

impl OptunaOptimizer {
    pub fn new(study_name: impl Into<String>) -> Self {
        Self {
            study_name: study_name.into(),
        }
    }

    pub fn suggest_params(
        &self,
        trial_number: u32,
        sl_pips_range: (f64, f64),
    ) -> Result<StrategyParams, EsseError> {
        Python::with_gil(|py| -> PyResult<StrategyParams> {
            append_python_bridge_path(py)?;

            let optuna_bridge = PyModule::import_bound(py, "optuna_bridge")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("study_name", &self.study_name)?;
            kwargs.set_item("trial_number", trial_number)?;
            kwargs.set_item("sl_min", sl_pips_range.0)?;
            kwargs.set_item("sl_max", sl_pips_range.1)?;

            let result = optuna_bridge
                .getattr("suggest_params")?
                .call((), Some(&kwargs))?;

            let sl_pips: f64 = result.get_item("sl_pips")?.extract()?;
            let tp_pips: f64 = result.get_item("tp_pips")?.extract()?;
            let signal_threshold: f64 = result.get_item("signal_threshold")?.extract()?;
            let feature_window: usize = result.get_item("feature_window")?.extract()?;

            Ok(StrategyParams {
                sl_pips,
                tp_pips,
                signal_threshold,
                feature_window,
            })
        })
        .map_err(EsseError::from)
    }

    pub fn report_result(&self, trial_number: u32, score: f64) -> Result<(), EsseError> {
        Python::with_gil(|py| -> PyResult<()> {
            append_python_bridge_path(py)?;

            let optuna_bridge = PyModule::import_bound(py, "optuna_bridge")?;
            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("study_name", &self.study_name)?;
            kwargs.set_item("trial_number", trial_number)?;
            kwargs.set_item("value", score)?;

            optuna_bridge
                .getattr("report_result")?
                .call((), Some(&kwargs))?;

            Ok(())
        })
        .map_err(EsseError::from)
    }
}

fn append_python_bridge_path(py: Python<'_>) -> PyResult<()> {
    let python_dir = env::current_dir()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?
        .join("python");
    let python_dir = python_dir.to_string_lossy();

    let sys = PyModule::import_bound(py, "sys")?;
    let path = sys.getattr("path")?;

    if !path
        .call_method1("__contains__", (python_dir.as_ref(),))?
        .extract::<bool>()?
    {
        path.call_method1("append", (python_dir.as_ref(),))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Python with optuna installed"]
    fn suggests_params_from_optuna_bridge() -> Result<(), EsseError> {
        let optimizer = OptunaOptimizer::new("esse_test_study");

        for trial_number in 0..3 {
            let params = optimizer.suggest_params(trial_number, (10.0, 50.0))?;
            assert!((10.0..=50.0).contains(&params.sl_pips));
            assert!(params.tp_pips >= params.sl_pips * 1.5);
            assert!((0.05..=0.95).contains(&params.signal_threshold));
            assert!((3..=300).contains(&params.feature_window));
            optimizer.report_result(trial_number, 1.0 + trial_number as f64)?;
        }

        Ok(())
    }
}
