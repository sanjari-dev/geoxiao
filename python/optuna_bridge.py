import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)

_studies: dict[str, optuna.Study] = {}


def get_study(study_name: str) -> optuna.Study:
    if study_name not in _studies:
        _studies[study_name] = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
    return _studies[study_name]


def suggest_params(
    study_name: str,
    trial_number: int,
    sl_min: float = 10.0,
    sl_max: float = 50.0,
) -> dict:
    study = get_study(study_name)
    trial = study.ask()
    sl_pips = trial.suggest_float("sl_pips", sl_min, sl_max)
    tp_pips = trial.suggest_float("tp_pips", sl_pips * 1.5, 150.0)
    signal_threshold = trial.suggest_float("signal_threshold", 0.05, 0.95)
    feature_window = trial.suggest_int("feature_window", 3, 300)
    return {
        "sl_pips": sl_pips,
        "tp_pips": tp_pips,
        "signal_threshold": signal_threshold,
        "feature_window": feature_window,
        "_trial_id": trial.number,
    }


def report_result(study_name: str, trial_number: int, value: float):
    study = get_study(study_name)
    study.tell(trial_number, value)
