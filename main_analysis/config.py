from pathlib import Path

import numpy as np
import pandas as pd

# This generator creates STEP UP-inspired fully synthetic data

CONFIG_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = CONFIG_DIRECTORY / "results"

# Random seed for reproducibility
RANDOM_SEED = 42

# Trial sizes
TOTAL_N = 1206
SEMAGLUTIDE_N = 1005
PLACEBO_N = 201
TREATMENT_PROBABILITY = SEMAGLUTIDE_N / TOTAL_N

# Simulation settings
PRIMARY_ANALYSIS_SCENARIOS = (
    "no_hte",
    "joint_subgroup",
    "published_scenario",
    "continuous"
)
SCENARIO_LABELS = {
    "no_hte": "No HTE",
    "joint_subgroup": "Joint subgroup",
    "published_scenario": "Published scenario",
    "continuous": "Continuous"
}
METHOD_LABELS = {
    "classical_interaction": "Classical interaction",
    "bayesian_hierarchical": "Bayesian hierarchical",
    "causal_forest": "Causal forest"
}
CALIBRATION_N = 200_000
MONTE_CARLO_REPLICATIONS = 500

# Small settings for checking the complete analysis workflow
PILOT_REPLICATIONS = 10
PRIOR_SENSITIVITY_REPLICATIONS = 10
EVALUATION_N = 1_000

# Keep treated probabilities below one in the No HTE scenario
# (treated probability = placebo probability + constant risk difference)
PLACEBO_SUBGROUP_EFFECT_SCALE = 0.6

# Published full arm targets
TREATED_RESPONSE_TARGET = 862 / SEMAGLUTIDE_N
PLACEBO_RESPONSE_TARGET = 63 / PLACEBO_N
ATE_TARGET = TREATED_RESPONSE_TARGET - PLACEBO_RESPONSE_TARGET

# Pre-truncation parameters calibrated to match the target mean and SD
AGE_GENERATING_MEAN = 46.831676
AGE_GENERATING_SD = 12.401212
BMI_GENERATING_MEAN = 33.814772
BMI_GENERATING_SD = 10.319237

# Columns available to an analyst
BASELINE_COLUMNS = ["age_years", "female", "bmi", "prediabetes"]
ANALYSIS_COLUMNS = ["participant_id", *BASELINE_COLUMNS, "treated", "weight_loss_5pct"]

def create_random_generators():
    """Create independent random streams from the single analysis seed"""
    stream_names = [
        "calibration",
        "evaluation",
        "main_trials",
        "main_models",
        "prior_trials",
        "prior_models",
        # Tuning trials and fits must not reuse the final 500-replication streams
        "tuning_trials",
        "tuning_models"
    ]
    seed_sequences = np.random.SeedSequence(RANDOM_SEED).spawn(len(stream_names))
    return {
        name: np.random.default_rng(seed_sequence)
        for name, seed_sequence in zip(stream_names, seed_sequences)
    }

# Glycaemic status was available for 1205 of the 1206 participants
FEMALE_PROBABILITY = (753 + 147) / TOTAL_N
PREDIABETES_PROBABILITY = (378 + 82) / 1205

# Published subgroup odds ratios from Supplementary Table 3
SUBGROUP_OR_TARGETS = {
    "sex": {"Female": 11.4, "Male": 14.4},
    "age_group": {"Under 65": 12.9, "65 or older": 4.6},
    "bmi_group": {"30 to under 35": 11.4, "35 to under 40": 15.7, "40 or higher": 10.7},
    "glycaemic_status": {"Normoglycaemia": 14.7, "Prediabetes": 9.1}
}

# Reference categories have no indicator in the outcome model
REFERENCE_LEVELS = {
    "sex": "Male",
    "age_group": "Under 65",
    "bmi_group": "30 to under 35",
    "glycaemic_status": "Normoglycaemia"
}

def make_model_term(factor, level, placebo_subgroup_log_or):
    """Compute the true interaction coefficients from the published odds ratios"""
    reference = REFERENCE_LEVELS[factor]
    interaction_log_or = np.log(
        SUBGROUP_OR_TARGETS[factor][level] / SUBGROUP_OR_TARGETS[factor][reference]
    )
    return {
        "factor": factor,
        "level": level,
        # Shrink placebo subgroup effects to keep No HTE probabilities valid
        "placebo_log_or": PLACEBO_SUBGROUP_EFFECT_SCALE * placebo_subgroup_log_or,
        "interaction_log_or": interaction_log_or
    }

def inverse_logit(values):
    """Convert log odds to probabilities"""
    # Limit extreme values to prevent numerical overflow
    values = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))

def sample_truncated_normal(random_generator, mean, sd, lower, upper, size):
    """Draw normally distributed values inside fixed limits"""
    values = random_generator.normal(mean, sd, size)
    outside = (values < lower) | (values >= upper)
    # Redraw only values that fall outside the allowed range
    while outside.any():
        values[outside] = random_generator.normal(mean, sd, outside.sum())
        outside = (values < lower) | (values >= upper)
    return values

def add_subgroup_labels(data):
    """Derive subgroup labels from the analysis columns"""
    labelled = data.copy()
    labelled["sex"] = np.where(labelled["female"] == 1, "Female", "Male")
    labelled["age_group"] = np.where(
        labelled["age_years"] < 65, "Under 65", "65 or older"
    )
    labelled["bmi_group"] = np.select(
        [labelled["bmi"] < 35, labelled["bmi"] < 40],
        ["30 to under 35", "35 to under 40"],
        default="40 or higher"
    )
    labelled["glycaemic_status"] = np.where(
        labelled["prediabetes"] == 1, "Prediabetes", "Normoglycaemia"
    )
    return labelled

def simulate_baseline(n, random_generator):
    """Generate baseline covariates using the supplied random generator"""
    # Match the published pooled category proportions
    female = random_generator.binomial(1, FEMALE_PROBABILITY, n)
    prediabetes = random_generator.binomial(1, PREDIABETES_PROBABILITY, n)

    # Apply the trial eligibility limits to age and BMI
    age = sample_truncated_normal(
        random_generator, AGE_GENERATING_MEAN, AGE_GENERATING_SD, 18, np.inf, n
    )
    bmi = sample_truncated_normal(
        random_generator, BMI_GENERATING_MEAN, BMI_GENERATING_SD, 30, np.inf, n
    )

    baseline = pd.DataFrame(
        {
            "participant_id": np.arange(1, n + 1),
            "age_years": age,
            "female": female,
            "bmi": bmi,
            "prediabetes": prediabetes
        }
    )
    return add_subgroup_labels(baseline)

# Create a large synthetic baseline population for calibrating the simulation
_calibration_data = simulate_baseline(
    CALIBRATION_N, create_random_generators()["calibration"]
)

# Fixed standardisation constants for the continuous age scenario
CALIBRATION_AGE_MEAN = _calibration_data["age_years"].mean()
CALIBRATION_AGE_SD = _calibration_data["age_years"].std(ddof=0)

# Strong prespecified synthetic joint effect scenario
JOINT_SUBGROUP_COEFFICIENT = 2.0
# Prespecified synthetic decrease in effectiveness per age SD
CONTINUOUS_AGE_COEFFICIENT = -0.5

def estimated_placebo_total(factor, level):
    """Estimate the number of placebo participants in a subgroup"""
    proportion = _calibration_data[factor].eq(level).mean()
    return PLACEBO_N * proportion

def placebo_log_or(responders, total, reference_responders, reference_total):
    """Calculate a placebo subgroup log odds ratio"""
    odds = responders / (total - responders)
    reference_odds = reference_responders / (reference_total - reference_responders)
    return np.log(odds / reference_odds)

# Compute the main and treatment interaction coefficients for each covariate
# Sex and glycaemic totals were reported directly
# Age and BMI totals use the simulated baseline proportions
MODEL_TERMS = [
    make_model_term("sex", "Female", placebo_log_or(50, 147, 13, 54)),
    make_model_term(
        "age_group",
        "65 or older",
        placebo_log_or(
            6,
            estimated_placebo_total("age_group", "65 or older"),
            57,
            estimated_placebo_total("age_group", "Under 65")
        )
    ),
    make_model_term(
        "bmi_group",
        "35 to under 40",
        placebo_log_or(
            19,
            estimated_placebo_total("bmi_group", "35 to under 40"),
            16,
            estimated_placebo_total("bmi_group", "30 to under 35")
        )
    ),
    make_model_term(
        "bmi_group",
        "40 or higher",
        placebo_log_or(
            28,
            estimated_placebo_total("bmi_group", "40 or higher"),
            16,
            estimated_placebo_total("bmi_group", "30 to under 35")
        )
    ),
    make_model_term("glycaemic_status", "Prediabetes", placebo_log_or(30, 82, 33, 118))
]

def subgroup_score(data, coefficient_name):
    """Construct subgroup contributions to the true logistic outcome model"""
    score = np.zeros(len(data))
    for term in MODEL_TERMS:
        # Reference categories keep a contribution of zero
        belongs_to_level = data[term["factor"]] == term["level"]
        score += belongs_to_level * term[coefficient_name]
    return score

def calibrate_intercept(base_score, target_probability):
    """Find an intercept that gives the requested average probability"""
    lower = -10.0
    upper = 10.0
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if inverse_logit(base_score + midpoint).mean() < target_probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2

# Calibrate the placebo model using the large fixed population
_placebo_score = subgroup_score(_calibration_data, "placebo_log_or")
PLACEBO_INTERCEPT = calibrate_intercept(_placebo_score, PLACEBO_RESPONSE_TARGET)

def joint_subgroup(data):
    """Identify participants in the joint BMI and prediabetes subgroup"""
    return (data["bmi"].ge(40) & data["prediabetes"].eq(1)).to_numpy()

def standardised_age(data):
    """Standardise age using the fixed calibration population"""
    return (data["age_years"].to_numpy() - CALIBRATION_AGE_MEAN) / CALIBRATION_AGE_SD

def define_subgroups(data, scenario):
    """Define the prespecified subgroup difference for each scenario"""
    if scenario in {"no_hte", "published_scenario"}:
        group_one = data["age_years"].ge(65).to_numpy()
        label = "age 65 or older minus under 65"
    elif scenario == "joint_subgroup":
        group_one = joint_subgroup(data)
        label = "BMI 40 or higher and prediabetes minus all others"
    elif scenario == "continuous":
        raise ValueError("Continuous subgroups use fixed evaluation age quartiles")
    else:
        raise ValueError(f"Unknown primary scenario {scenario}")
    return {"label": label, "group_one": group_one, "group_zero": ~group_one}

_calibration_placebo_log_odds = PLACEBO_INTERCEPT + _placebo_score

# Calibrate the joint subgroup treatment intercept
_joint_treatment_base_score = (
    _calibration_placebo_log_odds
    + JOINT_SUBGROUP_COEFFICIENT * joint_subgroup(_calibration_data)
)
JOINT_TREATMENT_INTERCEPT = calibrate_intercept(
    _joint_treatment_base_score, TREATED_RESPONSE_TARGET
)

# Calibrate the published additive treatment intercept
_interaction_score = subgroup_score(_calibration_data, "interaction_log_or")
_published_treatment_base_score = (
    PLACEBO_INTERCEPT + _placebo_score + _interaction_score
)
PUBLISHED_TREATMENT_INTERCEPT = calibrate_intercept(
    _published_treatment_base_score, TREATED_RESPONSE_TARGET
)

# Calibrate the continuous age treatment intercept
_continuous_treatment_base_score = (
    _calibration_placebo_log_odds
    + CONTINUOUS_AGE_COEFFICIENT * standardised_age(_calibration_data)
)
CONTINUOUS_TREATMENT_INTERCEPT = calibrate_intercept(
    _continuous_treatment_base_score, TREATED_RESPONSE_TARGET
)

def calculate_outcome_probabilities(data, scenario):
    """Calculate both potential outcome probabilities"""
    if scenario not in PRIMARY_ANALYSIS_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario}")

    placebo_log_odds = PLACEBO_INTERCEPT + subgroup_score(data, "placebo_log_or")
    placebo_probability = inverse_logit(placebo_log_odds)

    if scenario == "no_hte":
        treated_probability = placebo_probability + ATE_TARGET
    elif scenario == "joint_subgroup":
        treatment_log_odds = (
            placebo_log_odds
            + JOINT_TREATMENT_INTERCEPT
            + JOINT_SUBGROUP_COEFFICIENT * joint_subgroup(data)
        )
        treated_probability = inverse_logit(treatment_log_odds)
    elif scenario == "published_scenario":
        treatment_log_odds = (
            placebo_log_odds
            + PUBLISHED_TREATMENT_INTERCEPT
            + subgroup_score(data, "interaction_log_or")
        )
        treated_probability = inverse_logit(treatment_log_odds)
    else:
        treatment_log_odds = (
            placebo_log_odds
            + CONTINUOUS_TREATMENT_INTERCEPT
            + CONTINUOUS_AGE_COEFFICIENT * standardised_age(data)
        )
        treated_probability = inverse_logit(treatment_log_odds)

    # Check that all probabilities are valid (between zero and one)
    if ((treated_probability < 0) | (treated_probability > 1)).any():
        raise ValueError("Treatment probabilities must be between zero and one")
    return placebo_probability, treated_probability

def simulate_trial(baseline, random_generator, scenario):
    """Randomise treatment and outcomes using the supplied random generator"""
    data = baseline.copy()

    treated_n = round(TREATMENT_PROBABILITY * len(data))
    # Randomly assign treatment to the requested number of participants
    treated = np.r_[
        np.ones(treated_n, dtype=int), np.zeros(len(data) - treated_n, dtype=int)
    ]
    random_generator.shuffle(treated)

    placebo_probability, treated_probability = calculate_outcome_probabilities(
        data, scenario
    )

    # Select the response probability for the assigned treatment arm
    observed_probability = np.where(
        treated == 1, treated_probability, placebo_probability
    )

    data["treated"] = treated
    data["weight_loss_5pct"] = random_generator.binomial(1, observed_probability)

    # Store each participant's true probabilities and CATE
    truth = pd.DataFrame(
        {
            "participant_id": data["participant_id"],
            "scenario": scenario,
            "probability_placebo": placebo_probability,
            "probability_semaglutide": treated_probability,
            "true_cate_risk_difference": treated_probability - placebo_probability
        }
    )
    return data, truth
