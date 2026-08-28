from pathlib import Path

import numpy as np
import pandas as pd

# This generator creates STEP UP-inspired fully synthetic data

# Main analysis output directory
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
SCENARIO_SCALES = {"constant_risk_difference": 0.0, "published_heterogeneity": 1.0}
# Primary scenarios in the comparative simulation study
# The published scenario supplies the multiple additive subgroup case
PRIMARY_ANALYSIS_SCENARIOS = (
    "constant_risk_difference",
    "simple_age_subgroup",
    "published_heterogeneity",
    "age_bmi_interaction",
    "continuous_heterogeneity"
)
SCENARIO_LABELS = {
    "constant_risk_difference": "Constant effect",
    "simple_age_subgroup": "Age subgroup",
    "published_heterogeneity": "Published additive",
    "age_bmi_interaction": "Age and BMI interaction",
    "continuous_heterogeneity": "Continuous"
}
METHOD_LABELS = {
    "classical_interaction": "Classical interaction",
    "bayesian_hierarchical": "Bayesian hierarchical",
    "causal_forest": "Causal forest"
}
CUSTOM_SCENARIO_COEFFICIENTS = {
    "simple_age_subgroup": -1.2,
    "age_bmi_interaction": -1.3,
    "continuous_heterogeneity": 0.75
}
CALIBRATION_N = 200_000
MONTE_CARLO_REPLICATIONS = 500

# Small settings for checking the complete analysis workflow
PILOT_REPLICATIONS = 10
PRIOR_SENSITIVITY_REPLICATIONS = 10
EVALUATION_N = 1_000

# Keep treated probabilities below one in the constant-risk-difference scenario
# (treated probability = placebo probability + constant risk difference)
PLACEBO_MAIN_EFFECT_SCALE = 0.6

# Published full-arm targets
TREATED_RESPONSE_TARGET = 862 / SEMAGLUTIDE_N
PLACEBO_RESPONSE_TARGET = 63 / PLACEBO_N
AVERAGE_CATE_TARGET = TREATED_RESPONSE_TARGET - PLACEBO_RESPONSE_TARGET

# Pre-truncation parameters calibrated to match the target mean and SD
AGE_GENERATING_MEAN = 46.831676
AGE_GENERATING_SD = 12.401212
BMI_GENERATING_MEAN = 33.814772
BMI_GENERATING_SD = 10.319237

# Height assumptions to calculate bodyweight from BMI
FEMALE_HEIGHT_MEAN = 1.64
FEMALE_HEIGHT_SD = 0.0675
MALE_HEIGHT_MEAN = 1.79
MALE_HEIGHT_SD = 0.0775
MINIMUM_HEIGHT = 1.40
MAXIMUM_HEIGHT = 2.10

# Columns available to an analyst
BASELINE_COLUMNS = [
    "age_years",
    "female",
    "bodyweight_kg",
    "bmi",
    "waist_circumference_cm",
    "hba1c_pct",
    "prediabetes"
]
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

def pooled_mean_sd(n_1, mean_1, sd_1, n_0, mean_0, sd_0):
    """Combine two groups' means and sample standard deviations"""
    # Weight each arm by its sample size
    total = n_1 + n_0
    mean = (n_1 * mean_1 + n_0 * mean_0) / total
    # Combine variation within and between the two arms
    sum_squares = (
        (n_1 - 1) * sd_1**2
        + (n_0 - 1) * sd_0**2
        + n_1 * (mean_1 - mean) ** 2
        + n_0 * (mean_0 - mean) ** 2
    )
    return mean, np.sqrt(sum_squares / (total - 1))

# Pooled BMI and waist targets used by the baseline generator
CONTINUOUS_TARGETS = pd.DataFrame.from_dict(
    {
        "bmi": pooled_mean_sd(SEMAGLUTIDE_N, 39.8, 7.0, PLACEBO_N, 39.7, 6.6),
        "waist_circumference_cm": pooled_mean_sd(
            SEMAGLUTIDE_N, 118.4, 15.8, PLACEBO_N, 118.6, 14.5
        )
    },
    orient="index",
    columns=["target_mean", "target_sd"]
)

# Glycaemic status was available for 1205 of the 1206 participants
FEMALE_PROBABILITY = (753 + 147) / TOTAL_N
PREDIABETES_PROBABILITY = (378 + 82) / 1205

# Waist-generation assumptions chosen to match the pooled mean and SD
# Higher BMI is associated with a larger waist circumference
# Men have larger waists than women at the same BMI
# total waist variance = BMI-related variance + sex-related variance + noise variance
WAIST_BMI_SLOPE = 1.4
WAIST_MALE_EFFECT = 6.0
WAIST_NOISE_SD = np.sqrt(
    CONTINUOUS_TARGETS.loc["waist_circumference_cm", "target_sd"] ** 2
    - (WAIST_BMI_SLOPE * CONTINUOUS_TARGETS.loc["bmi", "target_sd"]) ** 2
    - WAIST_MALE_EFFECT**2 * FEMALE_PROBABILITY * (1 - FEMALE_PROBABILITY)
)

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

def make_model_term(factor, level, placebo_log_or):
    """Compute the true interaction coefficients from the published odds ratios"""
    reference = REFERENCE_LEVELS[factor]
    interaction_log_or = np.log(
        SUBGROUP_OR_TARGETS[factor][level] / SUBGROUP_OR_TARGETS[factor][reference]
    )
    return {
        "factor": factor,
        "level": level,
        # Shrink the placebo subgroup effects so the constant-risk-difference
        # scenario cannot produce probabilities above one
        "placebo_log_or": PLACEBO_MAIN_EFFECT_SCALE * placebo_log_or,
        "interaction_log_or": interaction_log_or
    }

def inverse_logit(values):
    """Convert log odds to probabilities"""
    # Limit extreme values to prevent numerical overflow
    values = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))

def sample_truncated_normal(rng, mean, sd, lower, upper, size):
    """Draw normally distributed values inside fixed limits"""
    values = rng.normal(mean, sd, size)
    outside = (values < lower) | (values >= upper)
    # Redraw only values that fall outside the allowed range
    while outside.any():
        values[outside] = rng.normal(mean, sd, outside.sum())
        outside = (values < lower) | (values >= upper)
    return values

def sample_height(rng, female):
    """Generate plausible sex-specific heights in metres"""
    means = np.where(female == 1, FEMALE_HEIGHT_MEAN, MALE_HEIGHT_MEAN)
    sds = np.where(female == 1, FEMALE_HEIGHT_SD, MALE_HEIGHT_SD)
    heights = rng.normal(means, sds)
    outside = (heights < MINIMUM_HEIGHT) | (heights >= MAXIMUM_HEIGHT)
    # Redraw only heights outside the plausible range
    while outside.any():
        heights[outside] = rng.normal(means[outside], sds[outside])
        outside = (heights < MINIMUM_HEIGHT) | (heights >= MAXIMUM_HEIGHT)
    return heights

def add_subgroup_labels(data):
    """Derive subgroup labels from the analysis columns"""
    # Keep the original data unchanged
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
    # Derive bodyweight from BMI and a plausible unobserved height
    height = sample_height(random_generator, female)
    bodyweight = bmi * height**2

    # Relate waist to BMI and sex while retaining unexplained variation
    male = 1 - female
    waist = (
        CONTINUOUS_TARGETS.loc["waist_circumference_cm", "target_mean"]
        + WAIST_BMI_SLOPE * (bmi - CONTINUOUS_TARGETS.loc["bmi", "target_mean"])
        + WAIST_MALE_EFFECT * (male - (1 - FEMALE_PROBABILITY))
        + random_generator.normal(0, WAIST_NOISE_SD, n)
    )

    # Generate HbA1c within the limits of each glycaemic category
    hba1c = np.empty(n)
    normoglycaemia = prediabetes == 0
    hba1c[normoglycaemia] = sample_truncated_normal(
        random_generator, 5.48, 0.20, 4.5, 5.7, normoglycaemia.sum()
    )
    hba1c[~normoglycaemia] = sample_truncated_normal(
        random_generator, 5.98, 0.20, 5.7, 6.5, (~normoglycaemia).sum()
    )

    # Create the baseline DataFrame
    baseline = pd.DataFrame(
        {
            "participant_id": np.arange(1, n + 1),
            "age_years": age,
            "female": female,
            "bodyweight_kg": bodyweight,
            "bmi": bmi,
            "waist_circumference_cm": waist,
            "hba1c_pct": hba1c,
            "prediabetes": prediabetes
        }
    )
    return add_subgroup_labels(baseline)

# Create a large synthetic baseline population for calibrating the simulation
_calibration_data = simulate_baseline(
    CALIBRATION_N, create_random_generators()["calibration"]
)

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
    # Perform a binary search repeatedly
    for _ in range(80):
        midpoint = (lower + upper) / 2
        if inverse_logit(base_score + midpoint).mean() < target_probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2

# Calibrate the placebo and treatment models using the large fixed population

# Calculate the placebo subgroup score
_placebo_score = subgroup_score(_calibration_data, "placebo_log_or")

# Calibrate the placebo intercept to match the published placebo response rate
PLACEBO_INTERCEPT = calibrate_intercept(_placebo_score, PLACEBO_RESPONSE_TARGET)

# Calculate the treatment-interaction score
_interaction_score = subgroup_score(_calibration_data, "interaction_log_or")

# Construct the treatment base score
_published_treatment_base_score = (
    PLACEBO_INTERCEPT + _placebo_score + _interaction_score
)

# Calibrate the common treatment effect to match the published semaglutide response rate
PUBLISHED_TREATMENT_INTERCEPT = calibrate_intercept(
    _published_treatment_base_score, TREATED_RESPONSE_TARGET
)

def custom_scenario_score(data, scenario):
    """Calculate the selected scenario's treatment-effect modifier"""
    if scenario == "simple_age_subgroup":
        return data["age_years"].ge(65).astype(float).to_numpy()
    if scenario == "age_bmi_interaction":
        return (data["age_years"].ge(60) & data["bmi"].ge(40)).astype(float).to_numpy()
    if scenario == "continuous_heterogeneity":
        return 0.65 * np.tanh((data["bmi"].to_numpy() - 40) / 7) + 0.35 * np.tanh(
            (data["age_years"].to_numpy() - 50) / 15
        )
    raise ValueError(f"Unknown custom scenario {scenario}")

# Calculate the continuous modifier score in the calibration population
_continuous_calibration_score = custom_scenario_score(
    _calibration_data, "continuous_heterogeneity"
)

# Store the thresholds for the low- and high-modifier groups
CONTINUOUS_MODIFIER_QUARTILES = tuple(
    np.quantile(_continuous_calibration_score, [0.25, 0.75])
)

def subgroup_comparison(data, scenario):
    """Define the prespecified subgroup comparison for each scenario"""
    if scenario in {
        "constant_risk_difference",
        "simple_age_subgroup",
        "published_heterogeneity"
    }:
        group_one = data["age_years"].ge(65).to_numpy()
        label = "age 65 or older minus under 65"
    elif scenario == "age_bmi_interaction":
        group_one = (data["age_years"].ge(60) & data["bmi"].ge(40)).to_numpy()
        label = "age 60 or older and BMI 40 or higher minus all others"
    elif scenario == "continuous_heterogeneity":
        score = custom_scenario_score(data, scenario)
        lower, upper = CONTINUOUS_MODIFIER_QUARTILES
        group_one = score >= upper
        group_zero = score <= lower
        return {
            "label": "upper minus lower modifier quartile",
            "group_one": group_one,
            "group_zero": group_zero
        }
    else:
        raise ValueError(f"Unknown primary scenario {scenario}")

    return {"label": label, "group_one": group_one, "group_zero": ~group_one}

# Calculate placebo log odds for every participant in the calibration population
_calibration_placebo_log_odds = PLACEBO_INTERCEPT + _placebo_score

# Calibrate a separate common treatment effect for each custom scenario
# Keep the heterogeneity pattern fixed while matching the published
# population-average semaglutide response
CUSTOM_TREATMENT_INTERCEPTS = {}

for _scenario, _coefficient in CUSTOM_SCENARIO_COEFFICIENTS.items():
    # Construct treatment log odds before adding the common treatment effect
    _custom_base_score = (
        _calibration_placebo_log_odds
        + _coefficient * custom_scenario_score(_calibration_data, _scenario)
    )
    # Find the common log-odds shift required to match the treatment target
    CUSTOM_TREATMENT_INTERCEPTS[_scenario] = calibrate_intercept(
        _custom_base_score, TREATED_RESPONSE_TARGET
    )

def calculate_outcome_probabilities(data, scenario):
    """Calculate both potential outcome probabilities"""
    known_scenarios = set(SCENARIO_SCALES) | set(CUSTOM_SCENARIO_COEFFICIENTS)
    if scenario not in known_scenarios:
        raise ValueError(f"Unknown scenario {scenario}")

    # Calculate the placebo probability
    placebo_log_odds = PLACEBO_INTERCEPT + subgroup_score(data, "placebo_log_or")
    placebo_probability = inverse_logit(placebo_log_odds)

    if scenario in CUSTOM_SCENARIO_COEFFICIENTS:
        # Calculate treatment log-odds
        treatment_log_odds = (
            placebo_log_odds
            + CUSTOM_TREATMENT_INTERCEPTS[scenario]
            + CUSTOM_SCENARIO_COEFFICIENTS[scenario]
            * custom_scenario_score(data, scenario)
        )
        # Calculate the treatment probability
        treated_probability = inverse_logit(treatment_log_odds)
        return placebo_probability, treated_probability

    # Use the published subgroup OR pattern to define the full scenario
    published_treatment_log_odds = (
        placebo_log_odds
        + PUBLISHED_TREATMENT_INTERCEPT
        + subgroup_score(data, "interaction_log_or")
    )
    published_treatment_probability = inverse_logit(published_treatment_log_odds)

    # Shrink individual risk differences towards a common risk difference

    # Calculate the published individual treatment effect (CATE)
    published_cate = published_treatment_probability - placebo_probability

    # Change the amount of heterogeneity in the treatment effect
    scale = SCENARIO_SCALES[scenario]
    scenario_cate = AVERAGE_CATE_TARGET + scale * (published_cate - AVERAGE_CATE_TARGET)

    # Calculate the final treatment probability
    treated_probability = placebo_probability + scenario_cate

    # Check that all probabilities are valid (between zero and one)
    if ((treated_probability < 0) | (treated_probability > 1)).any():
        raise ValueError("Treatment probabilities must be between zero and one")
    return placebo_probability, treated_probability

def simulate_trial(baseline, random_generator, scenario):
    """Randomise treatment and outcomes using the supplied random generator"""
    data = baseline.copy()

    # Calculate the number assigned to treatment
    treated_n = round(TREATMENT_PROBABILITY * len(data))
    # Randomly assign treatment to the requested number of participants
    treated = np.r_[
        np.ones(treated_n, dtype=int), np.zeros(len(data) - treated_n, dtype=int)
    ]
    random_generator.shuffle(treated)

    # Calculate both potential outcome probabilities
    placebo_probability, treated_probability = calculate_outcome_probabilities(
        data, scenario
    )

    # Select the response probability for the assigned treatment arm
    observed_probability = np.where(
        treated == 1, treated_probability, placebo_probability
    )

    # Generate the observed binary outcome
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
