"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import (
    BrainCorrelationSummary,
    BrainDataInput,
)


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(
        data.csv_path,
        sep="\t",
        na_values="NA",
    )

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_count = len(men)
    women_count = len(women)

    features = [
        "FSIQ",
        "VIQ",
        "PIQ",
        "Weight",
        "Height",
    ]

    women_mri_correlation = {}

    for feature in features:
        correlation = women[feature].corr(
            women["MRI_Count"],
            method="pearson",
        )
        women_mri_correlation[feature] = float(correlation)

    men_mri_correlation = {}

    for feature in features:
        correlation = men[feature].corr(
            men["MRI_Count"],
            method="pearson",
        )
        men_mri_correlation[feature] = float(correlation)

    strongest_mri_feature = max(
        features,
        key=lambda feature: max(
            abs(women_mri_correlation[feature]),
            abs(men_mri_correlation[feature]),
        ),
    )

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )