import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput

from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations


print("=== TITANIC ===")

titanic_result = analyze_titanic(
    TitanicInput(csv_path="titanic.csv")
)

print("row_count:")
print(titanic_result.row_count)

print("\nmissing_by_column:")
print(titanic_result.missing_by_column)

print("\nadults_over_30_count:")
print(titanic_result.adults_over_30_count)

print("\nmean_age_by_pclass:")
print(titanic_result.mean_age_by_pclass)

print("\nsurvival_rate_by_pclass:")
print(titanic_result.survival_rate_by_pclass)

print("\nhighest_fares:")
print(titanic_result.highest_fares)


print("\n=== BRAIN CORRELATIONS ===")

brain_result = analyze_brain_correlations(
    BrainDataInput(csv_path="brainsize.txt")
)

print("men_count:")
print(brain_result.men_count)

print("\nwomen_count:")
print(brain_result.women_count)

print("\nwomen_mri_correlation:")
print(brain_result.women_mri_correlation)

print("\nmen_mri_correlation:")
print(brain_result.men_mri_correlation)

print("\nstrongest_mri_feature:")
print(brain_result.strongest_mri_feature)