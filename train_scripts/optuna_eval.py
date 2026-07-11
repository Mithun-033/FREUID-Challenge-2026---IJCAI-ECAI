import os

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from optuna.importance import get_param_importances
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_rank,
    plot_slice,
)

# ==========================================================
# CHANGE THESE
# ==========================================================

STUDY_NAME="optuna_study"
STORAGE="sqlite:///optuna_study.db"

OUTPUT_DIR="optuna_analysis"

# ==========================================================

os.makedirs(OUTPUT_DIR,exist_ok=True)

study=optuna.load_study(
    study_name=STUDY_NAME,
    storage=STORAGE,
)

print("="*80)
print("STUDY SUMMARY")
print("="*80)

print(f"Study Name      : {study.study_name}")
print(f"Direction       : {study.direction}")
print(f"Trials          : {len(study.trials)}")
print()

best=study.best_trial

print("="*80)
print("BEST TRIAL")
print("="*80)

print(f"Trial Number : {best.number}")
print(f"Objective    : {best.value}")
print()

print("Parameters")
print("-"*40)
for k,v in best.params.items():
    print(f"{k:<25}{v}")

print()

print("User Attributes")
print("-"*40)
if len(best.user_attrs)==0:
    print("None")
else:
    for k,v in best.user_attrs.items():
        print(f"{k:<25}{v}")

print()

print("System Attributes")
print("-"*40)
if len(best.system_attrs)==0:
    print("None")
else:
    for k,v in best.system_attrs.items():
        print(f"{k:<25}{v}")

print()

print("Intermediate Values")
print("-"*40)
if len(best.intermediate_values)==0:
    print("None")
else:
    for step,val in best.intermediate_values.items():
        print(f"Epoch {step:<5} {val}")

print()

# ==========================================================
# DATAFRAME
# ==========================================================

df=study.trials_dataframe(attrs=(
    "number",
    "value",
    "state",
    "datetime_start",
    "datetime_complete",
    "duration",
    "params",
    "user_attrs",
))

df.to_csv(f"{OUTPUT_DIR}/all_trials.csv",index=False)

print("="*80)
print("TOP TRIALS")
print("="*80)

ascending=(study.direction.name=="MINIMIZE")

cols=["number","value"]

for c in df.columns:
    if c.startswith("user_attrs_"):
        cols.append(c)

display_df=df.sort_values("value",ascending=ascending)

print(display_df[cols].head(15))

# ==========================================================
# PARAM IMPORTANCE
# ==========================================================

print()
print("="*80)
print("PARAMETER IMPORTANCE")
print("="*80)

importance=get_param_importances(study)

for k,v in importance.items():
    print(f"{k:<25}{v:.4f}")

pd.DataFrame({
    "parameter":list(importance.keys()),
    "importance":list(importance.values())
}).to_csv(
    f"{OUTPUT_DIR}/parameter_importance.csv",
    index=False
)

# ==========================================================
# SAVE ALL PLOTS
# ==========================================================

plots={
    "optimization_history":plot_optimization_history(study),
    "parallel_coordinate":plot_parallel_coordinate(study),
    "slice":plot_slice(study),
    "contour":plot_contour(study),
    "edf":plot_edf(study),
    "param_importance":plot_param_importances(study),
    "rank":plot_rank(study),
}

try:
    plots["intermediate_values"]=plot_intermediate_values(study)
except:
    pass

for name,fig in plots.items():
    try:
        fig.write_image(f"{OUTPUT_DIR}/{name}.png")
    except Exception:
        pass

    try:
        fig.write_html(f"{OUTPUT_DIR}/{name}.html")
    except Exception:
        pass

# ==========================================================
# USER ATTRIBUTE TABLE
# ==========================================================

records=[]

for t in study.trials:

    row={
        "trial":t.number,
        "value":t.value,
        "state":str(t.state)
    }

    row.update(t.params)
    row.update(t.user_attrs)

    records.append(row)

user_df=pd.DataFrame(records)

user_df.to_csv(
    f"{OUTPUT_DIR}/trial_metrics.csv",
    index=False
)

# ==========================================================
# INTERMEDIATE VALUES TABLE
# ==========================================================

rows=[]

for t in study.trials:
    for step,val in t.intermediate_values.items():
        rows.append({
            "trial":t.number,
            "step":step,
            "value":val
        })

if len(rows)>0:
    pd.DataFrame(rows).to_csv(
        f"{OUTPUT_DIR}/intermediate_values.csv",
        index=False
    )

# ==========================================================
# CORRELATION
# ==========================================================

numeric=user_df.select_dtypes(include="number")

if len(numeric.columns)>1:
    corr=numeric.corr()
    corr.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")

    plt.figure(figsize=(10,8))
    plt.imshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns,rotation=90)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    plt.close()

print()
print("="*80)
print("Finished!")
print("="*80)
print(f"Outputs saved to: {OUTPUT_DIR}")