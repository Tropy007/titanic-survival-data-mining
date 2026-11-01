"""Export the small, pre-aggregated tables behind the Tableau and Power BI dashboards.

Run from the repository root:  python dashboard/export_dashboard_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "data"
OUT.mkdir(parents=True, exist_ok=True)
def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(name); print(df.to_string(index=False)); print()
t = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
cls = t.groupby(["Pclass", "Sex"])["Survived"].mean().round(4).reset_index()
cls["Passenger class"] = cls["Pclass"].map({1: "1st class", 2: "2nd class", 3: "3rd class"}); cls["Sex"] = cls["Sex"].str.capitalize()
save(cls[["Passenger class", "Sex", "Survived"]].rename(columns={"Survived": "Survival rate"}), "titanic_class_sex.csv")
port = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
pr = t.dropna(subset=["Embarked"]).groupby("Embarked")["Survived"].mean().round(4).reset_index()
pr["Port"] = pr["Embarked"].map(port)
save(pr[["Port", "Survived"]].rename(columns={"Survived": "Survival rate"}), "titanic_port.csv")
mix = pd.crosstab(t["Embarked"].map(port), t["Pclass"], normalize="index").round(4).stack().reset_index()
mix.columns = ["Port", "Pclass", "Share of passengers"]; mix["Passenger class"] = mix["Pclass"].map({1: "1st class", 2: "2nd class", 3: "3rd class"})
save(mix[["Port", "Passenger class", "Share of passengers"]], "titanic_port_class_mix.csv")
ab = pd.cut(t["Age"], [0, 12, 18, 35, 50, 80], labels=["1. Child (0-12)", "2. Teen (13-18)", "3. 19-35", "4. 36-50", "5. 51+"])
save(t.groupby(ab, observed=True)["Survived"].mean().round(4).reset_index().rename(columns={"Age": "Age group", "Survived": "Survival rate"}), "titanic_age.csv")
save(pd.DataFrame({"metric": ["Passengers", "Survived", "Women survived", "Men survived", "Random forest accuracy", "Top rule lift"],
                   "value": [f"{len(t)}", f"{t.Survived.mean():.1%}", f"{t[t.Sex == 'female'].Survived.mean():.1%}", f"{t[t.Sex == 'male'].Survived.mean():.1%}",
                             "83.9%", "2.54"]}), "titanic_kpis.csv")  # model results from the notebook
