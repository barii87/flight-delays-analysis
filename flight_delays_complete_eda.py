import os, re, gc, glob, time, math, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import missingno as msno

from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import plotly.express as px
import plotly.graph_objects as go

from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay,
                             PrecisionRecallDisplay, average_precision_score, f1_score, silhouette_score,
                             mean_absolute_error, r2_score)

warnings.filterwarnings("ignore")

SEED = 42
N_ROWS = 4_000_000
rng = np.random.default_rng(SEED)

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.3f}".format)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.figsize": (12, 5), "figure.dpi": 100, "axes.titlesize": 13, "axes.titleweight": "bold"})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def samp(frame, n):
    return frame.sample(min(n, len(frame)), random_state=SEED)

def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), bbox_inches="tight", dpi=110)

for lib in (np, pd, sns, plt.matplotlib, stats, sm):
    print(f"{getattr(lib, '__name__', ''):<22} {getattr(lib, '__version__', '')}")

def find_dir():
    env = os.environ.get("FLIGHTS_DIR")
    if env and os.path.exists(os.path.join(env, "flights.csv")):
        return env
    hits = glob.glob("/kaggle/input/**/flights.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    import kagglehub
    folder = kagglehub.dataset_download("usdot/flight-delays")
    return os.path.dirname(glob.glob(os.path.join(folder, "**", "flights.csv"), recursive=True)[0])

DATA_DIR = find_dir()
code_types = {"ORIGIN_AIRPORT": str, "DESTINATION_AIRPORT": str, "TAIL_NUMBER": str, "CANCELLATION_REASON": str}
flights_path = os.path.join(DATA_DIR, "flights.csv")
print("Reading:", flights_path, f"({os.path.getsize(flights_path)/1e6:,.0f} MB)")

t0 = time.time()
try:
    flights = pd.read_csv(flights_path, engine="pyarrow", dtype=code_types)
except Exception as e:
    print("pyarrow engine unavailable:", type(e).__name__)
    flights = pd.read_csv(flights_path, dtype=code_types, low_memory=False)
airlines = pd.read_csv(os.path.join(DATA_DIR, "airlines.csv"))
airports = pd.read_csv(os.path.join(DATA_DIR, "airports.csv"))
print(f"flights {flights.shape} | airlines {airlines.shape} | airports {airports.shape} | {time.time()-t0:.1f}s")

numeric_code = (flights["ORIGIN_AIRPORT"].str.fullmatch(r"\d+").fillna(False) |
                flights["DESTINATION_AIRPORT"].str.fullmatch(r"\d+").fillna(False))
print(f"Rows with numeric airport codes (October 2015 quirk), removed: {numeric_code.sum():,}")
flights = flights.loc[~numeric_code]

if N_ROWS and len(flights) > N_ROWS:
    df = flights.sample(n=N_ROWS, random_state=SEED).reset_index(drop=True)
else:
    df = flights.reset_index(drop=True)
del flights; gc.collect()
print(f"Working sample: {len(df):,} flights")

airlines = airlines.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"})
ap = airports.rename(columns=str.upper)
orig = ap.add_prefix("ORIG_").rename(columns={"ORIG_IATA_CODE": "ORIGIN_AIRPORT"})
dest = ap.add_prefix("DEST_").rename(columns={"DEST_IATA_CODE": "DESTINATION_AIRPORT"})

df = (df.merge(airlines, on="AIRLINE", how="left", validate="m:1")
        .merge(orig, on="ORIGIN_AIRPORT", how="left", validate="m:1")
        .merge(dest, on="DESTINATION_AIRPORT", how="left", validate="m:1"))
print(df.shape)
print("Unmatched airlines:", df["AIRLINE_NAME"].isna().sum(), "| unmatched origins:", df["ORIG_AIRPORT"].isna().sum())

print(df.shape)
df.head()

df.info(memory_usage="deep")

overview = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_%": df.isna().mean().mul(100).round(2),
    "n_unique": df.nunique(dropna=True),
    "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
})
overview

df.describe(percentiles=[.01, .25, .5, .75, .99]).T

df.describe(include=["object", "string"]).T

def mem_mb(frame):
    return frame.memory_usage(deep=True).sum() / 1024**2

before = mem_mb(df)
float_cols = df.select_dtypes(include="float").columns
int_cols = df.select_dtypes(include="integer").columns
df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")
df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")

CAT_COLS = ["AIRLINE", "AIRLINE_NAME", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TAIL_NUMBER", "CANCELLATION_REASON",
            "ORIG_AIRPORT", "ORIG_CITY", "ORIG_STATE", "ORIG_COUNTRY", "DEST_AIRPORT", "DEST_CITY", "DEST_STATE", "DEST_COUNTRY"]
for c in CAT_COLS:
    if c in df:
        df[c] = df[c].astype("category")

after = mem_mb(df)
print(f"Memory: {before:,.0f} MB → {after:,.0f} MB ({1-after/before:.0%} saved)")

missing = (df.isna().sum().to_frame("n_missing")
             .assign(pct=lambda d: d.n_missing / len(df) * 100)
             .query("n_missing > 0").sort_values("pct", ascending=False))

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(data=missing.reset_index(), y="index", x="pct", ax=ax, color="#4C72B0")
ax.set(title="Missing values by column", xlabel="% missing", ylabel="")
for p in ax.patches:
    ax.text(p.get_width() + .3, p.get_y() + p.get_height() / 2, f"{p.get_width():.1f}%", va="center", fontsize=9)
savefig("missing_bar"); plt.show()
missing

miss_sample = samp(df, 50_000)
msno.matrix(miss_sample[missing.index], figsize=(14, 6), fontsize=9, sparkline=False); plt.show()
msno.heatmap(miss_sample[missing.index], figsize=(12, 8), fontsize=9); plt.show()

flight_state = np.select([df["CANCELLED"] == 1, df["DIVERTED"] == 1], ["Cancelled", "Diverted"], "Completed")
(df[missing.index].isna().groupby(flight_state).mean().mul(100).round(1).T
   .style.background_gradient(cmap="Reds").format("{:.1f}%"))

df["FLIGHT_DATE"] = pd.to_datetime(pd.DataFrame({"year": df["YEAR"], "month": df["MONTH"], "day": df["DAY"]}))

def hhmm_to_minutes(s):
    s = s.astype("float64")
    return (s // 100 % 24) * 60 + s % 100

for c in ["SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "WHEELS_OFF", "WHEELS_ON", "SCHEDULED_ARRIVAL", "ARRIVAL_TIME"]:
    df[f"{c}_MIN"] = hhmm_to_minutes(df[c]).astype("float32")
df["SCHED_DEP_DT"] = df["FLIGHT_DATE"] + pd.to_timedelta(df["SCHEDULED_DEPARTURE_MIN"], unit="m")

df["STATUS"] = pd.Categorical(np.select([df["CANCELLED"] == 1, df["DIVERTED"] == 1], ["Cancelled", "Diverted"], "Completed"))
reason_map = {"A": "Airline/Carrier", "B": "Weather", "C": "National Air System", "D": "Security"}
df["CANCELLATION_REASON"] = df["CANCELLATION_REASON"].astype(str).map(reason_map).astype("category")

CAUSES = ["AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"]
done = df["STATUS"] == "Completed"
df.loc[done, CAUSES] = df.loc[done, CAUSES].fillna(0)

key = ["FLIGHT_DATE", "AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT", "SCHEDULED_DEPARTURE"]
dups = df.duplicated(subset=key).sum()
df = df.drop_duplicates(subset=key)
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
df = df.drop(columns=const_cols)
print(f"Duplicates removed: {dups:,} | constant columns dropped: {const_cols}")

c = df.loc[done]
checks = {
    "AIR_TIME <= ELAPSED_TIME": (c["AIR_TIME"] <= c["ELAPSED_TIME"]).mean(),
    "ELAPSED = AIR + TAXI_IN + TAXI_OUT": np.isclose(c["ELAPSED_TIME"], c["AIR_TIME"] + c["TAXI_IN"] + c["TAXI_OUT"]).mean(),
    "delay causes sum = ARRIVAL_DELAY (when ≥15)": np.isclose(c.loc[c["ARRIVAL_DELAY"] >= 15, CAUSES].sum(axis=1),
                                                              c.loc[c["ARRIVAL_DELAY"] >= 15, "ARRIVAL_DELAY"]).mean(),
}
pd.Series(checks, name="share_consistent").map("{:.2%}".format).to_frame()

num_check = ["DEPARTURE_DELAY", "ARRIVAL_DELAY", "TAXI_OUT", "TAXI_IN", "AIR_TIME", "DISTANCE"]
rows = []
for col in num_check:
    x = df[col].dropna().to_numpy(dtype="float64")
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    rows.append({"column": col, "Q1": q1, "median": np.median(x), "Q3": q3,
                 "IQR_outlier_%": ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100,
                 "z>3_%": (np.abs(stats.zscore(x)) > 3).mean() * 100,
                 "max": x.max(), "skew": stats.skew(x), "kurtosis": stats.kurtosis(x)})
pd.DataFrame(rows).set_index("column")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, num_check):
    sns.boxplot(x=samp(df[col].dropna(), 200_000), ax=ax, color="#55A868", fliersize=1)
    ax.set_title(col)
plt.tight_layout(); plt.show()

df["ARR_DELAY_CAPPED"] = df["ARRIVAL_DELAY"].clip(-60, df["ARRIVAL_DELAY"].quantile(.995))
df["DEP_DELAY_CAPPED"] = df["DEPARTURE_DELAY"].clip(-60, df["DEPARTURE_DELAY"].quantile(.995))

df["DEP_HOUR"] = (df["SCHEDULED_DEPARTURE"] // 100 % 24).astype("int8")
df["ARR_HOUR"] = (df["SCHEDULED_ARRIVAL"] // 100 % 24).astype("int8")
df["DEP_PERIOD"] = pd.cut(df["DEP_HOUR"], [-1, 5, 11, 17, 23], labels=["Night", "Morning", "Afternoon", "Evening"])
df["DAY_NAME"] = pd.Categorical(df["FLIGHT_DATE"].dt.day_name(),
                                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], ordered=True)
df["IS_WEEKEND"] = df["DAY_OF_WEEK"] >= 6
df["WEEK"] = df["FLIGHT_DATE"].dt.isocalendar().week.astype("int8")
df["DAY_OF_YEAR"] = df["FLIGHT_DATE"].dt.dayofyear.astype("int16")
season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
df["SEASON"] = pd.Categorical(df["MONTH"].map(season_map), ["Winter", "Spring", "Summer", "Fall"], ordered=True)

holidays = USFederalHolidayCalendar().holidays("2014-12-01", "2016-01-31").to_numpy()
day_vals = df["FLIGHT_DATE"].to_numpy()
pos = np.searchsorted(holidays, day_vals)
prev_h, next_h = holidays[np.clip(pos - 1, 0, len(holidays) - 1)], holidays[np.clip(pos, 0, len(holidays) - 1)]
df["DAYS_TO_HOLIDAY"] = np.minimum(np.abs((day_vals - prev_h).astype("timedelta64[D]").astype(int)),
                                   np.abs((next_h - day_vals).astype("timedelta64[D]").astype(int))).astype("int16")

df["ROUTE"] = (df["ORIGIN_AIRPORT"].astype(str) + "→" + df["DESTINATION_AIRPORT"].astype(str)).astype("category")
df["DELAYED_15"] = (df["ARRIVAL_DELAY"] >= 15).astype("float32").where(df["STATUS"] == "Completed")
df["AVG_SPEED_MPH"] = (df["DISTANCE"] / df["AIR_TIME"] * 60).where(df["AIR_TIME"] > 0)
df["DISTANCE_BAND"] = pd.cut(df["DISTANCE"], [0, 500, 1000, 2000, 5000], labels=["<500 mi", "500-1000", "1000-2000", ">2000"])
df["TIME_MADE_UP"] = df["DEPARTURE_DELAY"] - df["ARRIVAL_DELAY"]

ad = df["ARRIVAL_DELAY"].to_numpy()
df["DELAY_CLASS"] = pd.Categorical(np.select([np.isnan(ad), ad < 0, ad < 15, ad < 60, ad < 180],
                                             ["Not arrived", "Early", "On time", "Moderate", "Long"], "Very long"),
                                   ["Early", "On time", "Moderate", "Long", "Very long", "Not arrived"], ordered=True)

cause_vals = df[CAUSES].fillna(0).to_numpy()
main = np.array([c.replace("_DELAY", "").replace("_", " ").title() for c in CAUSES])[cause_vals.argmax(axis=1)]
df["MAIN_CAUSE"] = pd.Categorical(np.where(df["DELAYED_15"] == 1, main, "None"))

df[["FLIGHT_DATE", "DEP_HOUR", "DEP_PERIOD", "DAY_NAME", "SEASON", "DAYS_TO_HOLIDAY", "ROUTE",
    "DELAYED_15", "AVG_SPEED_MPH", "DELAY_CLASS", "MAIN_CAUSE"]].head()

geo_ok = df[["ORIG_LATITUDE", "ORIG_LONGITUDE", "DEST_LATITUDE", "DEST_LONGITUDE"]].notna().all(axis=1).to_numpy()
lat1, lon1, lat2, lon2 = (np.radians(df.loc[geo_ok, c].to_numpy(dtype="float64"))
                          for c in ["ORIG_LATITUDE", "ORIG_LONGITUDE", "DEST_LATITUDE", "DEST_LONGITUDE"])

def haversine_mi(lat1, lon1, lat2, lon2):
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))

t = time.time(); gc_dist = haversine_mi(lat1, lon1, lat2, lon2); t_np = time.time() - t
df.loc[geo_ok, "GREAT_CIRCLE_MI"] = gc_dist.astype("float32")

n = min(200_000, len(gc_dist))
t = time.time()
_ = [3958.8 * 2 * math.asin(math.sqrt(math.sin((lat2[i] - lat1[i]) / 2) ** 2 +
     math.cos(lat1[i]) * math.cos(lat2[i]) * math.sin((lon2[i] - lon1[i]) / 2) ** 2)) for i in range(n)]
t_loop = (time.time() - t) * len(gc_dist) / max(n, 1)
reported = df.loc[geo_ok, "DISTANCE"].to_numpy(dtype="float64")
print(f"NumPy {t_np:.3f}s vs estimated Python loop {t_loop:.1f}s → ~{t_loop / max(t_np, 1e-9):,.0f}× faster")
print(f"Correlation reported vs great-circle distance: {np.corrcoef(reported, gc_dist)[0, 1]:.4f}")
print(f"Median ratio reported / great-circle: {np.median(reported / np.maximum(gc_dist, 1)):.3f}")

arr = df["ARRIVAL_DELAY"].dropna().to_numpy(dtype="float64")
dep = df["DEPARTURE_DELAY"].dropna().to_numpy(dtype="float64")
print("Arrival delay percentiles (5,25,50,75,95,99):", np.percentile(arr, [5, 25, 50, 75, 95, 99]).round(1))
print(f"Mean {arr.mean():.2f} | std {arr.std():.2f} | trimmed mean (10%) {stats.trim_mean(arr, .1):.2f}")
print(f"Arrived early: {(arr < 0).mean():.1%} | on time (<15): {(arr < 15).mean():.1%} | over 3 hours late: {(arr >= 180).mean():.2%}")

X = df[["DEPARTURE_DELAY", "TAXI_OUT", "AIR_TIME", "TAXI_IN", "ARRIVAL_DELAY"]].dropna().to_numpy(dtype="float64")
Z = (X - X.mean(axis=0)) / X.std(axis=0)
print("Standardised means:", Z.mean(axis=0).round(6), "| stds:", Z.std(axis=0).round(3))
display(pd.DataFrame(np.corrcoef(Z, rowvar=False), index=["DEP", "TAXI_OUT", "AIR", "TAXI_IN", "ARR"],
                     columns=["DEP", "TAXI_OUT", "AIR", "TAXI_IN", "ARR"]).round(3))

counts, edges = np.histogram(arr, bins=[-np.inf, -15, 0, 15, 60, 180, np.inf])
labels = ["< -15", "-15..0", "0..15", "15..60", "60..180", "> 180"]
print("Arrival delay bins:", dict(zip(labels, counts.tolist())))
codes, cnts = np.unique(df["AIRLINE"].astype(str).to_numpy(), return_counts=True)
print("Flights per airline code:", dict(zip(codes.tolist(), cnts.tolist())))

route_counts = np.sort(df["ROUTE"].value_counts().to_numpy())[::-1]
cum_share = np.cumsum(route_counts) / route_counts.sum()
n80 = int(np.searchsorted(cum_share, .8)) + 1
plt.figure(figsize=(10, 4))
plt.plot(np.arange(1, len(cum_share) + 1), cum_share * 100)
plt.axhline(80, ls="--", c="red"); plt.axvline(n80, ls="--", c="red")
plt.xscale("log"); plt.xlabel("Number of routes (log scale)"); plt.ylabel("% of flights")
plt.title(f"{n80:,} of {len(route_counts):,} routes carry 80% of flights"); plt.show()

airline_stats = (df.groupby("AIRLINE_NAME", observed=True)
                   .agg(flights=("FLIGHT_NUMBER", "size"),
                        on_time_rate=("DELAYED_15", lambda s: 1 - s.mean()),
                        mean_arr_delay=("ARRIVAL_DELAY", "mean"),
                        median_arr_delay=("ARRIVAL_DELAY", "median"),
                        cancel_rate=("CANCELLED", "mean"),
                        divert_rate=("DIVERTED", "mean"),
                        avg_distance=("DISTANCE", "mean"),
                        avg_taxi_out=("TAXI_OUT", "mean"),
                        routes=("ROUTE", "nunique"),
                        aircraft=("TAIL_NUMBER", "nunique"))
                   .sort_values("flights", ascending=False))
airline_stats["share_%"] = airline_stats["flights"] / airline_stats["flights"].sum() * 100
airline_stats["flights_per_aircraft"] = airline_stats["flights"] / airline_stats["aircraft"]
airline_stats["punctuality_rank"] = airline_stats["on_time_rate"].rank(ascending=False).astype(int)
airline_stats.style.background_gradient(subset=["on_time_rate"], cmap="RdYlGn").background_gradient(
    subset=["cancel_rate", "mean_arr_delay"], cmap="Reds").format(precision=3)

pivot_am = df.pivot_table(index="AIRLINE", columns="MONTH", values="ARRIVAL_DELAY", aggfunc="mean", observed=True).round(1)
display(pivot_am.style.background_gradient(cmap="RdYlGn_r", axis=None).format("{:.1f}"))

display(pd.crosstab(df["MONTH"], df["CANCELLATION_REASON"], normalize="index").mul(100).round(1))

multi = df.groupby(["SEASON", "DEP_PERIOD"], observed=True)["DELAYED_15"].mean().unstack().mul(100).round(1)
display(multi)
display(multi.stack().rename("delayed_%").reset_index().head(8))

cause_long = (df.loc[df["DELAYED_15"] == 1, ["AIRLINE"] + CAUSES]
                .melt(id_vars="AIRLINE", var_name="cause", value_name="minutes"))
cause_share = (cause_long.groupby(["AIRLINE", "cause"], observed=True)["minutes"].sum()
                         .groupby(level=0, group_keys=False).apply(lambda s: s / s.sum() * 100).unstack().round(1))
cause_share

df["ROUTE_MEAN_DELAY"] = df.groupby("ROUTE", observed=True)["ARRIVAL_DELAY"].transform("mean")
df["DELAY_VS_ROUTE"] = df["ARRIVAL_DELAY"] - df["ROUTE_MEAN_DELAY"]
df["TAXI_OUT_Q"] = pd.qcut(df["TAXI_OUT"], 5, labels=["Q1 fast", "Q2", "Q3", "Q4", "Q5 slow"], duplicates="drop")

display(df.query("ARRIVAL_DELAY > 600 and DISTANCE < 500")
          [["FLIGHT_DATE", "AIRLINE_NAME", "ROUTE", "DEPARTURE_DELAY", "ARRIVAL_DELAY", "MAIN_CAUSE"]].head(8))

origin_stats = (df.groupby("ORIGIN_AIRPORT", observed=True)
                  .agg(departures=("FLIGHT_NUMBER", "size"), dep_delay=("DEPARTURE_DELAY", "mean"),
                       taxi_out=("TAXI_OUT", "mean"), cancel_rate=("CANCELLED", "mean"),
                       destinations=("DESTINATION_AIRPORT", "nunique"))
                  .reset_index()
                  .merge(ap, left_on="ORIGIN_AIRPORT", right_on="IATA_CODE", how="left")
                  .drop(columns="IATA_CODE")
                  .sort_values("departures", ascending=False))
display(origin_stats.head(15))

daily = (df.groupby("FLIGHT_DATE")
           .agg(flights=("FLIGHT_NUMBER", "size"), cancelled=("CANCELLED", "sum"),
                mean_arr_delay=("ARRIVAL_DELAY", "mean"), delayed_rate=("DELAYED_15", "mean")))
daily["ma_7"] = daily["mean_arr_delay"].rolling(7, min_periods=3).mean()
daily["ewm_14"] = daily["mean_arr_delay"].ewm(span=14).mean()
daily["cancel_rate"] = daily["cancelled"] / daily["flights"]
daily["flights_wow_%"] = daily["flights"].pct_change(7) * 100
daily.tail()

top_routes = (df.groupby(["AIRLINE", "ROUTE"], observed=True).size().rename("flights")
                .groupby(level=0, group_keys=False).nlargest(3).reset_index())
display(top_routes.head(15))
busiest_tails = (df.groupby(["TAIL_NUMBER", "AIRLINE"], observed=True).size().rename("flights")
                   .sort_values(ascending=False).head(10))
busiest_tails.to_frame()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
df["STATUS"].value_counts().plot.pie(autopct="%1.2f%%", ax=axes[0], startangle=90, cmap="Set2", explode=[0, .1, .1])
axes[0].set(title="Flight status", ylabel="")
sns.countplot(x="DELAY_CLASS", data=df, ax=axes[1], palette="RdYlGn_r")
axes[1].set_title("Arrival delay classes")
axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height() / len(df):.1%}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
plt.tight_layout(); savefig("status_delay_class"); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
air = df["AIRLINE_NAME"].value_counts()
sns.barplot(x=air.values, y=air.index.astype(str), ax=axes[0], palette="viridis")
axes[0].set_title("Flights per airline")
busiest = df["ORIG_AIRPORT"].value_counts().head(20)
sns.barplot(x=busiest.values, y=busiest.index.astype(str), ax=axes[1], palette="magma")
axes[1].set_title("20 busiest origin airports")
plt.tight_layout(); savefig("airlines_airports"); plt.show()

sample = samp(df.loc[df["STATUS"] == "Completed"], 300_000)
plots = [("DEP_DELAY_CAPPED", "Departure delay (min)"), ("ARR_DELAY_CAPPED", "Arrival delay (min)"),
         ("TAXI_OUT", "Taxi out (min)"), ("AIR_TIME", "Air time (min)"), ("DISTANCE", "Distance (mi)"),
         ("AVG_SPEED_MPH", "Average speed (mph)")]
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, (col, label) in zip(axes.flat, plots):
    data = sample[col].dropna()
    if "DELAY" in col:
        data = data[data.between(-60, 240)]
    sns.histplot(data, bins=70, kde=True, ax=ax, color="#4C72B0")
    ax.axvline(data.median(), color="red", ls="--", label=f"median {data.median():.0f}")
    ax.set_title(label); ax.legend()
plt.tight_layout(); savefig("numeric_distributions"); plt.show()

top5 = df["AIRLINE_NAME"].value_counts().index[:5]
plt.figure(figsize=(11, 5))
sns.ecdfplot(data=sample[sample["AIRLINE_NAME"].isin(top5)].assign(AIRLINE_NAME=lambda d: d["AIRLINE_NAME"].astype(str)),
             x="ARR_DELAY_CAPPED", hue="AIRLINE_NAME")
plt.axvline(15, ls="--", c="grey"); plt.xlim(-60, 180)
plt.title("ECDF of arrival delay, five largest airlines"); plt.xlabel("Arrival delay (min)"); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
df["CANCELLATION_REASON"].value_counts().plot.bar(ax=axes[0], color="#C44E52", rot=15, title="Cancellation reasons")
df.loc[df["MAIN_CAUSE"] != "None", "MAIN_CAUSE"].value_counts().plot.bar(ax=axes[1], color="#8172B2", rot=15,
                                                                         title="Main cause of delays ≥ 15 min")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sns.countplot(x="MONTH", data=df, ax=axes[0, 0], color="#4C72B0"); axes[0, 0].set_title("Flights per month")
sns.countplot(x="DAY_NAME", data=df, ax=axes[0, 1], palette="coolwarm"); axes[0, 1].set_title("Flights per weekday")
axes[0, 1].tick_params(axis="x", rotation=30)
sns.countplot(x="DEP_HOUR", data=df, ax=axes[1, 0], color="#55A868"); axes[1, 0].set_title("Flights by scheduled departure hour")
df.groupby("DEP_HOUR")["DELAYED_15"].mean().mul(100).plot(ax=axes[1, 1], marker="o", color="#C44E52")
axes[1, 1].set(title="% delayed by departure hour", ylabel="% delayed ≥ 15 min")
plt.tight_layout(); savefig("time_counts"); plt.show()

heat = df.pivot_table(index="DAY_NAME", columns="DEP_HOUR", values="ARRIVAL_DELAY", aggfunc="mean", observed=True)
plt.figure(figsize=(18, 5))
sns.heatmap(heat, cmap="RdYlGn_r", center=0, linewidths=.3, cbar_kws={"label": "mean arrival delay (min)"})
plt.title("Mean arrival delay: weekday × departure hour (delays build up during the day)")
savefig("weekday_hour_heatmap"); plt.show()

heat2 = df.pivot_table(index="MONTH", columns="DEP_HOUR", values="DELAYED_15", aggfunc="mean").mul(100)
plt.figure(figsize=(18, 5))
sns.heatmap(heat2, cmap="magma_r", cbar_kws={"label": "% delayed"})
plt.title("% delayed: month × departure hour"); plt.show()

fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
axes[0].plot(daily.index, daily["mean_arr_delay"], alpha=.35, label="daily")
axes[0].plot(daily.index, daily["ma_7"], label="7-day mean", lw=2)
for h in holidays[(holidays >= np.datetime64("2015-01-01")) & (holidays <= np.datetime64("2015-12-31"))]:
    axes[0].axvline(h, color="green", alpha=.3, ls=":")
axes[0].set(title="Mean arrival delay per day (green = federal holidays)", ylabel="minutes"); axes[0].legend()
axes[1].bar(daily.index, daily["cancel_rate"] * 100, color="#C44E52")
axes[1].set(title="Daily cancellation rate", ylabel="%")
plt.tight_layout(); savefig("daily_trend"); plt.show()

daily.nlargest(5, "cancel_rate")[["flights", "cancelled", "cancel_rate", "mean_arr_delay"]]

monthly_air = (df[df["AIRLINE_NAME"].isin(top5)].groupby(["MONTH", "AIRLINE_NAME"], observed=True)["DELAYED_15"]
                 .mean().mul(100).rename("delayed_%").reset_index())
plt.figure(figsize=(14, 5))
sns.lineplot(data=monthly_air, x="MONTH", y="delayed_%", hue="AIRLINE_NAME", marker="o")
plt.title("Monthly delay rate, five largest airlines"); plt.legend(bbox_to_anchor=(1.01, 1)); plt.show()

mix = pd.crosstab(df["WEEK"], df["DELAY_CLASS"], normalize="index")
mix.plot.area(figsize=(15, 5), cmap="RdYlGn_r", alpha=.85)
plt.title("Weekly mix of delay classes"); plt.ylabel("share"); plt.legend(bbox_to_anchor=(1.01, 1)); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ot = airline_stats["on_time_rate"].sort_values()
sns.barplot(x=ot.values * 100, y=ot.index.astype(str), ax=axes[0], palette="RdYlGn")
axes[0].axvline((1 - df["DELAYED_15"].mean()) * 100, ls="--", c="black", label="overall")
axes[0].set(title="On-time rate by airline", xlabel="% on time", xlim=(60, 95)); axes[0].legend()

cause_share.plot.barh(stacked=True, ax=axes[1], cmap="tab10")
axes[1].set(title="Delay minutes by cause (share per airline)", xlabel="%"); axes[1].legend(bbox_to_anchor=(1.01, 1))
plt.tight_layout(); savefig("airline_performance"); plt.show()

order = airline_stats.sort_values("median_arr_delay").index.astype(str)
plt.figure(figsize=(16, 6))
sns.boxplot(data=sample.assign(AIRLINE_NAME=lambda d: d["AIRLINE_NAME"].astype(str)),
            x="AIRLINE_NAME", y="ARR_DELAY_CAPPED", order=order, showfliers=False, palette="coolwarm")
plt.xticks(rotation=40, ha="right"); plt.ylabel("Arrival delay (min)"); plt.title("Arrival delay by airline (no outliers shown)"); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.violinplot(data=sample, x="DEP_PERIOD", y="ARR_DELAY_CAPPED", ax=axes[0], cut=0, palette="Set2")
sns.boxenplot(data=sample, x="SEASON", y="TAXI_OUT", ax=axes[1], palette="Set3")
sns.barplot(data=sample, x="DISTANCE_BAND", y="TIME_MADE_UP", ax=axes[2], palette="Blues")
axes[0].set_title("Arrival delay by time of day"); axes[1].set_title("Taxi-out by season")
axes[2].set_title("Minutes recovered in the air by distance"); axes[0].set_ylim(-60, 200)
plt.tight_layout(); plt.show()

NUM_FEATS = ["DEPARTURE_DELAY", "ARRIVAL_DELAY", "TAXI_OUT", "TAXI_IN", "AIR_TIME", "DISTANCE", "SCHEDULED_TIME",
             "DEP_HOUR", "MONTH", "DAY_OF_WEEK", "DAYS_TO_HOLIDAY", "AVG_SPEED_MPH"] + CAUSES
fig, axes = plt.subplots(1, 2, figsize=(24, 10))
for ax, method in zip(axes, ["pearson", "spearman"]):
    corr = sample[NUM_FEATS].corr(method=method)
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, square=True, cbar_kws={"shrink": .7}, annot_kws={"size": 8})
    ax.set_title(f"{method.title()} correlation")
plt.tight_layout(); savefig("correlations"); plt.show()

pp = samp(sample[["DEP_DELAY_CAPPED", "ARR_DELAY_CAPPED", "TAXI_OUT", "AIR_TIME", "DISTANCE", "DEP_PERIOD"]].dropna(), 4_000)
g = sns.pairplot(pp, hue="DEP_PERIOD", corner=True, plot_kws={"alpha": .4, "s": 10}, diag_kind="kde")
g.fig.suptitle("Pair plot (4k flights)", y=1.02); plt.show()

sns.jointplot(data=samp(sample, 50_000), x="DEP_DELAY_CAPPED", y="ARR_DELAY_CAPPED", kind="hex",
              height=7, cmap="mako_r", xlim=(-30, 200), ylim=(-60, 200))
plt.show()

sns.lmplot(data=samp(sample, 20_000), x="DISTANCE", y="AIR_TIME", hue="SEASON",
           scatter_kws={"s": 4, "alpha": .3}, height=5, aspect=1.6)
plt.show()

sns.catplot(data=sample, x="DEP_PERIOD", y="DELAYED_15", hue="IS_WEEKEND", col="SEASON", kind="point",
            col_wrap=2, height=3.5, aspect=1.4, errorbar=("ci", 95))
plt.show()

hol = df.groupby(df["DAYS_TO_HOLIDAY"].clip(upper=15))["DELAYED_15"].mean().mul(100)
plt.figure(figsize=(10, 4)); hol.plot(marker="o")
plt.title("% delayed vs days from nearest federal holiday"); plt.xlabel("days (15 = 15+)"); plt.ylabel("% delayed"); plt.show()

ap_map = origin_stats.dropna(subset=["LATITUDE", "LONGITUDE"])
plt.figure(figsize=(18, 9))
sc = plt.scatter(ap_map["LONGITUDE"], ap_map["LATITUDE"], s=ap_map["departures"] / ap_map["departures"].max() * 1500,
                 c=ap_map["dep_delay"], cmap="RdYlGn_r", alpha=.7, edgecolor="k", linewidth=.3)
for _, r in ap_map.head(15).iterrows():
    plt.annotate(r["ORIGIN_AIRPORT"], (r["LONGITUDE"], r["LATITUDE"]), fontsize=9, ha="center", va="bottom")
plt.colorbar(sc, label="mean departure delay (min)")
plt.xlim(-125, -66); plt.ylim(24, 50)
plt.title("Airports: size = departures, colour = mean departure delay (lower 48)"); savefig("airport_map"); plt.show()

fig = px.scatter_geo(ap_map.assign(ORIGIN_AIRPORT=lambda d: d["ORIGIN_AIRPORT"].astype(str)),
                     lat="LATITUDE", lon="LONGITUDE", size="departures", color="dep_delay",
                     hover_name="AIRPORT", hover_data={"ORIGIN_AIRPORT": True, "cancel_rate": ":.2%", "destinations": True},
                     color_continuous_scale="RdYlGn_r", scope="usa", size_max=40, title="Airports (hover for details)")
fig.show()

routes = (df.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], observed=True).size().rename("flights").reset_index()
            .nlargest(80, "flights")
            .merge(ap[["IATA_CODE", "LATITUDE", "LONGITUDE"]], left_on="ORIGIN_AIRPORT", right_on="IATA_CODE")
            .merge(ap[["IATA_CODE", "LATITUDE", "LONGITUDE"]], left_on="DESTINATION_AIRPORT", right_on="IATA_CODE",
                   suffixes=("_o", "_d")))
fig = go.Figure()
for _, r in routes.iterrows():
    fig.add_trace(go.Scattergeo(lon=[r["LONGITUDE_o"], r["LONGITUDE_d"]], lat=[r["LATITUDE_o"], r["LATITUDE_d"]],
                                mode="lines", line=dict(width=r["flights"] / routes["flights"].max() * 4, color="royalblue"),
                                opacity=.6, hoverinfo="text", text=f"{r['ORIGIN_AIRPORT']}→{r['DESTINATION_AIRPORT']}: {r['flights']:,}"))
fig.update_layout(title="80 busiest routes", geo=dict(scope="usa"), showlegend=False)
fig.show()

state_delay = (df.groupby("ORIG_STATE", observed=True)
                 .agg(flights=("FLIGHT_NUMBER", "size"), delayed_rate=("DELAYED_15", "mean"),
                      cancel_rate=("CANCELLED", "mean")).reset_index())
state_delay["ORIG_STATE"] = state_delay["ORIG_STATE"].astype(str)
px.choropleth(state_delay, locations="ORIG_STATE", locationmode="USA-states", scope="usa", color="delayed_rate",
              hover_data={"flights": ":,", "cancel_rate": ":.2%"}, color_continuous_scale="Reds",
              title="Share of delayed departures by origin state").show()

out_deg = df.groupby("ORIGIN_AIRPORT", observed=True)["DESTINATION_AIRPORT"].nunique().rename("out_degree")
in_deg = df.groupby("DESTINATION_AIRPORT", observed=True)["ORIGIN_AIRPORT"].nunique().rename("in_degree")
volume = df["ORIGIN_AIRPORT"].value_counts().rename("departures")
hubs = pd.concat([out_deg, in_deg, volume], axis=1).fillna(0)
hubs["degree"] = hubs["out_degree"] + hubs["in_degree"]
hubs = hubs.join(ap.set_index("IATA_CODE")[["AIRPORT", "CITY", "STATE"]]).sort_values("degree", ascending=False)
display(hubs.head(15))

hub_share = (df.groupby(["AIRLINE_NAME", "ORIGIN_AIRPORT"], observed=True).size().rename("n").reset_index()
               .sort_values("n", ascending=False).groupby("AIRLINE_NAME", observed=True)
               .agg(main_hub=("ORIGIN_AIRPORT", "first"), hub_flights=("n", "first"), total=("n", "sum")))
hub_share["hub_concentration_%"] = hub_share["hub_flights"] / hub_share["total"] * 100
hub_share.sort_values("hub_concentration_%", ascending=False)

pair = df.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], observed=True).size().rename("ab").reset_index()
pair = pair.merge(pair.rename(columns={"ORIGIN_AIRPORT": "DESTINATION_AIRPORT", "DESTINATION_AIRPORT": "ORIGIN_AIRPORT", "ab": "ba"}),
                  on=["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], how="left").fillna({"ba": 0})
pair["imbalance"] = (pair["ab"] - pair["ba"]) / (pair["ab"] + pair["ba"])
print(f"Routes: {len(pair):,} | one-directional routes: {(pair['ba'] == 0).mean():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(hubs["degree"], bins=50, log_scale=(False, True), ax=axes[0]); axes[0].set_title("Degree distribution (log y)")
sns.scatterplot(data=hubs, x="degree", y="departures", ax=axes[1], alpha=.6)
for name, r in hubs.head(8).iterrows():
    axes[1].annotate(name, (r["degree"], r["departures"]), fontsize=9)
axes[1].set_title("Connections vs traffic"); plt.tight_layout(); plt.show()

s = samp(df.loc[df["STATUS"] == "Completed"], 50_000)
print("Shapiro (arrival delay, 5k):", stats.shapiro(s["ARRIVAL_DELAY"].iloc[:5000]))
print("D'Agostino K² (arrival delay):", stats.normaltest(s["ARRIVAL_DELAY"]))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
stats.probplot(s["ARRIVAL_DELAY"], dist="norm", plot=axes[0]); axes[0].set_title("Q-Q arrival delay")
stats.probplot(np.log(s["TAXI_OUT"]), dist="norm", plot=axes[1]); axes[1].set_title("Q-Q log taxi-out")
plt.tight_layout(); plt.show()

def cramers_v(x, y):
    tbl = pd.crosstab(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(tbl)
    return chi2, p, dof, np.sqrt(chi2 / (tbl.to_numpy().sum() * (min(tbl.shape) - 1)))

rows = []
for col in ["AIRLINE", "DEP_PERIOD", "SEASON", "IS_WEEKEND", "DAY_OF_WEEK", "DISTANCE_BAND", "MONTH", "TAXI_OUT_Q"]:
    chi2, p, dof, v = cramers_v(s[col], s["DELAYED_15"])
    rows.append({"variable": col, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v})
pd.DataFrame(rows).sort_values("cramers_v", ascending=False)

for col, mask, label in [("ARRIVAL_DELAY", s["IS_WEEKEND"], "weekend vs weekday"),
                         ("TAXI_OUT", s["DELAYED_15"] == 1, "delayed vs on-time"),
                         ("DEPARTURE_DELAY", s["DEP_PERIOD"] == "Evening", "evening vs rest")]:
    a, b = s.loc[mask, col], s.loc[~mask, col]
    _, t_p = stats.ttest_ind(a, b, equal_var=False)
    _, u_p = stats.mannwhitneyu(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    print(f"{col:<16} {label:<20} Welch p={t_p:.2e} | Mann-Whitney p={u_p:.2e} | Cohen's d={d:+.3f}")

groups = [g["ARRIVAL_DELAY"].to_numpy() for _, g in s.groupby("AIRLINE", observed=True)]
print("\nOne-way ANOVA arrival delay ~ airline:", stats.f_oneway(*groups))
print("Kruskal-Wallis:", stats.kruskal(*groups))
print("Levene:", stats.levene(*groups))

r, p = stats.pearsonr(s["DEPARTURE_DELAY"], s["ARRIVAL_DELAY"]); print(f"Pearson dep~arr r={r:.3f}, p={p:.2e}")
rho, p = stats.spearmanr(s["TAXI_OUT"], s["ARRIVAL_DELAY"]); print(f"Spearman taxi_out~arr rho={rho:.3f}, p={p:.2e}")
tau, p = stats.kendalltau(s["DEP_HOUR"].iloc[:10000], s["DEPARTURE_DELAY"].iloc[:10000]); print(f"Kendall hour~dep tau={tau:.3f}")

boot = stats.bootstrap((s["ARRIVAL_DELAY"].to_numpy()[:10_000],), np.median, n_resamples=2000,
                       method="percentile", batch=200, random_state=SEED)
print(f"Median arrival delay 95% CI: [{boot.confidence_interval.low:.1f}, {boot.confidence_interval.high:.1f}] min")

done_all = df["DELAYED_15"].dropna()
k, n = int((done_all == 0).sum()), len(done_all)
ci = stats.binomtest(k, n).proportion_ci(method="wilson")
print(f"On-time rate {k / n:.3%}  95% CI [{ci.low:.3%}, {ci.high:.3%}]")

x = s["TAXI_OUT"].to_numpy(dtype="float64")
for name in ["norm", "lognorm", "gamma"]:
    dist = getattr(stats, name); params = dist.fit(x)
    ks = stats.kstest(x, lambda v, d=dist, p=params: d.cdf(v, *p))
    print(f"Taxi-out fit {name:<8} KS={ks.statistic:.4f}")

m = samp(df.loc[df["STATUS"] == "Completed"], 200_000).copy()
m["IS_WEEKEND"] = m["IS_WEEKEND"].astype(int)
m["AIRLINE"] = m["AIRLINE"].astype(str)
ols = smf.ols("ARRIVAL_DELAY ~ DEPARTURE_DELAY + TAXI_OUT + TAXI_IN + DISTANCE + C(AIRLINE) + C(DEP_PERIOD) + C(SEASON) + IS_WEEKEND",
              data=m).fit(cov_type="HC3")
print(ols.summary().tables[0]); print(ols.summary().tables[1])

logit = smf.logit("DELAYED_15 ~ C(AIRLINE) + C(DEP_PERIOD) + C(SEASON) + DISTANCE + IS_WEEKEND + DAYS_TO_HOLIDAY",
                  data=m).fit(disp=0, maxiter=200)
odds = pd.DataFrame({"odds_ratio": np.exp(logit.params), "ci_low": np.exp(logit.conf_int()[0]),
                     "ci_high": np.exp(logit.conf_int()[1]), "p_value": logit.pvalues}).drop("Intercept")
print(f"Pseudo R²: {logit.prsquared:.3f}")
odds.sort_values("odds_ratio")

or_plot = odds.sort_values("odds_ratio")
plt.figure(figsize=(9, 9))
plt.errorbar(or_plot["odds_ratio"], or_plot.index, fmt="o", capsize=3,
             xerr=[or_plot["odds_ratio"] - or_plot["ci_low"], or_plot["ci_high"] - or_plot["odds_ratio"]])
plt.axvline(1, color="red", ls="--"); plt.xscale("log"); plt.title("Odds ratios for a delay ≥ 15 min (95% CI)"); plt.show()

ts = daily["mean_arr_delay"].asfreq("D").interpolate()
decomp = seasonal_decompose(ts, model="additive", period=7)
fig = decomp.plot(); fig.set_size_inches(14, 9); plt.show()

adf = adfuller(ts.dropna())
print(f"ADF statistic={adf[0]:.3f}, p={adf[1]:.4f} → {'stationary' if adf[1] < .05 else 'non-stationary'}")
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
plot_acf(ts, lags=40, ax=axes[0]); plot_pacf(ts, lags=40, ax=axes[1], method="ywm"); plt.show()

vol = daily["flights"].asfreq("D").interpolate()
train, test = vol.iloc[:-28], vol.iloc[-28:]
sarima = SARIMAX(np.log(train), order=(1, 0, 1), seasonal_order=(1, 1, 1, 7),
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
fc = sarima.get_forecast(steps=len(test))
pred, ci = np.exp(fc.predicted_mean), np.exp(fc.conf_int())
pred.index = ci.index = test.index
mape = np.mean(np.abs((test - pred) / test)) * 100
plt.figure(figsize=(15, 5))
plt.plot(train.index[-90:], train.iloc[-90:], label="train")
plt.plot(test.index, test, label="actual", color="black")
plt.plot(pred.index, pred, label=f"forecast (MAPE {mape:.1f}%)", color="red")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="red", alpha=.15)
plt.title("SARIMA forecast of daily flight volume (last 4 weeks)"); plt.legend(); plt.show()

sb = samp(df.loc[df["STATUS"] == "Completed"], 500_000)[["AIRLINE_NAME", "DEP_PERIOD", "DELAY_CLASS"]].astype(str)
px.sunburst(sb, path=["AIRLINE_NAME", "DEP_PERIOD", "DELAY_CLASS"], title="Airline → time of day → delay class").show()

tm = origin_stats.dropna(subset=["STATE"]).astype({"ORIGIN_AIRPORT": str})
px.treemap(tm, path=["STATE", "ORIGIN_AIRPORT"], values="departures", color="dep_delay",
           color_continuous_scale="RdYlGn_r", title="Departures by state and airport (colour = mean delay)").show()

mon = df.groupby(["MONTH", "AIRLINE_NAME"], observed=True)["DELAYED_15"].mean().mul(100).rename("delayed_%").reset_index()
mon["AIRLINE_NAME"] = mon["AIRLINE_NAME"].astype(str)
px.line(mon, x="MONTH", y="delayed_%", color="AIRLINE_NAME", markers=True, title="Monthly delay rate by airline").show()

px.density_heatmap(samp(df, 500_000), x="DEP_HOUR", y="MONTH", z="ARR_DELAY_CAPPED", histfunc="avg",
                   nbinsx=24, nbinsy=12, color_continuous_scale="RdYlGn_r", title="Mean arrival delay: month × hour").show()

anim = (df.groupby(["MONTH", "ORIG_STATE"], observed=True)["DELAYED_15"].mean().mul(100)
          .rename("delayed_%").reset_index().astype({"ORIG_STATE": str}))
px.choropleth(anim, locations="ORIG_STATE", locationmode="USA-states", scope="usa", color="delayed_%",
              animation_frame="MONTH", range_color=(0, anim["delayed_%"].quantile(.95)), color_continuous_scale="Reds",
              title="Delay rate by state, month by month (press play)").show()

perf = airline_stats.reset_index().astype({"AIRLINE_NAME": str})
px.scatter(perf, x="cancel_rate", y="on_time_rate", size="flights", color="AIRLINE_NAME", text="AIRLINE_NAME",
           title="Airlines: cancellations vs punctuality (bubble = flights)").update_traces(textposition="top center").show()

prof_cols = ["departures", "dep_delay", "taxi_out", "cancel_rate", "destinations"]
prof = origin_stats.query("departures >= 1000").dropna(subset=prof_cols + ["LATITUDE", "LONGITUDE"]).copy()
Zp = StandardScaler().fit_transform(np.column_stack([np.log1p(prof["departures"]), prof[prof_cols[1:]]]))
pca = PCA().fit(Zp)
pcs = pca.transform(Zp)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].bar(range(1, len(prof_cols) + 1), pca.explained_variance_ratio_, label="per component")
axes[0].plot(range(1, len(prof_cols) + 1), np.cumsum(pca.explained_variance_ratio_), "o-r", label="cumulative")
axes[0].set(title="PCA explained variance (airport profiles)", xlabel="component"); axes[0].legend()
axes[1].scatter(pcs[:, 0], pcs[:, 1], alpha=.6)
for i, code_ in enumerate(prof["ORIGIN_AIRPORT"].astype(str)):
    if prof["departures"].iloc[i] > prof["departures"].quantile(.9):
        axes[1].annotate(code_, (pcs[i, 0], pcs[i, 1]), fontsize=8)
for j, name in enumerate(prof_cols):
    axes[1].arrow(0, 0, pca.components_[0, j] * 3, pca.components_[1, j] * 3, color="red", head_width=.08)
    axes[1].text(pca.components_[0, j] * 3.3, pca.components_[1, j] * 3.3, name, color="red")
axes[1].set(title="PCA biplot of airports", xlabel="PC1", ylabel="PC2")
plt.tight_layout(); plt.show()

ks = range(2, min(11, len(prof)))
inertia, sil = [], []
for k in ks:
    km_k = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Zp)
    inertia.append(km_k.inertia_); sil.append(silhouette_score(Zp, km_k.labels_))
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(list(ks), inertia, "o-"); axes[0].set_title("Elbow")
axes[1].plot(list(ks), sil, "o-", color="green"); axes[1].set_title("Silhouette score")
plt.show()

K = list(ks)[int(np.argmax(sil))]
prof["cluster"] = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit_predict(Zp)
display(prof.groupby("cluster")[prof_cols].mean().round(3).assign(n_airports=prof["cluster"].value_counts().sort_index()))

plt.figure(figsize=(16, 8))
sns.scatterplot(data=prof, x="LONGITUDE", y="LATITUDE", hue="cluster", size="departures", sizes=(20, 600), palette="tab10", alpha=.75)
plt.xlim(-125, -66); plt.ylim(24, 50); plt.title(f"Airport clusters (K = {K}, best silhouette)")
plt.legend(bbox_to_anchor=(1.01, 1)); plt.show()

coords = np.radians(ap.dropna(subset=["LATITUDE", "LONGITUDE"])[["LATITUDE", "LONGITUDE"]].to_numpy())
db = DBSCAN(eps=200 / 6371.0, min_samples=4, metric="haversine", algorithm="ball_tree").fit(coords)
labels = db.labels_
print(f"DBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} regional airport groups, {np.mean(labels == -1):.0%} isolated airports")
deg = np.degrees(coords)
plt.figure(figsize=(16, 8))
plt.scatter(deg[labels == -1, 1], deg[labels == -1, 0], s=12, c="lightgrey", label="isolated")
plt.scatter(deg[labels != -1, 1], deg[labels != -1, 0], s=20, c=labels[labels != -1], cmap="tab20")
plt.xlim(-125, -66); plt.ylim(24, 50); plt.title("DBSCAN: airports within 200 km of each other"); plt.legend(); plt.show()

NUM_ML = ["MONTH", "DAY", "DAY_OF_WEEK", "DAY_OF_YEAR", "DEP_HOUR", "ARR_HOUR", "SCHEDULED_TIME", "DISTANCE", "DAYS_TO_HOLIDAY"]
CAT_ML = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "DEP_PERIOD"]

ml = samp(df.loc[df["STATUS"] == "Completed"], 600_000)
X = ml[NUM_ML + CAT_ML].copy()
for c in CAT_ML:
    X[c] = X[c].astype(str)
y = ml["DELAYED_15"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
print(f"train {X_train.shape}, test {X_test.shape}, delayed rate {y.mean():.2%}")

results = {}
logreg = Pipeline([
    ("prep", ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_ML),
                                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), CAT_ML)])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
ord_prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                     encoded_missing_value=-1, max_categories=200), CAT_ML),
                              ("num", "passthrough", NUM_ML)])
hgb = Pipeline([("prep", ord_prep),
                ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=.08, max_leaf_nodes=63,
                                                       categorical_features=list(range(len(CAT_ML))),
                                                       class_weight="balanced", early_stopping=True, random_state=SEED))])
rf = Pipeline([("prep", ord_prep),
               ("clf", RandomForestClassifier(n_estimators=150, max_depth=18, min_samples_leaf=20,
                                              class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))])

for name, model in [("Logistic regression", logreg), ("Random forest", rf), ("Hist gradient boosting", hgb)]:
    t = time.time()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    results[name] = {"ROC_AUC": roc_auc_score(y_test, proba), "PR_AUC": average_precision_score(y_test, proba),
                     "F1": f1_score(y_test, proba >= .5), "fit_s": time.time() - t, "model": model}
    print(f"{name:<24} ROC-AUC {results[name]['ROC_AUC']:.3f} | PR-AUC {results[name]['PR_AUC']:.3f} | {results[name]['fit_s']:.0f}s")
pd.DataFrame(results).T.drop(columns="model")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for name, r in results.items():
    RocCurveDisplay.from_estimator(r["model"], X_test, y_test, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_estimator(r["model"], X_test, y_test, name=name, ax=axes[1])
axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set_title("ROC curves"); axes[1].set_title("Precision-recall curves")
best_name = max(results, key=lambda k: results[k]["ROC_AUC"]); best = results[best_name]["model"]
ConfusionMatrixDisplay.from_estimator(best, X_test, y_test, normalize="true", cmap="Blues", ax=axes[2],
                                      display_labels=["On time", "Delayed"])
axes[2].set_title(f"Confusion matrix: {best_name}"); axes[2].grid(False)
plt.tight_layout(); savefig("model_eval"); plt.show()
print(classification_report(y_test, best.predict(X_test), target_names=["On time", "Delayed"]))

proba = best.predict_proba(X_test)[:, 1]
thr = np.linspace(.05, .95, 91)
f1s = [f1_score(y_test, proba >= t) for t in thr]
best_t = thr[int(np.argmax(f1s))]
plt.figure(figsize=(9, 4)); plt.plot(thr, f1s); plt.axvline(best_t, ls="--", c="red")
plt.title(f"F1 vs threshold (best {best_t:.2f} → F1 {max(f1s):.3f})"); plt.xlabel("threshold"); plt.show()

idx = rng.choice(len(X_test), size=min(20_000, len(X_test)), replace=False)
pi = permutation_importance(best, X_test.iloc[idx], y_test.iloc[idx], n_repeats=3, scoring="roc_auc", random_state=SEED, n_jobs=-1)
pd.Series(pi.importances_mean, index=X_test.columns).sort_values().plot.barh(figsize=(9, 6), color="#4C72B0")
plt.title(f"Permutation importance ({best_name})"); plt.show()

cv = cross_val_score(hgb, X_train.iloc[:150_000], y_train.iloc[:150_000],
                     cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="roc_auc")
print(f"CV ROC-AUC: {cv.mean():.3f} ± {cv.std():.3f}")

NUM_LIVE = NUM_ML + ["DEPARTURE_DELAY", "TAXI_OUT"]
Xl = ml[NUM_LIVE + CAT_ML].copy()
for c in CAT_ML:
    Xl[c] = Xl[c].astype(str)
Xl_train, Xl_test = Xl.loc[X_train.index], Xl.loc[X_test.index]
live_prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                      encoded_missing_value=-1, max_categories=200), CAT_ML),
                               ("num", "passthrough", NUM_LIVE)])
live_clf = Pipeline([("prep", live_prep), ("clf", HistGradientBoostingClassifier(max_iter=300, categorical_features=list(range(len(CAT_ML))),
                                                                                random_state=SEED))]).fit(Xl_train, y_train)
print(f"After pushback (with departure delay + taxi-out): ROC-AUC {roc_auc_score(y_test, live_clf.predict_proba(Xl_test)[:, 1]):.3f}"
      f" vs at booking {results[best_name]['ROC_AUC']:.3f}")

y_reg = ml["ARRIVAL_DELAY"]
reg = Pipeline([("prep", live_prep), ("reg", HistGradientBoostingRegressor(max_iter=300, loss="absolute_error",
                                                                          categorical_features=list(range(len(CAT_ML))),
                                                                          random_state=SEED))])
reg.fit(Xl_train, y_reg.loc[X_train.index])
pred_reg = reg.predict(Xl_test)
true_reg = y_reg.loc[X_test.index]
print(f"Arrival-delay regression: MAE {mean_absolute_error(true_reg, pred_reg):.2f} min | R² {r2_score(true_reg, pred_reg):.3f}"
      f" | baseline MAE {mean_absolute_error(true_reg, np.full(len(true_reg), y_reg.loc[X_train.index].median())):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
k = rng.choice(len(true_reg), size=min(20_000, len(true_reg)), replace=False)
axes[0].scatter(true_reg.iloc[k], pred_reg[k], s=3, alpha=.3)
axes[0].plot([-60, 400], [-60, 400], "r--"); axes[0].set(xlim=(-60, 400), ylim=(-60, 400), xlabel="actual", ylabel="predicted",
                                                        title="Predicted vs actual arrival delay")
sns.histplot(true_reg - pred_reg, bins=100, ax=axes[1]); axes[1].set(xlim=(-100, 100), title="Residuals")
plt.tight_layout(); plt.show()

os.makedirs("outputs", exist_ok=True)
try:
    df.to_parquet("outputs/flights_clean.parquet", index=False)
    print("Saved outputs/flights_clean.parquet")
except ImportError:
    samp(df, 500_000).to_csv("outputs/flights_clean_sample.csv.gz", index=False)
    print("pyarrow not installed: saved a 500k-row CSV sample")
airline_stats.to_csv("outputs/airline_summary.csv")
origin_stats.to_csv("outputs/airport_summary.csv", index=False)
daily.to_csv("outputs/daily_summary.csv")
pd.DataFrame(results).T.drop(columns="model").to_csv("outputs/model_scores.csv")
print("Figures:", sorted(os.listdir(FIG_DIR)))

worst_hour = df.groupby("DEP_HOUR")["DELAYED_15"].mean().idxmax()
best_hour = df.groupby("DEP_HOUR")["DELAYED_15"].mean().idxmin()
worst_month = df.groupby("MONTH")["DELAYED_15"].mean().idxmax()
top_cause = df.loc[df["MAIN_CAUSE"] != "None", "MAIN_CAUSE"].value_counts().idxmax()
print("═" * 72)
print(" KEY FINDINGS")
print("═" * 72)
print(f"• Flights analysed: {len(df):,} on {df['ROUTE'].nunique():,} routes between {df['ORIGIN_AIRPORT'].nunique():,} airports")
print(f"• On-time rate (arrival < 15 min late): {1 - df['DELAYED_15'].mean():.1%} | cancelled: {df['CANCELLED'].mean():.2%}")
print(f"• Most punctual airline: {airline_stats['on_time_rate'].idxmax()} | least: {airline_stats['on_time_rate'].idxmin()}")
print(f"• Safest departure hour: {best_hour}:00 | worst: {worst_hour}:00 | worst month: {worst_month}")
print(f"• Most common main delay cause: {top_cause}")
print(f"• Busiest airport: {origin_stats.iloc[0]['AIRPORT']} ({origin_stats.iloc[0]['departures']:,} departures)")
print(f"• Best booking-time model: {best_name}, ROC-AUC {results[best_name]['ROC_AUC']:.3f}")
print("═" * 72)

