import pandas as pd

df = pd.read_csv("strategy_tqqq_reserve_debug.csv", parse_dates=["date"]).set_index("date")

print("Span:", df.index.min().date(), "to", df.index.max().date())
print("Avg deployed:", df["deployed_p"].mean())
print("Pct days >80% deployed:", (df["deployed_p"] > 0.8).mean())
print("Pct days 0% deployed:", (df["deployed_p"] < 1e-9).mean())

# Buys above T>1.3
viol_buy = (df["deployed_p"].diff() > 1e-3) & (df["temp"] > 1.3)
print("buys_above_1.3:", int(viol_buy.sum()))
if viol_buy.any():
    print(df.loc[viol_buy, ["temp", "deployed_p", "target_p", "base_p"]].head())

# Sells when sell-block was active
viol_sell = (df["deployed_p"].diff() < -1e-3) & (df["block_sell"] == 1)
print("sells_when_blocked:", int(viol_sell.sum()))
if viol_sell.any():
    print(df.loc[viol_sell, ["temp", "deployed_p", "target_p", "base_p", "ret_3", "ret_6", "ret_12", "ret_22"]].head())

print("first_nonzero_deploy_date:", df.index[df["deployed_p"] > 1e-9].min())
print("forced_derisk_events:", int(df["forced_derisk"].sum()))
if (df["forced_derisk"] == 1).any():
    print("first_forced_derisk_dates:", df.index[df["forced_derisk"] == 1][:5].tolist())



