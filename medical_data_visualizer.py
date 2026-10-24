
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# 1. Import the data
df = pd.read_csv("medical_examination.csv")


# 2. Add an overweight column
df["overweight"] = (
    df["weight"] / ((df["height"] / 100) ** 2) > 25
).astype(int)


# 3. Normalize cholesterol and glucose
df["cholesterol"] = (df["cholesterol"] > 1).astype(int)
df["gluc"] = (df["gluc"] > 1).astype(int)


# 4. Draw the categorical plot
def draw_cat_plot():
    # 5. Create DataFrame for cat plot
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight",
        ],
    )

    # 6. Group and reformat the data
    df_cat = (
        df_cat.groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7. Draw the categorical plot
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
    )

    fig.set_axis_labels("variable", "total")
    fig.set_titles("cardio = {col_name}")
    fig = fig.fig

    # 8. Get the figure
    # 9. Do not modify the next two lines
    fig.savefig("catplot.png")
    return fig


# 10. Draw the heat map
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Plot the correlation matrix
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        center=0,
        vmax=0.3,
        vmin=-0.1,
    )

    # 16. Do not modify the next two lines
    fig.savefig("heatmap.png")
    return fig
