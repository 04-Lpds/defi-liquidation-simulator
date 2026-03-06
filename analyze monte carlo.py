import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Load saved results
results = pd.read_csv("monte_carlo_price_stress_results.csv")  # change path if needed


bucket_order = [
    "1.0–1.5× (Mild)",
    "1.6–2.2× (Moderate)",
    "2.3–3.0× (Severe)",
    "3.1–4.0× (Extreme)",
    "4.1–5.0× (Tail/Capped)"
]

# Convert 'bucket' column to ordered categorical
results['bucket'] = pd.Categorical(
    results['bucket'],
    categories=bucket_order,
    ordered=True
)

sns.set_theme(style="darkgrid", palette="dark")
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='bucket',
    y='final_bad_debt',
    data=results,
    #ax=ax,
    palette="dark",  # or your preferred palette
    boxprops=dict(edgecolor="white"),  # box outline
    medianprops=dict(color="lime"),  # median line
    whiskerprops=dict(color="cyan", linewidth=1.5),  # <--- Whisker color here
    capprops=dict(color="cyan", linewidth=1.5),  # cap (end of whisker) color
    flierprops=dict(marker='o', markerfacecolor='red', markeredgecolor='white')
)
sns.pointplot(
    x='bucket',
    y='final_bad_debt',
    data=results,
    color='red',
    markers='o',
    linestyles='--',
    errorbar=None
)
plt.title(f"Final Bad Debt Distribution by Scaled Price Shock Severity", color='white')
plt.xlabel("Severity Bucket (Scale Factor)", color='white')
plt.ylabel("Final Bad Debt ($)", color='white')
plt.yticks(rotation=0, color='white')
plt.xticks(rotation=70, color='white')
plt.grid(True, alpha=0.3, color='grey')
plt.tight_layout()
plt.tight_layout()
# Save plot as .png
plt.gca().set_facecolor('#1e1e1e')
plt.gcf().set_facecolor('black')
plt.savefig("bad_debt_by_bucket.png", dpi=300, bbox_inches='tight')
print("Chart saved as bad_debt_by_bucket.png")

plt.show()
