from config import Config
from state import initialize_state
from sim import run_simulation
from metrics import get_research_summary, plot_key_metrics
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


RESULTS_DIR = Path("results")
CHARTS_DIR = RESULTS_DIR / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

dates = [
    "2026-03-26",   # Recent mild-moderate
    "2022-05-12",   # Terra (highly reflexive)
    "2022-06-12",   # Celsius
    "2022-11-08",   # FTX (systemic)
    "2025-10-10"    # Major recent liquidation event
]


def run_alpha_sweep():
    results = []
    beta_fixed = 0.3
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dates = ["2026-03-26", "2025-10-10", "2022-11-08", "2022-06-12", "2022-05-12"]

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = CHARTS_DIR / f"alpha_sweep_{run_timestamp}"
    current_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Charts will be saved to: {current_run_dir}\n")

    print("=== Alpha Sweep Started (beta fixed at 0.3) ===\n")

    for date in dates:
        print(f"\n=== DATE: {date} ===")

        # === Use the same reliable pattern as beta sweep ===
        config = Config()
        config.reload_for_new_date(date)

        config.use_hybrid_oracle = True
        config.oracle_amm_weight = beta_fixed
        config.plot_sim_metrics = False

        for alpha in alphas:
            print(f"  Running α = {alpha}")
            config.oracle_ema_alpha = alpha

            state = initialize_state(config)
            run_simulation(config, state)

            summary = get_research_summary(state, config)
            summary["alpha"] = alpha
            summary["beta"] = beta_fixed
            results.append(summary)

            if getattr(config, 'save_charts', True):
                plot_key_metrics(state, config, save_dir=current_run_dir)

            print(f"    → {summary['total_liquidations']:5,} liqs | Peak {summary['peak_liquidatable_pct']}%")

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values(by=["crisis_date", "alpha"]).reset_index(drop=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = RESULTS_DIR / f"alpha_sweep_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Alpha Sweep completed!")
    print(f"Charts saved to: {current_run_dir}")
    print(f"Results saved to: {csv_path}")
    print("\nSummary:")
    print(df[["crisis_date", "alpha", "total_liquidations", "peak_liquidatable_pct",
              "peak_pending_debt_usd"]].round(2).to_string(index=False))

    # Sensitivity plots
    plot_alpha_sensitivity(df, current_run_dir)

    return df

def plot_alpha_sensitivity(df: pd.DataFrame, save_dir: Path):
    """Similar to beta sensitivity plots"""
    metrics = [
        ("total_liquidations", "Total Liquidations"),
        ("peak_liquidatable_pct", "Peak % Liquidatable"),
        ("peak_pending_debt_usd", "Peak Pending Debt (USD)")
    ]

    colors = ['blue', 'orange', 'green', 'red', 'cyan']
    dates = sorted(df["crisis_date"].unique())

    for col, ylabel in metrics:
        plt.figure(figsize=(10, 6))

        for i, date in enumerate(dates):
            data = df[df["crisis_date"] == date].sort_values("alpha")
            plt.plot(data["alpha"], data[col],
                     marker='o', linewidth=2.5,
                     color=colors[i % len(colors)],
                     label=date)

        plt.xlabel("Alpha (EMA Responsiveness)")
        plt.ylabel(ylabel)
        plt.title(f"Effect of Alpha on {ylabel} (β fixed at 0.3)")
        plt.legend(title="Date", fontsize=10)
        plt.grid(True, alpha=0.3)

        plot_path = save_dir / f"alpha_sweep_{col}.png"
        plt.savefig(plot_path, dpi=220, bbox_inches='tight')
        plt.close()

        print(f"Plot saved: {plot_path.name}")

    print(f"\nAll alpha sensitivity plots saved to: {save_dir}")



def run_beta_sweep():
    results = []
    alpha_fixed = 0.3
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dates = ["2026-03-26", "2025-10-10", "2022-11-08", "2022-06-12", "2022-05-12"]

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = CHARTS_DIR / f"sweep_run_{run_timestamp}"
    current_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Charts will be saved to: {current_run_dir}\n")

    print("=== Beta Sweep Started ===\n")

    for date in dates:
        print(f"\n=== DATE: {date} ===")

        # === Fresh Config + Full Reload ===
        config = Config()
        config.reload_for_new_date(date)  # This handles prices + AMM + borrowers

        config.use_hybrid_oracle = True
        config.oracle_ema_alpha = alpha_fixed
        config.plot_sim_metrics = False

        for beta in betas:
            print(f"  Running β = {beta}")
            config.oracle_amm_weight = beta

            state = initialize_state(config)
            run_simulation(config, state)

            summary = get_research_summary(state, config)
            summary["alpha"] = alpha_fixed
            summary["beta"] = beta
            results.append(summary)

            if getattr(config, 'save_charts', True):
                plot_key_metrics(state, config, save_dir=current_run_dir)

            print(f"    → {summary['total_liquidations']:5,} liqs | Peak {summary['peak_liquidatable_pct']}%")

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values(by=["crisis_date", "beta"]).reset_index(drop=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    csv_path = RESULTS_DIR / f"beta_sweep_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nParameter Sweep completed.")
    print(f"Charts saved to: {current_run_dir}")
    print(f"Results saved to: {csv_path}")
    print("\nSummary:")
    print(df[["crisis_date", "beta", "total_liquidations", "peak_liquidatable_pct",
              "peak_pending_debt_usd"]].round(2).to_string(index=False))

    # Sensitivity plots:
    plot_beta_sensitivity(df, current_run_dir)
    return df


def plot_beta_sensitivity(df: pd.DataFrame, save_dir: Path):
    metrics = [
        ("total_liquidations", "Total Liquidations"),
        ("peak_liquidatable_pct", "Peak % Liquidatable"),
        ("peak_pending_debt_usd", "Peak Pending Debt (USD)")
    ]

    colors = ['blue', 'orange', 'green', 'red', 'cyan']
    dates = sorted(df["crisis_date"].unique())

    for col, ylabel in metrics:
        plt.figure(figsize=(10, 6))

        for i, date in enumerate(dates):
            data = df[df["crisis_date"] == date].sort_values("beta")
            plt.plot(data["beta"], data[col],
                     marker='o',
                     linewidth=2.5,
                     color=colors[i % len(colors)],
                     label=date)

        plt.xlabel("Beta (AMM Weight / Reflexivity)")
        plt.ylabel(ylabel)
        plt.title(f"Effect of Beta on {ylabel}")
        plt.legend(title="Date", fontsize=10)
        plt.grid(True, alpha=0.3)

        plot_path = save_dir / f"beta_sweep_{col}.png"
        plt.savefig(plot_path, dpi=220, bbox_inches='tight')
        plt.close()

        print(f"Plot saved: {plot_path.name}")

    print(f"\nAll sensitivity plots (with all dates) saved to: {save_dir}")


from tqdm import tqdm
import time


def run_2d_sweep():
    results = []

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    betas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    dates = ["2026-03-26", "2022-05-12", "2022-06-12", "2022-11-08", "2025-10-10"]

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_dir = CHARTS_DIR / f"2d_sweep_{run_timestamp}"
    current_run_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(dates) * len(alphas) * len(betas)
    print(f"Starting 2D Sweep → {total_runs} simulations")
    print(f"Output folder: {current_run_dir}\n")

    start_time = time.time()

    # Main progress bar
    with tqdm(total=total_runs, desc="2D Sweep Progress", unit="sim") as pbar:
        for date in dates:
            print(f"\n=== DATE: {date} ===")

            config = Config()
            config.reload_for_new_date(date)

            config.use_hybrid_oracle = True
            config.plot_sim_metrics = False

            for alpha in alphas:
                config.oracle_ema_alpha = alpha

                for beta in betas:
                    config.oracle_amm_weight = beta

                    state = initialize_state(config)
                    run_simulation(config, state)

                    summary = get_research_summary(state, config)
                    summary["alpha"] = alpha
                    summary["beta"] = beta
                    results.append(summary)

                    # Update progress bar
                    pbar.update(1)

                    # Optional: show current ETA
                    elapsed = time.time() - start_time
                    eta = (elapsed / pbar.n) * (total_runs - pbar.n) / 60
                    pbar.set_postfix({
                        'date': date[-5:],
                        'α': f"{alpha:.1f}",
                        'β': f"{beta:.1f}",
                        'ETA': f"{eta:.1f}min"
                    })

    # Save results
    df = pd.DataFrame(results)
    df = df.sort_values(by=["crisis_date", "alpha", "beta"]).reset_index(drop=True)

    csv_path = RESULTS_DIR / f"2d_sweep_results_{run_timestamp}.csv"
    df.to_csv(csv_path, index=False)

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ 2D Sweep completed in {total_time:.1f} minutes!")
    print(f"Results saved to: {csv_path}")

    # Generate visualizations
    print("\nGenerating plots...")
    plot_2d_heatmaps(df, current_run_dir)  # Individual heatmaps
    plot_2d_multi_panel(df, current_run_dir)  # Big 2D panel
    plot_2d_3d_surfaces(df, current_run_dir)  # Static 3D surfaces
    #plot_interactive_3d(df, current_run_dir)  # Interactive HTML plots

    return df


def plot_2d_heatmaps(df: pd.DataFrame, save_dir: Path):
    """Create heatmaps for key metrics showing Alpha vs Beta interaction"""

    # Pivot the data for heatmaps
    metrics = {
        "total_liquidations": "Total Liquidations",
        "peak_liquidatable_pct": "Peak % Liquidatable",
        "peak_pending_debt_usd": "Peak Pending Debt (USD)"
    }

    for date in df["crisis_date"].unique():
        date_df = df[df["crisis_date"] == date]

        for col, title in metrics.items():
            # Create pivot table
            pivot = date_df.pivot(index="alpha", columns="beta", values=col)

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt=".0f" if col != "peak_liquidatable_pct" else ".1f",
                        cmap="YlOrRd", linewidths=0.5)

            plt.title(f"{title}\n{date} (Higher = Worse)")
            plt.xlabel("Beta (AMM Reflexivity)")
            plt.ylabel("Alpha (EMA Responsiveness)")

            plot_path = save_dir / f"heatmap_{date}_{col}.png"
            plt.savefig(plot_path, dpi=220, bbox_inches='tight')
            plt.close()

            print(f"Heatmap saved: {plot_path.name}")

    print(f"\nAll 2D heatmaps saved to: {save_dir}")


def plot_2d_multi_panel(df: pd.DataFrame, save_dir: Path):
    """Creates one big multi-panel figure for the 2D sweep"""
    dates = sorted(df["crisis_date"].unique())
    metrics = [
        ("total_liquidations", "Total Liquidations"),
        ("peak_liquidatable_pct", "Peak % Liquidatable"),
        ("peak_pending_debt_usd", "Peak Pending Debt (USD)")
    ]

    n_dates = len(dates)
    fig, axes = plt.subplots(len(metrics), n_dates, figsize=(4 * n_dates, 4 * len(metrics)), dpi=200)

    if n_dates == 1:
        axes = axes.reshape(-1, 1)

    for row, (col, title) in enumerate(metrics):
        for col_idx, date in enumerate(dates):
            ax = axes[row, col_idx]
            date_df = df[df["crisis_date"] == date]
            pivot = date_df.pivot(index="alpha", columns="beta", values=col)

            sns.heatmap(pivot, annot=True, fmt=".0f" if "liquidations" in col else ".1f",
                        cmap="YlOrRd", linewidths=0.5, ax=ax)

            ax.set_title(f"{date}\n{title}")
            ax.set_xlabel("Beta (Reflexivity)")
            if col_idx == 0:
                ax.set_ylabel(f"Alpha (EMA)\n{title}")
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    plot_path = save_dir / "2d_multi_panel_heatmap.png"
    plt.savefig(plot_path, dpi=220, bbox_inches='tight')
    plt.close()

    print(f"Big multi-panel heatmap saved: {plot_path.name}")


def plot_2d_3d_surfaces(df: pd.DataFrame, save_dir: Path):
    """3D surface plots for key metrics"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    metrics = [
        ("total_liquidations", "Total Liquidations"),
        ("peak_liquidatable_pct", "Peak % Liquidatable"),
        ("peak_pending_debt_usd", "Peak Pending Debt (USD)")
    ]

    for date in df["crisis_date"].unique():
        date_df = df[df["crisis_date"] == date]

        for col, title in metrics:
            # Prepare grid
            pivot = date_df.pivot(index="alpha", columns="beta", values=col)
            X = pivot.columns.values
            Y = pivot.index.values
            X, Y = np.meshgrid(X, Y)
            Z = pivot.values

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            # Surface plot
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            ax.set_xlabel('Beta (AMM Reflexivity)')
            ax.set_ylabel('Alpha (EMA Responsiveness)')
            ax.set_zlabel(title)
            ax.set_title(f'{title} - {date}')

            plt.savefig(save_dir / f"3d_surface_{date}_{col}.png", dpi=200, bbox_inches='tight')
            plt.close()

            print(f"3D surface saved: {date} - {col}")


def plot_2d_3d_multi_surface(df: pd.DataFrame, save_dir: Path):
    """Dark-themed 3D multi-surface plots with all dates"""
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    metrics = [
        ("total_liquidations", "Total Liquidations"),
        ("peak_liquidatable_pct", "Peak % Liquidatable"),
        ("peak_pending_debt_usd", "Peak Pending Debt (USD)")
    ]

    dates = sorted(df["crisis_date"].unique())
    colors = ['cyan', 'magenta', 'lime', 'yellow', 'orange']

    for col, title in metrics:
        fig = plt.figure(figsize=(14, 10), facecolor='#0a0a0a')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#111111')
        fig.patch.set_facecolor('#0a0a0a')

        for i, date in enumerate(dates):
            date_df = df[df["crisis_date"] == date]
            pivot = date_df.pivot(index="alpha", columns="beta", values=col)

            X = pivot.columns.values
            Y = pivot.index.values
            X, Y = np.meshgrid(X, Y)
            Z = pivot.values

            ax.plot_surface(X, Y, Z,
                            alpha=0.75,
                            color=colors[i % len(colors)],
                            label=date,
                            edgecolor='none',
                            linewidth=0,
                            antialiased=True,
                            shade=True,  # enables basic lighting
                            rstride=1, cstride=1)

        # Styling
        ax.set_xlabel('Beta (AMM Reflexivity)', color='white', labelpad=15)
        ax.set_ylabel('Alpha (EMA Responsiveness)', color='white', labelpad=15)
        ax.set_zlabel(title, color='white', labelpad=15)
        ax.set_title(f'3D View: {title} Across Dates', color='white', pad=30, fontsize=16)

        ax.grid(True, alpha=0.2)

        # Legend
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=colors[i % len(colors)], alpha=0.75)
                 for i in range(len(dates))]
        ax.legend(proxy, dates, title="Date", loc='upper left', fontsize=10,
                  title_fontsize=11, facecolor='#1e1e1e', edgecolor='white', labelcolor='white')

        plt.savefig(save_dir / f"3d_multi_surface_{col}_dark.png",
                    dpi=250, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print(f"✅ Saved dark 3D multi-surface: {col}")

def plot_interactive_3d(df: pd.DataFrame, save_dir: Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics = ["total_liquidations", "peak_liquidatable_pct", "peak_pending_debt_usd"]
    titles = ["Total Liquidations", "Peak % Liquidatable", "Peak Pending Debt (USD)"]

    for date in df["crisis_date"].unique():
        date_df = df[df["crisis_date"] == date]

        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=titles
        )

        for i, col in enumerate(metrics):
            pivot = date_df.pivot(index="alpha", columns="beta", values=col)
            X = pivot.columns.values
            Y = pivot.index.values
            X, Y = np.meshgrid(X, Y)
            Z = pivot.values

            fig.add_trace(
                go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', showscale=False),
                row=1, col=i + 1
            )

        fig.update_layout(
            title=f"3D Parameter Space - {date}",
            height=600,
            width=1400,
            scene=dict(xaxis_title='Beta', yaxis_title='Alpha', zaxis_title='Metric')
        )

        html_path = save_dir / f"interactive_3d_{date}.html"
        fig.write_html(str(html_path))
        print(f"Interactive 3D plot saved: {html_path.name}")

# ====================== LOAD EXISTING 2D RESULTS & PLOT ======================

def plot_from_existing_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} simulations from {csv_path}")
    print(df[["crisis_date", "alpha", "beta"]].drop_duplicates().head())

    save_dir = Path(csv_path).parent / "3d_plots"
    save_dir.mkdir(exist_ok=True)

    plot_2d_3d_surfaces(df, save_dir)  # Your 3D surfaces
    plot_2d_multi_panel(df, save_dir)  # The big 2D heatmap panel
    plot_2d_3d_multi_surface(df, save_dir)

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    #run_beta_sweep()
    #run_alpha_sweep()
    run_2d_sweep()
    csv_file = "results/2d_sweep_results_20260512_021644.csv"  # ← Change this
    plot_from_existing_csv(csv_file)