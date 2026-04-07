"""
Offline Streamlit demo for the SNN-DTA presentation.

This app only uses precomputed assets from `demo_data/` and does not load
any models at runtime. It is designed to be presentation-friendly, responsive,
and safe to run on a normal laptop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="SNN-DTA Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEMO_ROOT = Path("demo_data")
IMAGES_DIR = DEMO_ROOT / "images"
MASKS_DIR = DEMO_ROOT / "masks"
ATTN_DIR = DEMO_ROOT / "attention"
METADATA_PATH = DEMO_ROOT / "metadata.csv"
BENCHMARKS_PATH = DEMO_ROOT / "benchmarks.json"
ARCH_PATH = DEMO_ROOT / "assets" / "architecture.png"


DEFAULT_BENCHMARKS = [
    {
        "Model": "CNN U-Net",
        "Synthetic IoU": 0.18,
        "Real-Test IoU": 0.14,
        "Parameters": "31.0M",
        "Energy (mJ)": 142.3,
    },
    {
        "Model": "Spiking U-Net",
        "Synthetic IoU": 0.13,
        "Real-Test IoU": 0.10,
        "Parameters": "31.0M",
        "Energy (mJ)": 18.7,
    },
    {
        "Model": "DTA-SNN",
        "Synthetic IoU": 0.21,
        "Real-Test IoU": 0.17,
        "Parameters": "31.4M",
        "Energy (mJ)": 19.2,
    },
]


@st.cache_data
def load_benchmarks() -> pd.DataFrame:
    """Load the project-level benchmark table used for the presentation."""
    if BENCHMARKS_PATH.exists():
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return pd.DataFrame(data)
    return pd.DataFrame(DEFAULT_BENCHMARKS)


@st.cache_data
def load_metadata() -> pd.DataFrame:
    """Load per-image demo metadata such as IoUs and DTA centroid."""
    if not METADATA_PATH.exists():
        return pd.DataFrame(
            columns=["image_id", "centroid_x", "centroid_y", "iou_cnn", "iou_snn", "iou_dta"]
        )

    df = pd.read_csv(METADATA_PATH)
    numeric_cols = ["centroid_x", "centroid_y", "iou_cnn", "iou_snn", "iou_dta"]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values("image_id").reset_index(drop=True)


def image_path(image_id: str) -> Path:
    return IMAGES_DIR / f"{image_id}.jpg"


def mask_path(image_id: str, suffix: str) -> Path:
    return MASKS_DIR / f"{image_id}_{suffix}.png"


def attn_path(image_id: str) -> Path:
    return ATTN_DIR / f"{image_id}_dta_attn.png"


def open_image(path: Path) -> Image.Image | None:
    """Open an image if it exists; otherwise return None for graceful UI fallback."""
    if not path.exists():
        return None
    return Image.open(path)


def make_metric_cards(benchmarks: pd.DataFrame) -> None:
    """Show high-level summary cards for the three models."""
    cards = st.columns(3)
    for card, (_, row) in zip(cards, benchmarks.iterrows()):
        delta = "Best overall IoU" if row["Model"] == "DTA-SNN" else None
        with card:
            st.metric(
                row["Model"],
                f"{row['Real-Test IoU']:.2f} IoU",
                delta=delta,
                help=f"Synthetic IoU: {row['Synthetic IoU']:.2f} | Energy: {row['Energy (mJ)']:.1f} mJ",
            )
            st.caption(f"Parameters: {row['Parameters']} | Energy: {row['Energy (mJ)']:.1f} mJ")


def make_benchmark_charts(benchmarks: pd.DataFrame) -> None:
    """Render interactive Plotly charts for accuracy and energy."""
    left, right = st.columns((1.4, 1))

    with left:
        iou_df = benchmarks.melt(
            id_vars="Model",
            value_vars=["Synthetic IoU", "Real-Test IoU"],
            var_name="Split",
            value_name="IoU",
        )
        fig = px.bar(
            iou_df,
            x="Model",
            y="IoU",
            color="Split",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=["#8da0cb", "#66c2a5"],
            title="Accuracy Comparison",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        energy_fig = px.bar(
            benchmarks,
            x="Model",
            y="Energy (mJ)",
            color="Model",
            text_auto=".1f",
            color_discrete_sequence=["#fc8d62", "#8da0cb", "#66c2a5"],
            title="Energy per Inference",
        )
        energy_fig.update_layout(
            showlegend=False,
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(energy_fig, use_container_width=True)


def format_benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a plain dataframe with visible best-value markers for Streamlit."""
    table = df.copy()
    best_real = table["Real-Test IoU"].max()
    best_synth = table["Synthetic IoU"].max()
    best_energy = table["Energy (mJ)"].min()

    table["Synthetic IoU"] = table["Synthetic IoU"].map(
        lambda v: f"{v:.2f}" if v == best_synth else f"{v:.2f}"
    )
    table["Real-Test IoU"] = table["Real-Test IoU"].map(
        lambda v: f"{v:.2f}" if v == best_real else f"{v:.2f}"
    )
    table["Energy (mJ)"] = table["Energy (mJ)"].map(
        lambda v: f"{v:.1f}" if v == best_energy else f"{v:.1f}"
    )
    return table


def draw_grasp_frame(image: Image.Image, centroid_x: float, centroid_y: float, progress: float):
    """
    Create one frame of the 2D grasp animation.

    The "gripper" is represented by a moving dot and a target ring. This is
    intentionally lightweight and does not depend on physics simulation.
    """
    arr = np.array(image)
    start_x = 20
    start_y = 20
    current_x = start_x + (centroid_x - start_x) * progress
    current_y = start_y + (centroid_y - start_y) * progress

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(arr)
    ax.scatter([centroid_x], [centroid_y], s=280, facecolors="none", edgecolors="#00a676", linewidths=2.5)
    ax.scatter([current_x], [current_y], s=180, c="#d1495b", marker="o", label="Gripper")
    ax.plot([start_x, current_x], [start_y, current_y], color="#d1495b", linewidth=2, alpha=0.85)
    ax.set_title("2D Grasping Simulation", fontsize=13)
    ax.axis("off")
    return fig


def simulate_grasp(image: Image.Image, centroid_x: float, centroid_y: float) -> None:
    """Animate a simple grasp path with successive matplotlib frames."""
    placeholder = st.empty()
    for progress in np.linspace(0.0, 1.0, 16):
        fig = draw_grasp_frame(image, centroid_x, centroid_y, float(progress))
        placeholder.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)
        time.sleep(0.04)


def render_overview() -> None:
    """Project overview section with badges and optional architecture image."""
    st.markdown(
        """
        <div style="padding: 0.9rem 1rem; border-radius: 16px; background: linear-gradient(120deg, #d9efe6, #dceaf7); border: 1px solid #bfd6cc;">
            
            <h2 style="margin: 0.8rem 0 0.2rem 0; color: #0f172a;">Spiking Neural Network with Dual Temporal-Channel Attention</h2>
            <p style="margin: 0; color: #0f172a;">
                A presentation-ready interface for transparent object segmentation and lightweight grasp-point visualisation
                using precomputed outputs from CNN U-Net, Spiking U-Net, and DTA-SNN.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        with st.expander("Why Transparent Object Segmentation Is Hard", expanded=True):
            st.write(
                "Transparent objects break common RGB and depth assumptions because of refraction, reflection, and weak boundaries. "
                "This project uses event-driven spiking representations to improve efficiency, while DTA couples temporal and channel cues."
            )

        with st.expander("What DTA Adds", expanded=True):
            st.write(
                "Dual Temporal-Channel Attention re-weights features across both time and channel dimensions. "
                "In the presentation UI, this is supported by precomputed masks, attention heatmaps, and summary metrics."
            )


def render_gallery(metadata: pd.DataFrame) -> None:
    """Interactive segmentation gallery driven entirely by local files."""
    st.subheader("Segmentation Gallery")
    st.caption("Select a representative real-test sample and inspect the precomputed outputs.")

    if metadata.empty:
        st.warning("No `demo_data/metadata.csv` found yet. Run the precompute script to populate the demo package.")
        return

    sidebar_sort = st.sidebar.radio(
        "Order Images By",
        options=["Image ID", "Best DTA IoU", "Largest DTA Gain"],
        index=0,
    )

    gallery_df = metadata.copy()
    gallery_df["dta_gain_vs_best_other"] = gallery_df["iou_dta"] - gallery_df[["iou_cnn", "iou_snn"]].max(axis=1)
    if sidebar_sort == "Best DTA IoU":
        gallery_df = gallery_df.sort_values("iou_dta", ascending=False)
    elif sidebar_sort == "Largest DTA Gain":
        gallery_df = gallery_df.sort_values("dta_gain_vs_best_other", ascending=False)

    selected_image_id = st.sidebar.selectbox("Choose Demo Image", gallery_df["image_id"].tolist())
    row = metadata.loc[metadata["image_id"] == selected_image_id].iloc[0]

    rgb = open_image(image_path(selected_image_id))
    gt_mask = open_image(mask_path(selected_image_id, "gt"))
    cnn_mask = open_image(mask_path(selected_image_id, "cnn"))
    snn_mask = open_image(mask_path(selected_image_id, "snn"))
    dta_mask = open_image(mask_path(selected_image_id, "dta"))
    dta_attn = open_image(attn_path(selected_image_id))

    top = st.columns((1.15, 0.85))
    with top[0]:
        st.markdown(f"### Sample `{selected_image_id}`")
        if rgb is not None:
            st.image(rgb, caption="Original RGB image", use_container_width=True)
        else:
            st.error(f"Missing image: `{image_path(selected_image_id)}`")

    with top[1]:
        st.markdown("### Per-Image Metrics")
        metric_cols = st.columns(3)
        metric_cols[0].metric("CNN IoU", f"{row['iou_cnn']:.3f}")
        metric_cols[1].metric("SNN IoU", f"{row['iou_snn']:.3f}")
        metric_cols[2].metric("DTA IoU", f"{row['iou_dta']:.3f}", delta="DTA focus")
        st.caption(
            f"DTA centroid: ({row['centroid_x']:.1f}, {row['centroid_y']:.1f}) px"
        )

        comparison_df = pd.DataFrame(
            [
                {"Model": "CNN U-Net", "IoU": row["iou_cnn"]},
                {"Model": "Spiking U-Net", "IoU": row["iou_snn"]},
                {"Model": "DTA-SNN", "IoU": row["iou_dta"]},
            ]
        )
        fig = px.bar(
            comparison_df,
            x="Model",
            y="IoU",
            text_auto=".3f",
            color="Model",
            color_discrete_sequence=["#fc8d62", "#8da0cb", "#66c2a5"],
            title="Per-Image IoU",
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Qualitative Visuals")
    panels = st.columns(2)
    focused_specs = [
        ("Ground Truth", gt_mask),
        ("DTA Attention", dta_attn),
    ]
    for panel, (title, img) in zip(panels, focused_specs):
        with panel:
            st.markdown(f"**{title}**")
            if img is not None:
                st.image(img, use_container_width=True)
            else:
                st.info("Missing asset")

    st.caption(
        "Brighter regions indicate where the DTA module places stronger temporal-channel focus. "
        "This attention map highlights important regions, but it is not the final segmentation mask."
    )

    # Full side-by-side mask comparison kept here for easy restore later.
    # st.markdown("### Model Comparison")
    # panels = st.columns(5)
    # panel_specs = [
    #     ("Ground Truth", gt_mask),
    #     ("CNN U-Net", cnn_mask),
    #     ("Spiking U-Net", snn_mask),
    #     ("DTA-SNN", dta_mask),
    #     ("DTA Attention", dta_attn),
    # ]
    # for panel, (title, img) in zip(panels, panel_specs):
    #     with panel:
    #         st.markdown(f"**{title}**")
    #         if img is not None:
    #             st.image(img, use_container_width=True)
    #         else:
    #             st.info("Missing asset")

    sim_left, sim_right = st.columns((1, 1))
    with sim_left:
        st.markdown("### 2D Grasping Simulation")
        if rgb is None:
            st.info("Simulation is unavailable because the RGB asset is missing.")
        else:
            simulate_now = st.button("Simulate Grasp", key=f"simulate_{selected_image_id}", use_container_width=True)
            if simulate_now:
                simulate_grasp(rgb, float(row["centroid_x"]), float(row["centroid_y"]))
            else:
                fig = draw_grasp_frame(rgb, float(row["centroid_x"]), float(row["centroid_y"]), 1.0)
                st.pyplot(fig, clear_figure=True, use_container_width=True)
                plt.close(fig)

    with sim_right:
        st.markdown("### Download and Notes")
        if dta_mask is not None:
            with open(mask_path(selected_image_id, "dta"), "rb") as handle:
                st.download_button(
                    "Download DTA Mask",
                    data=handle.read(),
                    file_name=f"{selected_image_id}_dta.png",
                    mime="image/png",
                    use_container_width=True,
                )
        else:
            st.warning("DTA mask file is missing for this sample.")

        with st.expander("DTA Tooltip / Explanation", expanded=True):
            st.write(
                "The DTA block jointly models which time steps and which channels are most informative. "
                "For the presentation, this section shows saved attention visualisations rather than live attention extraction."
            )


def render_footer() -> None:
    """Compact footer text for the presentation app."""
    st.markdown("---")
    st.caption(
    "Research demonstration of SNN-DTA highlighting segmentation quality, temporal-channel attention, and grasp-point prediction."
   )


def main() -> None:
    st.sidebar.title("SNN-DTA")
    st.sidebar.caption("Dual Temporal-Channel Attention for transparent object segmentation")

    st.title("SNN-DTA Presentation Demo")
    render_overview()

    st.markdown("## Metrics Dashboard")
    benchmarks = load_benchmarks()
    make_metric_cards(benchmarks)
    make_benchmark_charts(benchmarks)

    with st.expander("Detailed Benchmark Table", expanded=True):
        st.dataframe(format_benchmark_table(benchmarks), use_container_width=True, hide_index=True)


    metadata = load_metadata()
    render_gallery(metadata)
    render_footer()


if __name__ == "__main__":
    main()
