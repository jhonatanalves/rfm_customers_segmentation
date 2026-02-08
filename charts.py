import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def _segment_col(rfm_df: pd.DataFrame) -> str:
    return "SegmentoNome" if "SegmentoNome" in rfm_df.columns else "ClusterId"


def scatter_by_group(rfm_df: pd.DataFrame, x: str, y: str) -> None:
    group_col = _segment_col(rfm_df)
    groups = rfm_df[group_col].astype(str).unique().tolist()

    fig, ax = plt.subplots()
    for g in groups:
        part = rfm_df[rfm_df[group_col].astype(str) == str(g)]
        ax.scatter(part[x], part[y], label=str(g), alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def render_scatter_grid(rfm_df: pd.DataFrame) -> None:
    st.subheader("Visualizações (scatter)")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.caption("Recência x Frequência")
        scatter_by_group(rfm_df, "Recencia", "Frequencia")
    with p2:
        st.caption("Frequência x Receita")
        scatter_by_group(rfm_df, "Frequencia", "Receita")
    with p3:
        st.caption("Recência x Receita")
        scatter_by_group(rfm_df, "Recencia", "Receita")