import pandas as pd
import streamlit as st

from rfm import (
    build_rfm_table,
    calcular_wcss,
    get_numero_otimo_clusters,
    cluster_rfm_joint,
)
from charts import render_scatter_grid
from llm_gemini import (
    list_gemini_models,
    build_cluster_naming_prompt,
    gemini_generate_json,
    build_cluster_labels,
)

st.set_page_config(page_title="RFM Segmentation", layout="wide")

# Estado
for key, default in {
    "rfm_out": None,
    "cluster_profile": None,
    "cluster_labels": None,
    "gemini_models": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# Sidebar
with st.sidebar:
    st.header("📂 Dataset")
    up = st.file_uploader("Upload CSV", type="csv")
    sep = st.selectbox("Separador", [",", ";", "\t"], index=0)

    st.header("⚙️ Clusterização (KMeans conjunto)")
    k_mode = st.radio("Como escolher k?", ["Automático (cotovelo)", "Manual"], index=0)

    k_min, k_max = 2, 10
    if k_mode == "Manual":
        n_clusters = st.slider("Clusters (k)", 2, 10, 4)
    else:
        st.caption("Automático usa WCSS + knee detection no intervalo 2..10.")
        n_clusters = None

    random_state = st.number_input("Random state", value=42, step=1)

    st.header("🏢 Contexto do negócio")
    business_context = st.text_area(
        "Descreva o negócio/setor (opcional)",
        placeholder="Ex: E-commerce de moda feminina; ticket médio R$120; foco em recompra.\nEx: SaaS B2B; modelo de assinatura; foco em upgrade e retenção.",
        height=120,
    )

    st.header("🤖 Gemini (BYOK)")
    api_key = st.text_input("Gemini API Key", type="password")

    if st.button("Carregar modelos", disabled=not api_key):
        try:
            st.session_state["gemini_models"] = list_gemini_models(api_key)
            st.success("Modelos carregados")
        except Exception as e:
            st.session_state["gemini_models"] = []
            st.error(str(e))

    model = None
    if st.session_state["gemini_models"]:
        model = st.selectbox("Modelo", st.session_state["gemini_models"])

    # Persistir no estado (para não perder no rerun)
    st.session_state["business_context"] = business_context.strip()


# Main
st.title("📊 Segmentação RFM (clusterização conjunta + nomes descritivos)")

if not up:
    st.info("Faça upload de um CSV para começar.")
    st.stop()

df = pd.read_csv(up, sep=sep)
st.subheader("Prévia do dataset")
st.dataframe(df.head(20), use_container_width=True)

cols = df.columns.tolist()

with st.form("rfm_form"):
    st.subheader("Mapeamento de colunas")
    customer_col = st.selectbox("Cliente", cols)
    date_col = st.selectbox("Data", cols)
    monetary_col = st.selectbox("Valor", cols)

    order_col = st.selectbox("Pedido (opcional)", ["(não tenho)"] + cols)
    order_col = None if order_col == "(não tenho)" else order_col

    approved_col = st.selectbox("Status (opcional)", ["(não tenho)"] + cols)
    approved_col = None if approved_col == "(não tenho)" else approved_col

    approved_values = []
    if approved_col:
        uniq = pd.Series(df[approved_col].astype(str).unique()).sort_values().tolist()
        approved_values = st.multiselect("Valores aprovados", uniq)

    submit = st.form_submit_button("🚀 Rodar RFM + Clusterização")


if submit:
    try:
        rfm_table = build_rfm_table(
            data=df,
            customer_col=customer_col,
            date_col=date_col,
            monetary_col=monetary_col,
            order_col=order_col,
            approved_col=approved_col,
            approved_values=approved_values,
        )

        # escolher k (se automático)
        chosen_k = n_clusters
        if chosen_k is None:
            # Para WCSS precisamos do X padronizado; cluster_rfm_joint já faz internamente,
            # mas aqui faremos via chamada leve: cluster_rfm_joint exige k. Então:
            # -> calculamos WCSS em um X construído dentro do próprio rfm.py? Mantive simples:
            #    fazemos um "cluster_rfm_joint" só no k=2..10? Não: vamos usar helpers do rfm.py.
            # Como precisamos de Xs, usamos o próprio cluster_rfm_joint depois. Para WCSS, construímos Xs via rfm.py:
            from rfm import build_rfm_features  # import local para evitar circularidade
            Xs, _ = build_rfm_features(rfm_table)
            wcss = calcular_wcss(Xs, k_min=2, k_max=10, random_state=int(random_state))
            chosen_k = get_numero_otimo_clusters(wcss, k_min=2, k_max=10)

        rfm_out, cluster_profile = cluster_rfm_joint(
            rfm=rfm_table,
            n_clusters=int(chosen_k),
            random_state=int(random_state),
        )

        st.session_state["rfm_out"] = rfm_out
        st.session_state["cluster_profile"] = cluster_profile
        st.session_state["cluster_labels"] = None  # invalida rótulos antigos se re-rodar

        st.success(f"Clusterização concluída! k = {int(chosen_k)}")

    except Exception as e:
        st.error(f"Erro ao rodar pipeline: {e}")
        st.stop()


rfm_out = st.session_state.get("rfm_out")
cluster_profile = st.session_state.get("cluster_profile")

if rfm_out is None or cluster_profile is None:
    st.info("Configure o mapeamento e clique em **Rodar RFM + Clusterização**.")
    st.stop()


st.subheader("Perfil dos clusters (agregado)")
st.dataframe(cluster_profile, use_container_width=True)

st.subheader("Resultado por cliente (com ClusterId)")
st.dataframe(rfm_out, use_container_width=True)

st.download_button(
    "⬇️ Baixar CSV (clientes)",
    rfm_out.to_csv(index=False).encode("utf-8"),
    "rfm_clientes.csv",
    "text/csv",
)

st.download_button(
    "⬇️ Baixar CSV (perfil clusters)",
    cluster_profile.to_csv(index=False).encode("utf-8"),
    "rfm_clusters_perfil.csv",
    "text/csv",
)

st.divider()
st.header("🤖 Nomear e explicar clusters (Gemini)")

if not api_key or not model:
    st.info("Configure a Gemini API Key e selecione um modelo na barra lateral.")
else:
    if st.button("Gerar nomes e estratégias (LLM)", type="primary"):
        business_context = st.session_state.get("business_context", "")
        prompt = build_cluster_naming_prompt(cluster_profile, business_context)


        with st.expander("🔎 Prompt enviado (agregados)", expanded=False):
            st.code(prompt)

        try:
            with st.spinner("Chamando Gemini..."):
                llm_json = gemini_generate_json(api_key=api_key, model=model, prompt=prompt)
                labels_df = build_cluster_labels(cluster_profile, llm_json)

            st.session_state["cluster_labels"] = labels_df
            st.success("Rótulos gerados com sucesso!")

        except Exception as e:
            st.error(f"Falha na geração de rótulos: {e}")


labels_df = st.session_state.get("cluster_labels")

if labels_df is None:
    st.info("Gere os nomes via Gemini para enriquecer os gráficos e as descrições.")
else:
    # Merge dos rótulos nos clientes e no perfil
    rfm_named = rfm_out.merge(labels_df, on="ClusterId", how="left")
    prof_named = cluster_profile.merge(labels_df, on="ClusterId", how="left")

    st.subheader("Clusters nomeados (LLM)")
    st.dataframe(prof_named, use_container_width=True)

    st.subheader("Explicações por segmento")
    for _, row in prof_named.sort_values("RankQualidade").iterrows():
        nome = row.get("SegmentoNome", f"Cluster {int(row['ClusterId'])}")
        desc = row.get("SegmentoDescricao", "")
        estrategias = row.get("Estrategias", [])
        st.markdown(f"### {nome}")
        if desc:
            st.write(desc)
        if isinstance(estrategias, list) and estrategias:
            st.write("**Ações sugeridas:**")
            for a in estrategias:
                st.write(f"- {a}")
        st.write("---")

    st.divider()
    st.header("📈 Gráficos (com rótulos descritivos)")
    render_scatter_grid(rfm_named)