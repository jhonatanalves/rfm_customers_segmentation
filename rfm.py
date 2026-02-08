import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_rfm_table(
    data: pd.DataFrame,
    customer_col: str,
    date_col: str,
    monetary_col: str,
    order_col: str | None,
    approved_col: str | None,
    approved_values: list[str],
) -> pd.DataFrame:
    df = data.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[customer_col, date_col])

    if approved_col and approved_values:
        df = df[df[approved_col].astype(str).isin([str(v) for v in approved_values])]

    df[monetary_col] = pd.to_numeric(df[monetary_col], errors="coerce").fillna(0)

    # Recência
    last_buy = df.groupby(customer_col)[date_col].max().reset_index()
    ref_date = last_buy[date_col].max()
    last_buy["Recencia"] = (ref_date - last_buy[date_col]).dt.days
    last_buy = last_buy[[customer_col, "Recencia"]]

    # Frequência
    if order_col:
        freq = df.groupby(customer_col)[order_col].nunique().reset_index(name="Frequencia")
    else:
        freq = df.groupby(customer_col).size().reset_index(name="Frequencia")

    # Receita
    rev = df.groupby(customer_col)[monetary_col].sum().reset_index(name="Receita")

    rfm = last_buy.merge(freq, on=customer_col).merge(rev, on=customer_col)
    rfm = rfm.rename(columns={customer_col: "Cliente"})
    return rfm


def build_rfm_features(rfm: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Constrói features robustas para clusterização conjunta:
    - R_inv = -Recencia (maior = melhor)
    - F_log = log1p(Frequencia)
    - M_log = log1p(Receita)
    Depois padroniza via StandardScaler.
    Retorna (Xs, rfm_enriquecido).
    """
    out = rfm.copy()
    out["R_inv"] = -out["Recencia"]
    out["F_log"] = np.log1p(out["Frequencia"])
    out["M_log"] = np.log1p(out["Receita"])

    X = out[["R_inv", "F_log", "M_log"]].values
    Xs = StandardScaler().fit_transform(X)
    return Xs, out


def calcular_wcss(Xs: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> list[float]:
    wcss = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=random_state)
        km.fit(Xs)
        wcss.append(float(km.inertia_))
    return wcss


def get_numero_otimo_clusters(wcss: list[float], k_min: int = 2, k_max: int = 10) -> int:
    """
    Knee detection por distância à reta.
    wcss deve corresponder a k_min..k_max.
    """
    x1, y1 = k_min, wcss[0]
    x2, y2 = k_max, wcss[-1]

    distancias = []
    for i, y0 in enumerate(wcss):
        x0 = k_min + i
        numerador = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominador = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distancias.append(numerador / denominador)

    return (k_min + int(np.argmax(distancias)))


def cluster_rfm_joint(
    rfm: pd.DataFrame,
    n_clusters: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    KMeans conjunto no espaço padronizado (R_inv, F_log, M_log).
    Retorna:
    - rfm_clustered: tabela por cliente com ClusterId e ScoreComposto
    - cluster_profile: perfil agregado por cluster (insumo p/ LLM)
    """
    Xs, enriched = build_rfm_features(rfm)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    cluster_id = km.fit_predict(Xs)
    enriched["ClusterId"] = cluster_id

    # Score composto (maior = melhor) no espaço padronizado
    enriched["ScoreComposto"] = Xs.sum(axis=1)

    # Perfil por cluster
    prof = (
        enriched.groupby("ClusterId")
        .agg(
            Clientes=("Cliente", "count"),
            Recencia_media=("Recencia", "mean"),
            Recencia_mediana=("Recencia", "median"),
            Frequencia_media=("Frequencia", "mean"),
            Frequencia_mediana=("Frequencia", "median"),
            Receita_media=("Receita", "mean"),
            Receita_mediana=("Receita", "median"),
            ScoreComposto_medio=("ScoreComposto", "mean"),
        )
        .reset_index()
    )

    # Ordena clusters por ScoreComposto_medio (pior -> melhor)
    prof = prof.sort_values("ScoreComposto_medio", ascending=True).reset_index(drop=True)
    prof["RankQualidade"] = np.arange(len(prof))  # 0 pior, k-1 melhor
    total = prof["Clientes"].sum()
    prof["PctBase"] = (prof["Clientes"] / total).round(4)

    # junta o rank para cada cliente
    enriched = enriched.merge(prof[["ClusterId", "RankQualidade"]], on="ClusterId", how="left")

    # Limpa colunas auxiliares internas de features (mantém se quiser debug)
    enriched = enriched.drop(columns=["R_inv", "F_log", "M_log"], errors="ignore")

    # Reordena colunas principais
    cols = ["Cliente", "Recencia", "Frequencia", "Receita", "ClusterId", "RankQualidade", "ScoreComposto"]
    enriched = enriched[cols]

    # Arredonda perfil para display
    prof_display = prof.copy()
    for c in prof_display.columns:
        if c.endswith("_media") or c.endswith("_mediana") or "ScoreComposto" in c:
            prof_display[c] = prof_display[c].astype(float).round(2)

    return enriched, prof_display