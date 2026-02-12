import pandas as pd
from AutoClean import AutoClean

def apply_autoclean(
    df: pd.DataFrame,
    tratar_duplicados: bool,
    imputacao: str,           # "auto" | "median" | "most_frequent"
    tratar_outliers: bool,
    extrair_datas: bool,
) -> pd.DataFrame:
    """
    Aplica AutoClean em modo manual, configurando:
    - duplicates: True/False
    - missing_num: "auto" | "median" | "most_frequent"
    - missing_categ: "auto" | "most_frequent"
    - outliers: "auto" (quando ligado) / False
    - extract_datetime: "auto" (quando ligado) / False
    """

    if imputacao == "auto":
        missing_num = "auto"
        missing_categ = "auto"
    elif imputacao == "median":
        missing_num = "median"
        missing_categ = "most_frequent"
    elif imputacao == "most_frequent":
        missing_num = "most_frequent"
        missing_categ = "most_frequent"
    else:
        raise ValueError("Imputação inválida.")

    pipeline = AutoClean(
        df,
        mode="manual",
        duplicates=True if tratar_duplicados else False,
        missing_num=missing_num,
        missing_categ=missing_categ,
        outliers="auto" if tratar_outliers else False,
        extract_datetime="auto" if extrair_datas else False,
        logfile=False,
        verbose=False,
    )
    return pipeline.output