import json
import pandas as pd
from google import genai
from google.genai import errors as genai_errors


def list_gemini_models(api_key: str) -> list[str]:
    client = genai.Client(api_key=api_key)
    models: list[str] = []
    for m in client.models.list():
        if hasattr(m, "supported_actions") and "generateContent" in m.supported_actions:
            models.append(m.name.replace("models/", ""))
    return sorted(set(models))



def build_cluster_naming_prompt(cluster_profile: pd.DataFrame, business_context: str) -> str:
    prof_csv = cluster_profile.to_csv(index=False)

    contexto = business_context.strip()
    if not contexto:
        contexto = "Não informado."

    prompt = f"""
Você é um especialista em CRM, retenção de clientes e análise de dados, com foco em otimizar estratégias de negócios.

#Contexto:
{contexto}

#Tarefa:
Analisar dados agregados de clusters de clientes, segmentados por RFM (Recência, Frequência, Valor Monetário). Para cada cluster, realizar:
Nomeação: Criar um nome conciso e descritivo do comportamento do segmento.
Descrição: Elaborar uma descrição detalhada (4-6 linhas) do perfil do cluster, incluindo estatística descritiva.
Estratégias: Sugerir três ações práticas de CRM/marketing adequadas ao contexto do negócio.

#Restrições:
Usar linguagem clara e focada em negócios.
Basear-se exclusivamente nos dados fornecidos.
Não mencionar técnicas de modelagem de dados (ex: KMeans).
Fornecer a resposta APENAS em JSON válido.

#Formato de Saída (JSON):
{{
  "clusters": [
    {{
      "ClusterId": 0,
      "SegmentoNome": "string curta",
      "SegmentoDescricao": "4-6 linhas (incluindo informações de RFM e representatividade na base)",
      "Estrategias": ["ação 1", "ação 2", "ação 3"]
    }}
  ]
}}

Entradas:

#DADOS AGREGADOS (CSV por cluster):
{prof_csv}
""".strip()

    return prompt

    """
    Envia SOMENTE agregados por cluster para a LLM.
    Pede saída em JSON estrito para facilitar o merge.
    """
    prof_csv = cluster_profile.to_csv(index=False)

    return f"""
Você é especialista em CRM, retenção e analytics.

Vou te passar uma tabela agregada por cluster obtido via KMeans em RFM (Recência, Frequência, Receita).
Sua tarefa é:
1) Criar um nome curto e descritivo para cada cluster (não use rótulos clássicos tipo "Campeões", "Fiéis", etc).
2) Escrever uma descrição curta (2-4 linhas) explicando o perfil.
3) Sugerir 3 ações práticas (bullets) para CRM/marketing para esse cluster.

Restrições:
- Use linguagem clara e orientada ao negócio.
- Não invente números. Baseie-se apenas nos dados agregados.
- Não mencione KMeans, padronização ou técnicas.
- Retorne APENAS um JSON válido (sem texto fora do JSON).

Formato exigido do JSON:
{{
  "clusters": [
    {{
      "ClusterId": 0,
      "SegmentoNome": "string curta",
      "SegmentoDescricao": "2-4 linhas",
      "Estrategias": ["ação 1", "ação 2", "ação 3"]
    }}
  ]
}}

Tabela agregada (CSV):
{prof_csv}
""".strip()


def gemini_generate_json(api_key: str, model: str, prompt: str) -> dict:
    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", "")

        # Alguns modelos podem devolver cercas; tenta limpar de forma conservadora
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            # remove possível linguagem "json\n"
            text = text.replace("json\n", "", 1).strip()

        return json.loads(text)

    except genai_errors.ClientError as e:
        raise RuntimeError(f"Gemini ClientError: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Falha ao decodificar JSON da LLM. Resposta recebida:\n{text}") from e


def build_cluster_labels(cluster_profile: pd.DataFrame, llm_json: dict) -> pd.DataFrame:
    """
    Converte JSON da LLM para DataFrame e valida campos.
    """
    if "clusters" not in llm_json or not isinstance(llm_json["clusters"], list):
        raise ValueError("JSON inválido: chave 'clusters' ausente ou não é lista.")

    rows = []
    for item in llm_json["clusters"]:
        rows.append(
            {
                "ClusterId": int(item["ClusterId"]),
                "SegmentoNome": str(item["SegmentoNome"]).strip(),
                "SegmentoDescricao": str(item["SegmentoDescricao"]).strip(),
                "Estrategias": item.get("Estrategias", []),
            }
        )

    labels = pd.DataFrame(rows)

    # valida cobertura mínima
    missing = set(cluster_profile["ClusterId"].tolist()) - set(labels["ClusterId"].tolist())
    if missing:
        raise ValueError(f"LLM não retornou rótulos para ClusterId: {sorted(missing)}")

    return labels