import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool
import numpy as np
import json


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    openai_api_key=OPENAI_API_KEY
)


@tool
def identificar_tipos_dados(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar informações sobre os tipos de dados do DataFrame.
    A instrução pode conter pedidos como:
    - 'Quais são os tipos de dados (numéricos, categóricos)?'
    - 'Identifique os tipos de cada coluna'
    - 'Mostre a classificação dos dados'
    """
    
    # Coletar informações sobre tipos de dados
    tipos_dados = df.dtypes.astype(str).to_dict()
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    colunas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    colunas_datetime = df.select_dtypes(include=['datetime64']).columns.tolist()
    colunas_boolean = df.select_dtypes(include=['bool']).columns.tolist()
    
    # Informações detalhadas de cada coluna
    info_colunas = {}
    for col in df.columns:
        info_colunas[col] = {
            "tipo": str(df[col].dtype),
            "valores_unicos": int(df[col].nunique()),
            "valores_nulos": int(df[col].isna().sum())
        }
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de apresentar informações sobre tipos de dados de um DataFrame
        a partir de uma {pergunta} feita pelo usuário.
        
        A seguir, você encontrará as informações sobre os tipos de dados:
        
        1. Um título: ## Classificação dos Tipos de Dados do DataFrame
        2. Total de colunas: {total_colunas}
        3. Tipos de dados de cada coluna: {tipos_dados}
        4. Lista de colunas numéricas ({qtd_numericas}): {colunas_numericas}
        5. Lista de colunas categóricas ({qtd_categoricas}): {colunas_categoricas}
        6. Lista de colunas datetime ({qtd_datetime}): {colunas_datetime}
        7. Lista de colunas booleanas ({qtd_boolean}): {colunas_boolean}
        8. Informações detalhadas por coluna: {info_colunas}
        9. Escreva um parágrafo sobre a adequação dos tipos de dados identificados
        10. Escreva um parágrafo sobre possíveis conversões ou ajustes necessários nos tipos
        """,
        input_variables=["pergunta", "total_colunas", "tipos_dados", "qtd_numericas", 
                        "colunas_numericas", "qtd_categoricas", "colunas_categoricas",
                        "qtd_datetime", "colunas_datetime", "qtd_boolean", "colunas_boolean",
                        "info_colunas"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "total_colunas": len(df.columns),
        "tipos_dados": json.dumps(tipos_dados, ensure_ascii=False),
        "qtd_numericas": len(colunas_numericas),
        "colunas_numericas": ", ".join(colunas_numericas) if colunas_numericas else "Nenhuma",
        "qtd_categoricas": len(colunas_categoricas),
        "colunas_categoricas": ", ".join(colunas_categoricas) if colunas_categoricas else "Nenhuma",
        "qtd_datetime": len(colunas_datetime),
        "colunas_datetime": ", ".join(colunas_datetime) if colunas_datetime else "Nenhuma",
        "qtd_boolean": len(colunas_boolean),
        "colunas_boolean": ", ".join(colunas_boolean) if colunas_boolean else "Nenhuma",
        "info_colunas": json.dumps(info_colunas, indent=2, ensure_ascii=False)
    })
    
    return resposta


@tool
def analisar_distribuicao_variaveis(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar informações sobre a distribuição das variáveis.
    A instrução pode conter pedidos como:
    - 'Qual a distribuição de cada variável (histogramas, distribuições)?'
    - 'Mostre a distribuição dos dados'
    - 'Análise de frequência das variáveis'
    """
    
    # Estatísticas descritivas das variáveis numéricas
    desc_numericas = df.select_dtypes(include=[np.number]).describe().to_dict()
    
    # Informações adicionais de distribuição
    distribuicao_numericas = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        distribuicao_numericas[col] = {
            "assimetria": float(df[col].skew()),
            "curtose": float(df[col].kurtosis())
        }
    
    # Distribuição de variáveis categóricas
    distribuicao_categoricas = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[col].value_counts().head(10)
        distribuicao_categoricas[col] = {
            "valores_unicos": int(df[col].nunique()),
            "moda": df[col].mode().tolist()[0] if not df[col].mode().empty else None,
            "top_10_frequencias": value_counts.to_dict()
        }
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de apresentar informações sobre distribuição de variáveis
        a partir de uma {pergunta} feita pelo usuário.
        
        A seguir, você encontrará as informações sobre distribuição:
        
        1. Um título: ## Análise de Distribuição das Variáveis
        2. Estatísticas descritivas das variáveis numéricas: {desc_numericas}
        3. Medidas de forma (assimetria e curtose) das variáveis numéricas: {distribuicao_numericas}
        4. Distribuição de frequência das variáveis categóricas: {distribuicao_categoricas}
        5. Total de variáveis numéricas analisadas: {qtd_numericas}
        6. Total de variáveis categóricas analisadas: {qtd_categoricas}
        7. Escreva um parágrafo interpretando a forma das distribuições (simetria, assimetria, normalidade)
        8. Escreva um parágrafo sobre insights obtidos da análise de frequência das variáveis categóricas
        9. Sugira visualizações apropriadas (histogramas, boxplots, gráficos de barras)
        """,
        input_variables=["pergunta", "desc_numericas", "distribuicao_numericas", 
                        "distribuicao_categoricas", "qtd_numericas", "qtd_categoricas"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "desc_numericas": json.dumps(desc_numericas, indent=2, ensure_ascii=False),
        "distribuicao_numericas": json.dumps(distribuicao_numericas, indent=2, ensure_ascii=False),
        "distribuicao_categoricas": json.dumps(distribuicao_categoricas, indent=2, ensure_ascii=False),
        "qtd_numericas": len(df.select_dtypes(include=[np.number]).columns),
        "qtd_categoricas": len(df.select_dtypes(include=['object', 'category']).columns)
    })
    
    return resposta


@tool
def calcular_intervalo_variaveis(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar informações sobre intervalos das variáveis.
    A instrução pode conter pedidos como:
    - 'Qual o intervalo de cada variável (mínimo, máximo)?'
    - 'Mostre o range das variáveis'
    - 'Qual a amplitude dos dados?'
    """
    
    # Calcular intervalos de variáveis numéricas
    intervalos = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        intervalos[col] = {
            "minimo": float(df[col].min()),
            "maximo": float(df[col].max()),
            "amplitude": float(df[col].max() - df[col].min()),
            "percentil_5": float(df[col].quantile(0.05)),
            "percentil_25": float(df[col].quantile(0.25)),
            "percentil_50": float(df[col].quantile(0.50)),
            "percentil_75": float(df[col].quantile(0.75)),
            "percentil_95": float(df[col].quantile(0.95)),
            "amplitude_interquartil": float(df[col].quantile(0.75) - df[col].quantile(0.25))
        }
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de apresentar informações sobre intervalos das variáveis
        a partir de uma {pergunta} feita pelo usuário.
        
        A seguir, você encontrará as informações sobre intervalos:
        
        1. Um título: ## Análise de Intervalos das Variáveis Numéricas
        2. Total de variáveis numéricas analisadas: {total_variaveis}
        3. Intervalos detalhados (mínimo, máximo, amplitude, percentis): {intervalos}
        4. Escreva um parágrafo interpretando a amplitude de cada variável
        5. Escreva um parágrafo sobre a presença de possíveis outliers (baseado nos percentis)
        6. Escreva um parágrafo sobre a necessidade de normalização ou padronização
        7. Sugira tratamentos para variáveis com grande amplitude
        """,
        input_variables=["pergunta", "total_variaveis", "intervalos"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "total_variaveis": len(df.select_dtypes(include=[np.number]).columns),
        "intervalos": json.dumps(intervalos, indent=2, ensure_ascii=False)
    })
    
    return resposta


@tool
def calcular_tendencia_central(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar medidas de tendência central.
    A instrução pode conter pedidos como:
    - 'Quais são as medidas de tendência central (média, mediana)?'
    - 'Calcule média, mediana e moda'
    - 'Mostre as medidas de posição central'
    """
    
    # Calcular medidas de tendência central
    tendencia = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        moda_valores = df[col].mode()
        media = df[col].mean()
        mediana = df[col].median()
        
        tendencia[col] = {
            "media": float(media),
            "mediana": float(mediana),
            "moda": moda_valores.tolist() if not moda_valores.empty else None,
            "diferenca_media_mediana": float(abs(media - mediana)),
            "media_aparada_10pct": float(df[col].quantile([0.10, 0.90]).mean())
        }
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de apresentar medidas de tendência central
        a partir de uma {pergunta} feita pelo usuário.
        
        A seguir, você encontrará as medidas de tendência central:
        
        1. Um título: ## Medidas de Tendência Central das Variáveis
        2. Total de variáveis numéricas analisadas: {total_variaveis}
        3. Medidas de tendência (média, mediana, moda): {tendencia}
        4. Escreva um parágrafo comparando média e mediana de cada variável
        5. Escreva um parágrafo sobre a simetria das distribuições (baseado na relação média vs mediana)
        6. Escreva um parágrafo sobre qual medida é mais representativa para cada variável
        7. Sugira quando usar média aparada em vez de média simples
        """,
        input_variables=["pergunta", "total_variaveis", "tendencia"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "total_variaveis": len(df.select_dtypes(include=[np.number]).columns),
        "tendencia": json.dumps(tendencia, indent=2, ensure_ascii=False)
    })
    
    return resposta


@tool
def calcular_variabilidade(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar medidas de variabilidade ou dispersão.
    A instrução pode conter pedidos como:
    - 'Qual a variabilidade dos dados (desvio padrão, variância)?'
    - 'Calcule as medidas de dispersão'
    - 'Mostre o desvio padrão e coeficiente de variação'
    """
    
    # Calcular medidas de variabilidade
    variabilidade = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        media = df[col].mean()
        std = df[col].std()
        
        variabilidade[col] = {
            "variancia": float(df[col].var()),
            "desvio_padrao": float(std),
            "coeficiente_variacao_pct": float((std / media * 100)) if media != 0 else None,
            "amplitude_total": float(df[col].max() - df[col].min()),
            "intervalo_interquartil": float(df[col].quantile(0.75) - df[col].quantile(0.25)),
            "desvio_medio_absoluto": float(abs(df[col] - media).mean())
        }
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de apresentar medidas de variabilidade dos dados
        a partir de uma {pergunta} feita pelo usuário.
        
        A seguir, você encontrará as medidas de variabilidade:
        
        1. Um título: ## Medidas de Variabilidade e Dispersão dos Dados
        2. Total de variáveis numéricas analisadas: {total_variaveis}
        3. Medidas de dispersão (variância, desvio padrão, CV, IQR): {variabilidade}
        4. Escreva um parágrafo interpretando o coeficiente de variação de cada variável
        5. Escreva um parágrafo comparando variáveis homogêneas (baixo CV) vs heterogêneas (alto CV)
        6. Escreva um parágrafo sobre a relação entre desvio padrão e intervalo interquartil
        7. Sugira quais variáveis precisam de padronização ou transformação
        8. Identifique variáveis com alta dispersão que merecem atenção especial
        """,
        input_variables=["pergunta", "total_variaveis", "variabilidade"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "total_variaveis": len(df.select_dtypes(include=[np.number]).columns),
        "variabilidade": json.dumps(variabilidade, indent=2, ensure_ascii=False)
    })
    
    return resposta

@tool
def analisar_padroes_temporais(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar análise de padrões ou tendências temporais.
    A instrução pode conter pedidos como:
    - 'Existem padrões temporais nos dados?'
    - 'Há tendências ao longo do tempo?'
    - 'Analise a evolução temporal das variáveis'
    - 'Mostre padrões de sazonalidade'
    """
    
    # Identificar colunas de data/tempo
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Tentar converter colunas object que possam ser datas
    possible_date_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col], errors='raise')
            possible_date_cols.append(col)
        except:
            pass
    
    # Análise temporal
    analise_temporal = {}
    
    if datetime_cols or possible_date_cols:
        for col in datetime_cols + possible_date_cols:
            # Converter para datetime se necessário
            if col in possible_date_cols:
                df_temp = df.copy()
                df_temp[col] = pd.to_datetime(df_temp[col])
            else:
                df_temp = df
            
            # Estatísticas temporais
            analise_temporal[col] = {
                "data_minima": str(df_temp[col].min()),
                "data_maxima": str(df_temp[col].max()),
                "periodo_total_dias": (df_temp[col].max() - df_temp[col].min()).days,
                "total_registros": int(df_temp[col].count()),
                "registros_por_ano": df_temp[col].dt.year.value_counts().to_dict() if hasattr(df_temp[col].dt, 'year') else None,
                "registros_por_mes": df_temp[col].dt.month.value_counts().to_dict() if hasattr(df_temp[col].dt, 'year') else None,
                "registros_por_dia_semana": df_temp[col].dt.dayofweek.value_counts().to_dict() if hasattr(df_temp[col].dt, 'dayofweek') else None
            }
    
    # Análise de tendências em variáveis numéricas com índice temporal
    tendencias_numericas = {}
    if datetime_cols or possible_date_cols:
        col_temporal = (datetime_cols + possible_date_cols)[0]
        for col_num in df.select_dtypes(include=[np.number]).columns[:5]:  # Limitar a 5 colunas
            # Calcular correlação com tempo (se possível)
            df_sorted = df.sort_values(col_temporal)
            tendencias_numericas[col_num] = {
                "media_primeira_metade": float(df_sorted[col_num].iloc[:len(df_sorted)//2].mean()),
                "media_segunda_metade": float(df_sorted[col_num].iloc[len(df_sorted)//2:].mean()),
                "variacao_percentual": f"{((df_sorted[col_num].iloc[len(df_sorted)//2:].mean() - df_sorted[col_num].iloc[:len(df_sorted)//2].mean()) / df_sorted[col_num].iloc[:len(df_sorted)//2].mean() * 100):.2f}%"
            }
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados especializado em análise de séries temporais e identificação de padrões.
        
        Com base na {pergunta} do usuário, apresente uma análise de padrões temporais:
        
        1. Um título: ## Análise de Padrões e Tendências Temporais
        2. Colunas temporais identificadas: {colunas_temporais}
        3. Colunas que podem ser datas: {colunas_possiveis_datas}
        4. Análise temporal detalhada: {analise_temporal}
        5. Tendências em variáveis numéricas ao longo do tempo: {tendencias_numericas}
        6. Total de colunas temporais encontradas: {total_temporais}
        7. Escreva um parágrafo sobre os padrões temporais identificados (sazonalidade, tendências)
        8. Escreva um parágrafo sobre a evolução das variáveis numéricas ao longo do tempo
        9. Escreva um parágrafo sobre recomendações de análises temporais adicionais
        10. Sugira visualizações apropriadas (gráficos de linha, séries temporais, decomposição sazonal)
        
        IMPORTANTE: Se não houver colunas temporais, informe que não foi possível identificar padrões temporais
        e sugira análises alternativas.
        """,
        input_variables=["pergunta", "colunas_temporais", "colunas_possiveis_datas", 
                        "analise_temporal", "tendencias_numericas", "total_temporais"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "colunas_temporais": ", ".join(datetime_cols) if datetime_cols else "Nenhuma coluna datetime identificada",
        "colunas_possiveis_datas": ", ".join(possible_date_cols) if possible_date_cols else "Nenhuma coluna possível identificada",
        "analise_temporal": json.dumps(analise_temporal, indent=2, ensure_ascii=False) if analise_temporal else "Não há dados temporais para analisar",
        "tendencias_numericas": json.dumps(tendencias_numericas, indent=2, ensure_ascii=False) if tendencias_numericas else "Não foi possível calcular tendências",
        "total_temporais": len(datetime_cols) + len(possible_date_cols)
    })
    
    return resposta


@tool
def analisar_frequencias(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar análise de frequências e valores mais/menos comuns.
    A instrução pode conter pedidos como:
    - 'Quais os valores mais frequentes?'
    - 'Quais os valores menos frequentes?'
    - 'Mostre a distribuição de frequência'
    - 'Quais são os valores raros nos dados?'
    """
    
    # Análise de frequência para variáveis categóricas
    frequencias_categoricas = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[col].value_counts()
        frequencias_categoricas[col] = {
            "valores_unicos": int(df[col].nunique()),
            "top_5_mais_frequentes": value_counts.head(5).to_dict(),
            "top_5_menos_frequentes": value_counts.tail(5).to_dict(),
            "valor_mais_comum": str(value_counts.index[0]),
            "frequencia_mais_comum": int(value_counts.iloc[0]),
            "percentual_mais_comum": f"{(value_counts.iloc[0] / len(df) * 100):.2f}%",
            "valores_raros": value_counts[value_counts == 1].index.tolist()[:10]  # Valores que aparecem apenas 1 vez
        }
    
    # Análise de frequência para variáveis numéricas (discretização)
    frequencias_numericas = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        # Criar bins para análise de frequência
        try:
            bins = pd.cut(df[col], bins=10)
            value_counts = bins.value_counts().sort_values(ascending=False)
            
            frequencias_numericas[col] = {
                "valores_unicos": int(df[col].nunique()),
                "faixa_mais_frequente": str(value_counts.index[0]),
                "frequencia_faixa": int(value_counts.iloc[0]),
                "valor_mais_comum_exato": float(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                "frequencia_valor_mais_comum": int((df[col] == df[col].mode().iloc[0]).sum()) if not df[col].mode().empty else 0
            }
        except:
            frequencias_numericas[col] = {"erro": "Não foi possível criar bins para esta variável"}
    
    # Análise de combinações frequentes (top pares de valores categóricos)
    combinacoes_frequentes = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(cat_cols) >= 2:
        # Pegar primeiras 2 colunas categóricas
        col1, col2 = cat_cols[0], cat_cols[1]
        combinacoes = df.groupby([col1, col2]).size().sort_values(ascending=False).head(5)
        combinacoes_frequentes[f"{col1}_x_{col2}"] = {
            str(k): int(v) for k, v in combinacoes.items()
        }
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados especializado em análise de frequências e identificação de padrões.
        
        Com base na {pergunta} do usuário, apresente uma análise de frequências:
        
        1. Um título: ## Análise de Frequências e Valores Mais/Menos Comuns
        2. Total de variáveis categóricas analisadas: {total_categoricas}
        3. Frequências de variáveis categóricas (mais e menos frequentes): {frequencias_categoricas}
        4. Total de variáveis numéricas analisadas: {total_numericas}
        5. Frequências de variáveis numéricas (por faixas): {frequencias_numericas}
        6. Combinações mais frequentes entre variáveis: {combinacoes_frequentes}
        7. Escreva um parágrafo destacando os valores mais frequentes e seu significado
        8. Escreva um parágrafo sobre valores raros ou incomuns encontrados
        9. Escreva um parágrafo sobre a concentração dos dados (se há dominância de certos valores)
        10. Sugira análises adicionais baseadas nos padrões de frequência identificados
        """,
        input_variables=["pergunta", "total_categoricas", "frequencias_categoricas", 
                        "total_numericas", "frequencias_numericas", "combinacoes_frequentes"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "total_categoricas": len(df.select_dtypes(include=['object', 'category']).columns),
        "frequencias_categoricas": json.dumps(frequencias_categoricas, indent=2, ensure_ascii=False),
        "total_numericas": len(df.select_dtypes(include=[np.number]).columns),
        "frequencias_numericas": json.dumps(frequencias_numericas, indent=2, ensure_ascii=False),
        "combinacoes_frequentes": json.dumps(combinacoes_frequentes, indent=2, ensure_ascii=False) if combinacoes_frequentes else "Não há combinações para analisar"
    })
    
    return resposta


@tool
def identificar_agrupamentos(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta quando o usuário solicitar identificação de clusters ou agrupamentos.
    A instrução pode conter pedidos como:
    - 'Existem agrupamentos nos dados?'
    - 'Identifique clusters'
    - 'Há padrões de agrupamento?'
    - 'Mostre grupos similares nos dados'
    """
    
    # Análise de agrupamentos naturais usando estatísticas
    from scipy import stats
    
    agrupamentos_categoricos = {}
    # Agrupar por variáveis categóricas e analisar variáveis numéricas
    for cat_col in df.select_dtypes(include=['object', 'category']).columns[:3]:  # Limitar a 3 colunas
        grupos_stats = {}
        for num_col in df.select_dtypes(include=[np.number]).columns[:3]:  # Limitar a 3 colunas numéricas
            grouped = df.groupby(cat_col)[num_col]
            grupos_stats[num_col] = {
                "media_por_grupo": grouped.mean().to_dict(),
                "mediana_por_grupo": grouped.median().to_dict(),
                "desvio_padrao_por_grupo": grouped.std().to_dict(),
                "total_grupos": int(df[cat_col].nunique())
            }
        
        agrupamentos_categoricos[cat_col] = grupos_stats
    
    # Análise de correlação entre variáveis numéricas (indica agrupamentos)
    numeric_df = df.select_dtypes(include=[np.number])
    correlacoes = {}
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr()
        # Pegar pares com alta correlação (>0.7 ou <-0.7)
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        "variavel_1": corr_matrix.columns[i],
                        "variavel_2": corr_matrix.columns[j],
                        "correlacao": float(corr_val)
                    })
        correlacoes["pares_alta_correlacao"] = high_corr[:10]  # Top 10
    
    # Análise de outliers por grupo (usando IQR)
    outliers_por_grupo = {}
    for cat_col in df.select_dtypes(include=['object', 'category']).columns[:2]:
        for num_col in df.select_dtypes(include=[np.number]).columns[:2]:
            grupos = df.groupby(cat_col)[num_col]
            outliers_info = {}
            for grupo_nome, grupo_data in grupos:
                Q1 = grupo_data.quantile(0.25)
                Q3 = grupo_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = grupo_data[(grupo_data < Q1 - 1.5*IQR) | (grupo_data > Q3 + 1.5*IQR)]
                outliers_info[str(grupo_nome)] = {
                    "quantidade_outliers": len(outliers),
                    "percentual_outliers": f"{(len(outliers) / len(grupo_data) * 100):.2f}%"
                }
            outliers_por_grupo[f"{cat_col}_{num_col}"] = outliers_info
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados especializado em identificação de clusters e padrões de agrupamento.
        
        Com base na {pergunta} do usuário, apresente uma análise de agrupamentos:
        
        1. Um título: ## Identificação de Agrupamentos e Clusters nos Dados
        2. Agrupamentos naturais por variáveis categóricas: {agrupamentos_categoricos}
        3. Pares de variáveis com alta correlação (potenciais clusters): {correlacoes}
        4. Análise de outliers por grupo: {outliers_por_grupo}
        5. Total de variáveis categóricas usadas para agrupamento: {total_cat_cols}
        6. Total de variáveis numéricas analisadas: {total_num_cols}
        7. Escreva um parágrafo descrevendo os agrupamentos naturais identificados
        8. Escreva um parágrafo sobre diferenças significativas entre grupos
        9. Escreva um parágrafo sobre variáveis que têm forte relação entre si (clusters de variáveis)
        10. Sugira técnicas avançadas de clustering (K-means, DBSCAN, hierárquico) que poderiam ser aplicadas
        11. Recomende visualizações para explorar clusters (scatter plots, dendrogramas, heatmaps)
        
        IMPORTANTE: Se não houver variáveis categóricas, sugira análises de clustering não supervisionado.
        """,
        input_variables=["pergunta", "agrupamentos_categoricos", "correlacoes", "outliers_por_grupo",
                        "total_cat_cols", "total_num_cols"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "agrupamentos_categoricos": json.dumps(agrupamentos_categoricos, indent=2, ensure_ascii=False),
        "correlacoes": json.dumps(correlacoes, indent=2, ensure_ascii=False),
        "outliers_por_grupo": json.dumps(outliers_por_grupo, indent=2, ensure_ascii=False),
        "total_cat_cols": len(df.select_dtypes(include=['object', 'category']).columns),
        "total_num_cols": len(df.select_dtypes(include=[np.number]).columns)
    })
    
    return resposta

@tool
def informacoes_genericas_do_dataframe(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar uma análise simples e direta a partir de um DataFrame pandas.
    A instrução pode conter pedidos como:
    - 'Mostre informações gerais do DataFrame'
    - 'Descreva o dataset'
    - 'Quais são as informações básicas dos dados?'
    """
    
    # Coletar informações gerais
    shape = df.shape
    
    # Descrição das colunas
    columns = {}
    for col in df.columns:
        columns[col] = {
            "tipo": str(df[col].dtype),
            "nao_nulos": int(df[col].count()),
            "valores_unicos": int(df[col].nunique())
        }
    
    # Dados nulos
    nulos = df.isnull().sum().to_dict()
    nulos = {k: int(v) for k, v in nulos.items() if v > 0}
    
    # Strings 'nan'
    nans_str = {}
    for col in df.select_dtypes(include=['object']).columns:
        count_nan = df[col].astype(str).str.lower().isin(['nan', 'null', 'none']).sum()
        if count_nan > 0:
            nans_str[col] = int(count_nan)
    
    # Dados duplicados
    duplicados = int(df.duplicated().sum())
    
    # Template de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de apresentar informações sobre um DataFrame
        a partir de uma {pergunta} feita pelo usuário.
        
        A seguir, você encontrará as informações gerais da base de dados:
        
        1. Um título: ## Relatório de Informações Gerais sobre o Dataset
        2. A dimensão total do DataFrame: {shape} (linhas, colunas)
        3. A descrição de cada coluna (incluindo nome, tipo de dado e estatísticas básicas): {columns}
        4. As colunas que contêm dados nulos, com a respectiva quantidade: {nulos}
        5. As colunas que contêm strings 'nan', com a respectiva quantidade: {nans_str}
        6. E a existência (ou não) de dados duplicados: {duplicados} linhas duplicadas
        7. Escreva um parágrafo sobre análises que podem ser feitas com esses dados
        8. Escreva um parágrafo sobre tratamentos que podem ser feitos nos dados
        """,
        input_variables=["pergunta", "shape", "columns", "nulos", "nans_str", "duplicados"]
    )
    
    cadeia = template_resposta | llm | StrOutputParser()
    
    resposta = cadeia.invoke({
        "pergunta": pergunta,
        "shape": f"{shape[0]} linhas × {shape[1]} colunas",
        "columns": json.dumps(columns, indent=2, ensure_ascii=False),
        "nulos": json.dumps(nulos, ensure_ascii=False) if nulos else "Nenhuma coluna com valores nulos",
        "nans_str": json.dumps(nans_str, ensure_ascii=False) if nans_str else "Nenhuma coluna com strings 'nan'",
        "duplicados": duplicados
    })
    
    return resposta


# Relatório informações
@tool
def informacoes_dataframe(pergunta: str, df: pd.DataFrame) -> str:
    """Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
        incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
        nulos e duplicados para dar um panomara geral sobre o arquivo."""

    # Coleta de informações
    shape = df.shape
    columns = df.dtypes
    nulos = df.isnull().sum()
    nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
    duplicados = df.duplicated().sum()

   # Prompt de resposta 

    template_resposta = PromptTemplate( 

        template=""" 
        Você é um analista de dados encarregado de apresentar um resumo informativo sobre um DataFrame 
        a partir de uma {pergunta} feita pelo usuário. 

        A seguir, você encontrará as informações gerais da base de dados: 

        ================= INFORMAÇÕES DO DATAFRAME ================= 

        Dimensões: {shape}
 
        Colunas e tipos de dados: {columns} 

        Valores nulos por coluna: {nulos} 

        Strings 'nan' (qualquer capitalização) por coluna: {nans_str} 

        Linhas duplicadas: {duplicados} 

        ============================================================ 

        Com base nessas informações, escreva um resumo claro e organizado contendo: 

        1. Um título: ## Relatório de informações gerais sobre o dataset 
        2. A dimensão total do DataFrame; 
        3. A descrição de cada coluna (incluindo nome, tipo de dado e o que aquela coluna é) 
        4. As colunas que contêm dados nulos, com a respectiva quantidade.  
        5. As colunas que contêm strings 'nan', com a respectiva quantidade. 
        6. E a existência (ou não) de dados duplicados. 
        7. Escreva um parágrafo sobre análises que podem ser feitas com 
        esses dados. 
        8. Escreva um parágrafo sobre tratamentos que podem ser feitos nos dados. 
        """, 
        input_variables=["pergunta","shape", "columns", "nulos", "nans_str", "duplicados"] ) 

    cadeia = template_resposta | llm | StrOutputParser()

    resposta = cadeia.invoke({
              "pergunta": pergunta,
              "shape": shape,
              "columns": columns,
              "nulos": nulos,
              "nans_str": nans_str,
              "duplicados": duplicados
        })

    return resposta

# Relatório estatístico
@tool
def resumo_estatistico(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo e descritivo da base de dados,
    incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.).
    Não utilize esta ferramenta para calcular uma única métrica como 'qual é a média de X' ou 'qual a correlação das variáveis'.
    """
    # Coleta de estatísticas descritivas
    estatisticas_descritivas = df.describe(include='number').transpose().to_string()
    
    # Prompt de resposta
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de interpretar resultados estatísticos de uma base de dados
        a partir de uma {pergunta} feita pelo usuário.

        A seguir, você encontrará as estatísticas descritivas da base de dados:

        ================= ESTATÍSTICAS DESCRITIVAS =================

        {resumo}

        ============================================================

        Com base nesses dados, elabore um resumo explicativo com linguagem clara, acessível e fluida, destacando
        os principais pontos dos resultados. Inclua:

        1. Um título: ## Relatório de estatísticas descritivas
        2. Uma visão geral das estatísticas das colunas numéricas
        3. Um paráfrago sobre cada uma das colunas, comentando informações sobre seus valores.
        4. Identificação de possíveis outliers com base nos valores mínimo e máximo
        5. Recomendações de próximos passos na análise com base nos padrões identificados
        """,
        input_variables=["pergunta", "resumo"]
    )

    cadeia = template_resposta | llm | StrOutputParser()

    resposta = cadeia.invoke({"pergunta": pergunta, "resumo": estatisticas_descritivas})

    return resposta

# Gerador de gráficos 
@tool
def gerar_grafico(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuário.
    A instrução pode conter pedidos como: 'Crie um gráfico da média de tempo de entrega por clima','Plote a distribuição do tempo de entrega'"
    ou "Plote a relação entre a classifição dos agentes e o tempo de entrega. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
    'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição', 'represente graficamente', entre outros."""

 # Captura informações sobre o dataframe
    colunas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    amostra_dados = df.head(3).to_dict(orient='records')

  # Template otimizado para geração de código de gráficos
    template_resposta = PromptTemplate(
            template="""
            Você é um especialista em visualização de dados. Sua tarefa é gerar **apenas o código Python** para plotar um gráfico com base na solicitação do usuário.

            ## Solicitação do usuário:
            "{pergunta}"

            ## Metadados do DataFrame:
            {colunas}

            ## Amostra dos dados (3 primeiras linhas):
            {amostra}

            ## Instruções obrigatórias:
            1. Use as bibliotecas `matplotlib.pyplot` (como `plt`) e `seaborn` (como `sns`).
            2. Defina o tema com `sns.set_theme()`
            3. Certifique-se de que todas as colunas mencionadas na solicitação existem no DataFrame chamado `df`.
            4. Escolha o tipo de gráfico adequado conforme a análise solicitada:
            - **Distribuição de variáveis numéricas**: `histplot`, `kdeplot`, `boxplot` ou `violinplot`
            - **Distribuição de variáveis categóricas**: `countplot` 
            - **Comparação entre categorias**: `barplot`
            - **Relação entre variáveis**: `scatterplot` ou `lineplot`
            - **Séries temporais**: `lineplot`, com o eixo X formatado como datas
            5. Configure o tamanho do gráfico com `figsize=(8, 4)`.
            6. Adicione título e rótulos (`labels`) apropriados aos eixos.
            7. Posicione o título à esquerda com `loc='left'`, deixe o `pad=20` e use `fontsize=14`.
            8. Mantenha os ticks eixo X sem rotação com `plt.xticks(rotation=0)`
            9. Remova as bordas superior e direita do gráfico com `sns.despine()`.
            10. Finalize o código com `plt.show()`.

            Retorne APENAS o código Python, sem nenhum texto adicional ou explicação.

            Código Python:
            """, input_variables=["pergunta", "colunas", "amostra"]
        )

        # Gera o código
    cadeia = template_resposta | llm | StrOutputParser()
    codigo_bruto = cadeia.invoke({
            "pergunta": pergunta,
            "colunas": colunas_info,
            "amostra": amostra_dados
    })

        # Limpa o código gerado
    codigo_limpo = codigo_bruto.replace("```python", "").replace("```", "").strip()

        # Tenta executar o código para validação
    exec_globals = {'df': df, 'plt': plt, 'sns': sns}
    exec_locals = {}
    exec(codigo_limpo, exec_globals, exec_locals)

        # Mostra o gráfico 
    fig = plt.gcf()
    st.pyplot(fig)
        
    return "" 


# Função para criar ferramentas 
def criar_ferramentas(df):
    ferramenta_informacoes_dataframe = Tool(
        name="Informações Dataframe",
        func=lambda pergunta:informacoes_dataframe.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
        incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
        nulos e duplicados para dar um panomara geral sobre o arquivo.""",
        return_direct=True) # Para exibir o relatório gerado pela função
    ferramenta_informacoes_dataframe = Tool(
        name="Informações Dataframe",
        func=lambda pergunta:informacoes_dataframe.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
        incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
        nulos e duplicados para dar um panomara geral sobre o arquivo.""",
        return_direct=True) # Para exibir o relatório gerado pela função

    ferramenta_resumo_estatistico = Tool(
        name="Resumo Estatístico",
        func=lambda pergunta:resumo_estatistico.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo e descritivo da base de dados,
        incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.) e/ou múltiplas colunas numéricas.
        Não utilize esta ferramenta para calcular uma única métrica como 'qual é a média de X' ou 'qual a correlação das variáveis'.
        Para isso, use a ferramenta_python.""",
        return_direct=True) # Para exibir o relatório gerado pela função

    ferramenta_gerar_grafico = Tool(
        name="Gerar Gráfico",
        func=lambda pergunta:gerar_grafico.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuário.
        A instrução pode conter pedidos como: 'Crie um gráfico da média de tempo de entrega por clima','Plote a distribuição do tempo de entrega'"
        ou "Plote a relação entre a classificação dos agentes e o tempo de entrega. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
        'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição', 'represente graficamente', entre outros.""",
        return_direct=True)
    
    ferramenta_codigos_python = Tool(
        name="Códigos Python",
        func=PythonAstREPLTool(locals={"df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar cálculos, consultas ou transformações específicas usando Python diretamente sobre o DataFrame `df`.
        Exemplos de uso incluem: "Qual é a média da coluna X?", "Quais são os valores únicos da coluna Y?", "Qual a correlação entre A e B?". 
        Evite utilizar esta ferramenta para solicitações mais amplas ou descritivas, como informações gerais sobre o dataframe, resumos estatísticos completos ou geração de gráficos — nesses casos, use as ferramentas apropriadas.""")

    ferramenta_informacoes_genericas = Tool(
        name="Informações Gerais do DataFrame",
        func=lambda pergunta: informacoes_genericas_do_dataframe.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar uma análise simples e direta a partir de um DataFrame pandas.
        A instrução pode conter pedidos como: 'Mostre informações gerais do DataFrame', 'Descreva o dataset', 'Quais são as informações básicas dos dados?',
        'Mostre a dimensão do DataFrame', 'Há dados nulos?', 'Existem duplicados?'. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
        'informações gerais', 'visão geral', 'descrição do dataset', 'resumo dos dados', 'overview', 'informações básicas', entre outros.""",
        return_direct=True
    )
    
    ferramenta_tipos_dados = Tool(
        name="Identificar Tipos de Dados",
        func=lambda pergunta: identificar_tipos_dados.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar informações sobre os tipos de dados do DataFrame.
        A instrução pode conter pedidos como: 'Quais são os tipos de dados?', 'Quais colunas são numéricas?', 'Quais colunas são categóricas?',
        'Identifique os tipos de cada coluna', 'Mostre a classificação dos dados', 'Quais são as variáveis numéricas e categóricas?'.
        Palavras-chave comuns que indicam o uso desta ferramenta incluem: 'tipos de dados', 'tipo de cada coluna', 'variáveis numéricas',
        'variáveis categóricas', 'classificação dos dados', 'dtype', entre outros.""",
        return_direct=True
    )
    
    ferramenta_distribuicao = Tool(
        name="Analisar Distribuição das Variáveis",
        func=lambda pergunta: analisar_distribuicao_variaveis.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar informações sobre a distribuição das variáveis.
        A instrução pode conter pedidos como: 'Qual a distribuição de cada variável?', 'Mostre histogramas e distribuições',
        'Como estão distribuídos os dados?', 'Análise de frequência das variáveis', 'Mostre a distribuição dos dados',
        'Quais variáveis têm distribuição normal?', 'Mostre assimetria e curtose'. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
        'distribuição', 'histograma', 'frequência', 'assimetria', 'curtose', 'normalidade', 'describe', 'estatísticas descritivas', entre outros.""",
        return_direct=True
    )
    
    ferramenta_intervalo = Tool(
        name="Calcular Intervalos das Variáveis",
        func=lambda pergunta: calcular_intervalo_variaveis.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar informações sobre intervalos e ranges das variáveis.
        A instrução pode conter pedidos como: 'Qual o intervalo de cada variável?', 'Mostre o mínimo e máximo', 'Qual o range dos dados?',
        'Amplitude das variáveis', 'Mostre os valores mínimos e máximos', 'Qual a amplitude de cada coluna?', 'Mostre os percentis'.
        Palavras-chave comuns que indicam o uso desta ferramenta incluem: 'intervalo', 'range', 'amplitude', 'mínimo', 'máximo',
        'min', 'max', 'percentis', 'amplitude interquartil', 'IQR', entre outros.""",
        return_direct=True
    )
    
    ferramenta_tendencia_central = Tool(
        name="Calcular Medidas de Tendência Central",
        func=lambda pergunta: calcular_tendencia_central.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar medidas de tendência central.
        A instrução pode conter pedidos como: 'Quais são as medidas de tendência central?', 'Calcule média e mediana',
        'Mostre a moda dos dados', 'Medidas de posição central', 'Qual a média de cada variável?', 'Mostre a mediana',
        'Compare média e mediana'. Palavras-chave comuns que indicam o uso desta ferramenta incluem: 'média', 'mediana', 'moda',
        'tendência central', 'medidas de posição', 'valor central', 'média aparada', entre outros.""",
        return_direct=True
    )
    
    ferramenta_variabilidade = Tool(
        name="Calcular Medidas de Variabilidade",
        func=lambda pergunta: calcular_variabilidade.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar medidas de variabilidade ou dispersão.
        A instrução pode conter pedidos como: 'Qual a variabilidade dos dados?', 'Calcule desvio padrão e variância',
        'Mostre medidas de dispersão', 'Como variam os dados?', 'Qual o coeficiente de variação?', 'Mostre a dispersão dos dados',
        'Calcule o desvio padrão de cada variável'. Palavras-chave comuns que indicam o uso desta ferramenta incluem: 'variabilidade',
        'dispersão', 'variância', 'desvio padrão', 'std', 'coeficiente de variação', 'CV', 'desvio médio absoluto', entre outros.""",
        return_direct=True
    )
    
    ferramentas_padroes_temporais = Tool(
        name="Analisar Padrões Temporais",
        func=lambda pergunta: analisar_padroes_temporais.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar análise de padrões ou tendências temporais.
        A instrução pode conter pedidos como: 'Existem padrões temporais?', 'Há tendências ao longo do tempo?',
        'Analise a evolução temporal', 'Mostre sazonalidade', 'Como os dados variam no tempo?', 'Identifique tendências',
        'Análise de série temporal'. Palavras-chave comuns que indicam o uso desta ferramenta incluem: 'temporal', 'tempo',
        'tendência', 'evolução', 'sazonalidade', 'ao longo do tempo', 'histórico', 'cronológico', entre outros.""",
        return_direct=True
    )
    
    ferramentas_frequencias = Tool(
        name="Analisar Frequências",
        func=lambda pergunta: analisar_frequencias.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar análise de frequências e valores comuns/raros.
        A instrução pode conter pedidos como: 'Quais os valores mais frequentes?', 'Quais os valores menos frequentes?',
        'Mostre a distribuição de frequência', 'Quais são os valores raros?', 'Valores mais comuns', 'Top valores',
        'Valores únicos'. Palavras-chave comuns que indicam o uso desta ferramenta incluem: 'frequente', 'frequência',
        'comum', 'raro', 'top valores', 'mais aparecem', 'menos aparecem', 'value_counts', 'contagem', entre outros.""",
        return_direct=True
    )
    
    ferramentas_agrupamentos = Tool(
        name="Identificar Agrupamentos",
        func=lambda pergunta: identificar_agrupamentos.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta quando o usuário solicitar identificação de clusters ou agrupamentos.
        A instrução pode conter pedidos como: 'Existem agrupamentos?', 'Identifique clusters', 'Há padrões de agrupamento?',
        'Mostre grupos similares', 'Análise de clusters', 'Segmentação dos dados', 'Grupos naturais'.
        Palavras-chave comuns que indicam o uso desta ferramenta incluem: 'cluster', 'agrupamento', 'grupo', 'segmento',
        'similar', 'padrão', 'correlação entre grupos', 'diferenças entre grupos', entre outros.""",
        return_direct=True
    )

    return [
        ferramenta_informacoes_dataframe, 
        ferramenta_resumo_estatistico, 
        ferramenta_gerar_grafico,
        ferramenta_codigos_python,
        ferramenta_informacoes_genericas,
        ferramenta_tipos_dados,
        ferramenta_distribuicao,
        ferramenta_intervalo,
        ferramenta_tendencia_central,
        ferramenta_variabilidade,
        ferramentas_padroes_temporais,
        ferramentas_frequencias,
        ferramentas_agrupamentos
    ]  