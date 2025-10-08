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
        ferramenta_variabilidade
    ]  