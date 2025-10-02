import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from ferramentas import criar_ferramentas
import zipfile
import io
import os
import time

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REQUIRED_COLUMNS_NOTA_FISCAL = ['CHAVE DE ACESSO', 'MODELO_x', 'SÉRIE_x', 'NÚMERO_x']
REQUIRED_DF1_COLUMNS_NOTA_FISCAL = ['CHAVE DE ACESSO', 'MODELO', 'SÉRIE', 'NÚMERO']
REQUIRED_COLUMNS_CREDITCARD = ['Time', 'V1', 'V28', 'Amount', 'Class']

def flash_mensagem(mensagem, tipo):
    mensagem_temp = st.empty()
    if tipo:
        mensagem_temp.success(mensagem)
    else:
        mensagem_temp.error(mensagem)
    time.sleep(5)
    mensagem_temp.empty()



def extrair_zip_e_carregar_csvs_em_memoria(arquivo_carregado):
    """
    Extrai todos os arquivos CSV de um ZIP e os carrega em DataFrames Pandas em memória.
    Retorna um dicionário onde a chave é o nome do arquivo CSV (sem caminho) e o valor é o DataFrame.
    """

    dfs = {}
    with zipfile.ZipFile(arquivo_carregado, 'r') as zip_ref:
        csv_files_in_zip = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]

        if not csv_files_in_zip:
            flash_mensagem("❌ Nenhum arquivo CSV encontrado dentro do ZIP.", False)
        else:
            flash_mensagem(f"✅ Encontrados {len(csv_files_in_zip)} arquivo(s) CSV no ZIP", True)
                

        for csv_file_name in csv_files_in_zip:
            try:
                with zip_ref.open(csv_file_name) as file:
                    # Lê o arquivo CSV diretamente da memória
                    df = pd.read_csv(io.TextIOWrapper(file, 'utf-8'))
                    # Usa apenas o nome do arquivo, sem o caminho completo dentro do zip
                    df_name = os.path.basename(csv_file_name)
                    dfs[df_name] = df
            except Exception as e:
                print(f"Erro ao ler o CSV '{csv_file_name}' do ZIP: {e}")
                # Opcional: ignorar arquivos CSV que não podem ser lidos ou levantar um erro mais específico

    if not dfs:
        raise ValueError("Nenhum arquivo CSV válido pôde ser carregado do ZIP.")

    return dfs


def validate_columns(dataframe, column_list):
  return all(col in dataframe.columns for col in column_list)

def carregar_arquivo_zip_to_df(arquivo_carregado):
    dfs_dict = extrair_zip_e_carregar_csvs_em_memoria(arquivo_carregado)


    if len(dfs_dict) == 1:
        return dfs_dict[0]
    elif len(dfs_dict) == 2:
        df1_name = list(dfs_dict.keys())[0]
        df2_name = list(dfs_dict.keys())[1]
        df1 = dfs_dict[df1_name]
        df2 = dfs_dict[df2_name]
        if not validate_columns(df1, REQUIRED_DF1_COLUMNS_NOTA_FISCAL):
            st.error("❌ Não é um arquivo zip de Notafiscal Valido.")
        
        try:
            merged_df = pd.merge(df1, df2, on='CHAVE DE ACESSO', how='inner')
        except KeyError:
            st.error("❌ Não é uma combinação de arquivos csv de Notafiscal Valido.")
            return None
        return merged_df
    else:
        st.error("❌ O arquivo zip deve conter 1 ou 2 arquivos CSV.")
        return None


st.set_page_config(page_title="Assistente de Elite IA", layout="centered")
st.title("🧠 Assistente de Elite IA 🦜")

# Descrição da ferramenta
st.info("""
Este assistente utiliza um agente, criado com Langchain, para te ajudar a explorar, analisar e visualizar dados de forma interativa. Basta fazer o upload de um arquivo CSV.
""")

# Upload do CSV
st.markdown("### 📁 Faça upload do seu arquivo CSV")
arquivo_carregado = st.file_uploader("Selecione um arquivo CSV", type=["csv", "zip"], label_visibility="collapsed")

if arquivo_carregado:
    nome_arquivo = arquivo_carregado.name
    if nome_arquivo.lower().endswith('.zip'):
        df = carregar_arquivo_zip_to_df(arquivo_carregado)
    else:
        df = pd.read_csv(arquivo_carregado)    
        
    if df is None:
        flash_mensagem("❌ Arquivo inválido.", False)
    else:    
        if validate_columns(df, REQUIRED_COLUMNS_NOTA_FISCAL):
            tipo = "Nota Fiscal"
        elif validate_columns(df, REQUIRED_COLUMNS_CREDITCARD):
            tipo = "Credit Card"
        else:
            tipo = "Dados Comuns"

            flash_mensagem("✅ Arquivo carregado com sucesso!", True)  
            st.markdown(f"## 📄 Arquivo do tipo: {tipo}")
            
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                openai_api_key=OPENAI_API_KEY
            )

            # Ferramentas
            tools = criar_ferramentas(df)

            # Prompt react
            df_head = df.head().to_markdown()

            prompt_react_pt = PromptTemplate(
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
                partial_variables={"df_head": df_head},
                template="""
                Você é um assistente que sempre responde em português.

                Você tem acesso a um dataframe pandas chamado `df`.
                Aqui estão as primeiras linhas do DataFrame, obtidas com `df.head().to_markdown()`:

                {df_head}

                Responda às seguintes perguntas da melhor forma possível.

                Para isso, você tem acesso às seguintes ferramentas:

                {tools}

                Use o seguinte formato:

                Question: a pergunta de entrada que você deve responder  
                Thought: você deve sempre pensar no que fazer  
                Action: a ação a ser tomada, deve ser uma das [{tool_names}]  
                Action Input: a entrada para a ação  
                Observation: o resultado da ação  
                ... (este Thought/Action/Action Input/Observation pode se repetir 5 vezes)
                Thought: Agora eu sei a resposta final  
                Final Answer: a resposta final para a pergunta de entrada original.
                Quando usar a ferramenta_python: formate sua resposta final de forma clara, em lista, com valores separados por vírgulas e duas casas decimais sempre que apresentar números.

                Comece!

                Question: {input}  
                Thought: {agent_scratchpad}"""
            )

            
            # Agente
            agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt)
            orquestrador = AgentExecutor(agent=agente,
                                        tools=tools,
                                        verbose=True,
                                        handle_parsing_errors=True)
        
        # PERGUNTA SOBRE OS DADOS
            st.markdown("---")
            st.markdown("## 🔎 Perguntas sobre os dados ou peça um 📊 gráfico com base em uma pergunta")
            
            
            with st.form(key="form_pergunta", clear_on_submit=False):
                pergunta_sobre_dados = st.text_input(
                    "Faça uma pergunta sobre os dados (ex: 'Qual é a média do tempo de entrega?' ou 'Crie um gráfico da média de tempo de entrega por clima.')",
                    placeholder="Digite sua pergunta aqui...",
                    key="input_pergunta"
                )
                
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    submit_button = st.form_submit_button("Enviar 📨", use_container_width=True)

            if submit_button:
                if pergunta_sobre_dados.strip():  # Verifica se não está vazio
                    with st.spinner("Analisando os dados 🦜"):
                        resposta = orquestrador.invoke({"input": pergunta_sobre_dados})
                        st.markdown(resposta["output"])
                else:
                    st.warning("⚠️ Por favor, digite uma pergunta!")
