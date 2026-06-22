import streamlit as st
import pandas as pd
from pypmml import Model

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Logística Inteligente | E-Commerce", layout="wide")

# --- CARREGAMENTO DE DADOS E MODELO ---
@st.cache_data
def carregar_dados():
    return pd.read_csv("ECommerceShippingData.csv")

try:
    modelo = Model.load("modelo_mlp.pmml")
    erro_pmml = None
except Exception as e:
    modelo = None
    erro_pmml = str(e)

df = carregar_dados()

# --- BARRA LATERAL (MENU LATERAL) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/411/411712.png", width=100)
st.sidebar.title("Menu de Navegação")
menu = st.sidebar.radio("Selecione a página:", 
                        ["🚚 Simulador de Frete", 
                         "📊 Base de Dados", 
                         "📈 Dashboard de Métricas"])

st.sidebar.markdown("---")
st.sidebar.info("Trabalho de Inteligência de Negócios\n\nAluno(a): Raul Castro Brasiel\n\nProfessora: Dra. Danielli Araújo Lima")


# ==========================================
# TELA 1: SIMULADOR DE FRETE (TESTE AO VIVO)
# ==========================================
if menu == "🚚 Simulador de Frete":
    st.title("Simulador Logístico: Previsão de Atrasos")
    st.markdown("Insira os dados da encomenda para verificar o risco de atraso na entrega.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Dados do Produto")
        peso = st.number_input("Peso do Pacote (gramas)", min_value=0, max_value=10000, value=3000)
        custo = st.number_input("Custo do Produto (USD)", min_value=0, value=150)
        importancia_input = st.selectbox("Importância do Produto", ["low", "medium", "high"])
        desconto = st.number_input("Desconto Oferecido (%)", min_value=0, max_value=100, value=5)

    with col2:
        st.subheader("🚛 Logística e Cliente")
        armazem_input = st.selectbox("Bloco do Armazém", ["A", "B", "C", "D", "F"])
        transporte_input = st.selectbox("Meio de Transporte", ["Flight", "Ship", "Road"])
        chamadas = st.slider("Chamadas ao Suporte", 1, 10, 3)
        avaliacao = st.slider("Avaliação do Cliente (Rating)", 1, 5, 3)
        comprasAnteriores = st.number_input("Compras Anteriores do Cliente", min_value=0, value=3)
        genero_input = st.radio("Gênero do Cliente", ["F", "M"])

    st.markdown("---")
    
    if st.button("Executar Previsão de Entrega", type="primary"):
        if modelo is not None:
            
            # --- 1. TRADUTORES (Ordem Alfabética Padrão do KNIME) ---
            dict_armazem = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4}
            dict_transporte = {"Flight": 0, "Road": 1, "Ship": 2} 
            dict_importancia = {"high": 0, "low": 1, "medium": 2}
            dict_genero = {"F": 0, "M": 1}

            blocoArmazem = dict_armazem[armazem_input]
            modoEmbarque = dict_transporte[transporte_input]
            importancia = dict_importancia[importancia_input]
            genero = dict_genero[genero_input]
            
            # --- 2. A MÁGICA DA NORMALIZAÇÃO (0.0 a 1.0) ---
            # Função matemática igual ao nó Normalizer do KNIME
            def normalizar(valor, min_val, max_val):
                return (valor - min_val) / (max_val - min_val)

            # Estruturando os dados e já normalizando em tempo real com base no DF original
            dados_entrada = pd.DataFrame([{
                "Weight_in_gms": normalizar(peso, df["Weight_in_gms"].min(), df["Weight_in_gms"].max()),
                "Cost_of_the_Product": normalizar(custo, df["Cost_of_the_Product"].min(), df["Cost_of_the_Product"].max()),
                "Discount_offered": normalizar(desconto, df["Discount_offered"].min(), df["Discount_offered"].max()),
                "Customer_care_calls": normalizar(chamadas, df["Customer_care_calls"].min(), df["Customer_care_calls"].max()),
                "Customer_rating": normalizar(avaliacao, df["Customer_rating"].min(), df["Customer_rating"].max()),
                "Prior_purchases": normalizar(comprasAnteriores, df["Prior_purchases"].min(), df["Prior_purchases"].max()),
                "Warehouse_block (to number)": normalizar(blocoArmazem, 0, 4),
                "Mode_of_Shipment (to number)": normalizar(modoEmbarque, 0, 2),
                "Product_importance (to number)": normalizar(importancia, 0, 2),
                "Gender (to number)": normalizar(genero, 0, 1)
            }])

            # Executa a previsão no modelo com os dados em escala decimal perfeita
            resultado = modelo.predict(dados_entrada)
            
            # Lê o resultado
            previsao = str(resultado.iloc[0, 0]) 

            if previsao == "1" or previsao == "1.0":
                st.error("⚠️ ALTO RISCO DE ATRASO: Recomenda-se alterar a malha logística ou o tipo de frete.")
            else:
                st.success("✅ ENTREGA NO PRAZO: O fluxo logístico atual atende a esta encomenda de forma segura.")
        else:
            st.error(f"O arquivo PMML foi encontrado, mas ocorreu um erro ao tentar lê-lo: {erro_pmml}")
            st.info("💡 Dica técnica: A biblioteca PyPMML exige que o Java (JRE) esteja instalado e configurado no Windows para funcionar.")


# ==========================================
# TELA 2: BASE DE DADOS
# ==========================================
elif menu == "📊 Base de Dados":
    st.title("Explorador de Dados Logísticos")
    st.markdown("Base de dados **E-Commerce Shipping Data** extraída do Kaggle para treinamento do algoritmo supervisionado.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Registros", df.shape[0])
    col2.metric("Total de Atributos", df.shape[1])
    col3.metric("Entregas Atrasadas", df[df['Reached.on.Time_Y.N'] == 1].shape[0])
    
    st.dataframe(df, use_container_width=True)


# ==========================================
# TELA 3: DASHBOARD E MÉTRICAS
# ==========================================
elif menu == "📈 Dashboard de Métricas":
    st.title("Resultados do Aprendizado de Máquina")
    
    st.header("1. Visualização de Dados")
    st.markdown("Análise prévia para identificação de padrões:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Influência do Peso no Atraso (Box Plot)**")
        st.image("imagens/Box Plot.png")
    with col2:
        st.markdown("**Distribuição por Transporte (Pie Chart)**")
        st.image("imagens/Pie Chart.png")

    st.markdown("---")
    
    st.header("2. Desempenho do Algoritmo Ganhador")
    st.markdown("Após testar 6 algoritmos, a **Rede Neural Artificial (RProp MLP)** obteve o melhor desempenho.")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Matriz de Confusão**")
        st.image("imagens/matrizDeConfusao.png")
    with col4:
        st.markdown("**Estatísticas Finais**")
        st.metric("Acurácia", "66,2%")
        st.metric("Recall (Captura de Atrasos)", "82,4%")
        st.metric("Cohen's Kappa", "0.349")