# 🚚 Inteligência Artificial na Logística: Previsão de Atrasos no E-Commerce

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![KNIME](https://img.shields.io/badge/KNIME-Analytics_Platform-yellow.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Neural%20Networks-success.svg)

## 📌 Sobre o Projeto
Este projeto foi desenvolvido como Trabalho para a disciplina de Inteligência de Negócios.

O objetivo principal é mitigar riscos operacionais e financeiros no setor de e-commerce utilizando **Machine Learning**. Através da análise de uma base de dados histórica do Kaggle, o algoritmo é capaz de prever se uma nova encomenda chegará no prazo ou sofrerá atrasos, permitindo ações preventivas por parte da gestão logística.

## ⚙️ Arquitetura e Tecnologias
O projeto foi dividido em duas frentes principais (Engenharia de Dados e Engenharia de Software):

1. **KNIME Analytics Platform:** - Pré-processamento de dados (`Category to Number`, `Normalizer`).
   - Partição de dados (Treino e Teste).
   - Treinamento e avaliação de 6 modelos preditivos (Decision Tree, Random Forest, SVM, KNN, PNN e RProp MLP).
   - Exportação do modelo vencedor através do nó `PMML Writer`.
2. **Python & Streamlit (Web App):**
   - Construção do frontend/backend do simulador interativo.
   - Leitura do modelo preditivo exportado (`.pmml`) utilizando a biblioteca `PyPMML`.

## 📊 Resultados do Modelo
Após submeter os algoritmos ao nó *Scorer*, a **Rede Neural Artificial (RProp MLP)** obteve o melhor desempenho, especialmente na métrica de Recall, que é crucial para identificar falhas antes que ocorram.

* **Acurácia:** 66,2%
* **Recall (Capacidade de achar atrasos):** 82,4%
* **F1-Score:** 66,0%
* **Cohen's Kappa:** 0.349

**Insights de Negócios (Feature Importance):** O modelo revelou que atributos como *Peso da Encomenda* e o *Desconto Oferecido* (promoções muito agressivas) são os principais responsáveis por sobrecarregar a esteira logística e causar atrasos.

---

## 🚀 Como executar o projeto localmente

Para rodar o simulador logístico na sua máquina, siga as instruções abaixo:

### Pré-requisitos
* Ter o **Python 3.8+** instalado.
* Ter o **Java JRE ou JDK** instalado e configurado no Windows (Obrigatório para a biblioteca PyPMML funcionar).

### Passo a Passo da Instalação

1. **Clone o repositório:**
```bash
git clone [https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git](https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git)
cd NOME-DO-REPOSITORIO
```
2. Crie um Ambiente Virtual (Recomendado para evitar conflitos de bibliotecas):
```bash
python -m venv venv
```
3. Ative o Ambiente Virtual:
```bash
.\venv\Scripts\activate
```
(Nota: Se ocorrer um erro de permissão no Windows, rode Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass antes de ativar).

4. Instale as dependências:
```bash
pip install streamlit pandas pypmml py4j
```
5. Inicie a aplicação:
```bash
streamlit run app.py
```
O aplicativo abrirá automaticamente no seu navegador padrão no endereço http://localhost:8501.
