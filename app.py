import streamlit as st
from detector import DetectorVazamentosColeipa

st.title("💧 Sistema de Detecção de Vazamentos - Coleipa")

detector = DetectorVazamentosColeipa()

menu = st.sidebar.selectbox("Menu", [
    "Visualizar dados",
    "Criar sistema fuzzy",
    "Treinar Bayesiano",
    "Simular série temporal",
    "Gerar relatório"
])

if menu == "Visualizar dados":
    detector.visualizar_dados_coleipa()
elif menu == "Criar sistema fuzzy":
    detector.criar_sistema_fuzzy(visualizar=True)
elif menu == "Treinar Bayesiano":
    X, y, _ = detector.gerar_dados_baseados_coleipa()
    detector.treinar_modelo_bayesiano(X, y)
elif menu == "Simular série temporal":
    X, y, _ = detector.gerar_dados_baseados_coleipa()
    detector.treinar_modelo_bayesiano(X, y, visualizar=False)
    detector.simular_serie_temporal_coleipa()
elif menu == "Gerar relatório":
    detector.gerar_relatorio_coleipa()
