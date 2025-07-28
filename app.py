import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import seaborn as sns
from datetime import datetime, timedelta
import io
import os
import time
import json
# Função para download de arquivos
def download_button(object_to_download, download_filename, button_text):
    """
    Gera um botão que permite o download de um objeto
    """
    if isinstance(object_to_download, pd.DataFrame):
        # Se for um DataFrame
        buffer = io.BytesIO()
        
        if download_filename.endswith('.csv'):
            object_to_download.to_csv(buffer, index=False)
            mime_type = "text/csv"
        else:
            object_to_download.to_excel(buffer, index=False)
            mime_type = "application/vnd.ms-excel"
        
        buffer.seek(0)
        st.download_button(
            label=button_text,
            data=buffer,
            file_name=download_filename,
            mime=mime_type
        )
    else:
        # Se for outro tipo de objeto
        st.warning("Tipo de objeto não suportado para download")

# Removido plotly e matplotlib.dates para compatibilidade

# Configurações para matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# SISTEMA DE ALERTAS INTELIGENTES
# ============================================================================

class SistemaAlertas:
    """Sistema inteligente de alertas com diferentes níveis e notificações"""
    
    def __init__(self):
        self.niveis_alerta = {
            'muito_baixo': {
                'cor': '#2ECC71', 
                'icone': '🟢', 
                'som': False,
                'prioridade': 1,
                'titulo': 'Operação Normal'
            },
            'baixo': {
                'cor': '#F39C12', 
                'icone': '🟡', 
                'som': False,
                'prioridade': 2,
                'titulo': 'Atenção'
            },
            'medio': {
                'cor': '#E67E22', 
                'icone': '🟠', 
                'som': True,
                'prioridade': 3,
                'titulo': 'Risco Moderado'
            },
            'alto': {
                'cor': '#E74C3C', 
                'icone': '🔴', 
                'som': True, 
                'notificacao': True,
                'prioridade': 4,
                'titulo': 'Risco Alto'
            },
            'critico': {
                'cor': '#8E44AD', 
                'icone': '🚨', 
                'som': True, 
                'notificacao': True, 
                'email': True,
                'prioridade': 5,
                'titulo': 'CRÍTICO'
            }
        }
        self.historico_alertas = []
        
    def determinar_nivel_alerta(self, risco):
        """Determina o nível do alerta baseado no risco"""
        if risco < 20:
            return 'muito_baixo'
        elif risco < 40:
            return 'baixo'
        elif risco < 60:
            return 'medio'
        elif risco < 80:
            return 'alto'
        else:
            return 'critico'
    
    def processar_alerta(self, risco, dados_sistema=None, localizacao=None):
        """Processa um alerta e determina as ações necessárias"""
        nivel = self.determinar_nivel_alerta(risco)
        config_nivel = self.niveis_alerta[nivel]
        
        alerta = {
            'timestamp': datetime.now(),
            'nivel': nivel,
            'risco': risco,
            'titulo': config_nivel['titulo'],
            'icone': config_nivel['icone'],
            'cor': config_nivel['cor'],
            'localizacao': localizacao,
            'dados_sistema': dados_sistema,
            'processado': False
        }
        
        self.historico_alertas.append(alerta)
        
        # Limitar histórico a 100 alertas
        if len(self.historico_alertas) > 100:
            self.historico_alertas = self.historico_alertas[-100:]
        
        return alerta
    
    def obter_alertas_ativos(self, ultimas_horas=24):
        """Obtém alertas das últimas horas"""
        limite_tempo = datetime.now() - timedelta(hours=ultimas_horas)
        return [a for a in self.historico_alertas if a['timestamp'] > limite_tempo]
    
    def exibir_alertas_streamlit(self, container):
        """Exibe alertas na interface Streamlit"""
        alertas_ativos = self.obter_alertas_ativos()
        
        if not alertas_ativos:
            container.success("🟢 Nenhum alerta ativo no momento")
            return
        
        # Ordenar por prioridade e timestamp
        alertas_ordenados = sorted(
            alertas_ativos, 
            key=lambda x: (self.niveis_alerta[x['nivel']]['prioridade'], x['timestamp']), 
            reverse=True
        )
        
        for alerta in alertas_ordenados[:5]:  # Mostrar apenas os 5 mais importantes
            with container.container():
                self._exibir_alerta_individual(alerta)
    
    def _exibir_alerta_individual(self, alerta):
        """Exibe um alerta individual"""
        config = self.niveis_alerta[alerta['nivel']]
        
        if alerta['nivel'] == 'critico':
            st.error(f"""
            {config['icone']} **{config['titulo']}** - {alerta['timestamp'].strftime('%H:%M:%S')}
            
            **Risco**: {alerta['risco']:.1f}%
            
            **Localização**: {alerta.get('localizacao', 'Sistema Principal')}
            
            ⚠️ **AÇÃO IMEDIATA NECESSÁRIA**
            """)
        elif alerta['nivel'] == 'alto':
            st.warning(f"""
            {config['icone']} **{config['titulo']}** - {alerta['timestamp'].strftime('%H:%M:%S')}
            
            **Risco**: {alerta['risco']:.1f}%
            
            **Localização**: {alerta.get('localizacao', 'Sistema Principal')}
            """)
        else:
            st.info(f"""
            {config['icone']} **{config['titulo']}** - {alerta['timestamp'].strftime('%H:%M:%S')}
            
            **Risco**: {alerta['risco']:.1f}%
            """)

def mostrar_pagina_bayes(detector):
    """Página do modelo Bayesiano"""
    st.header("🔄 Modelo Bayesiano")
    st.markdown("Treinamento e avaliação do modelo Naive Bayes para detecção de vazamentos")
    
    # Parâmetros de treinamento
    st.subheader("Parâmetros de Treinamento")
    n_amostras = st.slider("Número de amostras sintéticas", 100, 2000, 500, 100)
    
    # Botão para treinar o modelo
    if st.button("Treinar Modelo Bayesiano"):
        with st.spinner("Gerando dados sintéticos e treinando modelo..."):
            X, y, _ = detector.gerar_dados_baseados_coleipa(n_amostras)
            modelo, cm, report = detector.treinar_modelo_bayesiano(X, y)
            
            # Exibir resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Matriz de Confusão")
                fig_cm = visualizar_matriz_confusao(cm)
                st.pyplot(fig_cm)
            
            with col2:
                st.subheader("Relatório de Classificação")
                # Converter relatório para DataFrame para melhor visualização
                df_report = pd.DataFrame(report).transpose()
                df_report = df_report.round(3)
                st.dataframe(df_report)
                
                # Características do sistema
                st.markdown("#### Características do Sistema Coleipa")
                st.markdown(f"""
                - **População**: {detector.caracteristicas_sistema['populacao']} habitantes
                - **Área**: {detector.caracteristicas_sistema['area_territorial']/1000:.1f} km²
                - **Perdas reais**: {detector.caracteristicas_sistema['percentual_perdas']:.1f}%
                - **IVI**: {detector.caracteristicas_sistema['ivi']:.2f} (Categoria D - Muito Ruim)
                """)
    
    # Explicação do modelo
    st.markdown("---")
    st.subheader("Sobre o Modelo Bayesiano")
    st.markdown("""
    O modelo Naive Bayes é treinado com dados sintéticos gerados a partir dos padrões observados no Sistema Coleipa.
    Ele considera três parâmetros principais:
    
    1. **Vazão** - Valores altos indicam possíveis vazamentos
    2. **Pressão** - Valores baixos indicam possíveis vazamentos
    3. **IVI** - Sistemas com IVI alto têm maior probabilidade de vazamentos
    
    Os dados de treinamento são gerados com base nas seguintes características:
    
    - **Operação Normal**: 
      - Vazão média mais baixa
      - Pressão média mais alta
      - IVI médio mais baixo (simulando sistemas mais eficientes)
      
    - **Vazamento**: 
      - Vazão média mais alta
      - Pressão média mais baixa
      - IVI médio próximo ao do Coleipa (16.33)
    
    O classificador é então treinado para reconhecer esses padrões e identificar situações de vazamento em dados novos.
    """)

def visualizar_matriz_confusao(cm):
    """Visualiza matriz de confusão"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Vazamento'],
                yticklabels=['Normal', 'Vazamento'])
    plt.title('Matriz de Confusão - Sistema Coleipa')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    return plt.gcf()

def mostrar_pagina_mapa_calor(detector):
    """Página dos mapas de calor IVI"""
    st.header("🔥 Mapas de Calor IVI")
    st.markdown("Análise de risco para diferentes combinações de vazão e pressão, considerando diferentes valores de IVI")
    
    # Configuração do mapa de calor
    st.subheader("Configuração")
    resolucao = st.slider("Resolução do mapa", 10, 50, 30, 5, 
                         help="Valores maiores geram mapas mais detalhados, mas aumentam o tempo de processamento")
    
    # Botão para gerar mapas de calor
    if st.button("Gerar Mapas de Calor"):
        with st.spinner("Gerando mapas de calor IVI... Isso pode demorar alguns segundos."):
            fig, ivi_valores = gerar_mapa_calor_ivi(detector, resolucao)
            st.pyplot(fig)
    
    # Análise detalhada do IVI
    st.markdown("---")
    st.subheader("Análise Detalhada do IVI - Sistema Coleipa")
    
    st.markdown(f"""
    ##### 🔍 IVI Calculado: {detector.caracteristicas_sistema['ivi']:.2f}
    ##### 📊 Classificação: Categoria D (Muito Ruim)
    ##### ⚠️ Interpretação: IVI > 16 indica uso extremamente ineficiente de recursos
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Comparação com outras categorias:")
        st.markdown("""
        - 🟢 **Categoria A (IVI 1-4)**: Sistema eficiente, perdas próximas ao inevitável
        - 🟡 **Categoria B (IVI 4-8)**: Sistema regular, melhorias recomendadas
        - 🟠 **Categoria C (IVI 8-16)**: Sistema ruim, ações urgentes necessárias
        - 🔴 **Categoria D (IVI >16)**: Sistema muito ruim, intervenção imediata
        """)
    
    with col2:
        st.markdown("##### 🎯 Análise específica do Coleipa (IVI = 16.33):")
        st.markdown("""
        - As perdas reais são 16.33 vezes maiores que as inevitáveis
        - Potencial de redução de perdas > 400 L/ramal.dia
        - Localização no mapa: zona vermelha (alto risco)
        - Combinação crítica: Vazão ALTA + Pressão BAIXA
        - Prioridade máxima: pesquisa e reparo imediato de vazamentos
        """)

def gerar_mapa_calor_ivi(detector, resolucao=30):
    """
    Gera mapas de calor mostrando o risco de vazamento para diferentes
    combinações de vazão e pressão, com diferentes valores de IVI
    """
    # Verificar se o sistema fuzzy está criado
    if detector.sistema_fuzzy is None:
        detector.criar_sistema_fuzzy()
    
    # Valores de IVI baseados na classificação do Banco Mundial
    ivi_valores = [2, 6, 12, 18]  # Representativos das categorias A, B, C, D
    ivi_categorias = ['BOM (2.0)', 'REGULAR (6.0)', 'RUIM (12.0)', 'MUITO RUIM (18.0)']
    ivi_classificacoes = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
    
    # Valores para o mapa de calor baseados nos dados Coleipa
    vazoes = np.linspace(7, 16, resolucao)
    pressoes = np.linspace(2.5, 8, resolucao)
    
    # Configurar figura com subplots 2x2
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()  # Facilitar o acesso aos subplots
    
    # Gerar um mapa de calor para cada valor de IVI
    for idx, (ax, ivi_valor, categoria, classificacao) in enumerate(zip(axes, ivi_valores, ivi_categorias, ivi_classificacoes)):
        
        # Criar grade para o mapa
        X, Y = np.meshgrid(vazoes, pressoes)
        Z = np.zeros_like(X)
        
        # Calcular risco para cada ponto na grade
        for ii in range(X.shape[0]):
            for jj in range(X.shape[1]):
                try:
                    # Garantir que os valores estão dentro dos limites
                    vazao_val = max(7, min(X[ii, jj], 16))
                    pressao_val = max(2.5, min(Y[ii, jj], 8))
                    ivi_val = max(1, min(ivi_valor, 25))
                    
                    # Calcular risco usando o sistema fuzzy
                    detector.sistema_fuzzy.input['vazao'] = vazao_val
                    detector.sistema_fuzzy.input['pressao'] = pressao_val
                    detector.sistema_fuzzy.input['ivi'] = ivi_val
                    
                    detector.sistema_fuzzy.compute()
                    risco = detector.sistema_fuzzy.output['risco_vazamento']
                    Z[ii, jj] = max(0, min(risco, 100))
                    
                except Exception as e:
                    # Heurística baseada nos padrões do Coleipa
                    vazao_norm = (X[ii, jj] - 7) / (16 - 7)  # Normalizar 0-1
                    pressao_norm = 1 - (Y[ii, jj] - 2.5) / (8 - 2.5)  # Inverter
                    
                    # Calcular risco base
                    risco_base = (vazao_norm * 0.6 + pressao_norm * 0.4) * 70
                    
                    # Ajustar pelo IVI
                    fator_ivi = ivi_valor / 10
                    Z[ii, jj] = min(100, risco_base * fator_ivi + 10)
        
        # Plotar mapa de calor
        im = ax.imshow(Z, cmap='RdYlGn_r', origin='lower', 
                      extent=[vazoes.min(), vazoes.max(), pressoes.min(), pressoes.max()],
                      aspect='auto', vmin=0, vmax=100, interpolation='bilinear')
        
        # Marcar o ponto característico do Coleipa
        if idx == 3:  # Último gráfico (IVI Muito Ruim) - destaque especial
            ax.scatter([14.5], [3.5], color='red', s=300, marker='*', 
                      edgecolors='darkred', linewidth=3, label='Ponto Coleipa', zorder=10)
            ax.legend(loc='upper left', fontsize=9)
        else:
            ax.scatter([14.5], [3.5], color='red', s=150, marker='*', 
                      edgecolors='darkred', linewidth=2, alpha=0.7, zorder=8)
        
        # Configurações dos eixos
        ax.set_xlabel('Vazão (m³/h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressão (mca)', fontsize=12, fontweight='bold')
        ax.set_title(f'Mapa de Risco - IVI {categoria}\n{classificacao}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(7, 16)
        ax.set_ylim(2.5, 8)
    
    # Criar barra de cores
    fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8, 
                label='Risco de Vazamento (%)')
    
    plt.tight_layout()
    return fig, ivi_valores

def mostrar_pagina_simulacao(detector):
    """Página de simulação temporal"""
    st.header("⏱️ Simulação Temporal")
    st.markdown("Simulação de série temporal com detecção de vazamentos")
    
    # Verificar se o modelo Bayes está treinado
    if detector.modelo_bayes is None:
        st.warning("O modelo Bayesiano não está treinado. Treinando modelo com parâmetros padrão...")
        with st.spinner("Treinando modelo..."):
            X, y, _ = detector.gerar_dados_baseados_coleipa()
            detector.treinar_modelo_bayesiano(X, y)
    
    # Botão para executar simulação
    if st.button("Executar Simulação"):
        with st.spinner("Simulando série temporal... Isso pode demorar alguns segundos."):
            fig, df = simular_serie_temporal_coleipa(detector)
            st.pyplot(fig)
            
            # Mostrar dados da simulação
            with st.expander("Ver dados da simulação"):
                # Formatar coluna de tempo para exibição
                df_display = df.copy()
                if 'Tempo' in df_display.columns:
                    df_display['Tempo'] = df_display['Tempo'].dt.strftime('%d/%m %H:%M')
                
                st.dataframe(df_display)
    
    # Explicação da simulação
    st.markdown("---")
    st.subheader("Sobre a Simulação Temporal")
    st.markdown("""
    A simulação temporal representa o comportamento do sistema ao longo de 3 dias completos, com um vazamento 
    simulado iniciando no segundo dia às 14h. Características da simulação:
    
    #### Comportamento Normal
    - Vazão e pressão seguem os padrões observados no sistema Coleipa
    - Variações aleatórias pequenas são adicionadas para simular flutuações naturais
    - Comportamento cíclico diário com picos de consumo durante o dia e vales durante a noite
    
    #### Vazamento Simulado
    - Inicia no segundo dia às 14h
    - Progressão gradual ao longo de várias horas (simulando vazamento crescente)
    - Causa aumento na vazão e redução na pressão simultaneamente
    
    #### Sistema de Detecção
    - Componente Fuzzy: Avalia o risco com base nas regras definidas
    - Componente Bayes: Calcula a probabilidade com base nos dados aprendidos
    - Sistema Híbrido: Combina ambas as abordagens (60% fuzzy + 40% bayes)
    - Limiar de detecção: Probabilidade > 0.5 indica vazamento
    """)

def simular_serie_temporal_coleipa(detector):
    """Simula série temporal baseada nos padrões reais do Coleipa"""
    # Criar série temporal expandida (3 dias completos)
    tempo = []
    vazao = []
    pressao = []
    
    for dia in range(3):
        for hora in range(24):
            timestamp = datetime(2024, 1, 1 + dia, hora, 0)
            tempo.append(timestamp)
            
            # Usar padrão simulado baseado no Coleipa
            if hora in range(6, 22):  # Período diurno
                v = 12 + np.random.normal(0, 2)
                p = 4 + np.random.normal(0, 1)
            else:  # Período noturno
                v = 8 + np.random.normal(0, 1)
                p = 6 + np.random.normal(0, 0.5)
            
            vazao.append(max(7, min(v, 16)))
            pressao.append(max(2, min(p, 10)))
    
    # Simular vazamento começando no segundo dia às 14h
    inicio_vazamento = 24 + 14  # índice correspondente
    for i in range(inicio_vazamento, len(vazao)):
        # Progressão do vazamento
        progresso = min(1.0, (i - inicio_vazamento) / 10)
        vazao[i] += 3 * progresso  # Aumento gradual
        pressao[i] -= 1.5 * progresso  # Diminuição gradual
    
    # Criar DataFrame
    df = pd.DataFrame({
        'Tempo': tempo,
        'Vazao': vazao,
        'Pressao': pressao,
        'IVI': [detector.caracteristicas_sistema['ivi']] * len(tempo),
        'Vazamento_Real': [0] * inicio_vazamento + [1] * (len(tempo) - inicio_vazamento)
    })
    
    # Calcular detecções se o modelo estiver treinado
    if detector.modelo_bayes is not None:
        deteccoes = []
        for _, row in df.iterrows():
            risco_fuzzy = detector.avaliar_risco_fuzzy(row['Vazao'], row['Pressao'], row['IVI'])
            prob_bayes = detector.modelo_bayes.predict_proba([[row['Vazao'], row['Pressao'], row['IVI']]])[0][1]
            prob_hibrida = 0.6 * (risco_fuzzy/100) + 0.4 * prob_bayes
            deteccoes.append({
                'Risco_Fuzzy': risco_fuzzy/100,
                'Prob_Bayes': prob_bayes,
                'Prob_Hibrida': prob_hibrida
            })
        
        for col in deteccoes[0].keys():
            df[col] = [d[col] for d in deteccoes]
    
    return visualizar_serie_temporal_coleipa(df, inicio_vazamento)

def visualizar_serie_temporal_coleipa(df, inicio_vazamento):
    """Visualiza série temporal baseada no Coleipa"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Gráfico 1: Vazão
    axes[0].plot(range(len(df)), df['Vazao'], 'b-', linewidth=1.5, label='Vazão')
    axes[0].axvline(x=inicio_vazamento, color='red', linestyle='--', 
                   label=f'Início Vazamento (Hora {inicio_vazamento})')
    axes[0].set_ylabel('Vazão (m³/h)')
    axes[0].set_title('Série Temporal - Sistema Coleipa: Vazão')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Pressão
    axes[1].plot(range(len(df)), df['Pressao'], 'r-', linewidth=1.5, label='Pressão')
    axes[1].axhline(y=10, color='orange', linestyle=':', label='Mínimo NBR (10 mca)')
    axes[1].axvline(x=inicio_vazamento, color='red', linestyle='--')
    axes[1].set_ylabel('Pressão (mca)')
    axes[1].set_title('Série Temporal - Sistema Coleipa: Pressão')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gráfico 3: Detecções (se disponível)
    if 'Prob_Hibrida' in df.columns:
        axes[2].plot(range(len(df)), df['Prob_Hibrida'], 'purple', linewidth=2, label='Detecção Híbrida')
        axes[2].plot(range(len(df)), df['Risco_Fuzzy'], 'green', alpha=0.7, label='Componente Fuzzy')
        axes[2].plot(range(len(df)), df['Prob_Bayes'], 'orange', alpha=0.7, label='Componente Bayes')
        axes[2].axhline(y=0.5, color='black', linestyle='-.', label='Limiar Detecção')
        axes[2].axvline(x=inicio_vazamento, color='red', linestyle='--')
        axes[2].set_ylabel('Probabilidade')
        axes[2].set_title('Detecção de Vazamentos - Sistema Híbrido')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Modelo Bayesiano não treinado\nApenas análise fuzzy disponível', 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Detecção não disponível')
    
    axes[2].set_xlabel('Tempo (horas)')
    plt.tight_layout()
    return fig, df

def mostrar_pagina_relatorio(detector):
    """Página de relatório completo"""
    st.header("📝 Relatório Completo")
    st.markdown("Relatório detalhado baseado nos dados do sistema Coleipa")
    
    # Botão para gerar relatório
    if st.button("Gerar Relatório Completo"):
        with st.spinner("Gerando relatório..."):
            relatorio = gerar_relatorio_coleipa(detector)
            
            # Cabeçalho do relatório
            st.markdown("---")
            st.subheader("RELATÓRIO DE ANÁLISE - SISTEMA COLEIPA")
            st.markdown("---")
            
            # 1. Características do Sistema
            st.subheader("1. CARACTERÍSTICAS DO SISTEMA")
            st.markdown(f"""
            - **Localização**: {relatorio['caracteristicas']['localizacao']}
            - **Área territorial**: {relatorio['caracteristicas']['area']:.1f} km²
            - **População atendida**: {relatorio['caracteristicas']['populacao']} habitantes
            - **Número de ligações**: {relatorio['caracteristicas']['ligacoes']}
            - **Extensão da rede**: {relatorio['caracteristicas']['rede']} km
            - **Densidade de ramais**: {relatorio['caracteristicas']['densidade_ramais']} ramais/km
            """)
            
            # 2. Resultados do Monitoramento
            st.subheader("2. RESULTADOS DO MONITORAMENTO (72 horas)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Volume médio demandado", f"{relatorio['monitoramento']['volume_demandado']:.1f} m³/dia")
                st.metric("Volume médio consumido", f"{relatorio['monitoramento']['volume_consumido']:.1f} m³/dia")
            
            with col2:
                st.metric("Perdas reais médias", f"{relatorio['monitoramento']['perdas_reais']:.1f} m³/dia")
                st.metric("Percentual de perdas", f"{relatorio['monitoramento']['percentual_perdas']:.1f}%")
            
            # 3. Indicadores de Desempenho
            st.subheader("3. INDICADORES DE DESEMPENHO")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("IPRL", f"{relatorio['indicadores']['iprl']} m³/lig.dia", "Perdas Reais por Ligação")
            
            with col2:
                st.metric("IPRI", f"{relatorio['indicadores']['ipri']} m³/lig.dia", "Perdas Reais Inevitáveis")
            
            with col3:
                st.metric("IVI", f"{relatorio['indicadores']['ivi']}", "Índice de Vazamentos na Infraestrutura")
            
            # 4. Classificação
            st.subheader("4. CLASSIFICAÇÃO (Banco Mundial)")
            st.markdown(f"""
            - **Categoria**: {relatorio['classificacao']['categoria']}
            - **Interpretação**: {relatorio['classificacao']['interpretacao']}
            - **Recomendação**: {relatorio['classificacao']['recomendacao']}
            """)
            
            # 5. Metodologia NPR - Priorização de Ações
            st.subheader("5. METODOLOGIA NPR - PRIORIZAÇÃO DE AÇÕES")
            
            # Criar tabela de prioridades
            df_prioridades = pd.DataFrame(relatorio['prioridades'])
            df_prioridades.columns = ["Ordem", "Ação", "Resultado"]
            
            # Gráfico de barras para prioridades
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(
                [p['acao'] for p in relatorio['prioridades']], 
                [p['resultado'] for p in relatorio['prioridades']],
                color=['#3498db', '#2980b9', '#1f618d', '#154360']
            )
            ax.set_xlabel('Resultado NPR')
            ax.set_title('Priorização de Ações (Metodologia NPR)')
            
            # Adicionar valores nas barras
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 1
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width}', 
                       va='center', fontweight='bold')
            
            st.pyplot(fig)
            st.dataframe(df_prioridades)
            
            # 6. Problemas Identificados
            st.subheader("6. PROBLEMAS IDENTIFICADOS")
            for i, problema in enumerate(relatorio['problemas'], 1):
                st.markdown(f"- {problema}")
            
            # 7. Recomendações
            st.subheader("7. RECOMENDAÇÕES")
            for i, recomendacao in enumerate(relatorio['recomendacoes'], 1):
                st.markdown(f"- **Recomendação {i}**: {recomendacao}")
            
            st.markdown("---")
            st.success("Relatório gerado com sucesso!")

def gerar_relatorio_coleipa(detector):
    """Gera relatório completo baseado nos dados do Coleipa"""
    relatorio = {
        "caracteristicas": {
            "localizacao": "Bairro Coleipa, Santa Bárbara do Pará-PA",
            "area": detector.caracteristicas_sistema['area_territorial']/1000,
            "populacao": detector.caracteristicas_sistema['populacao'],
            "ligacoes": detector.caracteristicas_sistema['numero_ligacoes'],
            "rede": detector.caracteristicas_sistema['comprimento_rede'],
            "densidade_ramais": detector.caracteristicas_sistema['densidade_ramais']
        },
        "monitoramento": {
            "volume_demandado": 273.5,
            "volume_consumido": detector.caracteristicas_sistema['volume_consumido_medio'],
            "perdas_reais": detector.caracteristicas_sistema['perdas_reais_media'],
            "percentual_perdas": detector.caracteristicas_sistema['percentual_perdas']
        },
        "indicadores": {
            "iprl": detector.caracteristicas_sistema['iprl'],
            "ipri": detector.caracteristicas_sistema['ipri'],
            "ivi": detector.caracteristicas_sistema['ivi']
        },
        "classificacao": {
            "categoria": "D (Muito Ruim)",
            "interpretacao": "Uso ineficiente de recursos",
            "recomendacao": "Programas de redução de perdas são imperiosos e prioritários"
        },
        "prioridades": [
            {"ordem": 1, "acao": "Pesquisa de vazamentos", "resultado": 40},
            {"ordem": 2, "acao": "Agilidade e qualidade dos reparos", "resultado": 32},
            {"ordem": 3, "acao": "Gerenciamento de infraestrutura", "resultado": 28},
            {"ordem": 4, "acao": "Gerenciamento de pressão", "resultado": 2}
        ],
        "problemas": [
            "Pressões abaixo do mínimo NBR 12218 (10 mca)",
            "Vazões mínimas noturnas elevadas (>50% da máxima)",
            "Comportamento inverso vazão-pressão característico de vazamentos",
            "IVI classificado como 'Muito Ruim' (>16)"
        ],
        "recomendacoes": [
            "Implementar programa intensivo de pesquisa de vazamentos",
            "Cadastrar e reparar vazamentos visíveis rapidamente",
            "Considerar aumento da altura do reservatório",
            "Substituir trechos com vazamentos recorrentes",
            "Mobilizar a comunidade para identificação de vazamentos"
        ]
    }
    
    return relatorio

def mostrar_pagina_configuracoes(detector):
    """Página de configurações do sistema"""
    st.header("⚙️ Configurações do Sistema")
    st.markdown("Personalize as características do sistema de abastecimento")
    
    # Exibir características atuais
    st.subheader("Características Atuais do Sistema")
    caracteristicas = detector.caracteristicas_sistema
    
    # Criar DataFrame para exibir as características atuais de forma organizada
    df_caracteristicas = pd.DataFrame({
        'Característica': list(caracteristicas.keys()),
        'Valor Atual': list(caracteristicas.values())
    })
    st.dataframe(df_caracteristicas)
    
    # Formulário para atualizar características
    st.subheader("Atualizar Características")
    st.markdown("Preencha os campos abaixo para atualizar as características do sistema. Deixe em branco para manter o valor atual.")
    
    # Lista das principais características para atualizar
    caracteristicas_para_atualizar = [
        'area_territorial', 'populacao', 'numero_ligacoes', 
        'comprimento_rede', 'densidade_ramais', 'percentual_perdas', 'ivi'
    ]
    
    # Criar formulário
    with st.form("form_caracteristicas"):
        # Dividir em colunas
        col1, col2 = st.columns(2)
        
        # Dicionário para armazenar novas características
        novas_caracteristicas = {}
        
        # Primeira coluna
        with col1:
            for carac in caracteristicas_para_atualizar[:3]:
                valor_atual = detector.caracteristicas_sistema[carac]
                valor = st.number_input(
                    f"{carac.replace('_', ' ').title()} (atual: {valor_atual})",
                    value=None,
                    placeholder=f"Valor atual: {valor_atual}"
                )
                if valor is not None:
                    novas_caracteristicas[carac] = valor
        
        # Segunda coluna
        with col2:
            for carac in caracteristicas_para_atualizar[3:]:
                valor_atual = detector.caracteristicas_sistema[carac]
                valor = st.number_input(
                    f"{carac.replace('_', ' ').title()} (atual: {valor_atual})",
                    value=None,
                    placeholder=f"Valor atual: {valor_atual}"
                )
                if valor is not None:
                    novas_caracteristicas[carac] = valor
        
        # Botão para atualizar
        botao_atualizar = st.form_submit_button("Atualizar Características")
    
    if botao_atualizar:
        if novas_caracteristicas:
            # Atualizar características
            for chave, valor in novas_caracteristicas.items():
                if chave in detector.caracteristicas_sistema:
                    detector.caracteristicas_sistema[chave] = valor
                    st.success(f"Característica '{chave}' atualizada para: {valor}")
            
            st.success("Características atualizadas com sucesso!")
        else:
            st.info("Nenhuma característica foi alterada.")
    
    # Opções adicionais
    st.markdown("---")
    st.subheader("Opções Adicionais")
    
    # Baixar template de dados
    st.markdown("##### Template de Dados")
    if st.button("📄 Gerar Template de Dados"):
        df_template = detector.gerar_dados_template()
        st.success("Template gerado com dados padrão do Coleipa!")
        st.dataframe(df_template)
    
    # Redefinir para valores padrão
    st.markdown("##### Redefinir Sistema")
    if st.button("Redefinir para Valores Padrão"):
        # Recriar detector com valores padrão
        st.session_state['detector'] = DetectorVazamentosColeipa()
        st.success("Sistema redefinido para valores padrão!")
        st.rerun()
# CONFIGURAÇÃO RESPONSIVA E MOBILE
# ============================================================================

def configurar_layout_responsivo():
    """Configura layout responsivo para diferentes dispositivos"""
    
    # Detectar largura da tela (aproximado)
    st.markdown("""
    <script>
    function detectScreenSize() {
        const width = window.innerWidth;
        const isMobile = width < 768;
        const isTablet = width >= 768 && width < 1024;
        
        if (isMobile) {
            document.body.classList.add('mobile-view');
        } else if (isTablet) {
            document.body.classList.add('tablet-view');
        }
    }
    detectScreenSize();
    window.addEventListener('resize', detectScreenSize);
    </script>
    """, unsafe_allow_html=True)
    
    # CSS responsivo
    st.markdown("""
    <style>
    /* Estilos base */
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
        min-width: 200px;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    
    .alert-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .dashboard-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-top: 4px solid #1f77b4;
    }
    
    /* Estilos para mobile */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
        }
        
        .metric-container {
            flex-direction: column;
            align-items: center;
        }
        
        .metric-card {
            min-width: 90%;
            margin: 0.5rem 0;
        }
        
        .dashboard-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Ajustar tamanho dos gráficos */
        .js-plotly-plot {
            width: 100% !important;
        }
        
        /* Sidebar responsiva */
        .css-1d391kg {
            width: 100% !important;
        }
    }
    
    /* Estilos para tablet */
    @media (min-width: 768px) and (max-width: 1024px) {
        .metric-container {
            justify-content: space-between;
        }
        
        .metric-card {
            min-width: 45%;
        }
    }
    
    /* Animações suaves */
    .metric-card, .dashboard-card, .alert-container {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover, .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Indicadores de status animados */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .status-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Botões melhorados */
    .stButton > button {
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# GERENCIADOR DE DADOS EM TEMPO REAL
# ============================================================================

class GerenciadorTempoReal:
    """Gerencia dados simulados para demonstração"""
    
    def __init__(self):
        self.dados_atuais = {}
        self.ativo = False
        
    def iniciar_monitoramento(self):
        """Inicia o modo de demonstração"""
        self.ativo = True
    
    def parar_monitoramento(self):
        """Para o modo de demonstração"""
        self.ativo = False
    
    def _simular_dados_sensor(self):
        """Simula dados de sensores"""
        import random
        
        # Adicionar alguma variação realística
        base_time = datetime.now()
        hour = base_time.hour
        
        # Padrão típico: vazão maior durante o dia, menor à noite
        if 6 <= hour <= 22:  # Período diurno
            vazao_base = 12 + random.normalvariate(0, 2)
            pressao_base = 4 + random.normalvariate(0, 1)
        else:  # Período noturno
            vazao_base = 8 + random.normalvariate(0, 1)
            pressao_base = 6 + random.normalvariate(0, 0.5)
        
        # Ocasionalmente simular uma anomalia
        if random.random() < 0.15:  # 15% de chance
            vazao_base += random.uniform(3, 6)  # Aumentar vazão
            pressao_base -= random.uniform(1, 2)  # Diminuir pressão
        
        dados = {
            'timestamp': base_time,
            'vazao': max(7, min(vazao_base, 16)),
            'pressao': max(2, min(pressao_base, 10)),
            'ivi': 16.33,  # IVI fixo do Coleipa
            'temperatura': random.uniform(20, 35),
            'ph': random.uniform(6.5, 8.5),
            'turbidez': random.uniform(0.1, 2.0)
        }
        
        self.dados_atuais = dados
        return dados
    
    def obter_dados_atuais(self):
        """Retorna os dados mais recentes"""
        if not self.dados_atuais:
            return self._simular_dados_sensor()
        return self.dados_atuais

# ============================================================================
# CLASSE PRINCIPAL MELHORADA
# ============================================================================

class DetectorVazamentosColeipa:
    """
    Sistema híbrido Fuzzy-Bayes para detecção de vazamentos baseado nos dados 
    do Sistema de Abastecimento de Água Potável (SAAP) do bairro da Coleipa
    """
    
    def __init__(self, arquivo_dados=None):
        """
        Inicializa o sistema baseado nos dados do artigo Coleipa
        """
        # Dados padrão do sistema Coleipa
        self.caracteristicas_sistema = {
            'area_territorial': 319000,  # m²
            'populacao': 1200,  # habitantes
            'numero_ligacoes': 300,  # ligações
            'comprimento_rede': 3,  # km
            'densidade_ramais': 100,  # ramais/km
            'vazao_media_normal': 3.17,  # l/s (média dos três dias)
            'pressao_media_normal': 5.22,  # mca (média dos três dias)
            'perdas_reais_media': 102.87,  # m³/dia
            'volume_consumido_medio': 128.29,  # m³/dia
            'percentual_perdas': 44.50,  # %
            'iprl': 0.343,  # m³/lig.dia
            'ipri': 0.021,  # m³/lig.dia
            'ivi': 16.33  # Índice de Vazamentos na Infraestrutura
        }
        
        # Dados padrão hardcoded (usados apenas se não for fornecido um arquivo)
        self.dados_coleipa_default = {
            'hora': list(range(1, 25)),
            'vazao_dia1': [8.11, 7.83, 7.76, 7.80, 8.08, 9.69, 11.52, 11.92, 13.08, 14.22, 15.68, 14.55, 14.78, 13.16, 12.81, 11.64, 13.02, 13.40, 13.55, 12.94, 12.63, 11.45, 9.88, 8.30],
            'pressao_dia1': [6.89, 7.25, 7.12, 6.51, 6.42, 6.18, 4.83, 3.57, 4.67, 3.92, 3.70, 3.11, 2.68, 2.55, 3.34, 3.77, 4.70, 4.66, 4.69, 3.77, 4.78, 5.73, 6.24, 6.36],
            'vazao_dia2': [7.42, 7.23, 7.20, 7.28, 7.53, 8.99, 10.50, 12.18, 13.01, 13.22, 14.27, 13.63, 13.21, 12.97, 12.36, 11.98, 13.50, 13.27, 14.85, 12.84, 11.29, 10.79, 9.55, 8.60],
            'pressao_dia2': [6.65, 7.41, 7.45, 7.46, 7.38, 6.85, 5.47, 5.69, 4.05, 3.99, 4.56, 4.25, 2.87, 3.37, 4.86, 5.99, 6.39, 6.16, 5.16, 3.86, 3.96, 5.49, 6.48, 6.62],
            'vazao_dia3': [7.67, 7.46, 7.76, 8.18, 8.44, 9.24, 11.05, 12.55, 13.65, 13.63, 14.98, 14.09, 14.00, 13.06, 12.58, 11.81, 13.40, 14.29, 14.06, 12.69, 12.12, 10.83, 8.93, 8.45],
            'pressao_dia3': [6.91, 7.26, 7.12, 7.30, 7.16, 7.05, 6.43, 3.96, 4.70, 3.77, 3.97, 4.06, 4.13, 3.78, 3.12, 3.34, 5.55, 5.41, 4.93, 3.81, 4.37, 5.61, 6.36, 6.49]
        }
        
        # Tentar carregar dados de arquivo se fornecido
        if arquivo_dados:
            self.dados_coleipa = self.carregar_dados_arquivo(arquivo_dados)
            st.success(f"Dados carregados do arquivo")
        else:
            self.dados_coleipa = self.dados_coleipa_default
            st.info("Usando dados padrão Coleipa (nenhum arquivo fornecido)")
        
        # Definição dos parâmetros fuzzy baseados nos dados reais do Coleipa
        # Vazão em m³/h (convertido de l/s)
        self.param_vazao = {
            'BAIXA': {'range': [7, 9, 11]},     # Vazões noturnas
            'NORMAL': {'range': [9, 11.5, 14]},  # Vazões de transição
            'ALTA': {'range': [12, 15, 16]}     # Vazões de pico
        }
        
        # Pressão em mca (dados originais do artigo)
        self.param_pressao = {
            'BAIXA': {'range': [0, 3, 5]},      # Abaixo do mínimo NBR (10 mca)
            'MEDIA': {'range': [4, 6, 8]},      # Faixa operacional observada
            'ALTA': {'range': [6, 8, 10]}      # Máximos observados
        }
        
        # IVI baseado na classificação do Banco Mundial
        self.param_ivi = {
            'BOM': {'range': [1, 2, 4]},        # Categoria A
            'REGULAR': {'range': [4, 6, 8]},    # Categoria B
            'RUIM': {'range': [8, 12, 16]},     # Categoria C
            'MUITO_RUIM': {'range': [16, 20, 25]}  # Categoria D (Coleipa = 16.33)
        }
        
        # Risco de vazamento
        self.param_risco = {
            'MUITO_BAIXO': {'range': [0, 10, 25]},
            'BAIXO': {'range': [15, 30, 45]},
            'MEDIO': {'range': [35, 50, 65]},
            'ALTO': {'range': [55, 70, 85]},
            'MUITO_ALTO': {'range': [75, 90, 100]}
        }
        
        # Inicialização dos componentes
        self.sistema_fuzzy = None
        self.modelo_bayes = None
        
        # Inicializar sistema de alertas
        self.sistema_alertas = SistemaAlertas()
    
    def gerar_dados_baseados_coleipa(self, n_amostras=500):
        """Gera dados sintéticos baseados nas características do sistema Coleipa"""
        # Simular dados baseados nos padrões observados
        vazao_normal_mean = 10.5
        vazao_normal_std = 2.0
        pressao_normal_mean = 5.5
        pressao_normal_std = 1.0
        
        vazao_vazamento_mean = 14.0
        vazao_vazamento_std = 1.5
        pressao_vazamento_mean = 3.5
        pressao_vazamento_std = 0.8
        
        # Gerar dados sintéticos
        n_normal = int(0.55 * n_amostras)  # 55% normal
        n_vazamento = n_amostras - n_normal
        
        # Dados normais
        vazao_normal = np.random.normal(vazao_normal_mean, vazao_normal_std, n_normal)
        pressao_normal = np.random.normal(pressao_normal_mean, pressao_normal_std, n_normal)
        ivi_normal = np.random.normal(8, 2, n_normal)
        
        # Dados de vazamento
        vazao_vazamento = np.random.normal(vazao_vazamento_mean, vazao_vazamento_std, n_vazamento)
        pressao_vazamento = np.random.normal(pressao_vazamento_mean, pressao_vazamento_std, n_vazamento)
        ivi_vazamento = np.random.normal(16.33, 3, n_vazamento)
        
        # Combinar dados
        X = np.vstack([
            np.column_stack([vazao_normal, pressao_normal, ivi_normal]),
            np.column_stack([vazao_vazamento, pressao_vazamento, ivi_vazamento])
        ])
        
        y = np.hstack([np.zeros(n_normal), np.ones(n_vazamento)])
        
        return X, y, None
    
    def treinar_modelo_bayesiano(self, X, y):
        """Treina modelo Bayesiano com dados baseados no Coleipa"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.modelo_bayes = GaussianNB()
        self.modelo_bayes.fit(X_train, y_train)
        
        y_pred = self.modelo_bayes.predict(X_test)
        
        # Calcular matriz de confusão e relatório de classificação
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Vazamento'], output_dict=True)
        
    def criar_dataframe_coleipa(self):
        """Cria DataFrame com os dados reais do monitoramento Coleipa"""
        df = pd.DataFrame()
        
        # Calcular médias e desvios padrão
        for hora in range(1, 25):
            idx = hora - 1
            vazao_valores = []
            pressao_valores = []
            
            # Verificar se temos dados para esta hora
            if idx < len(self.dados_coleipa['hora']):
                if 'vazao_dia1' in self.dados_coleipa and idx < len(self.dados_coleipa['vazao_dia1']):
                    vazao_valores.append(self.dados_coleipa['vazao_dia1'][idx])
                if 'vazao_dia2' in self.dados_coleipa and idx < len(self.dados_coleipa['vazao_dia2']):
                    vazao_valores.append(self.dados_coleipa['vazao_dia2'][idx])
                if 'vazao_dia3' in self.dados_coleipa and idx < len(self.dados_coleipa['vazao_dia3']):
                    vazao_valores.append(self.dados_coleipa['vazao_dia3'][idx])
                    
                if 'pressao_dia1' in self.dados_coleipa and idx < len(self.dados_coleipa['pressao_dia1']):
                    pressao_valores.append(self.dados_coleipa['pressao_dia1'][idx])
                if 'pressao_dia2' in self.dados_coleipa and idx < len(self.dados_coleipa['pressao_dia2']):
                    pressao_valores.append(self.dados_coleipa['pressao_dia2'][idx])
                if 'pressao_dia3' in self.dados_coleipa and idx < len(self.dados_coleipa['pressao_dia3']):
                    pressao_valores.append(self.dados_coleipa['pressao_dia3'][idx])
            
            # Se não temos dados suficientes, pular esta hora
            if len(vazao_valores) == 0 or len(pressao_valores) == 0:
                continue
                
            df = pd.concat([df, pd.DataFrame({
                'Hora': [hora],
                'Vazao_Dia1': [vazao_valores[0] if len(vazao_valores) > 0 else None],
                'Vazao_Dia2': [vazao_valores[1] if len(vazao_valores) > 1 else None],
                'Vazao_Dia3': [vazao_valores[2] if len(vazao_valores) > 2 else None],
                'Vazao_Media': [np.mean(vazao_valores)],
                'Vazao_DP': [np.std(vazao_valores)],
                'Pressao_Dia1': [pressao_valores[0] if len(pressao_valores) > 0 else None],
                'Pressao_Dia2': [pressao_valores[1] if len(pressao_valores) > 1 else None],
                'Pressao_Dia3': [pressao_valores[2] if len(pressao_valores) > 2 else None],
                'Pressao_Media': [np.mean(pressao_valores)],
                'Pressao_DP': [np.std(pressao_valores)],
                'IVI': [self.caracteristicas_sistema['ivi']],
                'Perdas_Detectadas': [1 if np.mean(vazao_valores) > 13 and np.mean(pressao_valores) < 5 else 0]
            })], ignore_index=True)
        
        return df
    
    def visualizar_dados_coleipa(self):
        """Visualiza os dados reais do monitoramento Coleipa"""
        df = self.criar_dataframe_coleipa()
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Gráfico 1: Vazões dos três dias
        if not df.empty:
            axes[0].plot(df['Hora'], df['Vazao_Dia1'], 'b-o', label='Dia 1', alpha=0.7)
            axes[0].plot(df['Hora'], df['Vazao_Dia2'], 'r-s', label='Dia 2', alpha=0.7)
            axes[0].plot(df['Hora'], df['Vazao_Dia3'], 'g-^', label='Dia 3', alpha=0.7)
            axes[0].plot(df['Hora'], df['Vazao_Media'], 'k-', linewidth=3, label='Média')
            axes[0].fill_between(df['Hora'], 
                               df['Vazao_Media'] - df['Vazao_DP'], 
                               df['Vazao_Media'] + df['Vazao_DP'], 
                               alpha=0.2, color='gray', label='±1σ')
        axes[0].set_title('Monitoramento de Vazão - SAAP Coleipa (72 horas)')
        axes[0].set_xlabel('Hora do Dia')
        axes[0].set_ylabel('Vazão (m³/h)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Pressões dos três dias
        if not df.empty:
            axes[1].plot(df['Hora'], df['Pressao_Dia1'], 'b-o', label='Dia 1', alpha=0.7)
            axes[1].plot(df['Hora'], df['Pressao_Dia2'], 'r-s', label='Dia 2', alpha=0.7)
            axes[1].plot(df['Hora'], df['Pressao_Dia3'], 'g-^', label='Dia 3', alpha=0.7)
            axes[1].plot(df['Hora'], df['Pressao_Media'], 'k-', linewidth=3, label='Média')
            axes[1].fill_between(df['Hora'], 
                               df['Pressao_Media'] - df['Pressao_DP'], 
                               df['Pressao_Media'] + df['Pressao_DP'], 
                               alpha=0.2, color='gray', label='±1σ')
        axes[1].axhline(y=10, color='red', linestyle='--', label='Mínimo NBR 12218 (10 mca)')
        axes[1].set_title('Monitoramento de Pressão - SAAP Coleipa (72 horas)')
        axes[1].set_xlabel('Hora do Dia')
        axes[1].set_ylabel('Pressão (mca)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gráfico 3: Comportamento inverso Vazão vs Pressão
        if not df.empty:
            ax2 = axes[2].twinx()
            line1 = axes[2].plot(df['Hora'], df['Vazao_Media'], 'b-', linewidth=2, label='Vazão Média')
            line2 = ax2.plot(df['Hora'], df['Pressao_Media'], 'r-', linewidth=2, label='Pressão Média')
            
            axes[2].set_xlabel('Hora do Dia')
            axes[2].set_ylabel('Vazão (m³/h)', color='b')
            ax2.set_ylabel('Pressão (mca)', color='r')
            axes[2].tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Combinar legendas
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[2].legend(lines, labels, loc='upper left')
        
        axes[2].set_title('Comportamento Inverso: Vazão × Pressão (Rede Setorizada)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calcular estatísticas
        stats = {}
        if not df.empty:
            stats = {
                "vazao_min": df['Vazao_Media'].min(),
                "vazao_min_hora": df.loc[df['Vazao_Media'].idxmin(), 'Hora'],
                "vazao_max": df['Vazao_Media'].max(),
                "vazao_max_hora": df.loc[df['Vazao_Media'].idxmax(), 'Hora'],
                "pressao_min": df['Pressao_Media'].min(),
                "pressao_min_hora": df.loc[df['Pressao_Media'].idxmin(), 'Hora'],
                "pressao_max": df['Pressao_Media'].max(),
                "pressao_max_hora": df.loc[df['Pressao_Media'].idxmax(), 'Hora'],
                "vazao_ratio": df['Vazao_Media'].min()/df['Vazao_Media'].max()*100,
                "horas_pressao_baixa": len(df[df['Pressao_Media'] < 10]),
                "perc_pressao_baixa": len(df[df['Pressao_Media'] < 10])/24*100
            }
        else:
            stats = {
                "vazao_min": 0, "vazao_min_hora": 0, "vazao_max": 0, "vazao_max_hora": 0,
                "pressao_min": 0, "pressao_min_hora": 0, "pressao_max": 0, "pressao_max_hora": 0,
                "vazao_ratio": 0, "horas_pressao_baixa": 0, "perc_pressao_baixa": 0
            }
        
        return fig, stats, df
        
    def visualizar_conjuntos_fuzzy(self):
        """Visualiza os conjuntos fuzzy baseados nos dados do Coleipa"""
        # Criar sistema fuzzy se ainda não existir
        vazao, pressao, ivi, risco_vazamento = self.criar_sistema_fuzzy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Vazão
        axes[0, 0].clear()
        for nome in self.param_vazao.keys():
            axes[0, 0].plot(vazao.universe, vazao[nome].mf, label=nome, linewidth=2)
        axes[0, 0].set_title('Conjuntos Fuzzy - Vazão (baseado nos dados Coleipa)')
        axes[0, 0].set_xlabel('Vazão (m³/h)')
        axes[0, 0].set_ylabel('Grau de Pertinência')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pressão
        axes[0, 1].clear()
        for nome in self.param_pressao.keys():
            axes[0, 1].plot(pressao.universe, pressao[nome].mf, label=nome, linewidth=2)
        axes[0, 1].set_title('Conjuntos Fuzzy - Pressão (baseado nos dados Coleipa)')
        axes[0, 1].set_xlabel('Pressão (mca)')
        axes[0, 1].set_ylabel('Grau de Pertinência')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IVI
        axes[1, 0].clear()
        for nome in self.param_ivi.keys():
            axes[1, 0].plot(ivi.universe, ivi[nome].mf, label=nome, linewidth=2)
        axes[1, 0].set_title('Conjuntos Fuzzy - IVI (Classificação Banco Mundial)')
        axes[1, 0].set_xlabel('IVI')
        axes[1, 0].set_ylabel('Grau de Pertinência')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=16.33, color='red', linestyle='--', label='Coleipa (16.33)')
        
        # Risco
        axes[1, 1].clear()
        for nome in self.param_risco.keys():
            axes[1, 1].plot(risco_vazamento.universe, risco_vazamento[nome].mf, label=nome, linewidth=2)
        axes[1, 1].set_title('Conjuntos Fuzzy - Risco de Vazamento')
        axes[1, 1].set_xlabel('Risco (%)')
        axes[1, 1].set_ylabel('Grau de Pertinência')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analisar_caso_coleipa(self, vazao=None, pressao=None, ivi=None):
        """Analisa um caso específico usando os padrões do Coleipa"""
        # Usar valores típicos do Coleipa se não fornecidos
        if vazao is None:
            vazao = 14.5  # Vazão típica de pico
        if pressao is None:
            pressao = 3.5   # Pressão baixa típica
        if ivi is None:
            ivi = 16.33   # IVI real do Coleipa
        
        # Classificação baseada nos dados Coleipa
        if vazao < 9:
            classe_vazao = "BAIXA (noturna)"
        elif vazao < 14:
            classe_vazao = "NORMAL (transição)"
        else:
            classe_vazao = "ALTA (pico/vazamento)"
        
        if pressao < 5:
            classe_pressao = "BAIXA (problema)"
        elif pressao < 8:
            classe_pressao = "MÉDIA (operacional)"
        else:
            classe_pressao = "ALTA (boa)"
        
        if ivi < 4:
            classe_ivi = "BOM (Categoria A)"
        elif ivi < 8:
            classe_ivi = "REGULAR (Categoria B)"
        elif ivi < 16:
            classe_ivi = "RUIM (Categoria C)"
        else:
            classe_ivi = "MUITO RUIM (Categoria D)"
        
        # Avaliação fuzzy
        risco_fuzzy = self.avaliar_risco_fuzzy(vazao, pressao, ivi)
        
        resultado = {}
        resultado['vazao'] = vazao
        resultado['pressao'] = pressao
        resultado['ivi'] = ivi
        resultado['classe_vazao'] = classe_vazao
        resultado['classe_pressao'] = classe_pressao
        resultado['classe_ivi'] = classe_ivi
        resultado['risco_fuzzy'] = risco_fuzzy
        
        # Avaliação Bayesiana (se disponível)
        if self.modelo_bayes is not None:
            dados = [vazao, pressao, ivi]
            prob_bayes = self.modelo_bayes.predict_proba([dados])[0][1]
            prob_hibrida = 0.6 * (risco_fuzzy/100) + 0.4 * prob_bayes
            
            resultado['prob_bayes'] = prob_bayes
            resultado['prob_hibrida'] = prob_hibrida
            
            if prob_hibrida > 0.5:
                resultado['status'] = "VAZAMENTO DETECTADO"
                resultado['cor'] = "🔴"
            elif prob_hibrida > 0.3:
                resultado['status'] = "RISCO ELEVADO - MONITORAR"
                resultado['cor'] = "🟡"
            else:
                resultado['status'] = "OPERAÇÃO NORMAL"
                resultado['cor'] = "🟢"
        else:
            if risco_fuzzy > 50:
                resultado['status'] = "RISCO ELEVADO (apenas análise fuzzy)"
                resultado['cor'] = "🟡"
            else:
                resultado['status'] = "RISCO BAIXO (apenas análise fuzzy)"
                resultado['cor'] = "🟢"
        
        # Comparação com dados reais do Coleipa
        resultado['percentual_perdas'] = self.caracteristicas_sistema['percentual_perdas']
        resultado['ivi_real'] = self.caracteristicas_sistema['ivi']
        
        return resultado
    
    def gerar_dados_template(self):
        """Gera os dados padrão para download como template"""
        df = pd.DataFrame(self.dados_coleipa_default)
        return df
    
    def carregar_dados_arquivo(self, arquivo_uploaded):
        """Carrega dados de monitoramento de um arquivo Excel ou CSV do Streamlit"""
        try:
            nome_arquivo = arquivo_uploaded.name
            nome, extensao = os.path.splitext(nome_arquivo)
            extensao = extensao.lower()
            
            if extensao == '.xlsx' or extensao == '.xls':
                df = pd.read_excel(arquivo_uploaded)
                st.success("Arquivo Excel carregado com sucesso")
            elif extensao == '.csv':
                df = pd.read_csv(arquivo_uploaded)
                st.success("Arquivo CSV carregado com sucesso")
            else:
                st.error(f"Formato de arquivo não suportado: {extensao}")
                return self.dados_coleipa_default
            
            # Converter DataFrame para dicionário
            dados = {}
            for coluna in df.columns:
                dados[coluna] = df[coluna].tolist()
            
            return dados
            
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")
            return self.dados_coleipa_default
    
    def criar_sistema_fuzzy(self):
        """Cria sistema fuzzy baseado nos dados do Coleipa"""
        # Definir universos baseados nos dados reais
        vazao = ctrl.Antecedent(np.arange(7, 17, 0.1), 'vazao')
        pressao = ctrl.Antecedent(np.arange(0, 11, 0.1), 'pressao')
        ivi = ctrl.Antecedent(np.arange(1, 26, 0.1), 'ivi')
        risco_vazamento = ctrl.Consequent(np.arange(0, 101, 1), 'risco_vazamento')
        
        # Definir conjuntos fuzzy
        for nome, param in self.param_vazao.items():
            vazao[nome] = fuzz.trimf(vazao.universe, param['range'])
        
        for nome, param in self.param_pressao.items():
            pressao[nome] = fuzz.trimf(pressao.universe, param['range'])
        
        for nome, param in self.param_ivi.items():
            ivi[nome] = fuzz.trimf(ivi.universe, param['range'])
        
        for nome, param in self.param_risco.items():
            risco_vazamento[nome] = fuzz.trimf(risco_vazamento.universe, param['range'])
        
        # Regras baseadas na análise do Coleipa
        regras = [
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['MUITO_ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['RUIM'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['MEDIA'] & ivi['BOM'], risco_vazamento['MUITO_BAIXO']),
            ctrl.Rule(vazao['BAIXA'] & pressao['ALTA'] & ivi['BOM'], risco_vazamento['MUITO_BAIXO']),
        ]
        
        # Criar sistema de controle
        sistema_ctrl = ctrl.ControlSystem(regras)
        self.sistema_fuzzy = ctrl.ControlSystemSimulation(sistema_ctrl)
        
        return vazao, pressao, ivi, risco_vazamento
    
    def avaliar_risco_fuzzy(self, vazao, pressao, ivi):
        """Avalia risco usando sistema fuzzy"""
        if self.sistema_fuzzy is None:
            self.criar_sistema_fuzzy()
        
        try:
            vazao_limitada = max(7, min(vazao, 16))
            pressao_limitada = max(0, min(pressao, 10))
            ivi_limitado = max(1, min(ivi, 25))
            
            self.sistema_fuzzy.input['vazao'] = vazao_limitada
            self.sistema_fuzzy.input['pressao'] = pressao_limitada
            self.sistema_fuzzy.input['ivi'] = ivi_limitado
            
            self.sistema_fuzzy.compute()
            
            return self.sistema_fuzzy.output['risco_vazamento']
        except Exception as e:
            st.error(f"Erro na avaliação fuzzy: {e}")
            return 50
    
    def analisar_caso_tempo_real(self, dados):
        """Analisa um caso em tempo real e gera alertas"""
        vazao = dados.get('vazao', 10)
        pressao = dados.get('pressao', 5)
        ivi = dados.get('ivi', 16.33)
        
        # Avaliação fuzzy
        risco_fuzzy = self.avaliar_risco_fuzzy(vazao, pressao, ivi)
        
        # Processar alerta
        alerta = self.sistema_alertas.processar_alerta(
            risco_fuzzy, 
            dados_sistema=dados,
            localizacao="Sistema Coleipa"
        )
        
        return {
            'risco': risco_fuzzy,
            'alerta': alerta,
            'dados': dados
        }

# ============================================================================
# DASHBOARD EM TEMPO REAL
# ============================================================================

def criar_dashboard_tempo_real(detector, gerenciador_tempo_real):
    """Cria dashboard com simulação de tempo real"""
    
    st.markdown('<h1 class="main-header">🚰 Monitor em Tempo Real - Sistema Coleipa</h1>', unsafe_allow_html=True)
    
    # Controles do dashboard
    col_control1, col_control2, col_control3 = st.columns(3)
    
    with col_control1:
        if st.button("▶️ Simular Monitoramento", key="start_monitor"):
            st.success("Simulação iniciada!")
    
    with col_control2:
        if st.button("🔄 Atualizar Dados", key="refresh_data"):
            st.info("Dados atualizados!")
    
    with col_control3:
        auto_refresh = st.checkbox("Modo Demonstração", value=False)
    
    # Simular dados atuais
    dados_atuais = gerenciador_tempo_real._simular_dados_sensor()
    resultado = detector.analisar_caso_tempo_real(dados_atuais)
    
    # Status geral
    exibir_status_sistema(resultado)
    
    # Métricas principais
    exibir_metricas_tempo_real(dados_atuais, resultado)
    
    # Alertas
    st.subheader("🚨 Sistema de Alertas")
    alerta = resultado['alerta']
    
    if alerta['nivel'] in ['alto', 'critico']:
        st.error(f"{alerta['icone']} {alerta['titulo']} - Risco: {resultado['risco']:.1f}%")
    elif alerta['nivel'] == 'medio':
        st.warning(f"{alerta['icone']} {alerta['titulo']} - Risco: {resultado['risco']:.1f}%")
    else:
        st.success(f"{alerta['icone']} {alerta['titulo']} - Risco: {resultado['risco']:.1f}%")
    
    # Gráficos em tempo real
    exibir_graficos_tempo_real(dados_atuais, resultado)

def exibir_status_sistema(resultado):
    """Exibe status geral do sistema"""
    alerta = resultado['alerta']
    config = alerta
    
    st.markdown(f"""
    <div class="dashboard-card">
        <h2>{config['icone']} Status do Sistema: {config['titulo']}</h2>
        <div style="display: flex; align-items: center; margin-top: 1rem;">
            <div class="status-indicator status-pulse" style="background-color: {config['cor']};"></div>
            <span style="font-size: 1.2rem; font-weight: bold; color: {config['cor']};">
                Risco: {resultado['risco']:.1f}%
            </span>
        </div>
        <p style="margin-top: 1rem; color: #666;">
            Última atualização: {config['timestamp'].strftime('%H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)

def exibir_metricas_tempo_real(dados, resultado):
    """Exibe métricas principais em tempo real"""
    st.subheader("📊 Métricas em Tempo Real")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💧 Vazão</h3>
            <h2 style="color: #1f77b4;">{dados['vazao']:.1f}</h2>
            <p>m³/h</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Pressão</h3>
            <h2 style="color: #ff7f0e;">{dados['pressao']:.1f}</h2>
            <p>mca</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 IVI</h3>
            <h2 style="color: #2ca02c;">{dados['ivi']:.2f}</h2>
            <p>Categoria D</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risco_cor = "#e74c3c" if resultado['risco'] > 60 else "#f39c12" if resultado['risco'] > 30 else "#2ecc71"
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Risco</h3>
            <h2 style="color: {risco_cor};">{resultado['risco']:.1f}%</h2>
            <p>Vazamento</p>
        </div>
        """, unsafe_allow_html=True)

def exibir_graficos_tempo_real(dados, resultado):
    """Exibe gráficos em tempo real usando Matplotlib"""
    st.subheader("📈 Gráficos em Tempo Real")
    
    # Simular histórico de dados (em produção, viria do banco de dados)
    historico = gerar_historico_simulado(24)  # Últimas 24 horas
    
    # Criar gráficos com matplotlib - versão simplificada
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Converter tempos para números simples para evitar problemas de formatação
    horas = list(range(len(historico['tempo'])))
    
    # Gráfico 1: Vazão vs Tempo
    axes[0, 0].plot(horas, historico['vazao'], 'b-', linewidth=2, label='Vazão')
    axes[0, 0].set_title('Vazão vs Tempo')
    axes[0, 0].set_xlabel('Horas atrás')
    axes[0, 0].set_ylabel('Vazão (m³/h)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Gráfico 2: Pressão vs Tempo
    axes[0, 1].plot(horas, historico['pressao'], 'orange', linewidth=2, label='Pressão')
    axes[0, 1].set_title('Pressão vs Tempo')
    axes[0, 1].set_xlabel('Horas atrás')
    axes[0, 1].set_ylabel('Pressão (mca)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Gráfico 3: Risco vs Tempo
    axes[1, 0].plot(horas, historico['risco'], 'red', linewidth=2, label='Risco')
    axes[1, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Limite')
    axes[1, 0].set_title('Risco vs Tempo')
    axes[1, 0].set_xlabel('Horas atrás')
    axes[1, 0].set_ylabel('Risco (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Gráfico 4: Correlação Vazão-Pressão
    scatter = axes[1, 1].scatter(historico['vazao'], historico['pressao'], 
                                alpha=0.6, c=historico['risco'], 
                                cmap='RdYlBu_r', s=50)
    axes[1, 1].set_title('Correlação Vazão-Pressão')
    axes[1, 1].set_xlabel('Vazão (m³/h)')
    axes[1, 1].set_ylabel('Pressão (mca)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def gerar_historico_simulado(horas):
    """Gera histórico simulado para os gráficos"""
    import random
    
    dados = {
        'tempo': [],
        'vazao': [],
        'pressao': [],
        'risco': []
    }
    
    for i in range(horas):
        # Usar apenas o índice da hora para simplificar
        dados['tempo'].append(f"H-{horas-i}")
        
        # Padrão diário simulado
        hour_of_day = (datetime.now().hour - (horas-i)) % 24
        
        if 6 <= hour_of_day <= 22:
            vazao = 12 + random.normalvariate(0, 2)
            pressao = 4 + random.normalvariate(0, 1)
        else:
            vazao = 8 + random.normalvariate(0, 1)
            pressao = 6 + random.normalvariate(0, 0.5)
        
        # Calcular risco baseado em vazão/pressão
        if vazao > 13 and pressao < 4:
            risco = random.uniform(60, 90)
        elif vazao > 11 and pressao < 5:
            risco = random.uniform(30, 60)
        else:
            risco = random.uniform(10, 30)
        
        dados['vazao'].append(max(7, min(vazao, 16)))
        dados['pressao'].append(max(2, min(pressao, 10)))
        dados['risco'].append(risco)
    
    return dados

# ============================================================================
# APLICAÇÃO PRINCIPAL MELHORADA
# ============================================================================

def app_main():
    """Função principal do aplicativo Streamlit melhorado"""
    
    # Configurar layout responsivo
    configurar_layout_responsivo()
    
    st.set_page_config(
        page_title="Sistema de Detecção de Vazamentos - Coleipa",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar componentes
    if 'detector' not in st.session_state:
        st.session_state.detector = DetectorVazamentosColeipa()
    
    if 'gerenciador_tempo_real' not in st.session_state:
        st.session_state.gerenciador_tempo_real = GerenciadorTempoReal()
    
    detector = st.session_state.detector
    gerenciador_tempo_real = st.session_state.gerenciador_tempo_real
    
    # Sidebar melhorada
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>💧 Sistema Coleipa</h2>
        <p style="color: #666;">Detecção Inteligente de Vazamentos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navegação
    paginas = {
        "🏠 Dashboard Tempo Real": "dashboard",
        "📊 Dados de Monitoramento": "dados",
        "🧠 Sistema Fuzzy": "fuzzy",
        "🔄 Modelo Bayesiano": "bayes",
        "🔥 Mapas de Calor IVI": "mapas",
        "⏱️ Simulação Temporal": "simulacao",
        "🔬 Análise de Caso": "analise",
        "📝 Relatório Completo": "relatorio",
        "⚙️ Configurações": "config"
    }
    
    pagina_selecionada = st.sidebar.radio(
        "Navegação:", 
        list(paginas.keys()),
        format_func=lambda x: x
    )
    
    # Upload de arquivo na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Dados de Entrada")
    arquivo_uploaded = st.sidebar.file_uploader(
        "Carregar dados de monitoramento", 
        type=["xlsx", "csv"],
        help="Faça upload dos seus dados ou use os dados padrão do Coleipa"
    )
    
    if arquivo_uploaded:
        detector = DetectorVazamentosColeipa(arquivo_uploaded)
        st.session_state.detector = detector
    
    # Status do sistema na sidebar (simplificado)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Status do Sistema")
    
    if st.sidebar.button("🔄 Obter Status Atual"):
        dados_simulados = gerenciador_tempo_real._simular_dados_sensor()
        st.sidebar.success("✅ Sistema Operacional")
        st.sidebar.metric("Vazão Atual", f"{dados_simulados['vazao']:.1f} m³/h")
        st.sidebar.metric("Pressão Atual", f"{dados_simulados['pressao']:.1f} mca")
    else:
        st.sidebar.info("⏸️ Clique para verificar status")
    
    # Informações do sistema na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Sistema Coleipa")
    st.sidebar.info(
        "Sistema baseado nos dados reais do SAAP do bairro da Coleipa, "
        "Santa Bárbara do Pará - PA. Utiliza lógica fuzzy e modelos bayesianos "
        "para detecção de vazamentos."
    )
    
    # Características principais
    st.sidebar.markdown("**Características:**")
    st.sidebar.markdown(f"- População: {detector.caracteristicas_sistema['populacao']} hab")
    st.sidebar.markdown(f"- IVI: {detector.caracteristicas_sistema['ivi']:.2f}")
    st.sidebar.markdown(f"- Perdas: {detector.caracteristicas_sistema['percentual_perdas']:.1f}%")
    
    # Roteamento de páginas
    pagina_codigo = paginas[pagina_selecionada]
    
    if pagina_codigo == "dashboard":
        criar_dashboard_tempo_real(detector, gerenciador_tempo_real)
    elif pagina_codigo == "dados":
        mostrar_pagina_dados_melhorada(detector)
    elif pagina_codigo == "fuzzy":
        mostrar_pagina_fuzzy_melhorada(detector)
    elif pagina_codigo == "bayes":
        mostrar_pagina_bayes(detector)
    elif pagina_codigo == "mapas":
        mostrar_pagina_mapa_calor(detector)
    elif pagina_codigo == "simulacao":
        mostrar_pagina_simulacao(detector)
    elif pagina_codigo == "analise":
        mostrar_pagina_analise_melhorada(detector)
    elif pagina_codigo == "relatorio":
        mostrar_pagina_relatorio(detector)
    elif pagina_codigo == "config":
        mostrar_pagina_configuracoes(detector)
    else:
        # Para outras páginas, manter funcionalidade básica
        st.info(f"Página {pagina_selecionada} em desenvolvimento...")
        st.markdown("Esta página manterá a funcionalidade original do sistema.")

# ============================================================================

def mostrar_pagina_dados_melhorada(detector):
    """Página de dados com interface melhorada"""
    st.markdown('<h1 class="main-header">📊 Dados de Monitoramento</h1>', unsafe_allow_html=True)
    
    # Cards informativos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <h3>🏢 Sistema</h3>
            <p><strong>Local:</strong> Bairro Coleipa</p>
            <p><strong>População:</strong> 1.200 hab</p>
            <p><strong>Ligações:</strong> 300</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📈 Indicadores</h3>
            <p><strong>IVI:</strong> 16.33 (Cat. D)</p>
            <p><strong>Perdas:</strong> 44.50%</p>
            <p><strong>IPRL:</strong> 0.343 m³/lig.dia</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <h3>⚡ Ações</h3>
            <p><strong>Status:</strong> Crítico</p>
            <p><strong>Prioridade:</strong> Alta</p>
            <p><strong>Ação:</strong> Pesquisa de vazamentos</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Botão melhorado para visualizar dados
    if st.button("📊 Visualizar Dados de Monitoramento", key="viz_dados"):
        with st.spinner("🔄 Processando dados de monitoramento..."):
            fig, stats, df = detector.visualizar_dados_coleipa()
            
            st.pyplot(fig)
            
            # Estatísticas em cards
            st.subheader("📋 Resumo Estatístico")
            
            col1, col2, col3, col4 = st.columns(4)
            
            cards_stats = [
                ("💧 Vazão Mín", f"{stats['vazao_min']:.1f} m³/h", f"Hora {int(stats['vazao_min_hora'])}"),
                ("💧 Vazão Máx", f"{stats['vazao_max']:.1f} m³/h", f"Hora {int(stats['vazao_max_hora'])}"),
                ("📊 Pressão Mín", f"{stats['pressao_min']:.1f} mca", f"Hora {int(stats['pressao_min_hora'])}"),
                ("⚠️ Horas < 10 mca", f"{stats['horas_pressao_baixa']}", f"{stats['perc_pressao_baixa']:.1f}%")
            ]
            
            for i, (titulo, valor, subtitulo) in enumerate(cards_stats):
                with [col1, col2, col3, col4][i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{titulo}</h4>
                        <h2 style="color: #1f77b4;">{valor}</h2>
                        <p>{subtitulo}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Tabela interativa
            st.subheader("📋 Dados Detalhados")
            st.dataframe(df, use_container_width=True)

def mostrar_pagina_fuzzy_melhorada(detector):
    """Página do sistema fuzzy com interface melhorada"""
    st.markdown('<h1 class="main-header">🧠 Sistema Fuzzy</h1>', unsafe_allow_html=True)
    
    # Tabs para organizar o conteúdo
    tab1, tab2, tab3 = st.tabs(["📊 Visualização", "🧪 Teste Interativo", "📚 Explicações"])
    
    with tab1:
        st.subheader("Conjuntos Fuzzy")
        if st.button("🎨 Visualizar Conjuntos Fuzzy", key="viz_fuzzy"):
            with st.spinner("🔄 Gerando visualização..."):
                fig = detector.visualizar_conjuntos_fuzzy()
                st.pyplot(fig)
    
    with tab2:
        st.subheader("🧪 Teste Interativo do Sistema Fuzzy")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vazao_teste = st.slider("💧 Vazão (m³/h)", 7.0, 16.0, 14.5, 0.1,
                                  help="Ajuste a vazão para testar o sistema")
        
        with col2:
            pressao_teste = st.slider("📊 Pressão (mca)", 0.0, 10.0, 3.5, 0.1,
                                    help="Ajuste a pressão para testar o sistema")
        
        with col3:
            ivi_teste = st.slider("📈 IVI", 1.0, 25.0, 16.33, 0.01,
                                help="Índice de Vazamentos na Infraestrutura")
        
        # Cálculo automático em tempo real
        risco = detector.avaliar_risco_fuzzy(vazao_teste, pressao_teste, ivi_teste)
        
        # Resultado visual melhorado
        st.markdown("### 🎯 Resultado da Avaliação")
        
        # Determinar categoria e cor
        if risco < 20:
            categoria, cor, icone = "MUITO BAIXO", "#2ECC71", "🟢"
        elif risco < 40:
            categoria, cor, icone = "BAIXO", "#F39C12", "🟡"
        elif risco < 60:
            categoria, cor, icone = "MÉDIO", "#E67E22", "🟠"
        elif risco < 80:
            categoria, cor, icone = "ALTO", "#E74C3C", "🔴"
        else:
            categoria, cor, icone = "MUITO ALTO", "#8E44AD", "🚨"
        
        # Card de resultado
        st.markdown(f"""
        <div class="dashboard-card" style="text-align: center; border-top: 4px solid {cor};">
            <h2>{icone} Risco de Vazamento</h2>
            <h1 style="color: {cor}; font-size: 3rem; margin: 1rem 0;">{risco:.1f}%</h1>
            <h3 style="color: {cor};">{categoria}</h3>
            <hr>
            <p><strong>Vazão:</strong> {vazao_teste:.1f} m³/h</p>
            <p><strong>Pressão:</strong> {pressao_teste:.1f} mca</p>
            <p><strong>IVI:</strong> {ivi_teste:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Barra de progresso visual
        st.progress(risco/100)
    
    with tab3:
        st.subheader("📚 Como Funciona o Sistema Fuzzy")
        
        st.markdown("""
        O sistema fuzzy utiliza regras baseadas no conhecimento especialista para avaliar o risco de vazamentos.
        
        ##### 🔧 Principais Regras:
        
        1. **Vazão ALTA + Pressão BAIXA + IVI MUITO_RUIM → Risco MUITO_ALTO**  
           *Situação típica de vazamento no sistema Coleipa*
           
        2. **Vazão NORMAL + Pressão BAIXA + IVI MUITO_RUIM → Risco ALTO**  
           *Sistemas com IVI alto têm maior risco*
           
        3. **Vazão NORMAL + Pressão ALTA + IVI BOM → Risco MUITO_BAIXO**  
           *Operação normal em sistemas bem mantidos*
        
        ##### 📊 Conjuntos Fuzzy Definidos:
        
        - **Vazão**: BAIXA (7-11), NORMAL (9-14), ALTA (12-16) m³/h
        - **Pressão**: BAIXA (0-5), MÉDIA (4-8), ALTA (6-10) mca  
        - **IVI**: BOM (1-4), REGULAR (4-8), RUIM (8-16), MUITO_RUIM (16-25)
        """)

def mostrar_pagina_analise_melhorada(detector):
    """Página de análise de caso com interface melhorada"""
    st.markdown('<h1 class="main-header">🔬 Análise de Caso Específico</h1>', unsafe_allow_html=True)
    
    # Verificar modelo Bayes
    if detector.modelo_bayes is None:
        st.warning("⚠️ Modelo Bayesiano não treinado. Treinando com parâmetros padrão...")
        with st.spinner("🔄 Treinando modelo..."):
            X, y, _ = detector.gerar_dados_baseados_coleipa()
            detector.treinar_modelo_bayesiano(X, y)
        st.success("✅ Modelo treinado com sucesso!")
    
    # Formulário melhorado
    st.subheader("⚙️ Parâmetros do Sistema")
    
    with st.form("analise_caso"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💧 Vazão")
            vazao = st.number_input(
                "Vazão (m³/h)", 
                min_value=7.0, max_value=16.0, value=14.5, step=0.1,
                help="Valor típico Coleipa: 14.5 m³/h"
            )
            st.caption(f"Faixa normal: 7-16 m³/h")
        
        with col2:
            st.markdown("#### 📊 Pressão")
            pressao = st.number_input(
                "Pressão (mca)", 
                min_value=0.0, max_value=10.0, value=3.5, step=0.1,
                help="Valor típico Coleipa: 3.5 mca"
            )
            st.caption(f"Mínimo NBR: 10 mca")
        
        with col3:
            st.markdown("#### 📈 IVI")
            ivi = st.number_input(
                "IVI", 
                min_value=1.0, max_value=25.0, value=16.33, step=0.01,
                help="IVI do Coleipa: 16.33 (Categoria D)"
            )
            st.caption(f"Categoria D: >16")
        
        # Botão de análise
        submitted = st.form_submit_button(
            "🚀 Analisar Caso", 
            use_container_width=True,
            type="primary"
        )
    
    if submitted:
        with st.spinner("🔄 Analisando caso..."):
            resultado = detector.analisar_caso_coleipa(vazao, pressao, ivi)
            
            # Resultado principal
            st.markdown("---")
            st.subheader("🎯 Resultado da Análise")
            
            # Status principal
            alerta = detector.sistema_alertas.processar_alerta(
                resultado['risco_fuzzy'], 
                {'vazao': vazao, 'pressao': pressao, 'ivi': ivi}
            )
            
            st.markdown(f"""
            <div class="dashboard-card" style="text-align: center; border-top: 6px solid {alerta['cor']};">
                <h1>{alerta['icone']} {alerta['titulo']}</h1>
                <h2 style="color: {alerta['cor']}; font-size: 2.5rem;">
                    Risco: {resultado['risco_fuzzy']:.1f}%
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Detalhes da análise
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Classificação dos Parâmetros")
                st.markdown(f"""
                - **Vazão**: {resultado['vazao']:.1f} m³/h → {resultado['classe_vazao']}
                - **Pressão**: {resultado['pressao']:.1f} mca → {resultado['classe_pressao']}  
                - **IVI**: {resultado['ivi']:.2f} → {resultado['classe_ivi']}
                """)
            
            with col2:
                st.markdown("#### 🔢 Resultados Numéricos")
                if 'prob_bayes' in resultado:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Fuzzy", f"{resultado['risco_fuzzy']:.1f}%")
                    with col_b:
                        st.metric("Bayes", f"{resultado['prob_bayes']:.3f}")
                    with col_c:
                        st.metric("Híbrido", f"{resultado['prob_hibrida']:.3f}")
                else:
                    st.metric("Risco Fuzzy", f"{resultado['risco_fuzzy']:.1f}%")
            
            # Recomendações
            st.markdown("#### 💡 Recomendações")
            if resultado['risco_fuzzy'] > 70:
                st.error("""
                🚨 **AÇÃO IMEDIATA NECESSÁRIA**
                - Investigar possível vazamento
                - Verificar integridade da rede
                - Mobilizar equipe de reparo
                """)
            elif resultado['risco_fuzzy'] > 40:
                st.warning("""
                ⚠️ **MONITORAMENTO INTENSIVO**
                - Aumentar frequência de leituras
                - Investigar padrões anômalos
                - Preparar equipe para possível intervenção
                """)
            else:
                st.success("""
                ✅ **OPERAÇÃO NORMAL**
                - Manter monitoramento de rotina
                - Continuar análises periódicas
                """)

# Iniciar aplicativo
if __name__ == "__main__":
    app_main()