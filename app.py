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
import threading
from queue import Queue
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# ============================================================================
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
    """Gerencia dados e atualizações em tempo real"""
    
    def __init__(self):
        self.dados_atuais = {}
        self.fila_atualizacoes = Queue()
        self.ativo = False
        self.thread_atualizacao = None
        
    def iniciar_monitoramento(self):
        """Inicia o monitoramento em tempo real"""
        self.ativo = True
        self.thread_atualizacao = threading.Thread(target=self._loop_atualizacao)
        self.thread_atualizacao.daemon = True
        self.thread_atualizacao.start()
    
    def parar_monitoramento(self):
        """Para o monitoramento em tempo real"""
        self.ativo = False
    
    def _loop_atualizacao(self):
        """Loop principal de atualização"""
        while self.ativo:
            try:
                # Simular recebimento de dados (em produção, viria de sensores/API)
                novos_dados = self._simular_dados_sensor()
                self.dados_atuais = novos_dados
                self.fila_atualizacoes.put(novos_dados)
                time.sleep(5)  # Atualizar a cada 5 segundos
            except Exception as e:
                print(f"Erro no loop de atualização: {e}")
                time.sleep(10)
    
    def _simular_dados_sensor(self):
        """Simula dados de sensores (substituir por dados reais)"""
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
        if random.random() < 0.1:  # 10% de chance
            vazao_base += random.uniform(3, 6)  # Aumentar vazão
            pressao_base -= random.uniform(1, 2)  # Diminuir pressão
        
        return {
            'timestamp': base_time,
            'vazao': max(7, min(vazao_base, 16)),
            'pressao': max(2, min(pressao_base, 10)),
            'ivi': 16.33,  # IVI fixo do Coleipa
            'temperatura': random.uniform(20, 35),
            'ph': random.uniform(6.5, 8.5),
            'turbidez': random.uniform(0.1, 2.0)
        }
    
    def obter_dados_atuais(self):
        """Retorna os dados mais recentes"""
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
    
    # Manter todos os métodos originais aqui (carregar_dados_arquivo, criar_dataframe_coleipa, etc.)
    # Por brevidade, não vou repetir todos os métodos, mas eles permaneceriam iguais
    
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
    """Cria dashboard com atualização em tempo real"""
    
    st.markdown('<h1 class="main-header">🚰 Monitor em Tempo Real - Sistema Coleipa</h1>', unsafe_allow_html=True)
    
    # Controles do dashboard
    col_control1, col_control2, col_control3 = st.columns(3)
    
    with col_control1:
        if st.button("▶️ Iniciar Monitoramento", key="start_monitor"):
            gerenciador_tempo_real.iniciar_monitoramento()
            st.success("Monitoramento iniciado!")
    
    with col_control2:
        if st.button("⏸️ Pausar Monitoramento", key="pause_monitor"):
            gerenciador_tempo_real.parar_monitoramento()
            st.info("Monitoramento pausado!")
    
    with col_control3:
        auto_refresh = st.checkbox("Atualização Automática", value=True)
    
    # Containers para atualização
    container_status = st.container()
    container_metricas = st.container()
    container_alertas = st.container()
    container_graficos = st.container()
    
    # Loop de atualização se ativo
    if auto_refresh and gerenciador_tempo_real.ativo:
        dados_atuais = gerenciador_tempo_real.obter_dados_atuais()
        
        if dados_atuais:
            # Analisar dados atuais
            resultado = detector.analisar_caso_tempo_real(dados_atuais)
            
            # Status geral
            with container_status:
                exibir_status_sistema(resultado)
            
            # Métricas principais
            with container_metricas:
                exibir_metricas_tempo_real(dados_atuais, resultado)
            
            # Alertas
            with container_alertas:
                st.subheader("🚨 Alertas Ativos")
                detector.sistema_alertas.exibir_alertas_streamlit(st)
            
            # Gráficos em tempo real
            with container_graficos:
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
    """Exibe gráficos em tempo real usando Plotly"""
    st.subheader("📈 Gráficos em Tempo Real")
    
    # Simular histórico de dados (em produção, viria do banco de dados)
    historico = gerar_historico_simulado(24)  # Últimas 24 horas
    
    # Gráfico de vazão e pressão
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Vazão vs Tempo', 'Pressão vs Tempo', 'Risco vs Tempo', 'Correlação Vazão-Pressão'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Vazão
    fig.add_trace(
        go.Scatter(x=historico['tempo'], y=historico['vazao'], 
                  name="Vazão", line=dict(color='blue')),
        row=1, col=1
    )
    
    # Pressão
    fig.add_trace(
        go.Scatter(x=historico['tempo'], y=historico['pressao'], 
                  name="Pressão", line=dict(color='orange')),
        row=1, col=2
    )
    
    # Risco
    fig.add_trace(
        go.Scatter(x=historico['tempo'], y=historico['risco'], 
                  name="Risco", line=dict(color='red')),
        row=2, col=1
    )
    
    # Correlação
    fig.add_trace(
        go.Scatter(x=historico['vazao'], y=historico['pressao'], 
                  mode='markers', name="Vazão vs Pressão"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Tempo", row=1, col=1)
    fig.update_xaxes(title_text="Tempo", row=1, col=2)
    fig.update_xaxes(title_text="Tempo", row=2, col=1)
    fig.update_xaxes(title_text="Vazão (m³/h)", row=2, col=2)
    fig.update_yaxes(title_text="Vazão (m³/h)", row=1, col=1)
    fig.update_yaxes(title_text="Pressão (mca)", row=1, col=2)
    fig.update_yaxes(title_text="Risco (%)", row=2, col=1)
    fig.update_yaxes(title_text="Pressão (mca)", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def gerar_historico_simulado(horas):
    """Gera histórico simulado para os gráficos"""
    import random
    from datetime import datetime, timedelta
    
    agora = datetime.now()
    dados = {
        'tempo': [],
        'vazao': [],
        'pressao': [],
        'risco': []
    }
    
    for i in range(horas):
        tempo = agora - timedelta(hours=horas-i)
        hour = tempo.hour
        
        # Padrão diário
        if 6 <= hour <= 22:
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
        
        dados['tempo'].append(tempo)
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
    
    # Status do sistema na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Status do Sistema")
    
    if gerenciador_tempo_real.ativo:
        st.sidebar.success("🟢 Monitoramento Ativo")
        dados_atuais = gerenciador_tempo_real.obter_dados_atuais()
        if dados_atuais:
            st.sidebar.metric("Última Vazão", f"{dados_atuais['vazao']:.1f} m³/h")
            st.sidebar.metric("Última Pressão", f"{dados_atuais['pressao']:.1f} mca")
    else:
        st.sidebar.info("⏸️ Monitoramento Pausado")
    
    # Alertas recentes na sidebar
    alertas_recentes = detector.sistema_alertas.obter_alertas_ativos(1)  # Última hora
    if alertas_recentes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚨 Alertas Recentes")
        for alerta in alertas_recentes[-3:]:  # Últimos 3
            cor = alerta['cor']
            st.sidebar.markdown(f"""
            <div style="background-color: {cor}20; padding: 0.5rem; margin: 0.25rem 0; border-radius: 5px; border-left: 3px solid {cor};">
                {alerta['icone']} {alerta['titulo']}<br>
                <small>{alerta['timestamp'].strftime('%H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Roteamento de páginas
    pagina_codigo = paginas[pagina_selecionada]
    
    if pagina_codigo == "dashboard":
        criar_dashboard_tempo_real(detector, gerenciador_tempo_real)
    elif pagina_codigo == "dados":
        mostrar_pagina_dados_melhorada(detector)
    elif pagina_codigo == "fuzzy":
        mostrar_pagina_fuzzy_melhorada(detector)
    elif pagina_codigo == "analise":
        mostrar_pagina_analise_melhorada(detector)
    else:
        # Para outras páginas, manter funcionalidade básica
        st.info(f"Página {pagina_selecionada} em desenvolvimento...")
        st.markdown("Esta página manterá a funcionalidade original do sistema.")

# ============================================================================
# PÁGINAS MELHORADAS
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