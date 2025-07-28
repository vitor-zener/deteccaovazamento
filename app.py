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

# Configurações para matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DetectorVazamentosColeipa:
    """
    Sistema híbrido Fuzzy-Bayes para detecção de vazamentos baseado nos dados 
    do Sistema de Abastecimento de Água Potável (SAAP) do bairro da Coleipa
    """
    
    def __init__(self, arquivo_dados=None):
        """
        Inicializa o sistema baseado nos dados do artigo Coleipa ou carrega de um arquivo
        
        Parâmetros:
        arquivo_dados (str): Caminho para arquivo Excel ou CSV contendo os dados de monitoramento
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
    
    def carregar_dados_arquivo(self, arquivo_uploaded):
        """
        Carrega dados de monitoramento de um arquivo Excel ou CSV do Streamlit
        
        Parâmetros:
        arquivo_uploaded: Arquivo enviado pelo Streamlit
        
        Retorna:
        dict: Dicionário com os dados carregados
        """
        try:
            # Determinar tipo de arquivo pela extensão
            nome_arquivo = arquivo_uploaded.name
            nome, extensao = os.path.splitext(nome_arquivo)
            extensao = extensao.lower()
            
            if extensao == '.xlsx' or extensao == '.xls':
                # Carregar arquivo Excel
                df = pd.read_excel(arquivo_uploaded)
                st.success("Arquivo Excel carregado com sucesso")
            elif extensao == '.csv':
                # Carregar arquivo CSV
                df = pd.read_csv(arquivo_uploaded)
                st.success("Arquivo CSV carregado com sucesso")
            else:
                st.error(f"Formato de arquivo não suportado: {extensao}. Use Excel (.xlsx, .xls) ou CSV (.csv)")
                return self.dados_coleipa_default
            
            # Validar estrutura dos dados
            # O arquivo deve ter as colunas: hora, vazao_dia1, pressao_dia1, etc.
            colunas_necessarias = ['hora', 'vazao_dia1', 'pressao_dia1', 'vazao_dia2', 
                                  'pressao_dia2', 'vazao_dia3', 'pressao_dia3']
            
            for coluna in colunas_necessarias:
                if coluna not in df.columns:
                    st.warning(f"Coluna '{coluna}' não encontrada no arquivo. Verifique o formato dos dados.")
            
            # Converter DataFrame para dicionário
            dados = {}
            for coluna in df.columns:
                dados[coluna] = df[coluna].tolist()
            
            # Verificar comprimento dos dados
            if len(dados['hora']) != 24:
                st.warning(f"O número de horas no arquivo ({len(dados['hora'])}) é diferente do esperado (24).")
            
            return dados
            
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")
            st.info("Usando dados padrão Coleipa como fallback")
            return self.dados_coleipa_default
    
    def gerar_dados_template(self):
        """
        Gera os dados padrão para download como template
        """
        df = pd.DataFrame(self.dados_coleipa_default)
        return df
    
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
        
        # Regras baseadas na análise do Coleipa e conhecimento do especialista
        regras = [
            # Regras para detecção de vazamentos baseadas no padrão Coleipa
            # Vazão ALTA + Pressão BAIXA = indicativo forte de vazamento
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['MUITO_ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['RUIM'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['REGULAR'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['BOM'], risco_vazamento['MEDIO']),
            
            # Vazão NORMAL + Pressão BAIXA = risco moderado
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['RUIM'], risco_vazamento['MEDIO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['REGULAR'], risco_vazamento['MEDIO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['BOM'], risco_vazamento['BAIXO']),
            
            # Vazão BAIXA (operação noturna normal)
            ctrl.Rule(vazao['BAIXA'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['MEDIO']),
            ctrl.Rule(vazao['BAIXA'] & pressao['MEDIA'] & ivi['MUITO_RUIM'], risco_vazamento['BAIXO']),
            ctrl.Rule(vazao['BAIXA'] & pressao['ALTA'] & ivi['MUITO_RUIM'], risco_vazamento['BAIXO']),
            
            # Operação normal
            ctrl.Rule(vazao['NORMAL'] & pressao['MEDIA'] & ivi['BOM'], risco_vazamento['MUITO_BAIXO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['ALTA'] & ivi['BOM'], risco_vazamento['MUITO_BAIXO']),
            
            # Regras específicas para o caso Coleipa (IVI = 16.33, categoria D)
            ctrl.Rule(ivi['MUITO_RUIM'], risco_vazamento['MEDIO']),  # IVI alto sempre indica risco
            
            # Padrão típico observado no Coleipa durante vazamentos
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'], risco_vazamento['ALTO'])
        ]
        
        # Criar sistema de controle
        sistema_ctrl = ctrl.ControlSystem(regras)
        self.sistema_fuzzy = ctrl.ControlSystemSimulation(sistema_ctrl)
        
        return vazao, pressao, ivi, risco_vazamento
    
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

    def visualizar_dados_coleipa(self):
        """Visualiza os dados reais do monitoramento Coleipa"""
        df = self.criar_dataframe_coleipa()
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Gráfico 1: Vazões dos três dias
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
        
        # Retornar a figura e as estatísticas
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
        
        return fig, stats, df
    
    def gerar_dados_baseados_coleipa(self, n_amostras=500):
        """Gera dados sintéticos baseados nas características do sistema Coleipa"""
        df_coleipa = self.criar_dataframe_coleipa()
        
        # Extrair padrões dos dados reais
        vazao_normal_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Vazao_Media'].mean()
        vazao_normal_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Vazao_Media'].std()
        pressao_normal_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Pressao_Media'].mean()
        pressao_normal_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Pressao_Media'].std()
        
        vazao_vazamento_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Vazao_Media'].mean()
        vazao_vazamento_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Vazao_Media'].std()
        pressao_vazamento_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Pressao_Media'].mean()
        pressao_vazamento_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Pressao_Media'].std()
        
        # Gerar dados sintéticos baseados nos padrões reais
        n_normal = int(0.55 * n_amostras)  # 55% normal (baseado nos dados Coleipa)
        n_vazamento = n_amostras - n_normal
        
        # Dados normais
        vazao_normal = np.random.normal(vazao_normal_mean, vazao_normal_std, n_normal)
        pressao_normal = np.random.normal(pressao_normal_mean, pressao_normal_std, n_normal)
        ivi_normal = np.random.normal(8, 2, n_normal)  # IVI melhor para operação normal
        
        # Dados de vazamento
        vazao_vazamento = np.random.normal(vazao_vazamento_mean, vazao_vazamento_std, n_vazamento)
        pressao_vazamento = np.random.normal(pressao_vazamento_mean, pressao_vazamento_std, n_vazamento)
        ivi_vazamento = np.random.normal(16.33, 3, n_vazamento)  # IVI similar ao Coleipa
        
        # Combinar dados
        X = np.vstack([
            np.column_stack([vazao_normal, pressao_normal, ivi_normal]),
            np.column_stack([vazao_vazamento, pressao_vazamento, ivi_vazamento])
        ])
        
        y = np.hstack([np.zeros(n_normal), np.ones(n_vazamento)])
        
        return X, y, df_coleipa
    
    def treinar_modelo_bayesiano(self, X, y):
        """Treina modelo Bayesiano com dados baseados no Coleipa"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.modelo_bayes = GaussianNB()
        self.modelo_bayes.fit(X_train, y_train)
        
        y_pred = self.modelo_bayes.predict(X_test)
        
        # Calcular matriz de confusão e relatório de classificação
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Vazamento'], output_dict=True)
        
        return self.modelo_bayes, cm, report
    
    def visualizar_matriz_confusao(self, cm):
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
    
    def avaliar_risco_fuzzy(self, vazao, pressao, ivi):
        """Avalia risco usando sistema fuzzy"""
        if self.sistema_fuzzy is None:
            self.criar_sistema_fuzzy()
        
        try:
            # Limitar valores aos intervalos dos dados Coleipa
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
    
    def simular_serie_temporal_coleipa(self):
        """Simula série temporal baseada nos padrões reais do Coleipa"""
        df_real = self.criar_dataframe_coleipa()
        
        # Criar série temporal expandida (3 dias completos)
        tempo = []
        vazao = []
        pressao = []
        
        for dia in range(3):
            for hora in range(24):
                timestamp = datetime(2024, 1, 1 + dia, hora, 0)
                tempo.append(timestamp)
                
                # Usar dados reais do Coleipa com variação
                idx = hora
                if idx < len(self.dados_coleipa['vazao_dia1']) and idx < len(self.dados_coleipa['pressao_dia1']) and dia == 0:
                    v = self.dados_coleipa['vazao_dia1'][idx] + np.random.normal(0, 0.1)
                    p = self.dados_coleipa['pressao_dia1'][idx] + np.random.normal(0, 0.05)
                elif idx < len(self.dados_coleipa['vazao_dia2']) and idx < len(self.dados_coleipa['pressao_dia2']) and dia == 1:
                    v = self.dados_coleipa['vazao_dia2'][idx] + np.random.normal(0, 0.1)
                    p = self.dados_coleipa['pressao_dia2'][idx] + np.random.normal(0, 0.05)
                elif idx < len(self.dados_coleipa['vazao_dia3']) and idx < len(self.dados_coleipa['pressao_dia3']) and dia == 2:
                    v = self.dados_coleipa['vazao_dia3'][idx] + np.random.normal(0, 0.1)
                    p = self.dados_coleipa['pressao_dia3'][idx] + np.random.normal(0, 0.05)
                else:
                    # Usar valores médios caso não tenhamos dados para esta hora/dia
                    if len(df_real) > 0:
                        hora_idx = hora % len(df_real)
                        v = df_real.iloc[hora_idx]['Vazao_Media'] + np.random.normal(0, 0.1)
                        p = df_real.iloc[hora_idx]['Pressao_Media'] + np.random.normal(0, 0.05)
                    else:
                        # Valores padrão se não temos nem dados médios
                        v = 10 + np.random.normal(0, 0.1)
                        p = 5 + np.random.normal(0, 0.05)
                
                vazao.append(v)
                pressao.append(p)
        
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
            'IVI': [self.caracteristicas_sistema['ivi']] * len(tempo),
            'Vazamento_Real': [0] * inicio_vazamento + [1] * (len(tempo) - inicio_vazamento)
        })
        
        # Calcular detecções se o modelo estiver treinado
        if self.modelo_bayes is not None:
            deteccoes = []
            for _, row in df.iterrows():
                risco_fuzzy = self.avaliar_risco_fuzzy(row['Vazao'], row['Pressao'], row['IVI'])
                prob_bayes = self.modelo_bayes.predict_proba([[row['Vazao'], row['Pressao'], row['IVI']]])[0][1]
                prob_hibrida = 0.6 * (risco_fuzzy/100) + 0.4 * prob_bayes
                deteccoes.append({
                    'Risco_Fuzzy': risco_fuzzy/100,
                    'Prob_Bayes': prob_bayes,
                    'Prob_Hibrida': prob_hibrida
                })
            
            for col in deteccoes[0].keys():
                df[col] = [d[col] for d in deteccoes]
        
        return self.visualizar_serie_temporal_coleipa(df, inicio_vazamento)
    
    def visualizar_serie_temporal_coleipa(self, df, inicio_vazamento):
        """Visualiza série temporal baseada no Coleipa"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Gráfico 1: Vazão
        axes[0].plot(df['Tempo'], df['Vazao'], 'b-', linewidth=1.5, label='Vazão')
        axes[0].axvline(x=df['Tempo'][inicio_vazamento], color='red', linestyle='--', 
                       label=f'Início Vazamento ({df["Tempo"][inicio_vazamento].strftime("%d/%m %H:%M")})')
        axes[0].set_ylabel('Vazão (m³/h)')
        axes[0].set_title('Série Temporal - Sistema Coleipa: Vazão')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Pressão
        axes[1].plot(df['Tempo'], df['Pressao'], 'r-', linewidth=1.5, label='Pressão')
        axes[1].axhline(y=10, color='orange', linestyle=':', label='Mínimo NBR (10 mca)')
        axes[1].axvline(x=df['Tempo'][inicio_vazamento], color='red', linestyle='--')
        axes[1].set_ylabel('Pressão (mca)')
        axes[1].set_title('Série Temporal - Sistema Coleipa: Pressão')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gráfico 3: Detecções (se disponível)
        if 'Prob_Hibrida' in df.columns:
            axes[2].plot(df['Tempo'], df['Prob_Hibrida'], 'purple', linewidth=2, label='Detecção Híbrida')
            axes[2].plot(df['Tempo'], df['Risco_Fuzzy'], 'green', alpha=0.7, label='Componente Fuzzy')
            axes[2].plot(df['Tempo'], df['Prob_Bayes'], 'orange', alpha=0.7, label='Componente Bayes')
            axes[2].axhline(y=0.5, color='black', linestyle='-.', label='Limiar Detecção')
            axes[2].axvline(x=df['Tempo'][inicio_vazamento], color='red', linestyle='--')
            axes[2].set_ylabel('Probabilidade')
            axes[2].set_title('Detecção de Vazamentos - Sistema Híbrido')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Modelo Bayesiano não treinado\nApenas análise fuzzy disponível', 
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
            axes[2].set_title('Detecção não disponível')
        
        axes[2].set_xlabel('Tempo')
        plt.tight_layout()
        return fig, df
    
    def gerar_mapa_calor_ivi(self, resolucao=30):
        """
        Gera mapas de calor mostrando o risco de vazamento para diferentes
        combinações de vazão e pressão, com diferentes valores de IVI baseados
        na classificação do Banco Mundial
        """
        # Verificar se o sistema fuzzy está criado
        if self.sistema_fuzzy is None:
            self.criar_sistema_fuzzy()
        
        # Valores de IVI baseados na classificação do Banco Mundial
        ivi_valores = [2, 6, 12, 18]  # Representativos das categorias A, B, C, D
        ivi_categorias = ['BOM (2.0)', 'REGULAR (6.0)', 'RUIM (12.0)', 'MUITO RUIM (18.0)']
        ivi_classificacoes = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
        
        # Valores para o mapa de calor baseados nos dados Coleipa
        vazoes = np.linspace(7, 16, resolucao)
        pressoes = np.linspace(2.5, 8, resolucao)
        
        # Configurar figura com subplots 2x2 para os mapas + 1 subplot para barra de cores
        fig = plt.figure(figsize=(18, 16))
        
        # Criar grid: 3 linhas, 2 colunas
        # Linha 1: 2 mapas superiores
        # Linha 2: 2 mapas inferiores  
        # Linha 3: barra de cores centralizada
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.15], hspace=0.3, wspace=0.2)
        
        # Criar os 4 subplots para os mapas
        axes = [
            fig.add_subplot(gs[0, 0]),  # Superior esquerdo
            fig.add_subplot(gs[0, 1]),  # Superior direito
            fig.add_subplot(gs[1, 0]),  # Inferior esquerdo
            fig.add_subplot(gs[1, 1])   # Inferior direito
        ]
        
        # Subplot para a barra de cores (ocupando toda a largura)
        cbar_ax = fig.add_subplot(gs[2, :])
        
        # Gerar um mapa de calor para cada valor de IVI
        im = None  # Para capturar a última imagem para a barra de cores
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
                        self.sistema_fuzzy.input['vazao'] = vazao_val
                        self.sistema_fuzzy.input['pressao'] = pressao_val
                        self.sistema_fuzzy.input['ivi'] = ivi_val
                        
                        self.sistema_fuzzy.compute()
                        risco = self.sistema_fuzzy.output['risco_vazamento']
                        Z[ii, jj] = max(0, min(risco, 100))
                        
                    except Exception as e:
                        # Heurística baseada nos padrões do Coleipa
                        vazao_norm = (X[ii, jj] - 7) / (16 - 7)  # Normalizar 0-1
                        pressao_norm = 1 - (Y[ii, jj] - 2.5) / (8 - 2.5)  # Inverter: pressão baixa = risco alto
                        
                        # Calcular risco base
                        risco_base = (vazao_norm * 0.6 + pressao_norm * 0.4) * 70
                        
                        # Ajustar pelo IVI
                        fator_ivi = ivi_valor / 10  # IVI 2=0.2, IVI 18=1.8
                        Z[ii, jj] = min(100, risco_base * fator_ivi + 10)
            
            # Plotar mapa de calor com escala de cores melhorada
            im = ax.imshow(Z, cmap='RdYlGn_r', origin='lower', 
                          extent=[vazoes.min(), vazoes.max(), pressoes.min(), pressoes.max()],
                          aspect='auto', vmin=0, vmax=100, interpolation='bilinear')
            
            # Adicionar contornos mais suaves
            try:
                contour_levels = [20, 40, 60, 80]
                contours = ax.contour(X, Y, Z, levels=contour_levels, colors='black', alpha=0.4, linewidths=1.5)
                ax.clabel(contours, inline=True, fontsize=10, fmt='%d%%', 
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            except:
                pass
            
            # Linhas de divisão dos conjuntos fuzzy - mais visíveis
            # Vazão: BAIXA (7-9), NORMAL (9-14), ALTA (14-16)
            ax.axvline(x=9, color='navy', linestyle=':', alpha=0.8, linewidth=2)
            ax.axvline(x=14, color='navy', linestyle=':', alpha=0.8, linewidth=2)
            
            # Pressão: BAIXA (2.5-4.5), NORMAL (4.5-6), ALTA (6-8)
            ax.axhline(y=4.5, color='darkgreen', linestyle=':', alpha=0.8, linewidth=2)
            ax.axhline(y=6.0, color='darkgreen', linestyle=':', alpha=0.8, linewidth=2)
            
            # Labels dos conjuntos fuzzy com tamanho reduzido
            ax.text(8, 7.5, 'VAZÃO\nBAIXA', color='navy', fontsize=9, fontweight='bold', 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            ax.text(11.5, 7.5, 'VAZÃO\nNORMAL', color='navy', fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            ax.text(15, 7.5, 'VAZÃO\nALTA', color='navy', fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            
            # Labels de pressão com tamanho reduzido
            ax.text(15.5, 3.5, 'PRESSÃO\nBAIXA', color='darkgreen', fontsize=9, fontweight='bold', 
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            ax.text(15.5, 5.2, 'PRESSÃO\nNORMAL', color='darkgreen', fontsize=9, fontweight='bold',
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            ax.text(15.5, 7, 'PRESSÃO\nALTA', color='darkgreen', fontsize=9, fontweight='bold',
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            
            # Marcar o ponto característico do Coleipa em todos os gráficos
            if idx == 3:  # Último gráfico (IVI Muito Ruim) - destaque especial
                ax.scatter([14.5], [3.5], color='red', s=300, marker='*', 
                          edgecolors='darkred', linewidth=3, label='Ponto Coleipa\n(IVI=16.33)', zorder=10)
                ax.annotate('SISTEMA COLEIPA\n(Vazão=14.5, Pressão=3.5)\nIVI=16.33 - CRÍTICO', 
                           xy=(14.5, 3.5), xytext=(11, 2.8),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=9, fontweight='bold', color='red',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9))
                ax.legend(loc='upper left', fontsize=9)
            else:
                # Marcar o ponto Coleipa nos outros gráficos também
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
            
            # Melhorar ticks
            ax.set_xticks(np.arange(7, 17, 1))
            ax.set_yticks(np.arange(3, 9, 1))
        
        # Criar barra de cores separada no subplot dedicado
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Risco de Vazamento (%)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Adicionar ticks personalizados na barra de cores
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%\n(Muito Baixo)', '20%\n(Baixo)', '40%\n(Médio)', 
                            '60%\n(Alto)', '80%\n(Muito Alto)', '100%\n(Crítico)'])
        
        # Título principal melhorado
        fig.suptitle('Mapa de Risco para diferentes IVIs\nClassificação Banco Mundial - Sistema Coleipa', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        return fig, ivi_valores
    
    def gerar_relatorio_coleipa(self):
        """Gera relatório completo baseado nos dados do Coleipa"""
        relatorio = {
            "caracteristicas": {
                "localizacao": "Bairro Coleipa, Santa Bárbara do Pará-PA",
                "area": self.caracteristicas_sistema['area_territorial']/1000,
                "populacao": self.caracteristicas_sistema['populacao'],
                "ligacoes": self.caracteristicas_sistema['numero_ligacoes'],
                "rede": self.caracteristicas_sistema['comprimento_rede'],
                "densidade_ramais": self.caracteristicas_sistema['densidade_ramais']
            },
            "monitoramento": {
                "volume_demandado": 273.5,
                "volume_consumido": self.caracteristicas_sistema['volume_consumido_medio'],
                "perdas_reais": self.caracteristicas_sistema['perdas_reais_media'],
                "percentual_perdas": self.caracteristicas_sistema['percentual_perdas']
            },
            "indicadores": {
                "iprl": self.caracteristicas_sistema['iprl'],
                "ipri": self.caracteristicas_sistema['ipri'],
                "ivi": self.caracteristicas_sistema['ivi']
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
    
    def atualizar_caracteristicas_sistema(self, novas_caracteristicas):
        """
        Atualiza as características do sistema com novos valores
        
        Parâmetros:
        novas_caracteristicas (dict): Dicionário com novas características
        """
        for chave, valor in novas_caracteristicas.items():
            if chave in self.caracteristicas_sistema:
                self.caracteristicas_sistema[chave] = valor
                st.success(f"Característica '{chave}' atualizada para: {valor}")
            else:
                st.warning(f"Aviso: Característica '{chave}' não existe no sistema")


# Configuração da página Streamlit
st.set_page_config(
    page_title="Sistema de Detecção de Vazamentos - Coleipa",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variável global para armazenar a instância do detector
@st.cache_resource
def get_detector(arquivo_uploaded=None):
    return DetectorVazamentosColeipa(arquivo_uploaded)

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


def app_main():
    """Função principal do aplicativo Streamlit"""
    st.title("💧 Sistema de Detecção de Vazamentos - SAAP Coleipa")
    st.markdown("##### Sistema híbrido Fuzzy-Bayes para detecção de vazamentos em redes de abastecimento")
    
    # Sidebar para navegação
    st.sidebar.title("Navegação")
    paginas = [
        "Início",
        "Dados de Monitoramento",
        "Sistema Fuzzy",
        "Modelo Bayesiano",
        "Mapas de Calor IVI",
        "Simulação Temporal",
        "Análise de Caso",
        "Relatório Completo",
        "Configurações"
    ]
    pagina_selecionada = st.sidebar.radio("Selecione uma página:", paginas)
    
    # Upload de arquivo na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dados de Entrada")
    arquivo_uploaded = st.sidebar.file_uploader("Carregar dados de monitoramento", type=["xlsx", "csv"])
    
    # Inicializar ou obter detector
    detector = get_detector(arquivo_uploaded)
    
    # Template para download na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Template de Dados")
    formato = st.sidebar.radio("Formato:", ["Excel (.xlsx)", "CSV (.csv)"], horizontal=True)
    nome_arquivo = "template_dados_coleipa." + ("xlsx" if formato == "Excel (.xlsx)" else "csv")
    df_template = detector.gerar_dados_template()
    download_button(df_template, nome_arquivo, "⬇️ Download Template")
    
    # Informações na sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sobre o Sistema")
    st.sidebar.info(
        "Sistema baseado nos dados reais do SAAP do bairro da Coleipa, "
        "Santa Bárbara do Pará - PA. Utiliza lógica fuzzy e modelos bayesianos "
        "para detecção de vazamentos em redes de abastecimento de água."
    )
    
    # Conteúdo principal baseado na página selecionada
    if pagina_selecionada == "Início":
        mostrar_pagina_inicio()
    
    elif pagina_selecionada == "Dados de Monitoramento":
        mostrar_pagina_dados(detector)
    
    elif pagina_selecionada == "Sistema Fuzzy":
        mostrar_pagina_fuzzy(detector)
    
    elif pagina_selecionada == "Modelo Bayesiano":
        mostrar_pagina_bayes(detector)
    
    elif pagina_selecionada == "Mapas de Calor IVI":
        mostrar_pagina_mapa_calor(detector)
    
    elif pagina_selecionada == "Simulação Temporal":
        mostrar_pagina_simulacao(detector)
    
    elif pagina_selecionada == "Análise de Caso":
        mostrar_pagina_analise_caso(detector)
    
    elif pagina_selecionada == "Relatório Completo":
        mostrar_pagina_relatorio(detector)
    
    elif pagina_selecionada == "Configurações":
        mostrar_pagina_configuracoes(detector)


def mostrar_pagina_inicio():
    """Página inicial do aplicativo"""
    st.header("Bem-vindo ao Sistema de Detecção de Vazamentos")
    
    # Descrição do sistema
    st.markdown("""
    Este sistema utiliza uma abordagem híbrida combinando lógica fuzzy e análise bayesiana para 
    detectar vazamentos em redes de abastecimento de água com base em dados de monitoramento.
    """)
    
    # Visão geral em 3 colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Análise de Dados")
        st.markdown("""
        - Visualização dos dados de monitoramento
        - Análise estatística de vazão e pressão
        - Identificação de padrões críticos
        """)
        st.image("https://via.placeholder.com/300x200?text=Dados+de+Monitoramento", use_container_width=True)
    
    with col2:
        st.subheader("🧠 Inteligência Híbrida")
        st.markdown("""
        - Sistema fuzzy baseado em conhecimento especialista
        - Modelo bayesiano para classificação
        - Mapas de calor para análise de riscos
        """)
        st.image("https://via.placeholder.com/300x200?text=Sistema+Fuzzy", use_container_width=True)
    
    with col3:
        st.subheader("📊 Resultados e Relatórios")
        st.markdown("""
        - Simulação de vazamentos em tempo real
        - Análise de casos específicos
        - Relatórios detalhados com recomendações
        """)
        st.image("https://via.placeholder.com/300x200?text=Relatorios", use_container_width=True)
    
    # Sobre o caso Coleipa
    st.markdown("---")
    st.subheader("Sobre o Sistema Coleipa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        O SAAP (Sistema de Abastecimento de Água Potável) do bairro da Coleipa, localizado em Santa 
        Bárbara do Pará, apresenta características típicas de sistemas com perdas significativas:
        
        - **IVI (Índice de Vazamentos na Infraestrutura)**: 16.33 (Categoria D - Muito Ruim)
        - **Perdas reais**: 44.50% do volume distribuído
        - **Pressões**: Consistentemente abaixo do mínimo recomendado (10 mca)
        - **Padrão característico**: Vazões altas com pressões baixas
        
        Este sistema foi desenvolvido a partir da análise detalhada desses dados e busca fornecer ferramentas
        para identificação, análise e gestão de vazamentos em redes semelhantes.
        """)
    
    with col2:
        st.markdown("""
        #### Características do Sistema
        
        - **Área territorial**: 319.000 m²
        - **População atendida**: 1.200 habitantes
        - **Número de ligações**: 300
        - **Extensão da rede**: 3 km
        - **Densidade de ramais**: 100 ramais/km
        
        #### Classificação Banco Mundial para IVI
        - **Categoria A (1-4)**: Sistema eficiente
        - **Categoria B (4-8)**: Sistema regular
        - **Categoria C (8-16)**: Sistema ruim
        - **Categoria D (>16)**: Sistema muito ruim
        """)
    
    # Como usar o sistema
    st.markdown("---")
    st.subheader("Como usar este sistema")
    st.markdown("""
    1. Utilize a barra lateral para navegar entre as diferentes funcionalidades
    2. Carregue seus dados de monitoramento ou use os dados padrão do Coleipa
    3. Explore os gráficos e análises disponíveis em cada seção
    4. Gere relatórios e recomendações para seu sistema específico
    """)
    
    # Rodapé
    st.markdown("---")
    st.caption("Sistema de Detecção de Vazamentos Coleipa | Baseado em técnicas híbridas Fuzzy-Bayes")


def mostrar_pagina_dados(detector):
    """Página de visualização dos dados de monitoramento"""
    st.header("📊 Dados de Monitoramento")
    st.markdown("Visualização dos dados reais de monitoramento do Sistema Coleipa")
    
    # Botão para processar os dados
    if st.button("Visualizar Dados de Monitoramento"):
        with st.spinner("Processando dados de monitoramento..."):
            fig, stats, df = detector.visualizar_dados_coleipa()
            
            # Exibir os gráficos
            st.pyplot(fig)
            
            # Exibir estatísticas em colunas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Estatísticas de Vazão")
                st.metric("Vazão mínima", f"{stats['vazao_min']:.2f} m³/h", f"Hora {int(stats['vazao_min_hora'])}")
                st.metric("Vazão máxima", f"{stats['vazao_max']:.2f} m³/h", f"Hora {int(stats['vazao_max_hora'])}")
                st.metric("Razão mín/máx", f"{stats['vazao_ratio']:.1f}%")
            
            with col2:
                st.subheader("Estatísticas de Pressão")
                st.metric("Pressão mínima", f"{stats['pressao_min']:.2f} mca", f"Hora {int(stats['pressao_min_hora'])}")
                st.metric("Pressão máxima", f"{stats['pressao_max']:.2f} mca", f"Hora {int(stats['pressao_max_hora'])}")
                st.metric("Horas com pressão < 10 mca", f"{stats['horas_pressao_baixa']} de 24", f"{stats['perc_pressao_baixa']:.1f}%")
            
            # Exibir tabela com os dados
            st.subheader("Dados de Monitoramento")
            st.dataframe(df)


def mostrar_pagina_fuzzy(detector):
    """Página do sistema fuzzy"""
    st.header("🧠 Sistema Fuzzy")
    st.markdown("Visualização e configuração do sistema fuzzy para detecção de vazamentos")
    
    # Visualização dos conjuntos fuzzy
    st.subheader("Conjuntos Fuzzy")
    if st.button("Visualizar Conjuntos Fuzzy"):
        with st.spinner("Gerando visualização dos conjuntos fuzzy..."):
            fig = detector.visualizar_conjuntos_fuzzy()
            st.pyplot(fig)
    
    # Explicação sobre as regras fuzzy
    st.subheader("Regras do Sistema Fuzzy")
    st.markdown("""
    O sistema fuzzy utiliza regras baseadas na análise do comportamento hidráulico da rede Coleipa.
    Algumas das principais regras são:
    
    1. **Vazão ALTA + Pressão BAIXA + IVI MUITO_RUIM → Risco MUITO_ALTO**  
       *Esta é a situação típica de vazamento no sistema Coleipa*
       
    2. **Vazão ALTA + Pressão BAIXA + IVI REGULAR/RUIM → Risco ALTO**  
       *Indicação forte de vazamento mesmo em sistemas com melhores condições*
       
    3. **Vazão NORMAL + Pressão BAIXA + IVI MUITO_RUIM → Risco ALTO**  
       *Sistemas com IVI alto têm maior risco mesmo com vazões normais*
       
    4. **Vazão NORMAL + Pressão ALTA + IVI BOM → Risco MUITO_BAIXO**  
       *Operação normal em sistemas bem mantidos*
    """)
    
    # Teste interativo do sistema fuzzy
    st.subheader("Teste Interativo")
    st.markdown("Ajuste os parâmetros abaixo para testar o comportamento do sistema fuzzy:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vazao_teste = st.slider("Vazão (m³/h)", 7.0, 16.0, 14.5, 0.1)
    
    with col2:
        pressao_teste = st.slider("Pressão (mca)", 0.0, 10.0, 3.5, 0.1)
    
    with col3:
        ivi_teste = st.slider("IVI", 1.0, 25.0, 16.33, 0.01)
    
    if st.button("Calcular Risco Fuzzy"):
        with st.spinner("Calculando risco..."):
            risco = detector.avaliar_risco_fuzzy(vazao_teste, pressao_teste, ivi_teste)
            
            # Determinar categoria de risco
            categoria_risco = ""
            cor_risco = ""
            if risco < 20:
                categoria_risco = "MUITO BAIXO"
                cor_risco = "green"
            elif risco < 40:
                categoria_risco = "BAIXO"
                cor_risco = "lightgreen"
            elif risco < 60:
                categoria_risco = "MÉDIO"
                cor_risco = "orange"
            elif risco < 80:
                categoria_risco = "ALTO"
                cor_risco = "darkorange"
            else:
                categoria_risco = "MUITO ALTO"
                cor_risco = "red"
            
            # Exibir resultado
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                #### Resultado da Avaliação
                - **Vazão**: {vazao_teste:.1f} m³/h
                - **Pressão**: {pressao_teste:.1f} mca
                - **IVI**: {ivi_teste:.2f}
                """)
            
            with col2:
                st.markdown(f"#### Risco de Vazamento")
                st.markdown(f"<h2 style='color:{cor_risco};'>{risco:.1f}% - {categoria_risco}</h2>", unsafe_allow_html=True)


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
                fig_cm = detector.visualizar_matriz_confusao(cm)
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
            fig, ivi_valores = detector.gerar_mapa_calor_ivi(resolucao)
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
    
    st.markdown("##### 🔧 Impacto visual nos mapas:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**IVI BOM (2.0):**  \nPredominantemente verde (baixo risco)")
    
    with col2:
        st.markdown("**IVI REGULAR (6.0):**  \nVerde-amarelo (risco moderado)")
    
    with col3:
        st.markdown("**IVI RUIM (12.0):**  \nAmarelo-laranja (risco elevado)")
    
    with col4:
        st.markdown("**IVI MUITO RUIM (18.0):**  \nVermelho intenso (risco crítico)")


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
            fig, df = detector.simular_serie_temporal_coleipa()
            st.pyplot(fig)
            
            # Mostrar dados da simulação
            with st.expander("Ver dados da simulação"):
                # Formatar coluna de tempo para exibição
                df_display = df.copy()
                df_display['Tempo'] = df_display['Tempo'].dt.strftime('%d/%m %H:%M')
                
                # Selecionar colunas relevantes
                if 'Prob_Hibrida' in df.columns:
                    df_display = df_display[['Tempo', 'Vazao', 'Pressao', 'IVI', 'Vazamento_Real', 'Prob_Hibrida']]
                else:
                    df_display = df_display[['Tempo', 'Vazao', 'Pressao', 'IVI', 'Vazamento_Real']]
                
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
    
    # Animação de vazamento (opcional, usando código HTML)
    with st.expander("Visualização Conceitual de Vazamento"):
        st.markdown("""
        <div style="width:100%;height:200px;background:linear-gradient(90deg, #3498db 0%, #2980b9 100%);border-radius:10px;position:relative;overflow:hidden;">
            <div style="position:absolute;width:30px;height:30px;background:#e74c3c;border-radius:50%;top:50%;left:70%;transform:translate(-50%,-50%);box-shadow:0 0 20px #e74c3c;">
                <div style="position:absolute;width:40px;height:40px;border:2px solid #e74c3c;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);animation:pulse 1.5s infinite;"></div>
            </div>
            <div style="position:absolute;width:100%;bottom:0;font-family:sans-serif;color:white;text-align:center;padding:10px;">
                Representação conceitual de vazamento na rede
            </div>
        </div>
        <style>
        @keyframes pulse {
            0% { width:40px; height:40px; opacity:1; }
            100% { width:70px; height:70px; opacity:0; }
        }
        </style>
        """, unsafe_allow_html=True)


def mostrar_pagina_analise_caso(detector):
    """Página de análise de caso específico"""
    st.header("🔬 Análise de Caso Específico")
    st.markdown("Analise um caso específico de operação com base nos parâmetros informados")
    
    # Verificar se o modelo Bayes está treinado
    if detector.modelo_bayes is None:
        st.warning("O modelo Bayesiano não está treinado. Alguns resultados estarão limitados apenas à análise fuzzy.")
        usar_bayes = st.checkbox("Treinar modelo Bayesiano agora", value=True)
        if usar_bayes:
            with st.spinner("Treinando modelo..."):
                X, y, _ = detector.gerar_dados_baseados_coleipa()
                detector.treinar_modelo_bayesiano(X, y)
    
    # Formulário para entrada de dados
    st.subheader("Parâmetros do Sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vazao = st.number_input("Vazão (m³/h)", min_value=7.0, max_value=16.0, value=14.5, step=0.1,
                              help="Valor típico para o Coleipa: 14.5 m³/h")
    
    with col2:
        pressao = st.number_input("Pressão (mca)", min_value=0.0, max_value=10.0, value=3.5, step=0.1,
                                help="Valor típico para o Coleipa: 3.5 mca")
    
    with col3:
        ivi = st.number_input("IVI", min_value=1.0, max_value=25.0, value=16.33, step=0.01,
                            help="IVI do Coleipa: 16.33 (Categoria D)")
    
    # Botão para executar análise
    if st.button("Analisar Caso"):
        with st.spinner("Analisando caso..."):
            resultado = detector.analisar_caso_coleipa(vazao, pressao, ivi)
            
            # Exibir resultados em colunas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classificação")
                st.markdown(f"""
                - **Vazão**: {resultado['vazao']:.1f} m³/h → {resultado['classe_vazao']}
                - **Pressão**: {resultado['pressao']:.1f} mca → {resultado['classe_pressao']}
                - **IVI**: {resultado['ivi']:.2f} → {resultado['classe_ivi']}
                """)
                
                # Resultado da análise
                st.subheader("Resultado da Análise")
                st.markdown(f"### {resultado['cor']} {resultado['status']}")
            
            with col2:
                st.subheader("Resultados Numéricos")
                
                # Mostrar diferentes componentes da análise
                if 'prob_bayes' in resultado:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Risco Fuzzy", f"{resultado['risco_fuzzy']:.1f}%")
                    with col_b:
                        st.metric("Prob. Bayesiana", f"{resultado['prob_bayes']:.3f}")
                    with col_c:
                        st.metric("Prob. Híbrida", f"{resultado['prob_hibrida']:.3f}")
                else:
                    st.metric("Risco Fuzzy", f"{resultado['risco_fuzzy']:.1f}%")
                    st.info("Modelo Bayesiano não disponível - apenas análise fuzzy")
                
                # Comparação com Coleipa
                st.subheader("Comparação com Sistema Coleipa")
                st.markdown(f"""
                - **Perdas reais**: {resultado['percentual_perdas']:.1f}%
                - **IVI real**: {resultado['ivi_real']:.2f} (Categoria D - Muito Ruim)
                - **Prioridade recomendada**: Pesquisa de vazamentos
                """)
    
    # Explicação sobre a análise de caso
    st.markdown("---")
    st.subheader("Como interpretar os resultados")
    
    st.markdown("""
    #### Classificação de Vazão
    - **BAIXA (noturna)**: Valores abaixo de 9 m³/h, típicos de períodos noturnos
    - **NORMAL (transição)**: Valores entre 9-14 m³/h, compatíveis com operação normal
    - **ALTA (pico/vazamento)**: Valores acima de 14 m³/h, indicam pico de consumo ou vazamento
    
    #### Classificação de Pressão
    - **BAIXA (problema)**: Valores abaixo de 5 mca, indicam problemas na rede
    - **MÉDIA (operacional)**: Valores entre 5-8 mca, dentro da faixa operacional observada
    - **ALTA (boa)**: Valores acima de 8 mca, próximos do recomendado pela NBR
    
    #### Classificação de IVI
    - **BOM (Categoria A)**: IVI entre 1-4, sistema eficiente
    - **REGULAR (Categoria B)**: IVI entre 4-8, sistema regular
    - **RUIM (Categoria C)**: IVI entre 8-16, sistema ruim
    - **MUITO RUIM (Categoria D)**: IVI acima de 16, sistema muito ruim
    
    #### Interpretação do Status
    - 🟢 **OPERAÇÃO NORMAL**: Baixa probabilidade de vazamentos
    - 🟡 **RISCO ELEVADO - MONITORAR**: Situação de atenção, monitoramento recomendado
    - 🔴 **VAZAMENTO DETECTADO**: Alta probabilidade de vazamento, intervenção necessária
    """)


def mostrar_pagina_relatorio(detector):
    """Página de relatório completo"""
    st.header("📝 Relatório Completo")
    st.markdown("Relatório detalhado baseado nos dados do sistema Coleipa")
    
    # Botão para gerar relatório
    if st.button("Gerar Relatório Completo"):
        with st.spinner("Gerando relatório..."):
            relatorio = detector.gerar_relatorio_coleipa()
            
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
            detector.atualizar_caracteristicas_sistema(novas_caracteristicas)
            st.success("Características atualizadas com sucesso!")
            
            # Exibir novas características
            st.subheader("Novas Características do Sistema")
            df_caracteristicas_atualizadas = pd.DataFrame({
                'Característica': list(detector.caracteristicas_sistema.keys()),
                'Valor Atualizado': list(detector.caracteristicas_sistema.values())
            })
            st.dataframe(df_caracteristicas_atualizadas)
        else:
            st.info("Nenhuma característica foi alterada.")
    
    # Opções adicionais
    st.markdown("---")
    st.subheader("Opções Adicionais")
    
    # Baixar template de dados
    st.markdown("##### Template de Dados")
    formato = st.radio("Formato do template:", ["Excel (.xlsx)", "CSV (.csv)"], horizontal=True)
    nome_arquivo = "template_dados_coleipa." + ("xlsx" if formato == "Excel (.xlsx)" else "csv")
    df_template = detector.gerar_dados_template()
    download_button(df_template, nome_arquivo, "⬇️ Download Template de Dados")
    
    # Redefinir para valores padrão
    st.markdown("##### Redefinir Sistema")
    if st.button("Redefinir para Valores Padrão"):
        # Recriar detector com valores padrão
        new_detector = DetectorVazamentosColeipa()
        # Substituir o detector no cache
        st.session_state['detector'] = new_detector
        st.success("Sistema redefinido para valores padrão!")
        st.experimental_rerun()


# Iniciar aplicativo
if __name__ == "__main__":
    app_main()
