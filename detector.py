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
            print(f"Dados carregados do arquivo: {arquivo_dados}")
        else:
            self.dados_coleipa = self.dados_coleipa_default
            print("Usando dados padrão Coleipa (nenhum arquivo fornecido)")
        
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
    
    def carregar_dados_arquivo(self, arquivo):
        """
        Carrega dados de monitoramento de um arquivo Excel ou CSV
        
        Parâmetros:
        arquivo (str): Caminho para o arquivo Excel ou CSV
        
        Retorna:
        dict: Dicionário com os dados carregados
        """
        try:
            # Determinar tipo de arquivo pela extensão
            nome, extensao = os.path.splitext(arquivo)
            extensao = extensao.lower()
            
            if extensao == '.xlsx' or extensao == '.xls':
                # Carregar arquivo Excel
                df = pd.read_excel(arquivo)
                print("Arquivo Excel carregado com sucesso")
            elif extensao == '.csv':
                # Carregar arquivo CSV
                df = pd.read_csv(arquivo)
                print("Arquivo CSV carregado com sucesso")
            else:
                raise ValueError(f"Formato de arquivo não suportado: {extensao}. Use Excel (.xlsx, .xls) ou CSV (.csv)")
            
            # Validar estrutura dos dados
            # O arquivo deve ter as colunas: hora, vazao_dia1, pressao_dia1, etc.
            colunas_necessarias = ['hora', 'vazao_dia1', 'pressao_dia1', 'vazao_dia2', 
                                  'pressao_dia2', 'vazao_dia3', 'pressao_dia3']
            
            for coluna in colunas_necessarias:
                if coluna not in df.columns:
                    raise ValueError(f"Coluna '{coluna}' não encontrada no arquivo. Verifique o formato dos dados.")
            
            # Converter DataFrame para dicionário
            dados = {}
            for coluna in df.columns:
                dados[coluna] = df[coluna].tolist()
            
            # Verificar comprimento dos dados
            if len(dados['hora']) != 24:
                print(f"Aviso: O número de horas no arquivo ({len(dados['hora'])}) é diferente do esperado (24).")
            
            return dados
            
        except Exception as e:
            print(f"Erro ao carregar arquivo: {e}")
            print("Usando dados padrão Coleipa como fallback")
            return self.dados_coleipa_default
    
    def salvar_dados_template(self, arquivo="template_dados_coleipa.xlsx"):
        """
        Salva os dados padrão em um arquivo Excel como template
        
        Parâmetros:
        arquivo (str): Nome do arquivo a ser salvo
        """
        try:
            df = pd.DataFrame(self.dados_coleipa_default)
            
            # Determinar tipo de arquivo pela extensão
            nome, extensao = os.path.splitext(arquivo)
            extensao = extensao.lower()
            
            if extensao == '.xlsx' or extensao == '.xls':
                df.to_excel(arquivo, index=False)
                print(f"Template de dados salvo em '{arquivo}'")
            elif extensao == '.csv':
                df.to_csv(arquivo, index=False)
                print(f"Template de dados salvo em '{arquivo}'")
            else:
                raise ValueError("Formato não suportado. Use .xlsx, .xls ou .csv")
                
        except Exception as e:
            print(f"Erro ao salvar template: {e}")
    
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
    
    def criar_sistema_fuzzy(self, visualizar=False):
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
        
        if visualizar:
            self.visualizar_conjuntos_fuzzy(vazao, pressao, ivi, risco_vazamento)
        
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
        
        return self.sistema_fuzzy
    
    def visualizar_conjuntos_fuzzy(self, vazao, pressao, ivi, risco_vazamento):
        """Visualiza os conjuntos fuzzy baseados nos dados do Coleipa"""
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
        plt.show()

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
        plt.show()
        
        # Mostrar estatísticas
        print("\n=== ESTATÍSTICAS DOS DADOS COLEIPA ===")
        print(f"Vazão mínima observada: {df['Vazao_Media'].min():.2f} m³/h (hora {df.loc[df['Vazao_Media'].idxmin(), 'Hora']})")
        print(f"Vazão máxima observada: {df['Vazao_Media'].max():.2f} m³/h (hora {df.loc[df['Vazao_Media'].idxmax(), 'Hora']})")
        print(f"Pressão mínima observada: {df['Pressao_Media'].min():.2f} mca (hora {df.loc[df['Pressao_Media'].idxmin(), 'Hora']})")
        print(f"Pressão máxima observada: {df['Pressao_Media'].max():.2f} mca (hora {df.loc[df['Pressao_Media'].idxmax(), 'Hora']})")
        print(f"Razão vazão min/max: {(df['Vazao_Media'].min()/df['Vazao_Media'].max()*100):.1f}%")
        print(f"Horas com pressão < 10 mca: {len(df[df['Pressao_Media'] < 10])}/24 ({len(df[df['Pressao_Media'] < 10])/24*100:.1f}%)")
    
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
    
    def treinar_modelo_bayesiano(self, X, y, visualizar=True):
        """Treina modelo Bayesiano com dados baseados no Coleipa"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.modelo_bayes = GaussianNB()
        self.modelo_bayes.fit(X_train, y_train)
        
        y_pred = self.modelo_bayes.predict(X_test)
        
        print("\n=== AVALIAÇÃO DO MODELO BAYESIANO ===")
        print("Baseado nos dados do SAAP Coleipa")
        print(f"Características do sistema:")
        print(f"- População: {self.caracteristicas_sistema['populacao']} habitantes")
        print(f"- Área: {self.caracteristicas_sistema['area_territorial']/1000:.1f} km²")
        print(f"- Perdas reais: {self.caracteristicas_sistema['percentual_perdas']:.1f}%")
        print(f"- IVI: {self.caracteristicas_sistema['ivi']:.2f} (Categoria D - Muito Ruim)")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Vazamento']))
        
        if visualizar:
            cm = confusion_matrix(y_test, y_pred)
            self.visualizar_matriz_confusao(cm)
        
        return self.modelo_bayes
    
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
        plt.show()
    
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
            print(f"Erro na avaliação fuzzy: {e}")
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
        
        print(f"\n=== ANÁLISE DE CASO - PADRÃO COLEIPA ===")
        print(f"Entrada:")
        print(f"  Vazão: {vazao:.1f} m³/h")
        print(f"  Pressão: {pressao:.1f} mca")
        print(f"  IVI: {ivi:.2f}")
        
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
        
        print(f"\nClassificação:")
        print(f"  Vazão: {classe_vazao}")
        print(f"  Pressão: {classe_pressao}")
        print(f"  IVI: {classe_ivi}")
        
        # Avaliação fuzzy
        risco_fuzzy = self.avaliar_risco_fuzzy(vazao, pressao, ivi)
        print(f"\nRisco Fuzzy: {risco_fuzzy:.1f}%")
        
        # Avaliação Bayesiana (se disponível)
        if self.modelo_bayes is not None:
            dados = [vazao, pressao, ivi]
            prob_bayes = self.modelo_bayes.predict_proba([dados])[0][1]
            prob_hibrida = 0.6 * (risco_fuzzy/100) + 0.4 * prob_bayes
            
            print(f"Probabilidade Bayesiana: {prob_bayes:.3f}")
            print(f"Probabilidade Híbrida: {prob_hibrida:.3f}")
            
            if prob_hibrida > 0.5:
                resultado = "VAZAMENTO DETECTADO"
                cor = "🔴"
            elif prob_hibrida > 0.3:
                resultado = "RISCO ELEVADO - MONITORAR"
                cor = "🟡"
            else:
                resultado = "OPERAÇÃO NORMAL"
                cor = "🟢"
            
            print(f"\n{cor} RESULTADO: {resultado}")
        else:
            if risco_fuzzy > 50:
                resultado = "RISCO ELEVADO (apenas análise fuzzy)"
                cor = "🟡"
            else:
                resultado = "RISCO BAIXO (apenas análise fuzzy)"
                cor = "🟢"
            
            print(f"\n{cor} RESULTADO: {resultado}")
        
        # Comparação com dados reais do Coleipa
        print(f"\n=== COMPARAÇÃO COM DADOS COLEIPA ===")
        print(f"Sistema Coleipa:")
        print(f"  Perdas reais: {self.caracteristicas_sistema['percentual_perdas']:.1f}%")
        print(f"  IVI real: {self.caracteristicas_sistema['ivi']:.2f}")
        print(f"  Classificação: Categoria D (Muito Ruim)")
        print(f"  Prioridade recomendada: Pesquisa de vazamentos")
    
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
        
        self.visualizar_serie_temporal_coleipa(df, inicio_vazamento)
        return df
    
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
        plt.show()
    
    def gerar_mapa_calor_ivi(self, resolucao=50):
        """
        Gera mapas de calor mostrando o risco de vazamento para diferentes
        combinações de vazão e pressão, com diferentes valores de IVI baseados
        na classificação do Banco Mundial
        """
        # Verificar se o sistema fuzzy está criado
        if self.sistema_fuzzy is None:
            print("Criando sistema fuzzy...")
            self.criar_sistema_fuzzy(visualizar=False)
        
        # Valores de IVI baseados na classificação do Banco Mundial
        ivi_valores = [2, 6, 12, 18]  # Representativos das categorias A, B, C, D
        ivi_categorias = ['BOM (2.0)', 'REGULAR (6.0)', 'RUIM (12.0)', 'MUITO RUIM (18.0)']
        ivi_classificacoes = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
        
        # Valores para o mapa de calor baseados nos dados Coleipa
        vazoes = np.linspace(7, 16, resolucao)
        pressoes = np.linspace(2.5, 8, resolucao)
        
        print(f"Gerando mapas de calor IVI com resolução {resolucao}x{resolucao}...")
        print("Isso pode demorar alguns segundos...")
        
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
            print(f"Processando mapa {idx+1}/4: IVI {categoria}")
            
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
        
        # Título principal melhorado - CORREÇÃO: removido o parâmetro pad que causa o erro
        fig.suptitle('Mapa de Risco para diferentes IVIs\nClassificação Banco Mundial - Sistema Coleipa', 
                     fontsize=16, fontweight='bold', y=0.96)
        plt.show()
        
        # Análise detalhada do IVI
        print("\n" + "="*70)
        print("ANÁLISE DETALHADA DO IVI - SISTEMA COLEIPA")
        print("="*70)
        print(f"🔍 IVI Calculado: {self.caracteristicas_sistema['ivi']:.2f}")
        print(f"📊 Classificação: Categoria D (Muito Ruim)")
        print(f"⚠️  Interpretação: IVI > 16 indica uso extremamente ineficiente de recursos")
        print(f"")
        print(f"📈 Comparação com outras categorias:")
        print(f"   🟢 Categoria A (IVI 1-4): Sistema eficiente, perdas próximas ao inevitável")
        print(f"   🟡 Categoria B (IVI 4-8): Sistema regular, melhorias recomendadas")
        print(f"   🟠 Categoria C (IVI 8-16): Sistema ruim, ações urgentes necessárias")
        print(f"   🔴 Categoria D (IVI >16): Sistema muito ruim, intervenção imediata")
        print(f"")
        print(f"🎯 Análise específica do Coleipa (IVI = 16.33):")
        print(f"   • As perdas reais são 16.33 vezes maiores que as inevitáveis")
        print(f"   • Potencial de redução de perdas > 400 L/ramal.dia")
        print(f"   • Localização no mapa: zona vermelha (alto risco)")
        print(f"   • Combinação crítica: Vazão ALTA + Pressão BAIXA")
        print(f"   • Prioridade máxima: pesquisa e reparo imediato de vazamentos")
        print(f"")
        print(f"🔧 Impacto visual nos mapas:")
        print(f"   • IVI BOM (2.0): Predominantemente verde (baixo risco)")
        print(f"   • IVI REGULAR (6.0): Verde-amarelo (risco moderado)")
        print(f"   • IVI RUIM (12.0): Amarelo-laranja (risco elevado)")
        print(f"   • IVI MUITO RUIM (18.0): Vermelho intenso (risco crítico)")
        print("="*70)
    
    def gerar_relatorio_coleipa(self):
        """Gera relatório completo baseado nos dados do Coleipa"""
        print("="*60)
        print("RELATÓRIO DE ANÁLISE - SISTEMA COLEIPA")
        print("="*60)
        
        print(f"\n1. CARACTERÍSTICAS DO SISTEMA:")
        print(f"   Localização: Bairro Coleipa, Santa Bárbara do Pará-PA")
        print(f"   Área territorial: {self.caracteristicas_sistema['area_territorial']/1000:.1f} km²")
        print(f"   População atendida: {self.caracteristicas_sistema['populacao']} habitantes")
        print(f"   Número de ligações: {self.caracteristicas_sistema['numero_ligacoes']}")
        print(f"   Extensão da rede: {self.caracteristicas_sistema['comprimento_rede']} km")
        print(f"   Densidade de ramais: {self.caracteristicas_sistema['densidade_ramais']} ramais/km")
        
        print(f"\n2. RESULTADOS DO MONITORAMENTO (72 horas):")
        print(f"   Volume médio demandado: 273,5 m³/dia")
        print(f"   Volume médio consumido: {self.caracteristicas_sistema['volume_consumido_medio']} m³/dia")
        print(f"   Perdas reais médias: {self.caracteristicas_sistema['perdas_reais_media']} m³/dia")
        print(f"   Percentual de perdas: {self.caracteristicas_sistema['percentual_perdas']:.1f}%")
        
        print(f"\n3. INDICADORES DE DESEMPENHO:")
        print(f"   IPRL (Índice de Perdas Reais por Ligação): {self.caracteristicas_sistema['iprl']} m³/lig.dia")
        print(f"   IPRI (Índice de Perdas Reais Inevitáveis): {self.caracteristicas_sistema['ipri']} m³/lig.dia")
        print(f"   IVI (Índice de Vazamentos na Infraestrutura): {self.caracteristicas_sistema['ivi']}")
        
        print(f"\n4. CLASSIFICAÇÃO (Banco Mundial):")
        print(f"   Categoria: D (Muito Ruim)")
        print(f"   Interpretação: Uso ineficiente de recursos")
        print(f"   Recomendação: Programas de redução de perdas são imperiosos e prioritários")
        
        print(f"\n5. METODOLOGIA NPR - PRIORIZAÇÃO DE AÇÕES:")
        print(f"   1º Prioridade: Pesquisa de vazamentos (Resultado: 40)")
        print(f"   2º Prioridade: Agilidade e qualidade dos reparos (Resultado: 32)")
        print(f"   3º Prioridade: Gerenciamento de infraestrutura (Resultado: 28)")
        print(f"   4º Prioridade: Gerenciamento de pressão (Resultado: 2)")
        
        print(f"\n6. PROBLEMAS IDENTIFICADOS:")
        print(f"   - Pressões abaixo do mínimo NBR 12218 (10 mca)")
        print(f"   - Vazões mínimas noturnas elevadas (>50% da máxima)")
        print(f"   - Comportamento inverso vazão-pressão característico de vazamentos")
        print(f"   - IVI classificado como 'Muito Ruim' (>16)")
        
        print(f"\n7. RECOMENDAÇÕES:")
        print(f"   - Implementar programa intensivo de pesquisa de vazamentos")
        print(f"   - Cadastrar e reparar vazamentos visíveis rapidamente")
        print(f"   - Considerar aumento da altura do reservatório")
        print(f"   - Substituir trechos com vazamentos recorrentes")
        print(f"   - Mobilizar a comunidade para identificação de vazamentos")
        
        print("="*60)
    
    def atualizar_caracteristicas_sistema(self, novas_caracteristicas):
        """
        Atualiza as características do sistema com novos valores
        
        Parâmetros:
        novas_caracteristicas (dict): Dicionário com novas características
        """
        for chave, valor in novas_caracteristicas.items():
            if chave in self.caracteristicas_sistema:
                self.caracteristicas_sistema[chave] = valor
                print(f"Característica '{chave}' atualizada para: {valor}")
            else:
                print(f"Aviso: Característica '{chave}' não existe no sistema")


# Função principal modificada
def main():
    """Função principal com menu interativo e suporte a arquivos"""
    print("="*60)
    print("SISTEMA DE DETECÇÃO DE VAZAMENTOS - COLEIPA")
    print("Baseado nos dados reais do SAAP do bairro da Coleipa")
    print("Santa Bárbara do Pará - PA")
    print("="*60)
    
    # Perguntar se deseja carregar dados de um arquivo
    usar_arquivo = input("\nDeseja carregar dados de um arquivo? (s/n): ").lower()
    
    arquivo_dados = None
    if usar_arquivo == 's' or usar_arquivo == 'sim':
        arquivo_dados = input("Digite o caminho do arquivo Excel ou CSV: ")
        # Verificar se o arquivo existe
        if not os.path.exists(arquivo_dados):
            print(f"Arquivo não encontrado: {arquivo_dados}")
            criar_template = input("Deseja criar um template para preencher? (s/n): ").lower()
            if criar_template == 's' or criar_template == 'sim':
                formato = input("Formato desejado (xlsx/csv): ").lower()
                if formato == 'csv':
                    nome_arquivo = "template_dados_coleipa.csv"
                else:
                    nome_arquivo = "template_dados_coleipa.xlsx"
                
                # Criar uma instância temporária para salvar o template
                temp_detector = DetectorVazamentosColeipa()
                temp_detector.salvar_dados_template(nome_arquivo)
                print(f"Por favor, preencha o arquivo '{nome_arquivo}' e execute o programa novamente.")
                return
            arquivo_dados = None
    
    # Inicializar o detector com ou sem arquivo
    detector = DetectorVazamentosColeipa(arquivo_dados)
    
    while True:
        print("\nMENU DE OPÇÕES:")
        print("1. Visualizar dados do monitoramento")
        print("2. Criar e visualizar sistema fuzzy")
        print("3. Treinar modelo Bayesiano")
        print("4. Gerar mapa de calor IVI (Análise crucial)")
        print("5. Simular série temporal")
        print("6. Analisar caso específico")
        print("7. Gerar relatório completo")
        print("8. Análise completa (todas as opções)")
        print("9. Salvar template de dados para preenchimento")
        print("10. Atualizar características do sistema")
        print("0. Sair")
        
        try:
            opcao = input("\nSelecione uma opção: ")
            
            if opcao == "1":
                print("\nVisualizando dados do monitoramento...")
                detector.visualizar_dados_coleipa()
                
            elif opcao == "2":
                print("\nCriando sistema fuzzy...")
                detector.criar_sistema_fuzzy(visualizar=True)
                
            elif opcao == "3":
                print("\nGerando dados e treinando modelo Bayesiano...")
                X, y, df_real = detector.gerar_dados_baseados_coleipa()
                detector.treinar_modelo_bayesiano(X, y)
                
            elif opcao == "4":
                print("\nGerando mapa de calor IVI...")
                if detector.sistema_fuzzy is None:
                    print("Criando sistema fuzzy primeiro...")
                    detector.criar_sistema_fuzzy(visualizar=False)
                detector.gerar_mapa_calor_ivi()
                
            elif opcao == "5":
                print("\nSimulando série temporal...")
                if detector.modelo_bayes is None:
                    print("Treinando modelo primeiro...")
                    X, y, _ = detector.gerar_dados_baseados_coleipa()
                    detector.treinar_modelo_bayesiano(X, y, visualizar=False)
                detector.simular_serie_temporal_coleipa()
                
            elif opcao == "6":
                print("\nAnálise de caso específico:")
                try:
                    vazao = float(input("Digite a vazão (m³/h) [Enter para usar 14.5]: ") or "14.5")
                    pressao = float(input("Digite a pressão (mca) [Enter para usar 3.5]: ") or "3.5")
                    ivi = float(input("Digite o IVI [Enter para usar 16.33]: ") or "16.33")
                    
                    if detector.modelo_bayes is None:
                        print("Treinando modelo primeiro...")
                        X, y, _ = detector.gerar_dados_baseados_coleipa()
                        detector.treinar_modelo_bayesiano(X, y, visualizar=False)
                    
                    detector.analisar_caso_coleipa(vazao, pressao, ivi)
                except ValueError:
                    print("Entrada inválida. Usando valores padrão do Coleipa.")
                    detector.analisar_caso_coleipa()
                
            elif opcao == "7":
                detector.gerar_relatorio_coleipa()
                
            elif opcao == "8":
                print("\nExecutando análise completa...")
                
                # 1. Dados reais
                print("\n1/7 - Visualizando dados reais...")
                detector.visualizar_dados_coleipa()
                
                # 2. Sistema fuzzy
                print("\n2/7 - Criando sistema fuzzy...")
                detector.criar_sistema_fuzzy(visualizar=True)
                
                # 3. Modelo Bayesiano
                print("\n3/7 - Treinando modelo Bayesiano...")
                X, y, df_real = detector.gerar_dados_baseados_coleipa()
                detector.treinar_modelo_bayesiano(X, y)
                
                # 4. Mapa de calor IVI (ANÁLISE CRUCIAL)
                print("\n4/7 - Gerando mapa de calor IVI (Análise crucial)...")
                detector.gerar_mapa_calor_ivi()
                
                # 5. Série temporal
                print("\n5/7 - Simulando série temporal...")
                detector.simular_serie_temporal_coleipa()
                
                # 6. Caso específico
                print("\n6/7 - Analisando caso típico do Coleipa...")
                detector.analisar_caso_coleipa()
                
                # 7. Relatório
                print("\n7/7 - Gerando relatório...")
                detector.gerar_relatorio_coleipa()
                
                print("\n✅ ANÁLISE COMPLETA FINALIZADA!")
                print("📊 O mapa de calor IVI demonstra a criticidade do sistema Coleipa")
                print("🔴 IVI = 16.33 (Categoria D) confirma necessidade urgente de intervenção")
                
            elif opcao == "9":
                formato = input("Formato desejado (xlsx/csv) [xlsx]: ").lower() or "xlsx"
                if formato == "csv":
                    nome_arquivo = input("Nome do arquivo [template_dados_coleipa.csv]: ") or "template_dados_coleipa.csv"
                else:
                    nome_arquivo = input("Nome do arquivo [template_dados_coleipa.xlsx]: ") or "template_dados_coleipa.xlsx"
                detector.salvar_dados_template(nome_arquivo)
                
            elif opcao == "10":
                print("\nAtualização das características do sistema:")
                print("Características atuais:")
                for chave, valor in detector.caracteristicas_sistema.items():
                    print(f"  {chave}: {valor}")
                
                print("\nDigite as características que deseja atualizar (deixe em branco para manter o valor atual)")
                novas_caracteristicas = {}
                
                # Lista das principais características para atualizar
                caracteristicas_para_atualizar = [
                    'area_territorial', 'populacao', 'numero_ligacoes', 
                    'comprimento_rede', 'densidade_ramais', 'percentual_perdas', 'ivi'
                ]
                
                for carac in caracteristicas_para_atualizar:
                    valor_atual = detector.caracteristicas_sistema[carac]
                    try:
                        entrada = input(f"{carac} [{valor_atual}]: ")
                        if entrada.strip():
                            novas_caracteristicas[carac] = float(entrada)
                    except ValueError:
                        print(f"Valor inválido para {carac}. Mantendo valor original.")
                
                detector.atualizar_caracteristicas_sistema(novas_caracteristicas)
                
            elif opcao == "0":
                print("Encerrando o programa...")
                break
                
            else:
                print("Opção inválida. Tente novamente.")
                
        except KeyboardInterrupt:
            print("\n\nPrograma interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"Erro: {e}")
            print("Tente novamente.")

if __name__ == "__main__":
    main()