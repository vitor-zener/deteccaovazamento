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

# Matplotlib configuration
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DetectorVazamentosColeipa:
    """
    Fuzzy-Bayes hybrid system for leak detection based on data from 
    the Coleipa Drinking Water Supply System (SAAP)
    """
    
    def __init__(self, arquivo_dados=None):
        """
        Initializes the system based on Coleipa article data or loads from a file
        
        Parameters:
        arquivo_dados (str): Path to Excel or CSV file containing monitoring data
        """
        # Default system characteristics
        self.caracteristicas_sistema = {
            'area_territorial': 319000,  # m²
            'populacao': 1200,  # inhabitants
            'numero_ligacoes': 300,  # connections
            'comprimento_rede': 3,  # km
            'densidade_ramais': 100,  # branches/km
            'vazao_media_normal': 3.17,  # l/s (average of three days)
            'pressao_media_normal': 5.22,  # mca (average of three days)
            'perdas_reais_media': 102.87,  # m³/day
            'volume_consumido_medio': 128.29,  # m³/day
            'percentual_perdas': 44.50,  # %
            'iprl': 0.343,  # m³/connection.day
            'ipri': 0.021,  # m³/connection.day
            'ivi': 16.33  # Infrastructure Leakage Index
        }
        
        # Default hardcoded data (used only if no file is provided)
        self.dados_coleipa_default = {
            'hora': list(range(1, 25)),
            'vazao_dia1': [8.11, 7.83, 7.76, 7.80, 8.08, 9.69, 11.52, 11.92, 13.08, 14.22, 15.68, 14.55, 14.78, 13.16, 12.81, 11.64, 13.02, 13.40, 13.55, 12.94, 12.63, 11.45, 9.88, 8.30],
            'pressao_dia1': [6.89, 7.25, 7.12, 6.51, 6.42, 6.18, 4.83, 3.57, 4.67, 3.92, 3.70, 3.11, 2.68, 2.55, 3.34, 3.77, 4.70, 4.66, 4.69, 3.77, 4.78, 5.73, 6.24, 6.36],
            'vazao_dia2': [7.42, 7.23, 7.20, 7.28, 7.53, 8.99, 10.50, 12.18, 13.01, 13.22, 14.27, 13.63, 13.21, 12.97, 12.36, 11.98, 13.50, 13.27, 14.85, 12.84, 11.29, 10.79, 9.55, 8.60],
            'pressao_dia2': [6.65, 7.41, 7.45, 7.46, 7.38, 6.85, 5.47, 5.69, 4.05, 3.99, 4.56, 4.25, 2.87, 3.37, 4.86, 5.99, 6.39, 6.16, 5.16, 3.86, 3.96, 5.49, 6.48, 6.62],
            'vazao_dia3': [7.67, 7.46, 7.76, 8.18, 8.44, 9.24, 11.05, 12.55, 13.65, 13.63, 14.98, 14.09, 14.00, 13.06, 12.58, 11.81, 13.40, 14.29, 14.06, 12.69, 12.12, 10.83, 8.93, 8.45],
            'pressao_dia3': [6.91, 7.26, 7.12, 7.30, 7.16, 7.05, 6.43, 3.96, 4.70, 3.77, 3.97, 4.06, 4.13, 3.78, 3.12, 3.34, 5.55, 5.41, 4.93, 3.81, 4.37, 5.61, 6.36, 6.49]
        }
        
        # Try to load data from file if provided
        if arquivo_dados:
            self.dados_coleipa = self.carregar_dados_arquivo(arquivo_dados)
            st.success(f"Data loaded from file")
        else:
            self.dados_coleipa = self.dados_coleipa_default
            st.info("Using default Coleipa data (no file provided)")
        
        # Definition of fuzzy parameters based on real Coleipa data
        # Flow rate in m³/h (converted from l/s)
        self.param_vazao = {
            'BAIXA': {'range': [7, 9, 11]},     # Night flow rates
            'NORMAL': {'range': [9, 11.5, 14]},  # Transition flow rates
            'ALTA': {'range': [12, 15, 16]}     # Peak flow rates
        }
        
        # Pressure in mca (original data from the article)
        self.param_pressao = {
            'BAIXA': {'range': [0, 3, 5]},      # Below NBR minimum (10 mca)
            'MEDIA': {'range': [4, 6, 8]},      # Observed operational range
            'ALTA': {'range': [6, 8, 10]}      # Observed maximums
        }
        
        # IVI based on World Bank classification
        self.param_ivi = {
            'BOM': {'range': [1, 2, 4]},        # Category A
            'REGULAR': {'range': [4, 6, 8]},    # Category B
            'RUIM': {'range': [8, 12, 16]},     # Category C
            'MUITO_RUIM': {'range': [16, 20, 25]}  # Category D (Coleipa = 16.33)
        }
        
        # Leak risk
        self.param_risco = {
            'MUITO_BAIXO': {'range': [0, 10, 25]},
            'BAIXO': {'range': [15, 30, 45]},
            'MEDIO': {'range': [35, 50, 65]},
            'ALTO': {'range': [55, 70, 85]},
            'MUITO_ALTO': {'range': [75, 90, 100]}
        }
        
        # Component initialization
        self.sistema_fuzzy = None
        self.modelo_bayes = None
    
    def carregar_dados_arquivo(self, arquivo_uploaded):
        """
        Loads monitoring data from an Excel or CSV file from Streamlit
        
        Parameters:
        arquivo_uploaded: File uploaded by Streamlit
        
        Returns:
        dict: Dictionary with loaded data
        """
        try:
            # Determine file type by extension
            nome_arquivo = arquivo_uploaded.name
            nome, extensao = os.path.splitext(nome_arquivo)
            extensao = extensao.lower()
            
            if extensao == '.xlsx' or extensao == '.xls':
                # Load Excel file
                df = pd.read_excel(arquivo_uploaded)
                st.success("Excel file loaded successfully")
            elif extensao == '.csv':
                # Load CSV file
                df = pd.read_csv(arquivo_uploaded)
                st.success("CSV file loaded successfully")
            else:
                st.error(f"Unsupported file format: {extensao}. Use Excel (.xlsx, .xls) or CSV (.csv)")
                return self.dados_coleipa_default
            
            # Validate data structure
            # The file must have columns: hora, vazao_dia1, pressao_dia1, etc.
            colunas_necessarias = ['hora', 'vazao_dia1', 'pressao_dia1', 'vazao_dia2', 
                                  'pressao_dia2', 'vazao_dia3', 'pressao_dia3']
            
            for coluna in colunas_necessarias:
                if coluna not in df.columns:
                    st.warning(f"Column '{coluna}' not found in the file. Please check the data format.")
            
            # Convert DataFrame to dictionary
            dados = {}
            for coluna in df.columns:
                dados[coluna] = df[coluna].tolist()
            
            # Check data length
            if len(dados['hora']) != 24:
                st.warning(f"The number of hours in the file ({len(dados['hora'])}) is different from expected (24).")
            
            # Reset fuzzy system to force recreation with new data
            self.sistema_fuzzy = None
            
            return dados
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Using default Coleipa data as fallback")
            return self.dados_coleipa_default
    
    def gerar_dados_template(self):
        """
        Generates default data for download as template
        """
        df = pd.DataFrame(self.dados_coleipa_default)
        return df
    
    def criar_dataframe_coleipa(self):
        """Creates DataFrame with real Coleipa monitoring data"""
        df = pd.DataFrame()
        
        # Calculate means and standard deviations
        for hora in range(1, 25):
            idx = hora - 1
            vazao_valores = []
            pressao_valores = []
            
            # Check if we have data for this hour
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
            
            # If we don't have enough data, skip this hour
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
        """Creates fuzzy system based on Coleipa data"""
        # Define universes based on real data
        vazao = ctrl.Antecedent(np.arange(7, 17, 0.1), 'vazao')
        pressao = ctrl.Antecedent(np.arange(0, 11, 0.1), 'pressao')
        ivi = ctrl.Antecedent(np.arange(1, 26, 0.1), 'ivi')
        risco_vazamento = ctrl.Consequent(np.arange(0, 101, 1), 'risco_vazamento')
        
        # Define fuzzy sets
        for nome, param in self.param_vazao.items():
            vazao[nome] = fuzz.trimf(vazao.universe, param['range'])
        
        for nome, param in self.param_pressao.items():
            pressao[nome] = fuzz.trimf(pressao.universe, param['range'])
        
        for nome, param in self.param_ivi.items():
            ivi[nome] = fuzz.trimf(ivi.universe, param['range'])
        
        for nome, param in self.param_risco.items():
            risco_vazamento[nome] = fuzz.trimf(risco_vazamento.universe, param['range'])
        
        # Rules based on Coleipa analysis and expert knowledge
        regras = [
            # Rules for leak detection based on Coleipa pattern
            # HIGH flow + LOW pressure = strong indication of leak
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['MUITO_ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['RUIM'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['REGULAR'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['BOM'], risco_vazamento['MEDIO']),
            
            # NORMAL flow + LOW pressure = moderate risk
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['RUIM'], risco_vazamento['MEDIO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['REGULAR'], risco_vazamento['MEDIO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['BAIXA'] & ivi['BOM'], risco_vazamento['BAIXO']),
            
            # LOW flow (normal night operation)
            ctrl.Rule(vazao['BAIXA'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['MEDIO']),
            ctrl.Rule(vazao['BAIXA'] & pressao['MEDIA'] & ivi['MUITO_RUIM'], risco_vazamento['BAIXO']),
            ctrl.Rule(vazao['BAIXA'] & pressao['ALTA'] & ivi['MUITO_RUIM'], risco_vazamento['BAIXO']),
            
            # Normal operation
            ctrl.Rule(vazao['NORMAL'] & pressao['MEDIA'] & ivi['BOM'], risco_vazamento['MUITO_BAIXO']),
            ctrl.Rule(vazao['NORMAL'] & pressao['ALTA'] & ivi['BOM'], risco_vazamento['MUITO_BAIXO']),
            
            # Specific rules for the Coleipa case (IVI = 16.33, category D)
            ctrl.Rule(ivi['MUITO_RUIM'], risco_vazamento['MEDIO']),  # High IVI always indicates risk
            
            # Typical pattern observed in Coleipa during leaks
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'], risco_vazamento['ALTO'])
        ]
        
        # Create control system
        sistema_ctrl = ctrl.ControlSystem(regras)
        self.sistema_fuzzy = ctrl.ControlSystemSimulation(sistema_ctrl)
        
        return vazao, pressao, ivi, risco_vazamento
    
    def visualizar_conjuntos_fuzzy(self):
        """Visualizes fuzzy sets based on Coleipa data"""
        # Create fuzzy system if it doesn't exist yet
        vazao, pressao, ivi, risco_vazamento = self.criar_sistema_fuzzy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Flow rate
        axes[0, 0].clear()
        for nome in self.param_vazao.keys():
            axes[0, 0].plot(vazao.universe, vazao[nome].mf, label=nome, linewidth=2)
        axes[0, 0].set_title('Fuzzy Sets - Flow Rate (based on Coleipa data)')
        axes[0, 0].set_xlabel('Flow Rate (m³/h)')
        axes[0, 0].set_ylabel('Membership Degree')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pressure
        axes[0, 1].clear()
        for nome in self.param_pressao.keys():
            axes[0, 1].plot(pressao.universe, pressao[nome].mf, label=nome, linewidth=2)
        axes[0, 1].set_title('Fuzzy Sets - Pressure (based on Coleipa data)')
        axes[0, 1].set_xlabel('Pressure (mca)')
        axes[0, 1].set_ylabel('Membership Degree')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IVI
        axes[1, 0].clear()
        for nome in self.param_ivi.keys():
            axes[1, 0].plot(ivi.universe, ivi[nome].mf, label=nome, linewidth=2)
        axes[1, 0].set_title('Fuzzy Sets - IVI (World Bank Classification)')
        axes[1, 0].set_xlabel('IVI')
        axes[1, 0].set_ylabel('Membership Degree')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=self.caracteristicas_sistema['ivi'], color='red', linestyle='--', 
                          label=f"Coleipa ({self.caracteristicas_sistema['ivi']:.2f})")
        
        # Risk
        axes[1, 1].clear()
        for nome in self.param_risco.keys():
            axes[1, 1].plot(risco_vazamento.universe, risco_vazamento[nome].mf, label=nome, linewidth=2)
        axes[1, 1].set_title('Fuzzy Sets - Leak Risk')
        axes[1, 1].set_xlabel('Risk (%)')
        axes[1, 1].set_ylabel('Membership Degree')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def visualizar_dados_coleipa(self):
        """Visualizes real Coleipa monitoring data"""
        df = self.criar_dataframe_coleipa()
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Chart 1: Flow rates for the three days
        axes[0].plot(df['Hora'], df['Vazao_Dia1'], 'b-o', label='Day 1', alpha=0.7)
        axes[0].plot(df['Hora'], df['Vazao_Dia2'], 'r-s', label='Day 2', alpha=0.7)
        axes[0].plot(df['Hora'], df['Vazao_Dia3'], 'g-^', label='Day 3', alpha=0.7)
        axes[0].plot(df['Hora'], df['Vazao_Media'], 'k-', linewidth=3, label='Average')
        axes[0].fill_between(df['Hora'], 
                           df['Vazao_Media'] - df['Vazao_DP'], 
                           df['Vazao_Media'] + df['Vazao_DP'], 
                           alpha=0.2, color='gray', label='±1σ')
        axes[0].set_title('Flow Rate Monitoring - SAAP Coleipa (72 hours)')
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Flow Rate (m³/h)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Chart 2: Pressures for the three days
        axes[1].plot(df['Hora'], df['Pressao_Dia1'], 'b-o', label='Day 1', alpha=0.7)
        axes[1].plot(df['Hora'], df['Pressao_Dia2'], 'r-s', label='Day 2', alpha=0.7)
        axes[1].plot(df['Hora'], df['Pressao_Dia3'], 'g-^', label='Day 3', alpha=0.7)
        axes[1].plot(df['Hora'], df['Pressao_Media'], 'k-', linewidth=3, label='Average')
        axes[1].fill_between(df['Hora'], 
                           df['Pressao_Media'] - df['Pressao_DP'], 
                           df['Pressao_Media'] + df['Pressao_DP'], 
                           alpha=0.2, color='gray', label='±1σ')
        axes[1].axhline(y=10, color='red', linestyle='--', label='NBR 12218 Minimum (10 mca)')
        axes[1].set_title('Pressure Monitoring - SAAP Coleipa (72 hours)')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Pressure (mca)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Chart 3: Inverse relationship Flow vs Pressure
        ax2 = axes[2].twinx()
        line1 = axes[2].plot(df['Hora'], df['Vazao_Media'], 'b-', linewidth=2, label='Average Flow')
        line2 = ax2.plot(df['Hora'], df['Pressao_Media'], 'r-', linewidth=2, label='Average Pressure')
        
        axes[2].set_xlabel('Hour of Day')
        axes[2].set_ylabel('Flow Rate (m³/h)', color='b')
        ax2.set_ylabel('Pressure (mca)', color='r')
        axes[2].tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[2].legend(lines, labels, loc='upper left')
        axes[2].set_title('Inverse Relationship: Flow × Pressure (Sectorized Network)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Return the figure and statistics
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
    
    def calcular_ivi_automatico(self, arquivo_uploaded=None):
        """
        Automatically calculates IVI (Infrastructure Leakage Index) 
        from flow and pressure data
        
        Parameters:
        arquivo_uploaded: Optional file with additional data for IVI calculation
        
        Returns:
        float: Calculated IVI value
        dict: Dictionary with calculation components (CARL, UARL, etc.)
        """
        # Create dataframe with monitoring data
        df_monitoramento = self.criar_dataframe_coleipa()
        
        # Extract system parameters
        comprimento_rede = self.caracteristicas_sistema['comprimento_rede']  # km
        numero_ligacoes = self.caracteristicas_sistema['numero_ligacoes']    # connections
        pressao_media = df_monitoramento['Pressao_Media'].mean()            # mca
        
        # Calculate minimum night flow (average of hours 1-4)
        horas_noturnas = df_monitoramento[(df_monitoramento['Hora'] >= 1) & (df_monitoramento['Hora'] <= 4)]
        vazao_minima_noturna = horas_noturnas['Vazao_Media'].mean()  # m³/h
        
        # Try to load additional data from file if provided
        dados_adicionais = {}
        if arquivo_uploaded:
            try:
                # Determine file type by extension
                nome_arquivo = arquivo_uploaded.name
                nome, extensao = os.path.splitext(nome_arquivo)
                extensao = extensao.lower()
                
                if extensao == '.xlsx' or extensao == '.xls':
                    df_ivi = pd.read_excel(arquivo_uploaded, sheet_name='Calculo_IVI')
                    st.success("Data for IVI calculation loaded successfully")
                elif extensao == '.csv':
                    df_ivi = pd.read_csv(arquivo_uploaded)
                    st.success("Data for IVI calculation loaded successfully")
                else:
                    st.warning(f"Unsupported format for automatic IVI calculation: {extensao}")
                    df_ivi = None
                
                # If we were able to load the file, extract relevant data
                if df_ivi is not None:
                    # Look for specific columns in the file
                    colunas_esperadas = ['volume_diario', 'consumo_autorizado', 'perdas_aparentes']
                    if all(col in df_ivi.columns for col in colunas_esperadas):
                        dados_adicionais = {
                            'volume_diario': df_ivi['volume_diario'].mean(),
                            'consumo_autorizado': df_ivi['consumo_autorizado'].mean(),
                            'perdas_aparentes': df_ivi['perdas_aparentes'].mean()
                        }
                        st.success("Additional data for IVI calculation found!")
                    else:
                        st.info("File format recognized, but required columns not found.")
                        st.info("Using alternative method for IVI calculation.")
                        
            except Exception as e:
                st.warning(f"Error processing file for IVI calculation: {e}")
                st.info("Using alternative method for IVI calculation.")
        
        # Method 1: If we have complete data from the file
        if 'volume_diario' in dados_adicionais and 'consumo_autorizado' in dados_adicionais and 'perdas_aparentes' in dados_adicionais:
            # Calculate real losses (CARL) in m³/day
            volume_diario = dados_adicionais['volume_diario']  # m³/day
            consumo_autorizado = dados_adicionais['consumo_autorizado']  # m³/day
            perdas_aparentes = dados_adicionais['perdas_aparentes']  # m³/day
            
            perdas_reais = volume_diario - consumo_autorizado - perdas_aparentes  # m³/day
            
        # Method 2: Based on minimum night flow and estimates
        else:
            # Estimate legitimate night consumption (typically 6-8% of daily consumption)
            consumo_noturno_perc = 0.07  # 7% is a typical value
            consumo_legitimo_noturno = vazao_minima_noturna * consumo_noturno_perc  # m³/h
            
            # Estimate night leakage
            vazamento_noturno = vazao_minima_noturna - consumo_legitimo_noturno  # m³/h
            
            # Convert to daily volume (N1 factor from FAVAD methodology)
            # N1 factor relates leakage variation with pressure
            fator_n1 = 1.15  # Typical value between 0.5 and 1.5
            fator_dia_noite = 24 * ((pressao_media / pressao_media) ** fator_n1)
            
            # Calculate daily real losses
            perdas_reais = vazamento_noturno * fator_dia_noite  # m³/day
        
        # Calculate UARL (Unavoidable Annual Real Losses) using the standard IWA formula
        # UARL (liters/day) = (18 × Lm + 0.8 × Nc + 25 × Lp) × P
        # Where:
        # Lm = network length (km)
        # Nc = number of connections
        # Lp = total length of service connections (km) - estimated as Nc/density_connections
        # P = average pressure (mca)
        
        densidade_ramais = self.caracteristicas_sistema['densidade_ramais']
        comprimento_ramais = numero_ligacoes / densidade_ramais  # km
        
        # UARL calculation in liters/day
        uarl_litros_dia = (18 * comprimento_rede + 0.8 * numero_ligacoes + 25 * comprimento_ramais) * pressao_media
        
        # Convert to m³/day
        uarl_m3_dia = uarl_litros_dia / 1000
        
        # Calculate IPRL (Real Losses per Connection Index)
        iprl = perdas_reais / numero_ligacoes  # m³/connection.day
        
        # Calculate IPRI (Unavoidable Real Losses Index)
        ipri = uarl_m3_dia / numero_ligacoes  # m³/connection.day
        
        # Finally, calculate IVI
        ivi = iprl / ipri if ipri > 0 else 0
        
        # Update system characteristics
        self.caracteristicas_sistema['perdas_reais_media'] = perdas_reais
        self.caracteristicas_sistema['iprl'] = iprl
        self.caracteristicas_sistema['ipri'] = ipri
        self.caracteristicas_sistema['ivi'] = ivi
        
        # Reset fuzzy system to reflect the new IVI
        self.sistema_fuzzy = None
        
        # Prepare detailed results
        resultados = {
            'vazao_minima_noturna': vazao_minima_noturna,
            'pressao_media': pressao_media,
            'perdas_reais': perdas_reais,
            'uarl_m3_dia': uarl_m3_dia,
            'iprl': iprl,
            'ipri': ipri,
            'ivi': ivi
        }
        
        return ivi, resultados
    
    def gerar_dados_baseados_coleipa(self, n_amostras=500):
        """Generates synthetic data based on Coleipa system characteristics"""
        df_coleipa = self.criar_dataframe_coleipa()
        
        # Extract patterns from real data
        vazao_normal_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Vazao_Media'].mean()
        vazao_normal_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Vazao_Media'].std()
        pressao_normal_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Pressao_Media'].mean()
        pressao_normal_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]['Pressao_Media'].std()
        
        vazao_vazamento_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Vazao_Media'].mean()
        vazao_vazamento_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Vazao_Media'].std()
        pressao_vazamento_mean = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Pressao_Media'].mean()
        pressao_vazamento_std = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]['Pressao_Media'].std()
        
        # Generate synthetic data based on real patterns
        n_normal = int(0.55 * n_amostras)  # 55% normal (based on Coleipa data)
        n_vazamento = n_amostras - n_normal
        
        # Normal data
        vazao_normal = np.random.normal(vazao_normal_mean, vazao_normal_std, n_normal)
        pressao_normal = np.random.normal(pressao_normal_mean, pressao_normal_std, n_normal)
        ivi_normal = np.random.normal(8, 2, n_normal)  # Better IVI for normal operation
        
        # Leak data
        vazao_vazamento = np.random.normal(vazao_vazamento_mean, vazao_vazamento_std, n_vazamento)
        pressao_vazamento = np.random.normal(pressao_vazamento_mean, pressao_vazamento_std, n_vazamento)
        ivi_vazamento = np.random.normal(self.caracteristicas_sistema['ivi'], 3, n_vazamento)  # IVI similar to Coleipa
        
        # Combine data
        X = np.vstack([
            np.column_stack([vazao_normal, pressao_normal, ivi_normal]),
            np.column_stack([vazao_vazamento, pressao_vazamento, ivi_vazamento])
        ])
        
        y = np.hstack([np.zeros(n_normal), np.ones(n_vazamento)])
        
        return X, y, df_coleipa
    
    def treinar_modelo_bayesiano(self, X, y):
        """Trains Bayesian model with data based on Coleipa"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.modelo_bayes = GaussianNB()
        self.modelo_bayes.fit(X_train, y_train)
        
        y_pred = self.modelo_bayes.predict(X_test)
        
        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Leak'], output_dict=True)
        
        return self.modelo_bayes, cm, report
    
    def visualizar_matriz_confusao(self, cm):
        """Visualizes confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Normal', 'Leak'],
                    yticklabels=['Normal', 'Leak'])
        plt.title('Confusion Matrix - Coleipa System')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return plt.gcf()
    
    def avaliar_risco_fuzzy(self, vazao, pressao, ivi):
        """Evaluates risk using fuzzy system"""
        if self.sistema_fuzzy is None:
            self.criar_sistema_fuzzy()
        
        try:
            # Limit values to Coleipa data ranges
            vazao_limitada = max(7, min(vazao, 16))
            pressao_limitada = max(0, min(pressao, 10))
            ivi_limitado = max(1, min(ivi, 25))
            
            self.sistema_fuzzy.input['vazao'] = vazao_limitada
            self.sistema_fuzzy.input['pressao'] = pressao_limitada
            self.sistema_fuzzy.input['ivi'] = ivi_limitado
            
            self.sistema_fuzzy.compute()
            
            return self.sistema_fuzzy.output['risco_vazamento']
        except Exception as e:
            st.error(f"Error in fuzzy evaluation: {e}")
            return 50
    
    def analisar_caso_coleipa(self, vazao=None, pressao=None, ivi=None):
        """Analyzes a specific case using Coleipa patterns"""
        # Use typical Coleipa values if not provided
        if vazao is None:
            vazao = 14.5  # Typical peak flow
        if pressao is None:
            pressao = 3.5   # Typical low pressure
        if ivi is None:
            ivi = self.caracteristicas_sistema['ivi']   # Real Coleipa IVI
        
        # Classification based on Coleipa data
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
        
        # Fuzzy evaluation
        risco_fuzzy = self.avaliar_risco_fuzzy(vazao, pressao, ivi)
        
        resultado = {}
        resultado['vazao'] = vazao
        resultado['pressao'] = pressao
        resultado['ivi'] = ivi
        resultado['classe_vazao'] = classe_vazao
        resultado['classe_pressao'] = classe_pressao
        resultado['classe_ivi'] = classe_ivi
        resultado['risco_fuzzy'] = risco_fuzzy
        
        # Bayesian evaluation (if available)
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
        
        # Comparison with real Coleipa data
        resultado['percentual_perdas'] = self.caracteristicas_sistema['percentual_perdas']
        resultado['ivi_real'] = self.caracteristicas_sistema['ivi']
        
        return resultado
    
    def simular_serie_temporal_coleipa(self):
        """Simulates time series based on real Coleipa patterns"""
        df_real = self.criar_dataframe_coleipa()
        
        # Create expanded time series (3 complete days)
        tempo = []
        vazao = []
        pressao = []
        
        for dia in range(3):
            for hora in range(24):
                timestamp = datetime(2024, 1, 1 + dia, hora, 0)
                tempo.append(timestamp)
                
                # Use real Coleipa data with variation
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
                    # Use average values if we don't have data for this hour/day
                    if len(df_real) > 0:
                        hora_idx = hora % len(df_real)
                        v = df_real.iloc[hora_idx]['Vazao_Media'] + np.random.normal(0, 0.1)
                        p = df_real.iloc[hora_idx]['Pressao_Media'] + np.random.normal(0, 0.05)
                    else:
                        # Default values if we don't even have average data
                        v = 10 + np.random.normal(0, 0.1)
                        p = 5 + np.random.normal(0, 0.05)
                
                vazao.append(v)
                pressao.append(p)
        
        # Simulate leak starting on the second day at 2pm
        inicio_vazamento = 24 + 14  # corresponding index
        for i in range(inicio_vazamento, len(vazao)):
            # Leak progression
            progresso = min(1.0, (i - inicio_vazamento) / 10)
            vazao[i] += 3 * progresso  # Gradual increase
            pressao[i] -= 1.5 * progresso  # Gradual decrease
        
        # Create DataFrame
        df = pd.DataFrame({
            'Tempo': tempo,
            'Vazao': vazao,
            'Pressao': pressao,
            'IVI': [self.caracteristicas_sistema['ivi']] * len(tempo),
            'Vazamento_Real': [0] * inicio_vazamento + [1] * (len(tempo) - inicio_vazamento)
        })
        
        # Calculate detections if the model is trained
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
        """Visualizes time series based on Coleipa"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Chart 1: Flow Rate
        axes[0].plot(df['Tempo'], df['Vazao'], 'b-', linewidth=1.5, label='Flow Rate')
        axes[0].axvline(x=df['Tempo'][inicio_vazamento], color='red', linestyle='--', 
                       label=f'Leak Start ({df["Tempo"][inicio_vazamento].strftime("%d/%m %H:%M")})')
        axes[0].set_ylabel('Flow Rate (m³/h)')
        axes[0].set_title('Time Series - Coleipa System: Flow Rate')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Chart 2: Pressure
        axes[1].plot(df['Tempo'], df['Pressao'], 'r-', linewidth=1.5, label='Pressure')
        axes[1].axhline(y=10, color='orange', linestyle=':', label='NBR Minimum (10 mca)')
        axes[1].axvline(x=df['Tempo'][inicio_vazamento], color='red', linestyle='--')
        axes[1].set_ylabel('Pressure (mca)')
        axes[1].set_title('Time Series - Coleipa System: Pressure')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Chart 3: Detections (if available)
        if 'Prob_Hibrida' in df.columns:
            axes[2].plot(df['Tempo'], df['Prob_Hibrida'], 'purple', linewidth=2, label='Hybrid Detection')
            axes[2].plot(df['Tempo'], df['Risco_Fuzzy'], 'green', alpha=0.7, label='Fuzzy Component')
            axes[2].plot(df['Tempo'], df['Prob_Bayes'], 'orange', alpha=0.7, label='Bayes Component')
            axes[2].axhline(y=0.5, color='black', linestyle='-.', label='Detection Threshold')
            axes[2].axvline(x=df['Tempo'][inicio_vazamento], color='red', linestyle='--')
            axes[2].set_ylabel('Probability')
            axes[2].set_title('Leak Detection - Hybrid System')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Bayesian model not trained\nOnly fuzzy analysis available', 
                        ha='center', va='center', transform=axes[2].transAxes, fontsize=14)
            axes[2].set_title('Detection not available')
        
        axes[2].set_xlabel('Time')
        plt.tight_layout()
        return fig, df
    
    def gerar_mapa_calor_ivi(self, resolucao=30):
        """
        Generates heat maps showing leak risk for different
        combinations of flow and pressure, with different IVI values based
        on World Bank classification
        """
        # Check if fuzzy system is created
        if self.sistema_fuzzy is None:
            self.criar_sistema_fuzzy()
        
        # Use current IVI for "MUITO_RUIM" category
        current_ivi = self.caracteristicas_sistema['ivi']
        
        # IVI values based on World Bank classification
        ivi_valores = [2, 6, 12, 16]  # Representative of categories A, B, C, D
        ivi_categorias = ['BOM (2.0)', 'REGULAR (6.0)', 'RUIM (12.0)', f'MUITO RUIM (16.0)']
        ivi_classificacoes = ['Category A', 'Category B', 'Category C', 'Category D']
        
        # Values for heat map based on Coleipa data
        vazoes = np.linspace(7, 16, resolucao)
        pressoes = np.linspace(2.5, 8, resolucao)
        
        # Set up figure with 2x2 subplots for maps + 1 subplot for color bar
        fig = plt.figure(figsize=(18, 16))
        
        # Create grid: 3 rows, 2 columns
        # Row 1: 2 upper maps
        # Row 2: 2 lower maps  
        # Row 3: centered color bar
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.15], hspace=0.3, wspace=0.2)
        
        # Create 4 subplots for the maps
        axes = [
            fig.add_subplot(gs[0, 0]),  # Upper left
            fig.add_subplot(gs[0, 1]),  # Upper right
            fig.add_subplot(gs[1, 0]),  # Lower left
            fig.add_subplot(gs[1, 1])   # Lower right
        ]
        
        # Subplot for color bar (occupying full width)
        cbar_ax = fig.add_subplot(gs[2, :])
        
        # Generate a heat map for each IVI value
        im = None  # To capture the last image for the color bar
        for idx, (ax, ivi_valor, categoria, classificacao) in enumerate(zip(axes, ivi_valores, ivi_categorias, ivi_classificacoes)):
            
            # Create grid for the map
            X, Y = np.meshgrid(vazoes, pressoes)
            Z = np.zeros_like(X)
            
            # Calculate risk for each point on the grid
            for ii in range(X.shape[0]):
                for jj in range(X.shape[1]):
                    try:
                        # Ensure values are within limits
                        vazao_val = max(7, min(X[ii, jj], 16))
                        pressao_val = max(2.5, min(Y[ii, jj], 8))
                        ivi_val = max(1, min(ivi_valor, 25))
                        
                        # Calculate risk using the fuzzy system
                        self.sistema_fuzzy.input['vazao'] = vazao_val
                        self.sistema_fuzzy.input['pressao'] = pressao_val
                        self.sistema_fuzzy.input['ivi'] = ivi_val
                        
                        self.sistema_fuzzy.compute()
                        risco = self.sistema_fuzzy.output['risco_vazamento']
                        Z[ii, jj] = max(0, min(risco, 100))
                        
                    except Exception as e:
                        # Heuristic based on Coleipa patterns
                        vazao_norm = (X[ii, jj] - 7) / (16 - 7)  # Normalize 0-1
                        pressao_norm = 1 - (Y[ii, jj] - 2.5) / (8 - 2.5)  # Invert: low pressure = high risk
                        
                        # Calculate base risk
                        risco_base = (vazao_norm * 0.6 + pressao_norm * 0.4) * 70
                        
                        # Adjust by IVI
                        fator_ivi = ivi_valor / 10  # IVI 2=0.2, IVI 18=1.8
                        Z[ii, jj] = min(100, risco_base * fator_ivi + 10)
            
            # Plot heat map with improved color scale
            im = ax.imshow(Z, cmap='RdYlGn_r', origin='lower', 
                          extent=[vazoes.min(), vazoes.max(), pressoes.min(), pressoes.max()],
                          aspect='auto', vmin=0, vmax=100, interpolation='bilinear')
            
            # Add smoother contours
            try:
                contour_levels = [20, 40, 60, 80]
                contours = ax.contour(X, Y, Z, levels=contour_levels, colors='black', alpha=0.4, linewidths=1.5)
                ax.clabel(contours, inline=True, fontsize=10, fmt='%d%%', 
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            except:
                pass
            
            # Fuzzy set division lines - more visible
            # Flow: LOW (7-9), NORMAL (9-14), HIGH (14-16)
            ax.axvline(x=9, color='navy', linestyle=':', alpha=0.8, linewidth=2)
            ax.axvline(x=14, color='navy', linestyle=':', alpha=0.8, linewidth=2)
            
            # Pressure: LOW (2.5-4.5), NORMAL (4.5-6), HIGH (6-8)
            ax.axhline(y=4.5, color='darkgreen', linestyle=':', alpha=0.8, linewidth=2)
            ax.axhline(y=6.0, color='darkgreen', linestyle=':', alpha=0.8, linewidth=2)
            
            # Fuzzy set labels with reduced size
            ax.text(8, 7.5, 'FLOW\nLOW', color='navy', fontsize=9, fontweight='bold', 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            ax.text(11.5, 7.5, 'FLOW\nNORMAL', color='navy', fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            ax.text(15, 7.5, 'FLOW\nHIGH', color='navy', fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            
            # Pressure labels with reduced size
            ax.text(15.5, 3.5, 'PRESSURE\nLOW', color='darkgreen', fontsize=9, fontweight='bold', 
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            ax.text(15.5, 5.2, 'PRESSURE\nNORMAL', color='darkgreen', fontsize=9, fontweight='bold',
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            ax.text(15.5, 7, 'PRESSURE\nHIGH', color='darkgreen', fontsize=9, fontweight='bold',
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            
            # Mark the characteristic point of Coleipa in all charts
            if idx == 3:  # Last chart (Very Bad IVI) - special highlight
                ax.scatter([14.5], [3.5], color='red', s=300, marker='*', 
                          edgecolors='darkred', linewidth=3, label='Coleipa Point\n(IVI=16.33)', zorder=10)
                ax.annotate(f'COLEIPA SYSTEM\n(Flow=14.5, Pressure=3.5)\nIVI={current_ivi:.2f} - CRITICAL', 
                           xy=(14.5, 3.5), xytext=(11, 2.8),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=9, fontweight='bold', color='red',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9))
                ax.legend(loc='upper left', fontsize=9)
            else:
                # Mark the Coleipa point in other charts too
                ax.scatter([14.5], [3.5], color='red', s=150, marker='*', 
                          edgecolors='darkred', linewidth=2, alpha=0.7, zorder=8)
            
            # Axis settings
            ax.set_xlabel('Flow Rate (m³/h)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Pressure (mca)', fontsize=12, fontweight='bold')
            ax.set_title(f'Risk Map - IVI {categoria}\n{classificacao}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_xlim(7, 16)
            ax.set_ylim(2.5, 8)
            
            # Improve ticks
            ax.set_xticks(np.arange(7, 17, 1))
            ax.set_yticks(np.arange(3, 9, 1))
        
        # Create separate color bar in the dedicated subplot
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Leak Risk (%)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)
        
        # Add custom ticks to color bar
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%\n(Very Low)', '20%\n(Low)', '40%\n(Medium)', 
                            '60%\n(High)', '80%\n(Very High)', '100%\n(Critical)'])
        
        # Improved main title
        fig.suptitle('Risk Map for different IVIs\nWorld Bank Classification - Coleipa System', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        return fig, ivi_valores
    
    def gerar_relatorio_coleipa(self):
        """Generates complete report based on Coleipa system data"""
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
        Updates system characteristics with new values
        
        Parameters:
        novas_caracteristicas (dict): Dictionary with new characteristics
        """
        for chave, valor in novas_caracteristicas.items():
            if chave in self.caracteristicas_sistema:
                self.caracteristicas_sistema[chave] = valor
                st.success(f"Characteristic '{chave}' updated to: {valor}")
            else:
                st.warning(f"Warning: Characteristic '{chave}' does not exist in the system")
        
        # Reset fuzzy system to force recreation with new parameters
        self.sistema_fuzzy = None


# Streamlit page configuration
st.set_page_config(
    page_title="Leak Detection System - Coleipa",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variable to store detector instance
@st.cache_resource
def get_detector(arquivo_uploaded=None):
    return DetectorVazamentosColeipa(arquivo_uploaded)

# Function for downloading files
def download_button(object_to_download, download_filename, button_text):
    """
    Generates a button that allows downloading an object
    """
    if isinstance(object_to_download, pd.DataFrame):
        # If it's a DataFrame
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
        # If it's another type of object
        st.warning("Object type not supported for download")


def app_main():
    """Main function of the Streamlit application"""
    st.title("💧 Leak Detection System - SAAP Coleipa")
    st.markdown("##### Fuzzy-Bayes hybrid system for leak detection in supply networks")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    paginas = [
        "Home",
        "Monitoring Data",
        "Fuzzy System",
        "Bayesian Model",
        "IVI Heat Maps",
        "Temporal Simulation",
        "Case Analysis",
        "Complete Report",
        "Settings"
    ]
    pagina_selecionada = st.sidebar.radio("Select a page:", paginas)
    
    # File upload in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Input Data")
    arquivo_uploaded = st.sidebar.file_uploader("Load monitoring data", type=["xlsx", "csv"])
    
    # Initialize or get detector
    detector = get_detector(arquivo_uploaded)
    
    # Template for download in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Template")
    formato = st.sidebar.radio("Format:", ["Excel (.xlsx)", "CSV (.csv)"], horizontal=True)
    nome_arquivo = "template_dados_coleipa." + ("xlsx" if formato == "Excel (.xlsx)" else "csv")
    df_template = detector.gerar_dados_template()
    download_button(df_template, nome_arquivo, "⬇️ Download Template")
    
    # Information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About the System")
    st.sidebar.info(
        "System based on real data from the SAAP of Coleipa neighborhood, "
        "Santa Bárbara do Pará - PA. Uses fuzzy logic and Bayesian models "
        "for leak detection in water supply networks."
    )
    
    # Main content based on selected page
    if pagina_selecionada == "Home":
        mostrar_pagina_inicio()
    
    elif pagina_selecionada == "Monitoring Data":
        mostrar_pagina_dados(detector)
    
    elif pagina_selecionada == "Fuzzy System":
        mostrar_pagina_fuzzy(detector)
    
    elif pagina_selecionada == "Bayesian Model":
        mostrar_pagina_bayes(detector)
    
    elif pagina_selecionada == "IVI Heat Maps":
        mostrar_pagina_mapa_calor(detector)
    
    elif pagina_selecionada == "Temporal Simulation":
        mostrar_pagina_simulacao(detector)
    
    elif pagina_selecionada == "Case Analysis":
        mostrar_pagina_analise_caso(detector)
    
    elif pagina_selecionada == "Complete Report":
        mostrar_pagina_relatorio(detector)
    
    elif pagina_selecionada == "Settings":
        mostrar_pagina_configuracoes(detector)


def mostrar_pagina_inicio():
    """Home page of the application"""
    st.header("Welcome to the Leak Detection System")
    
    # System description
    st.markdown("""
    This system uses a hybrid approach combining fuzzy logic and Bayesian analysis to 
    detect leaks in water supply networks based on monitoring data.
    """)
    
    # Overview in 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Data Analysis")
        st.markdown("""
        - Visualization of monitoring data
        - Statistical analysis of flow and pressure
        - Critical pattern identification
        """)
        st.image("https://via.placeholder.com/300x200?text=Monitoring+Data", use_container_width=True)
    
    with col2:
        st.subheader("🧠 Hybrid Intelligence")
        st.markdown("""
        - Fuzzy system based on expert knowledge
        - Bayesian model for classification
        - Heat maps for risk analysis
        """)
        st.image("https://via.placeholder.com/300x200?text=Fuzzy+System", use_container_width=True)
    
    with col3:
        st.subheader("📊 Results and Reports")
        st.markdown("""
        - Real-time leak simulation
        - Analysis of specific cases
        - Detailed reports with recommendations
        """)
        st.image("https://via.placeholder.com/300x200?text=Reports", use_container_width=True)
    
    # About the Coleipa case
    st.markdown("---")
    st.subheader("About the Coleipa System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        The SAAP (Drinking Water Supply System) of Coleipa neighborhood, located in Santa 
        Bárbara do Pará, presents typical characteristics of systems with significant losses:
        
        - **IVI (Infrastructure Leakage Index)**: {current_ivi:.2f}
        - **Real losses**: 44.50% of the distributed volume
        - **Pressures**: Consistently below the recommended minimum (10 mca)
        - **Characteristic pattern**: High flows with low pressures
        
        This system was developed from the detailed analysis of these data and seeks to provide tools
        for identification, analysis, and management of leaks in similar networks.
        """)
    
    with col2:
        st.markdown("""
        #### System Characteristics
        
        - **Territorial area**: 319,000 m²
        - **Population served**: 1,200 inhabitants
        - **Number of connections**: 300
        - **Network length**: 3 km
        - **Branch density**: 100 branches/km
        
        #### World Bank Classification for IVI
        - **Category A (1-4)**: Efficient system
        - **Category B (4-8)**: Regular system
        - **Category C (8-16)**: Bad system
        - **Category D (>16)**: Very bad system
        """)
    
    # How to use the system
    st.markdown("---")
    st.subheader("How to use this system")
    st.markdown("""
    1. Use the sidebar to navigate between different functionalities
    2. Load your monitoring data or use the default Coleipa data
    3. Explore the charts and analyses available in each section
    4. Generate reports and recommendations for your specific system
    """)
    
    # Footer
    st.markdown("---")
    st.caption("Coleipa Leak Detection System | Based on Fuzzy-Bayes hybrid techniques")


def mostrar_pagina_dados(detector):
    """Monitoring data visualization page"""
    st.header("📊 Monitoring Data")
    st.markdown("Visualization of real monitoring data from the Coleipa System")
    
    # Button to process data
    if st.button("View Monitoring Data"):
        with st.spinner("Processing monitoring data..."):
            fig, stats, df = detector.visualizar_dados_coleipa()
            
            # Display charts
            st.pyplot(fig)
            
            # Display statistics in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Flow Statistics")
                st.metric("Minimum flow", f"{stats['vazao_min']:.2f} m³/h", f"Hour {int(stats['vazao_min_hora'])}")
                st.metric("Maximum flow", f"{stats['vazao_max']:.2f} m³/h", f"Hour {int(stats['vazao_max_hora'])}")
                st.metric("Min/max ratio", f"{stats['vazao_ratio']:.1f}%")
            
            with col2:
                st.subheader("Pressure Statistics")
                st.metric("Minimum pressure", f"{stats['pressao_min']:.2f} mca", f"Hour {int(stats['pressao_min_hora'])}")
                st.metric("Maximum pressure", f"{stats['pressao_max']:.2f} mca", f"Hour {int(stats['pressao_max_hora'])}")
                st.metric("Hours with pressure < 10 mca", f"{stats['horas_pressao_baixa']} of 24", f"{stats['perc_pressao_baixa']:.1f}%")
            
            # Display data table
            st.subheader("Monitoring Data")
            st.dataframe(df)


def mostrar_pagina_fuzzy(detector):
    """Fuzzy system page"""
    st.header("🧠 Fuzzy System")
    st.markdown("Visualization and configuration of the fuzzy system for leak detection")
    
    # Fuzzy sets visualization
    st.subheader("Fuzzy Sets")
    if st.button("View Fuzzy Sets"):
        with st.spinner("Generating fuzzy sets visualization..."):
            fig = detector.visualizar_conjuntos_fuzzy()
            st.pyplot(fig)
    
    # Explanation about fuzzy rules
    st.subheader("Fuzzy System Rules")
    st.markdown("""
    The fuzzy system uses rules based on the analysis of the hydraulic behavior of the Coleipa network.
    Some of the main rules are:
    
    1. **HIGH Flow + LOW Pressure + VERY_BAD IVI → VERY_HIGH Risk**  
       *This is the typical leak situation in the Coleipa system*
       
    2. **HIGH Flow + LOW Pressure + REGULAR/BAD IVI → HIGH Risk**  
       *Strong indication of leak even in systems with better conditions*
       
    3. **NORMAL Flow + LOW Pressure + VERY_BAD IVI → HIGH Risk**  
       *Systems with high IVI have higher risk even with normal flows*
       
    4. **NORMAL Flow + HIGH Pressure + GOOD IVI → VERY_LOW Risk**  
       *Normal operation in well-maintained systems*
    """)
    
    # Interactive test of the fuzzy system
    st.subheader("Interactive Test")
    st.markdown("Adjust the parameters below to test the behavior of the fuzzy system:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vazao_teste = st.slider("Flow (m³/h)", 7.0, 16.0, 14.5, 0.1)
    
    with col2:
        pressao_teste = st.slider("Pressure (mca)", 0.0, 10.0, 3.5, 0.1)
    
    with col3:
        ivi_teste = st.slider("IVI", 1.0, 25.0, detector.caracteristicas_sistema['ivi'], 0.01)
    
    if st.button("Calculate Fuzzy Risk"):
        with st.spinner("Calculating risk..."):
            risco = detector.avaliar_risco_fuzzy(vazao_teste, pressao_teste, ivi_teste)
            
            # Determine risk category
            categoria_risco = ""
            cor_risco = ""
            if risco < 20:
                categoria_risco = "VERY LOW"
                cor_risco = "green"
            elif risco < 40:
                categoria_risco = "LOW"
                cor_risco = "lightgreen"
            elif risco < 60:
                categoria_risco = "MEDIUM"
                cor_risco = "orange"
            elif risco < 80:
                categoria_risco = "HIGH"
                cor_risco = "darkorange"
            else:
                categoria_risco = "VERY HIGH"
                cor_risco = "red"
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                #### Evaluation Result
                - **Flow**: {vazao_teste:.1f} m³/h
                - **Pressure**: {pressao_teste:.1f} mca
                - **IVI**: {ivi_teste:.2f}
                """)
            
            with col2:
                st.markdown(f"#### Leak Risk")
                st.markdown(f"<h2 style='color:{cor_risco};'>{risco:.1f}% - {categoria_risco}</h2>", unsafe_allow_html=True)


def mostrar_pagina_bayes(detector):
    """Bayesian model page"""
    st.header("🔄 Bayesian Model")
    st.markdown("Training and evaluation of the Naive Bayes model for leak detection")
    
    # Training parameters
    st.subheader("Training Parameters")
    n_amostras = st.slider("Number of synthetic samples", 100, 2000, 500, 100)
    
    # Button to train the model
    if st.button("Train Bayesian Model"):
        with st.spinner("Generating synthetic data and training model..."):
            X, y, _ = detector.gerar_dados_baseados_coleipa(n_amostras)
            modelo, cm, report = detector.treinar_modelo_bayesiano(X, y)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig_cm = detector.visualizar_matriz_confusao(cm)
                st.pyplot(fig_cm)
            
            with col2:
                st.subheader("Classification Report")
                # Convert report to DataFrame for better visualization
                df_report = pd.DataFrame(report).transpose()
                df_report = df_report.round(3)
                st.dataframe(df_report)
                
                # System characteristics
                st.markdown("#### Coleipa System Characteristics")
                st.markdown(f"""
                - **Population**: {detector.caracteristicas_sistema['populacao']} inhabitants
                - **Area**: {detector.caracteristicas_sistema['area_territorial']/1000:.1f} km²
                - **Real losses**: {detector.caracteristicas_sistema['percentual_perdas']:.1f}%
                - **IVI**: {detector.caracteristicas_sistema['ivi']:.2f} (Category D - Very Bad)
                """)
    
    # Model explanation
    st.markdown("---")
    st.subheader("About the Bayesian Model")
    st.markdown("""
    The Naive Bayes model is trained with synthetic data generated from the patterns observed in the Coleipa System.
    It considers three main parameters:
    
    1. **Flow** - High values indicate possible leaks
    2. **Pressure** - Low values indicate possible leaks
    3. **IVI** - Systems with high IVI have a higher probability of leaks
    
    The training data are generated based on the following characteristics:
    
    - **Normal Operation**: 
      - Lower average flow
      - Higher average pressure
      - Lower average IVI (simulating more efficient systems)
      
    - **Leak**: 
      - Higher average flow
      - Lower average pressure
      - Average IVI close to Coleipa's (16.33)
    
    The classifier is then trained to recognize these patterns and identify leak situations in new data.
    """)


def mostrar_pagina_mapa_calor(detector):
    """IVI heat maps page"""
    st.header("🔥 IVI Heat Maps")
    st.markdown("Risk analysis for different combinations of flow and pressure, considering different IVI values")
    
    # Heat map configuration
    st.subheader("Configuration")
    resolucao = st.slider("Map resolution", 10, 50, 30, 5, 
                         help="Higher values generate more detailed maps, but increase processing time")
    
    # Button to generate heat maps
    if st.button("Generate Heat Maps"):
        with st.spinner("Generating IVI heat maps... This may take a few seconds."):
            fig, ivi_valores = detector.gerar_mapa_calor_ivi(resolucao)
            st.pyplot(fig)
    
    # Detailed IVI analysis
    st.markdown("---")
    st.subheader("Detailed IVI Analysis - Coleipa System")
    
    st.markdown(f"""
    ##### 🔍 Calculated IVI: {detector.caracteristicas_sistema['ivi']:.2f}
    ##### 📊 Classification: Category D (Very Bad)
    ##### ⚠️ Interpretation: IVI > 16 indicates extremely inefficient use of resources
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Comparison with other categories:")
        st.markdown("""
        - 🟢 **Category A (IVI 1-4)**: Efficient system, losses close to unavoidable
        - 🟡 **Category B (IVI 4-8)**: Regular system, improvements recommended
        - 🟠 **Category C (IVI 8-16)**: Bad system, urgent actions needed
        - 🔴 **Category D (IVI >16)**: Very bad system, immediate intervention
        """)
    
    with col2:
        st.markdown("##### 🎯 Specific Coleipa analysis (IVI = 16.33):")
        st.markdown(f"""
        - Real losses are {detector.caracteristicas_sistema['ivi']:.2f} times higher than unavoidable
        - Loss reduction potential > 400 L/connection.day
        - Location on the map: red zone (high risk)
        - Critical combination: HIGH Flow + LOW Pressure
        - Maximum priority: immediate leak detection and repair
        """)
    
    st.markdown("##### 🔧 Visual impact on maps:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**GOOD IVI (2.0):**  \nPredominantly green (low risk)")
    
    with col2:
        st.markdown("**REGULAR IVI (6.0):**  \nGreen-yellow (moderate risk)")
    
    with col3:
        st.markdown("**BAD IVI (12.0):**  \nYellow-orange (high risk)")
    
    with col4:
        st.markdown(f"**VERY BAD IVI ({detector.caracteristicas_sistema['ivi']:.2f}):**  \nIntense red (critical risk)")


def mostrar_pagina_simulacao(detector):
    """Temporal simulation page"""
    st.header("⏱️ Temporal Simulation")
    st.markdown("Time series simulation with leak detection")
    
    # Check if Bayes model is trained
    if detector.modelo_bayes is None:
        st.warning("Bayesian model is not trained. Training model with default parameters...")
        with st.spinner("Training model..."):
            X, y, _ = detector.gerar_dados_baseados_coleipa()
            detector.treinar_modelo_bayesiano(X, y)
    
    # Button to run simulation
    if st.button("Run Simulation"):
        with st.spinner("Simulating time series... This may take a few seconds."):
            fig, df = detector.simular_serie_temporal_coleipa()
            st.pyplot(fig)
            
            # Show simulation data
            with st.expander("View simulation data"):
                # Format time column for display
                df_display = df.copy()
                df_display['Tempo'] = df_display['Tempo'].dt.strftime('%d/%m %H:%M')
                
                # Select relevant columns
                if 'Prob_Hibrida' in df.columns:
                    df_display = df_display[['Tempo', 'Vazao', 'Pressao', 'IVI', 'Vazamento_Real', 'Prob_Hibrida']]
                else:
                    df_display = df_display[['Tempo', 'Vazao', 'Pressao', 'IVI', 'Vazamento_Real']]
                
                st.dataframe(df_display)
    
    # Simulation explanation
    st.markdown("---")
    st.subheader("About the Temporal Simulation")
    st.markdown("""
    The temporal simulation represents the system's behavior over 3 complete days, with a simulated leak
    starting on the second day at 2 PM. Simulation characteristics:
    
    #### Normal Behavior
    - Flow and pressure follow the patterns observed in the Coleipa system
    - Small random variations are added to simulate natural fluctuations
    - Daily cyclic behavior with consumption peaks during the day and valleys during the night
    
    #### Simulated Leak
    - Starts on the second day at 2 PM
    - Gradual progression over several hours (simulating growing leak)
    - Causes simultaneous increase in flow and decrease in pressure
    
    #### Detection System
    - Fuzzy Component: Evaluates risk based on defined rules
    - Bayes Component: Calculates probability based on learned data
    - Hybrid System: Combines both approaches (60% fuzzy + 40% bayes)
    - Detection threshold: Probability > 0.5 indicates leak
    """)
    
    # Leak animation (optional, using HTML code)
    with st.expander("Conceptual Leak Visualization"):
        st.markdown("""
        <div style="width:100%;height:200px;background:linear-gradient(90deg, #3498db 0%, #2980b9 100%);border-radius:10px;position:relative;overflow:hidden;">
            <div style="position:absolute;width:30px;height:30px;background:#e74c3c;border-radius:50%;top:50%;left:70%;transform:translate(-50%,-50%);box-shadow:0 0 20px #e74c3c;">
                <div style="position:absolute;width:40px;height:40px;border:2px solid #e74c3c;border-radius:50%;top:50%;left:50%;transform:translate(-50%,-50%);animation:pulse 1.5s infinite;"></div>
            </div>
            <div style="position:absolute;width:100%;bottom:0;font-family:sans-serif;color:white;text-align:center;padding:10px;">
                Conceptual representation of network leak
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
    """Case analysis page"""
    st.header("🔬 Specific Case Analysis")
    st.markdown("Analyze a specific operation case based on the provided parameters")
    
    # Check if Bayes model is trained
    if detector.modelo_bayes is None:
        st.warning("Bayesian model is not trained. Some results will be limited to fuzzy analysis only.")
        usar_bayes = st.checkbox("Train Bayesian model now", value=True)
        if usar_bayes:
            with st.spinner("Training model..."):
                X, y, _ = detector.gerar_dados_baseados_coleipa()
                detector.treinar_modelo_bayesiano(X, y)
    
    # Form for data input
    st.subheader("System Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vazao = st.number_input("Flow (m³/h)", min_value=7.0, max_value=16.0, value=14.5, step=0.1,
                              help="Typical value for Coleipa: 14.5 m³/h")
    
    with col2:
        pressao = st.number_input("Pressure (mca)", min_value=0.0, max_value=10.0, value=3.5, step=0.1,
                                help="Typical value for Coleipa: 3.5 mca")
    
    with col3:
        ivi = st.number_input("IVI", min_value=1.0, max_value=25.0, value=detector.caracteristicas_sistema['ivi'], step=0.01,
                            help="Coleipa IVI: 16.33 (Category D)")
    
    # Button to run analysis
    if st.button("Analyze Case"):
        with st.spinner("Analyzing case..."):
            resultado = detector.analisar_caso_coleipa(vazao, pressao, ivi)
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification")
                st.markdown(f"""
                - **Flow**: {resultado['vazao']:.1f} m³/h → {resultado['classe_vazao']}
                - **Pressure**: {resultado['pressao']:.1f} mca → {resultado['classe_pressao']}
                - **IVI**: {resultado['ivi']:.2f} → {resultado['classe_ivi']}
                """)
                
                # Analysis result
                st.subheader("Analysis Result")
                st.markdown(f"### {resultado['cor']} {resultado['status']}")
            
            with col2:
                st.subheader("Numerical Results")
                
                # Show different analysis components
                if 'prob_bayes' in resultado:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Fuzzy Risk", f"{resultado['risco_fuzzy']:.1f}%")
                    with col_b:
                        st.metric("Bayesian Prob.", f"{resultado['prob_bayes']:.3f}")
                    with col_c:
                        st.metric("Hybrid Prob.", f"{resultado['prob_hibrida']:.3f}")
                else:
                    st.metric("Fuzzy Risk", f"{resultado['risco_fuzzy']:.1f}%")
                    st.info("Bayesian model not available - fuzzy analysis only")
                
                # Comparison with Coleipa
                st.subheader("Comparison with Coleipa System")
                st.markdown(f"""
                - **Real losses**: {resultado['percentual_perdas']:.1f}%
                - **Real IVI**: {resultado['ivi_real']:.2f} (Category D - Very Bad)
                - **Recommended priority**: Leak detection
                """)
    
    # Explanation about case analysis
    st.markdown("---")
    st.subheader("How to interpret the results")
    
    st.markdown("""
    #### Flow Classification
    - **LOW (night)**: Values below 9 m³/h, typical of night periods
    - **NORMAL (transition)**: Values between 9-14 m³/h, compatible with normal operation
    - **HIGH (peak/leak)**: Values above 14 m³/h, indicate consumption peak or leak
    
    #### Pressure Classification
    - **LOW (problem)**: Values below 5 mca, indicate network problems
    - **MEDIUM (operational)**: Values between 5-8 mca, within the observed operational range
    - **HIGH (good)**: Values above 8 mca, close to NBR recommendation
    
    #### IVI Classification
    - **GOOD (Category A)**: IVI between 1-4, efficient system
    - **REGULAR (Category B)**: IVI between 4-8, regular system
    - **BAD (Category C)**: IVI between 8-16, bad system
    - **VERY BAD (Category D)**: IVI above 16, very bad system
    
    #### Status Interpretation
    - 🟢 **NORMAL OPERATION**: Low probability of leaks
    - 🟡 **HIGH RISK - MONITOR**: Attention situation, monitoring recommended
    - 🔴 **LEAK DETECTED**: High probability of leak, intervention needed
    """)


def mostrar_pagina_relatorio(detector):
    """Complete report page"""
    st.header("📝 Complete Report")
    st.markdown("Detailed report based on Coleipa system data")
    
    # Button to generate report
    if st.button("Generate Complete Report"):
        with st.spinner("Generating report..."):
            relatorio = detector.gerar_relatorio_coleipa()
            
            # Report header
            st.markdown("---")
            st.subheader("ANALYSIS REPORT - COLEIPA SYSTEM")
            st.markdown("---")
            
            # 1. System Characteristics
            st.subheader("1. SYSTEM CHARACTERISTICS")
            st.markdown(f"""
            - **Location**: {relatorio['caracteristicas']['localizacao']}
            - **Territorial area**: {relatorio['caracteristicas']['area']:.1f} km²
            - **Population served**: {relatorio['caracteristicas']['populacao']} inhabitants
            - **Number of connections**: {relatorio['caracteristicas']['ligacoes']}
            - **Network length**: {relatorio['caracteristicas']['rede']} km
            - **Branch density**: {relatorio['caracteristicas']['densidade_ramais']} branches/km
            """)
            
            # 2. Monitoring Results
            st.subheader("2. MONITORING RESULTS (72 hours)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average demanded volume", f"{relatorio['monitoramento']['volume_demandado']:.1f} m³/day")
                st.metric("Average consumed volume", f"{relatorio['monitoramento']['volume_consumido']:.1f} m³/day")
            
            with col2:
                st.metric("Average real losses", f"{relatorio['monitoramento']['perdas_reais']:.1f} m³/day")
                st.metric("Loss percentage", f"{relatorio['monitoramento']['percentual_perdas']:.1f}%")
            
            # 3. Performance Indicators
            st.subheader("3. PERFORMANCE INDICATORS")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("IPRL", f"{relatorio['indicadores']['iprl']} m³/conn.day", "Real Losses per Connection")
            
            with col2:
                st.metric("IPRI", f"{relatorio['indicadores']['ipri']} m³/conn.day", "Unavoidable Real Losses")
            
            with col3:
                st.metric("IVI", f"{relatorio['indicadores']['ivi']}", "Infrastructure Leakage Index")
            
            # 4. Classification
            st.subheader("4. CLASSIFICATION (World Bank)")
            st.markdown(f"""
            - **Category**: {relatorio['classificacao']['categoria']}
            - **Interpretation**: {relatorio['classificacao']['interpretacao']}
            - **Recommendation**: {relatorio['classificacao']['recomendacao']}
            """)
            
           # 5. NPR Methodology - Action Prioritization
            st.subheader("5. NPR METHODOLOGY - ACTION PRIORITIZATION")
            
            # Create priorities table
            df_prioridades = pd.DataFrame(relatorio['prioridades'])
            df_prioridades.columns = ["Order", "Action", "Result"]
            
            # Bar chart for priorities
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(
                [p['acao'] for p in relatorio['prioridades']], 
                [p['resultado'] for p in relatorio['prioridades']],
                color=['#3498db', '#2980b9', '#1f618d', '#154360']
            )
            ax.set_xlabel('NPR Result')
            ax.set_title('Action Prioritization (NPR Methodology)')
            
            # Add values on bars
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 1
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width}', 
                       va='center', fontweight='bold')
            
            st.pyplot(fig)
            st.dataframe(df_prioridades)
            
            # 6. Identified Problems
            st.subheader("6. IDENTIFIED PROBLEMS")
            for i, problema in enumerate(relatorio['problemas'], 1):
                st.markdown(f"- {problema}")
            
            # 7. Recommendations
            st.subheader("7. RECOMMENDATIONS")
            for i, recomendacao in enumerate(relatorio['recomendacoes'], 1):
                st.markdown(f"- **Recommendation {i}**: {recomendacao}")
            
            # 8. Economic Impact Analysis
            st.subheader("8. ECONOMIC IMPACT ANALYSIS")
            
            # Estimate annual water loss
            perda_anual_m3 = relatorio['monitoramento']['perdas_reais'] * 365  # m³/year
            
            # Reference values for costs
            custo_agua_tratada = 1.50  # R$/m³ (average value for treated water)
            custo_energia = 0.80  # R$/m³ (energy cost for pumping)
            custo_manutencao = 0.50  # R$/m³ (maintenance cost related to losses)
            
            # Cost calculation
            custo_anual_agua = perda_anual_m3 * custo_agua_tratada
            custo_anual_energia = perda_anual_m3 * custo_energia
            custo_anual_manutencao = perda_anual_m3 * custo_manutencao
            custo_anual_total = custo_anual_agua + custo_anual_energia + custo_anual_manutencao
            
            # Estimated savings with IVI reduction
            ivi_atual = relatorio['indicadores']['ivi']
            ivi_alvo = 8.0  # Target: reduction to Category B
            reducao_percentual = max(0, (ivi_atual - ivi_alvo) / ivi_atual * 100)
            economia_potencial = custo_anual_total * (reducao_percentual / 100)
            
            # Display economic results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Estimated annual loss", f"{perda_anual_m3:.0f} m³/year")
                st.metric("Annual cost of treated water", f"R$ {custo_anual_agua:.2f}")
                st.metric("Annual energy cost", f"R$ {custo_anual_energia:.2f}")
                st.metric("Annual maintenance cost", f"R$ {custo_anual_manutencao:.2f}")
            
            with col2:
                st.metric("Total annual cost", f"R$ {custo_anual_total:.2f}")
                st.metric("IVI reduction target", f"{ivi_atual:.2f} → {ivi_alvo:.2f} ({reducao_percentual:.1f}%)")
                st.metric("Potential annual savings", f"R$ {economia_potencial:.2f}")
                payback_anos = 100000 / economia_potencial if economia_potencial > 0 else float('inf')
                st.metric("Estimated payback (R$ 100,000 investment)", f"{payback_anos:.1f} years")
            
            # Cost composition chart
            fig_custos, ax_custos = plt.subplots(figsize=(10, 6))
            custos = [custo_anual_agua, custo_anual_energia, custo_anual_manutencao]
            labels = ['Treated Water', 'Energy', 'Maintenance']
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            
            ax_custos.pie(custos, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                         wedgeprops=dict(width=0.5, edgecolor='w'))
            ax_custos.axis('equal')
            ax_custos.set_title('Composition of Costs Related to Losses')
            
            st.pyplot(fig_custos)
            
            # 9. Action Plan
            st.subheader("9. ACTION PLAN")
            
            # Action plan table
            plano_acao = [
                {
                    "Etapa": "Short Term (0-6 months)",
                    "Ação": "Search for non-visible leaks in the network",
                    "Custo Estimado": "R$ 25,000.00",
                    "Impacto Esperado": "20% reduction in losses"
                },
                {
                    "Etapa": "Short Term (0-6 months)",
                    "Ação": "Improve repair time for visible leaks",
                    "Custo Estimado": "R$ 10,000.00",
                    "Impacto Esperado": "5% reduction in losses"
                },
                {
                    "Etapa": "Medium Term (6-18 months)",
                    "Ação": "Installation of PRVs at critical points",
                    "Custo Estimado": "R$ 40,000.00",
                    "Impacto Esperado": "15% reduction in losses"
                },
                {
                    "Etapa": "Medium Term (6-18 months)",
                    "Ação": "Sectorization of the distribution network",
                    "Custo Estimado": "R$ 60,000.00",
                    "Impacto Esperado": "20% reduction in losses"
                },
                {
                    "Etapa": "Long Term (18-36 months)",
                    "Ação": "Replacement of critical network sections",
                    "Custo Estimado": "R$ 120,000.00",
                    "Impacto Esperado": "25% reduction in losses"
                }
            ]
            
            df_plano = pd.DataFrame(plano_acao)
            st.dataframe(df_plano, use_container_width=True)
            
            # Gantt chart for schedule
            fig_gantt, ax_gantt = plt.subplots(figsize=(12, 5))
            
            # Gantt data
            etapas = ['Leak detection', 'Improve repair time', 'Install PRVs', 
                    'Network sectorization', 'Replace critical sections']
            inicio = [0, 0, 6, 8, 18]
            duracao = [6, 3, 6, 10, 18]
            cores = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
            
            # Plot bars
            for i, (etapa, start, dur, cor) in enumerate(zip(etapas, inicio, duracao, cores)):
                ax_gantt.barh(i, dur, left=start, color=cor, alpha=0.8)
                # Add text on bar
                ax_gantt.text(start + dur/2, i, etapa, ha='center', va='center', fontsize=9, fontweight='bold')
            
            # Axis settings
            ax_gantt.set_yticks([])
            ax_gantt.set_xlabel('Months')
            ax_gantt.set_title('Implementation Schedule')
            ax_gantt.grid(axis='x', alpha=0.3)
            ax_gantt.set_axisbelow(True)
            
            # Add time markers
            for i in range(0, 37, 6):
                ax_gantt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
                ax_gantt.text(i, -0.5, f'{i}m', ha='center', va='top')
            
            st.pyplot(fig_gantt)
            
            # 10. Final Considerations
            st.subheader("10. FINAL CONSIDERATIONS")
            st.markdown("""
            The detailed analysis of the Drinking Water Supply System (SAAP) of Coleipa neighborhood reveals
            a critical condition regarding water losses, with classification D (very bad) according to World Bank criteria.
            This condition results in significant waste of water and financial resources.
            
            The implementation of the actions recommended in this report has the potential to:
            
            1. **Reduce IVI** from {:.2f} to values below 8 (Category B)
            2. **Save approximately R$ {:.2f} per year** in operational costs
            3. **Postpone investments** in production system expansion
            4. **Improve pressure and continuity** of supply for users
            
            It is strongly recommended to immediately adopt short-term measures, with special focus on detecting
            non-visible leaks, which constitutes the action with the greatest immediate impact according to the NPR Methodology.
            
            **Important note:** The success of the loss reduction program is directly linked to
            management commitment and allocation of necessary resources for its implementation.
            """.format(relatorio['indicadores']['ivi'], economia_potencial))
            
            # Signature and date
            st.markdown("---")
            data_atual = datetime.now().strftime("%d/%m/%Y")
            st.markdown(f"""
            **Report generated on:** {data_atual}
            
            **Leak Detection System - SAAP Coleipa**  
            *Based on Fuzzy-Bayes hybrid techniques and real monitoring data analysis*
            """)
            
            st.markdown("---")
            st.success("Complete report generated successfully!")


def mostrar_pagina_configuracoes(detector):
    """Settings page"""
    st.header("⚙️ Settings")
    st.markdown("Configure system parameters")
    
    # System characteristics form
    st.subheader("System Characteristics")
    
    # Create two columns for form organization
    col1, col2 = st.columns(2)
    
    with col1:
        area_territorial = st.number_input("Territorial Area (m²)", 
                                           value=detector.caracteristicas_sistema['area_territorial'],
                                           step=1000)
        
        populacao = st.number_input("Population", 
                                    value=detector.caracteristicas_sistema['populacao'],
                                    step=100)
        
        numero_ligacoes = st.number_input("Number of Connections", 
                                          value=detector.caracteristicas_sistema['numero_ligacoes'],
                                          step=10)
        
        comprimento_rede = st.number_input("Network Length (km)", 
                                           value=detector.caracteristicas_sistema['comprimento_rede'],
                                           step=0.1)
        
        densidade_ramais = st.number_input("Branch Density (branches/km)", 
                                           value=detector.caracteristicas_sistema['densidade_ramais'],
                                           step=10)
    
    with col2:
        vazao_media_normal = st.number_input("Normal Average Flow (l/s)", 
                                             value=detector.caracteristicas_sistema['vazao_media_normal'],
                                             step=0.01)
        
        pressao_media_normal = st.number_input("Normal Average Pressure (mca)", 
                                               value=detector.caracteristicas_sistema['pressao_media_normal'],
                                               step=0.01)
        
        perdas_reais_media = st.number_input("Average Real Losses (m³/day)", 
                                             value=detector.caracteristicas_sistema['perdas_reais_media'],
                                             step=0.1)
        
        volume_consumido_medio = st.number_input("Average Consumed Volume (m³/day)", 
                                                 value=detector.caracteristicas_sistema['volume_consumido_medio'],
                                                 step=0.1)
        
        percentual_perdas = st.number_input("Loss Percentage (%)", 
                                             value=detector.caracteristicas_sistema['percentual_perdas'],
                                             step=0.1)
    
    # Button to update characteristics
    if st.button("Update System Characteristics"):
        novas_caracteristicas = {
            'area_territorial': area_territorial,
            'populacao': populacao,
            'numero_ligacoes': numero_ligacoes,
            'comprimento_rede': comprimento_rede,
            'densidade_ramais': densidade_ramais,
            'vazao_media_normal': vazao_media_normal,
            'pressao_media_normal': pressao_media_normal,
            'perdas_reais_media': perdas_reais_media,
            'volume_consumido_medio': volume_consumido_medio,
            'percentual_perdas': percentual_perdas
        }
        
        detector.atualizar_caracteristicas_sistema(novas_caracteristicas)
        st.success("System characteristics updated successfully!")
    
    # IVI calculation
    st.markdown("---")
    st.subheader("Automatic IVI Calculation")
    st.markdown("Calculate IVI based on current data and parameters")
    
    if st.button("Calculate IVI"):
        with st.spinner("Calculating IVI..."):
            ivi, resultados = detector.calcular_ivi_automatico()
            
            st.success(f"IVI calculated successfully: {ivi:.2f}")
            
            # Display detailed results
            st.subheader("Calculation Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Minimum Night Flow", f"{resultados['vazao_minima_noturna']:.2f} m³/h")
                st.metric("Average Pressure", f"{resultados['pressao_media']:.2f} mca")
                st.metric("Real Losses", f"{resultados['perdas_reais']:.2f} m³/day")
                
            with col2:
                st.metric("UARL", f"{resultados['uarl_m3_dia']:.2f} m³/day", "Unavoidable Annual Real Losses")
                st.metric("IPRL", f"{resultados['iprl']:.3f} m³/conn.day", "Real Losses per Connection")
                st.metric("IPRI", f"{resultados['ipri']:.3f} m³/conn.day", "Unavoidable Real Losses per Connection")
    
    # Advanced options
    st.markdown("---")
    st.subheader("Advanced Options")
    
    # Fuzzy system configuration
    st.markdown("##### Fuzzy System Configuration")
    
    with st.expander("Configure Fuzzy System Parameters"):
        # Flow parameters
        st.markdown("**Flow Rate Parameters (m³/h)**")
        vazao_baixa = st.slider("LOW Flow", 5.0, 10.0, 
                               (detector.param_vazao['BAIXA']['range'][0], 
                                detector.param_vazao['BAIXA']['range'][2]),
                               0.1)
        
        vazao_normal = st.slider("NORMAL Flow", 8.0, 15.0, 
                                (detector.param_vazao['NORMAL']['range'][0],
                                 detector.param_vazao['NORMAL']['range'][2]),
                                0.1)
        
        vazao_alta = st.slider("HIGH Flow", 12.0, 18.0, 
                              (detector.param_vazao['ALTA']['range'][0],
                               detector.param_vazao['ALTA']['range'][2]),
                              0.1)
        
        # Pressure parameters
        st.markdown("**Pressure Parameters (mca)**")
        pressao_baixa = st.slider("LOW Pressure", 0.0, 6.0, 
                                 (detector.param_pressao['BAIXA']['range'][0],
                                  detector.param_pressao['BAIXA']['range'][2]),
                                 0.1)
        
        pressao_media = st.slider("MEDIUM Pressure", 3.0, 9.0, 
                                 (detector.param_pressao['MEDIA']['range'][0],
                                  detector.param_pressao['MEDIA']['range'][2]),
                                 0.1)
        
        pressao_alta = st.slider("HIGH Pressure", 6.0, 12.0, 
                                (detector.param_pressao['ALTA']['range'][0],
                                 detector.param_pressao['ALTA']['range'][2]),
                                0.1)
        
        if st.button("Update Fuzzy Parameters"):
            # Update fuzzy parameters
            detector.param_vazao = {
                'BAIXA': {'range': [vazao_baixa[0], (vazao_baixa[0] + vazao_baixa[1]) / 2, vazao_baixa[1]]},
                'NORMAL': {'range': [vazao_normal[0], (vazao_normal[0] + vazao_normal[1]) / 2, vazao_normal[1]]},
                'ALTA': {'range': [vazao_alta[0], (vazao_alta[0] + vazao_alta[1]) / 2, vazao_alta[1]]}
            }
            
            detector.param_pressao = {
                'BAIXA': {'range': [pressao_baixa[0], (pressao_baixa[0] + pressao_baixa[1]) / 2, pressao_baixa[1]]},
                'MEDIA': {'range': [pressao_media[0], (pressao_media[0] + pressao_media[1]) / 2, pressao_media[1]]},
                'ALTA': {'range': [pressao_alta[0], (pressao_alta[0] + pressao_alta[1]) / 2, pressao_alta[1]]}
            }
            
            # Reset fuzzy system to force recreation with new parameters
            detector.sistema_fuzzy = None
            
            st.success("Fuzzy parameters updated successfully!")
    
    # Reset system to default values
    st.markdown("##### Reset System")
    if st.button("Reset System to Default Values", type="primary", use_container_width=True):
        # Create a new detector with default values
        st.session_state['detector'] = DetectorVazamentosColeipa()
        st.success("System reset to default values!")
        st.info("Refresh the page to see the changes.")


# Run the application
if __name__ == "__main__":
    app_main()