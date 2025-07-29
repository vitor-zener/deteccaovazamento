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

# Configuração do Matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class DetectorVazamentosColeipa:
    """
    Sistema híbrido Fuzzy-Bayes para detecção de vazamentos baseado em dados do 
    Sistema de Abastecimento de Água Potável (SAAP) de Coleipa
    """
    
    def __init__(self, arquivo_dados=None):
        """
        Inicializa o sistema baseado nos dados do artigo de Coleipa ou carrega de arquivo
        
        Parâmetros:
        arquivo_dados (str): Caminho para arquivo Excel ou CSV contendo dados de monitoramento
        """
        # Características padrão do sistema baseadas no cálculo das imagens
        self.caracteristicas_sistema = {
            'area_territorial': 319000,  # m² (int)
            'populacao': 1200,  # habitantes (int)
            'numero_ligacoes': 300,  # ligações (int)
            'comprimento_rede': 3.0,  # km (float)
            'densidade_ramais': 100,  # ramais/km (int)
            'vazao_media_normal': 3.17,  # l/s (float)
            'pressao_media_normal': 5.22,  # mca (float)
            'perdas_reais_media': 102.87,  # m³/dia (float) - 37547.55/365
            'volume_consumido_medio': 128.29,  # m³/dia (float)
            'percentual_perdas': 44.50,  # % (float)
            'iprl': 0.343,  # m³/ligação.dia (float) - conforme imagem
            'ipri': 0.021,  # m³/ligação.dia (float) - conforme imagem
            'ivi': 16.33,  # Índice de Vazamentos da Infraestrutura (float) - resultado correto
            # Parâmetros para cálculo de IVI (parametrizáveis) - Nova fórmula
            'volume_perdido_anual': 37547.55,  # Vp - Volume perdido anual (m³/ano)
            'distancia_lote_medidor': 0.001,  # Lp - Distância entre limite do lote e medidor (km)
            'pressao_operacao_adequada': 20.0,  # P - Pressão média de operação adequada (mca)
            'coeficiente_rede': 18.0,  # Coeficiente para comprimento da rede (fixo: 18)
            'coeficiente_ligacoes': 0.8,  # Coeficiente para número de ligações (fixo: 0.8)
            'coeficiente_ramais': 25.0  # Coeficiente para distância dos ramais (fixo: 25)
        }
        
        # Garantir compatibilidade com versões anteriores
        self._garantir_parametros_ivi()
        
        # Dados padrão codificados (usados apenas se nenhum arquivo for fornecido)
        self.dados_coleipa_padrao = {
            'hora': list(range(1, 25)),
            'vazao_dia1': [8.11, 7.83, 7.76, 7.80, 8.08, 9.69, 11.52, 11.92, 13.08, 14.22, 15.68, 14.55, 14.78, 13.16, 12.81, 11.64, 13.02, 13.40, 13.55, 12.94, 12.63, 11.45, 9.88, 8.30],
            'pressao_dia1': [6.89, 7.25, 7.12, 6.51, 6.42, 6.18, 4.83, 3.57, 4.67, 3.92, 3.70, 3.11, 2.68, 2.55, 3.34, 3.77, 4.70, 4.66, 4.69, 3.77, 4.78, 5.73, 6.24, 6.36],
            'vazao_dia2': [7.42, 7.23, 7.20, 7.28, 7.53, 8.99, 10.50, 12.18, 13.01, 13.22, 14.27, 13.63, 13.21, 12.97, 12.36, 11.98, 13.50, 13.27, 14.85, 12.84, 11.29, 10.79, 9.55, 8.60],
            'pressao_dia2': [6.65, 7.41, 7.45, 7.46, 7.38, 6.85, 5.47, 5.69, 4.05, 3.99, 4.56, 4.25, 2.87, 3.37, 4.86, 5.99, 6.39, 6.16, 5.16, 3.86, 3.96, 5.49, 6.48, 6.62],
            'vazao_dia3': [7.67, 7.46, 7.76, 8.18, 8.44, 9.24, 11.05, 12.55, 13.65, 13.63, 14.98, 14.09, 14.00, 13.06, 12.58, 11.81, 13.40, 14.29, 14.06, 12.69, 12.12, 10.83, 8.93, 8.45],
            'pressao_dia3': [6.91, 7.26, 7.12, 7.30, 7.16, 7.05, 6.43, 3.96, 4.70, 3.77, 3.97, 4.06, 4.13, 3.78, 3.12, 3.34, 5.55, 5.41, 4.93, 3.81, 4.37, 5.61, 6.36, 6.49]
        }
        
        # Tentar carregar dados do arquivo se fornecido
        if arquivo_dados:
            self.dados_coleipa = self.carregar_dados_arquivo(arquivo_dados)
            st.success("Dados carregados do arquivo")
        else:
            self.dados_coleipa = self.dados_coleipa_padrao
            st.info("Usando dados padrão de Coleipa (nenhum arquivo fornecido)")
        
        # Definição dos parâmetros fuzzy baseados nos dados reais de Coleipa
        # Vazão em m³/h (convertida de l/s)
        self.param_vazao = {
            'BAIXA': {'faixa': [7, 9, 11]},     # Vazões noturnas
            'NORMAL': {'faixa': [9, 11.5, 14]},  # Vazões de transição
            'ALTA': {'faixa': [12, 15, 16]}     # Vazões de pico
        }
        
        # Pressão em mca (dados originais do artigo)
        self.param_pressao = {
            'BAIXA': {'faixa': [0, 3, 5]},      # Abaixo do mínimo NBR (10 mca)
            'MEDIA': {'faixa': [4, 6, 8]},      # Faixa operacional observada
            'ALTA': {'faixa': [6, 8, 10]}      # Máximos observados
        }
        
        # IVI baseado na classificação do Banco Mundial
        self.param_ivi = {
            'BOM': {'faixa': [1, 2, 4]},        # Categoria A
            'REGULAR': {'faixa': [4, 6, 8]},    # Categoria B
            'RUIM': {'faixa': [8, 12, 16]},     # Categoria C
            'MUITO_RUIM': {'faixa': [16, 20, 25]}  # Categoria D (Coleipa = 16.33)
        }
        
        # Risco de vazamento
        self.param_risco = {
            'MUITO_BAIXO': {'faixa': [0, 10, 25]},
            'BAIXO': {'faixa': [15, 30, 45]},
            'MEDIO': {'faixa': [35, 50, 65]},
            'ALTO': {'faixa': [55, 70, 85]},
            'MUITO_ALTO': {'faixa': [75, 90, 100]}
        }
        
        # Inicialização dos componentes
        self.sistema_fuzzy = None
        self.modelo_bayes = None
    
    def classificar_ivi_seguro(self, ivi_valor):
        """
        Método seguro para classificar IVI que funciona mesmo com problemas de cache
        """
        try:
            ivi_valor = float(ivi_valor)
        except (ValueError, TypeError):
            ivi_valor = 16.33
            
        if ivi_valor <= 4:
            return {
                'categoria': 'A - Eficiente',
                'categoria_simples': 'BOM',
                'cor': '🟢',
                'cor_streamlit': 'success',
                'interpretacao': 'Sistema eficiente com perdas próximas às inevitáveis',
                'recomendacao': 'Manter práticas atuais de gestão'
            }
        elif ivi_valor <= 8:
            return {
                'categoria': 'B - Regular',
                'categoria_simples': 'REGULAR',
                'cor': '🟡',
                'cor_streamlit': 'info',
                'interpretacao': 'Sistema regular, melhorias recomendadas',
                'recomendacao': 'Implementar melhorias graduais no sistema'
            }
        elif ivi_valor <= 16:
            return {
                'categoria': 'C - Ruim',
                'categoria_simples': 'RUIM',
                'cor': '🟠',
                'cor_streamlit': 'warning',
                'interpretacao': 'Sistema ruim, ações urgentes necessárias',
                'recomendacao': 'Implementar programa de redução de perdas urgente'
            }
        else:
            return {
                'categoria': 'D - Muito Ruim',
                'categoria_simples': 'MUITO RUIM',
                'cor': '🔴',
                'cor_streamlit': 'error',
                'interpretacao': 'Sistema muito ruim, intervenção imediata necessária',
                'recomendacao': 'Programas de redução de perdas são imperiosos e prioritários'
            }
    
    def classificar_ivi(self, ivi_valor):
        """
        Função utilitária para classificar o IVI de forma consistente
        
        Parâmetros:
        ivi_valor (float): Valor do IVI a ser classificado
        
        Retorna:
        dict: Dicionário com categoria, cor, interpretação e recomendação
        """
        if ivi_valor <= 4:
            return {
                'categoria': 'A - Eficiente',
                'categoria_simples': 'BOM',
                'cor': '🟢',
                'cor_streamlit': 'success',
                'interpretacao': 'Sistema eficiente com perdas próximas às inevitáveis',
                'recomendacao': 'Manter práticas atuais de gestão'
            }
        elif ivi_valor <= 8:
            return {
                'categoria': 'B - Regular',
                'categoria_simples': 'REGULAR',
                'cor': '🟡',
                'cor_streamlit': 'info',
                'interpretacao': 'Sistema regular, melhorias recomendadas',
                'recomendacao': 'Implementar melhorias graduais no sistema'
            }
        elif ivi_valor <= 16:
            return {
                'categoria': 'C - Ruim',
                'categoria_simples': 'RUIM',
                'cor': '🟠',
                'cor_streamlit': 'warning',
                'interpretacao': 'Sistema ruim, ações urgentes necessárias',
                'recomendacao': 'Implementar programa de redução de perdas urgente'
            }
        else:
            return {
                'categoria': 'D - Muito Ruim',
                'categoria_simples': 'MUITO RUIM',
                'cor': '🔴',
                'cor_streamlit': 'error',
                'interpretacao': 'Sistema muito ruim, intervenção imediata necessária',
                'recomendacao': 'Programas de redução de perdas são imperiosos e prioritários'
            }
    
    def exibir_status_ivi_streamlit(self, ivi_valor, prefixo="IVI ATUAL DO SISTEMA"):
        """
        Exibe o status do IVI no Streamlit com formatação consistente
        
        Parâmetros:
        ivi_valor (float): Valor do IVI
        prefixo (str): Texto que precede o valor do IVI
        """
        classificacao = self.classificar_ivi(ivi_valor)
        
        if classificacao['cor_streamlit'] == 'success':
            st.success(f"✅ **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
        elif classificacao['cor_streamlit'] == 'info':
            st.info(f"ℹ️ **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
        elif classificacao['cor_streamlit'] == 'warning':
            st.warning(f"⚠️ **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
        else:  # error
            st.error(f"🚨 **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
        
        return classificacao
    
    def _garantir_parametros_ivi(self):
        """
        Garante que todos os parâmetros necessários para cálculo de IVI existem
        Usado para compatibilidade com versões anteriores
        """
        parametros_padrao_ivi = {
            'volume_perdido_anual': 37547.55,
            'distancia_lote_medidor': 0.001,
            'pressao_operacao_adequada': 20.0,
            'coeficiente_rede': 18.0,      # Novo valor conforme fórmula atualizada
            'coeficiente_ligacoes': 0.8,
            'coeficiente_ramais': 25.0
        }
        
        for chave, valor_padrao in parametros_padrao_ivi.items():
            if chave not in self.caracteristicas_sistema:
                self.caracteristicas_sistema[chave] = valor_padrao
    
    def carregar_dados_arquivo(self, arquivo_uploaded):
        """
        Carrega dados de monitoramento de um arquivo Excel ou CSV do Streamlit
        
        Parâmetros:
        arquivo_uploaded: Arquivo carregado pelo Streamlit
        
        Retorna:
        dict: Dicionário com os dados carregados
        """
        try:
            # Determinar o tipo de arquivo pela extensão
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
                return self.dados_coleipa_padrao
            
            # Validar estrutura dos dados
            # O arquivo deve ter colunas: hora, vazao_dia1, pressao_dia1, etc.
            colunas_necessarias = ['hora', 'vazao_dia1', 'pressao_dia1', 'vazao_dia2', 
                                  'pressao_dia2', 'vazao_dia3', 'pressao_dia3']
            
            for coluna in colunas_necessarias:
                if coluna not in df.columns:
                    st.warning(f"Coluna '{coluna}' não encontrada no arquivo. Verifique o formato dos dados.")
            
            # Converter DataFrame para dicionário
            dados = {}
            for coluna in df.columns:
                dados[coluna] = df[coluna].tolist()
            
            # Verificar tamanho dos dados
            if len(dados.get('hora', [])) != 24:
                st.warning(f"O número de horas no arquivo ({len(dados.get('hora', []))}) é diferente do esperado (24).")
            
            # Resetar sistema fuzzy para forçar recriação com novos dados
            self.sistema_fuzzy = None
            
            return dados
            
        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")
            st.info("Usando dados padrão de Coleipa como alternativa")
            return self.dados_coleipa_padrao
    
    def gerar_dados_modelo(self):
        """
        Gera dados padrão para download como modelo
        """
        df = pd.DataFrame(self.dados_coleipa_padrao)
        return df
    
    def criar_dataframe_coleipa(self):
        """Cria DataFrame com dados reais de monitoramento de Coleipa"""
        df = pd.DataFrame()
        
        # Calcular médias e desvios padrão
        for hora in range(1, 25):
            idx = hora - 1
            vazao_valores = []
            pressao_valores = []
            
            # Verificar se temos dados para esta hora
            if idx < len(self.dados_coleipa.get('hora', [])):
                for dia in ['dia1', 'dia2', 'dia3']:
                    col_vazao = f'vazao_{dia}'
                    col_pressao = f'pressao_{dia}'
                    
                    if col_vazao in self.dados_coleipa and idx < len(self.dados_coleipa[col_vazao]):
                        vazao_valores.append(self.dados_coleipa[col_vazao][idx])
                    
                    if col_pressao in self.dados_coleipa and idx < len(self.dados_coleipa[col_pressao]):
                        pressao_valores.append(self.dados_coleipa[col_pressao][idx])
            
            # Se não temos dados suficientes, pular esta hora
            if len(vazao_valores) == 0 or len(pressao_valores) == 0:
                continue
                
            df = pd.concat([df, pd.DataFrame({
                'Hora': [hora],
                'Vazao_Dia1': [vazao_valores[0] if len(vazao_valores) > 0 else None],
                'Vazao_Dia2': [vazao_valores[1] if len(vazao_valores) > 1 else None],
                'Vazao_Dia3': [vazao_valores[2] if len(vazao_valores) > 2 else None],
                'Vazao_Media': [np.mean(vazao_valores)],
                'Vazao_DP': [np.std(vazao_valores) if len(vazao_valores) > 1 else 0],
                'Pressao_Dia1': [pressao_valores[0] if len(pressao_valores) > 0 else None],
                'Pressao_Dia2': [pressao_valores[1] if len(pressao_valores) > 1 else None],
                'Pressao_Dia3': [pressao_valores[2] if len(pressao_valores) > 2 else None],
                'Pressao_Media': [np.mean(pressao_valores)],
                'Pressao_DP': [np.std(pressao_valores) if len(pressao_valores) > 1 else 0],
                'IVI': [self.caracteristicas_sistema['ivi']],
                'Perdas_Detectadas': [1 if np.mean(vazao_valores) > 13 and np.mean(pressao_valores) < 5 else 0]
            })], ignore_index=True)
        
        return df
    
    def criar_sistema_fuzzy(self):
        """Cria sistema fuzzy baseado nos dados de Coleipa"""
        # Definir universos baseados nos dados reais
        vazao = ctrl.Antecedent(np.arange(7, 17, 0.1), 'vazao')
        pressao = ctrl.Antecedent(np.arange(0, 11, 0.1), 'pressao')
        ivi = ctrl.Antecedent(np.arange(1, 26, 0.1), 'ivi')
        risco_vazamento = ctrl.Consequent(np.arange(0, 101, 1), 'risco_vazamento')
        
        # Definir conjuntos fuzzy
        for nome, param in self.param_vazao.items():
            vazao[nome] = fuzz.trimf(vazao.universe, param['faixa'])
        
        for nome, param in self.param_pressao.items():
            pressao[nome] = fuzz.trimf(pressao.universe, param['faixa'])
        
        for nome, param in self.param_ivi.items():
            ivi[nome] = fuzz.trimf(ivi.universe, param['faixa'])
        
        for nome, param in self.param_risco.items():
            risco_vazamento[nome] = fuzz.trimf(risco_vazamento.universe, param['faixa'])
        
        # Regras baseadas na análise de Coleipa e conhecimento especialista
        regras = [
            # Regras para detecção de vazamentos baseadas no padrão de Coleipa
            # Vazão ALTA + pressão BAIXA = forte indicação de vazamento
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['MUITO_RUIM'], risco_vazamento['MUITO_ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['RUIM'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['REGULAR'], risco_vazamento['ALTO']),
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'] & ivi['BOM'], risco_vazamento['MEDIO']),
            
            # Vazão NORMAL + pressão BAIXA = risco moderado
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
            
            # Padrão típico observado em Coleipa durante vazamentos
            ctrl.Rule(vazao['ALTA'] & pressao['BAIXA'], risco_vazamento['ALTO'])
        ]
        
        # Criar sistema de controle
        sistema_ctrl = ctrl.ControlSystem(regras)
        self.sistema_fuzzy = ctrl.ControlSystemSimulation(sistema_ctrl)
        
        return vazao, pressao, ivi, risco_vazamento
    
    def visualizar_conjuntos_fuzzy(self):
        """Visualiza conjuntos fuzzy baseados nos dados de Coleipa"""
        # Criar sistema fuzzy se ainda não existir
        vazao, pressao, ivi, risco_vazamento = self.criar_sistema_fuzzy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Vazão
        axes[0, 0].clear()
        for nome in self.param_vazao.keys():
            axes[0, 0].plot(vazao.universe, vazao[nome].mf, label=nome, linewidth=2)
        axes[0, 0].set_title('Conjuntos Fuzzy - Vazão (baseado em dados de Coleipa)')
        axes[0, 0].set_xlabel('Vazão (m³/h)')
        axes[0, 0].set_ylabel('Grau de Pertinência')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pressão
        axes[0, 1].clear()
        for nome in self.param_pressao.keys():
            axes[0, 1].plot(pressao.universe, pressao[nome].mf, label=nome, linewidth=2)
        axes[0, 1].set_title('Conjuntos Fuzzy - Pressão (baseado em dados de Coleipa)')
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
        axes[1, 0].axvline(x=self.caracteristicas_sistema['ivi'], color='red', linestyle='--', 
                          label=f"Coleipa ({self.caracteristicas_sistema['ivi']:.2f})")
        
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
        """Visualiza dados reais de monitoramento de Coleipa"""
        df = self.criar_dataframe_coleipa()
        
        if df.empty:
            st.error("Não foi possível criar o DataFrame com os dados. Verifique os dados de entrada.")
            return None, {}, df
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Gráfico 1: Vazões dos três dias
        axes[0].plot(df['Hora'], df['Vazao_Dia1'], 'b-o', label='Dia 1', alpha=0.7)
        if 'Vazao_Dia2' in df.columns and not df['Vazao_Dia2'].isna().all():
            axes[0].plot(df['Hora'], df['Vazao_Dia2'], 'r-s', label='Dia 2', alpha=0.7)
        if 'Vazao_Dia3' in df.columns and not df['Vazao_Dia3'].isna().all():
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
        if 'Pressao_Dia2' in df.columns and not df['Pressao_Dia2'].isna().all():
            axes[1].plot(df['Hora'], df['Pressao_Dia2'], 'r-s', label='Dia 2', alpha=0.7)
        if 'Pressao_Dia3' in df.columns and not df['Pressao_Dia3'].isna().all():
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
        
        # Gráfico 3: Relação inversa Vazão vs Pressão
        ax2 = axes[2].twinx()
        linha1 = axes[2].plot(df['Hora'], df['Vazao_Media'], 'b-', linewidth=2, label='Vazão Média')
        linha2 = ax2.plot(df['Hora'], df['Pressao_Media'], 'r-', linewidth=2, label='Pressão Média')
        
        axes[2].set_xlabel('Hora do Dia')
        axes[2].set_ylabel('Vazão (m³/h)', color='b')
        ax2.set_ylabel('Pressão (mca)', color='r')
        axes[2].tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combinar legendas
        linhas = linha1 + linha2
        rotulos = [l.get_label() for l in linhas]
        axes[2].legend(linhas, rotulos, loc='upper left')
        axes[2].set_title('Relação Inversa: Vazão × Pressão (Rede Setorizada)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Retornar a figura e estatísticas
        estatisticas = {
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
            "perc_pressao_baixa": len(df[df['Pressao_Media'] < 10])/len(df)*100
        }
        
        return fig, estatisticas, df
    
    def calcular_ivi_automatico(self, arquivo_uploaded=None):
        """
        Calcula automaticamente o IVI (Índice de Vazamentos da Infraestrutura) 
        usando parâmetros configuráveis do sistema
        
        Parâmetros:
        arquivo_uploaded: Arquivo opcional com dados adicionais para cálculo do IVI
        
        Retorna:
        float: Valor do IVI calculado
        dict: Dicionário com componentes do cálculo (IPRL, IPRI, etc.)
        """
        # Garantir que todos os parâmetros existem
        self._garantir_parametros_ivi()
        
        # Usar parâmetros configuráveis do sistema com valores de fallback
        Vp_anual = self.caracteristicas_sistema.get('volume_perdido_anual', 37547.55)
        Nc = self.caracteristicas_sistema.get('numero_ligacoes', 300)
        Lm = self.caracteristicas_sistema.get('comprimento_rede', 3.0)
        Lp = self.caracteristicas_sistema.get('distancia_lote_medidor', 0.001)
        P = self.caracteristicas_sistema.get('pressao_operacao_adequada', 20.0)
        
        # Coeficientes da fórmula IPRI (parametrizáveis) com valores de fallback
        coef_rede = self.caracteristicas_sistema.get('coeficiente_rede', 8.0)
        coef_ligacoes = self.caracteristicas_sistema.get('coeficiente_ligacoes', 0.8)
        coef_ramais = self.caracteristicas_sistema.get('coeficiente_ramais', 25.0)
        
        # Cálculo do IPRL (Índice de Perdas Reais por Ligação) - Equação 3
        # IPRL = Vp / (Nc × 365)
        iprl = Vp_anual / (Nc * 365) if Nc > 0 else 0  # m³/lig.dia
        
        # Cálculo do IPRI (Índice de Perdas Reais Inevitáveis) - Equação 4 (Nova Fórmula)
        # IPRI = (18 × Lm + 0,8 × Nc + 25 × Lp × Nc) × P / (Nc × 1000)
        numerador_ipri = (coef_rede * Lm + coef_ligacoes * Nc + coef_ramais * Lp * Nc) * P
        denominador_ipri = Nc * 1000
        ipri = numerador_ipri / denominador_ipri if denominador_ipri > 0 else 0  # m³/lig.dia
        
        # Cálculo do IVI (Índice de Vazamentos na Infraestrutura) - Equação 5
        # IVI = IPRL / IPRI
        ivi = iprl / ipri if ipri > 0 else 0
        
        # Calcular perdas reais diárias para compatibilidade
        perdas_reais_diarias = Vp_anual / 365  # m³/dia
        
        # Atualizar características do sistema com tipos corretos
        self.caracteristicas_sistema['perdas_reais_media'] = float(perdas_reais_diarias)
        self.caracteristicas_sistema['iprl'] = float(iprl)
        self.caracteristicas_sistema['ipri'] = float(ipri)
        self.caracteristicas_sistema['ivi'] = float(ivi)
        
        # Resetar sistema fuzzy para refletir o novo IVI
        self.sistema_fuzzy = None
        
        # Preparar resultados detalhados
        resultados = {
            'volume_perdido_anual': Vp_anual,
            'numero_ligacoes': Nc,
            'comprimento_rede': Lm,
            'distancia_lote_medidor': Lp,
            'pressao_operacao': P,
            'coeficiente_rede': coef_rede,
            'coeficiente_ligacoes': coef_ligacoes,
            'coeficiente_ramais': coef_ramais,
            'perdas_reais_diarias': perdas_reais_diarias,
            'iprl': iprl,
            'ipri': ipri,
            'ivi': ivi,
            'calculo_iprl': f"{Vp_anual:.2f} / ({Nc} × 365) = {iprl:.3f} m³/lig.dia",
            'calculo_ipri': f"({coef_rede} × {Lm} + {coef_ligacoes} × {Nc} + {coef_ramais} × {Lp} × {Nc}) × {P} / ({Nc} × 1000) = {ipri:.6f} m³/lig.dia",
            'calculo_ivi': f"{iprl:.3f} / {ipri:.6f} = {ivi:.2f}"
        }
        
        return ivi, resultados
    
    def gerar_dados_baseados_coleipa(self, n_amostras=500):
        """Gera dados sintéticos baseados nas características do sistema Coleipa"""
        df_coleipa = self.criar_dataframe_coleipa()
        
        if df_coleipa.empty:
            # Se não temos dados, usar valores padrão
            vazao_normal_mean, vazao_normal_std = 10.0, 1.5
            pressao_normal_mean, pressao_normal_std = 6.0, 1.0
            vazao_vazamento_mean, vazao_vazamento_std = 14.0, 1.0
            pressao_vazamento_mean, pressao_vazamento_std = 3.5, 0.5
        else:
            # Extrair padrões dos dados reais
            dados_normais = df_coleipa[df_coleipa['Perdas_Detectadas'] == 0]
            dados_vazamento = df_coleipa[df_coleipa['Perdas_Detectadas'] == 1]
            
            if not dados_normais.empty:
                vazao_normal_mean = dados_normais['Vazao_Media'].mean()
                vazao_normal_std = dados_normais['Vazao_Media'].std()
                pressao_normal_mean = dados_normais['Pressao_Media'].mean()
                pressao_normal_std = dados_normais['Pressao_Media'].std()
            else:
                vazao_normal_mean, vazao_normal_std = 10.0, 1.5
                pressao_normal_mean, pressao_normal_std = 6.0, 1.0
            
            if not dados_vazamento.empty:
                vazao_vazamento_mean = dados_vazamento['Vazao_Media'].mean()
                vazao_vazamento_std = dados_vazamento['Vazao_Media'].std()
                pressao_vazamento_mean = dados_vazamento['Pressao_Media'].mean()
                pressao_vazamento_std = dados_vazamento['Pressao_Media'].std()
            else:
                vazao_vazamento_mean, vazao_vazamento_std = 14.0, 1.0
                pressao_vazamento_mean, pressao_vazamento_std = 3.5, 0.5
        
        # Gerar dados sintéticos baseados nos padrões reais
        n_normal = int(0.55 * n_amostras)  # 55% normal (baseado nos dados de Coleipa)
        n_vazamento = n_amostras - n_normal
        
        # Dados normais
        vazao_normal = np.random.normal(vazao_normal_mean, vazao_normal_std, n_normal)
        pressao_normal = np.random.normal(pressao_normal_mean, pressao_normal_std, n_normal)
        ivi_normal = np.random.normal(8, 2, n_normal)  # IVI melhor para operação normal
        
        # Dados de vazamento
        vazao_vazamento = np.random.normal(vazao_vazamento_mean, vazao_vazamento_std, n_vazamento)
        pressao_vazamento = np.random.normal(pressao_vazamento_mean, pressao_vazamento_std, n_vazamento)
        ivi_vazamento = np.random.normal(self.caracteristicas_sistema.get('ivi', 16.33), 3, n_vazamento)  # IVI dinâmico similar ao de Coleipa
        
        # Combinar dados
        X = np.vstack([
            np.column_stack([vazao_normal, pressao_normal, ivi_normal]),
            np.column_stack([vazao_vazamento, pressao_vazamento, ivi_vazamento])
        ])
        
        y = np.hstack([np.zeros(n_normal), np.ones(n_vazamento)])
        
        return X, y, df_coleipa
    
    def treinar_modelo_bayesiano(self, X, y):
        """Treina modelo Bayesiano com dados baseados em Coleipa"""
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
            # Limitar valores às faixas dos dados de Coleipa
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
        """Analisa um caso específico usando padrões de Coleipa"""
        # Usar valores típicos de Coleipa se não fornecidos
        if vazao is None:
            vazao = 14.5  # Vazão típica de pico
        if pressao is None:
            pressao = 3.5   # Pressão típica baixa
        if ivi is None:
            ivi = self.caracteristicas_sistema.get('ivi', 16.33)   # IVI atual calculado dinamicamente
        
        # Classificação baseada nos dados de Coleipa
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
        
        # Usar classificação segura para evitar problemas de cache
        if ivi <= 4:
            classe_ivi = "BOM (Categoria A - Eficiente)"
        elif ivi <= 8:
            classe_ivi = "REGULAR (Categoria B - Regular)"
        elif ivi <= 16:
            classe_ivi = "RUIM (Categoria C - Ruim)"
        else:
            classe_ivi = "MUITO RUIM (Categoria D - Muito Ruim)"
        
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
        
        # Comparação com dados reais de Coleipa
        resultado['percentual_perdas'] = self.caracteristicas_sistema.get('percentual_perdas', 44.50)
        resultado['ivi_real'] = self.caracteristicas_sistema.get('ivi', 16.33)
        
        return resultado
    
    def simular_serie_temporal_coleipa(self):
        """Simula série temporal baseada nos padrões reais de Coleipa"""
        df_real = self.criar_dataframe_coleipa()
        
        # Criar série temporal expandida (3 dias completos)
        tempo = []
        vazao = []
        pressao = []
        
        for dia in range(3):
            for hora in range(24):
                timestamp = datetime(2024, 1, 1 + dia, hora, 0)
                tempo.append(timestamp)
                
                # Usar dados reais de Coleipa com variação
                idx = hora
                if not df_real.empty and idx < len(df_real):
                    v = df_real.iloc[idx]['Vazao_Media'] + np.random.normal(0, 0.1)
                    p = df_real.iloc[idx]['Pressao_Media'] + np.random.normal(0, 0.05)
                else:
                    # Valores padrão se não temos dados
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
            'IVI': [self.caracteristicas_sistema.get('ivi', 16.33)] * len(tempo),  # IVI dinâmico
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
        """Visualiza série temporal baseada em Coleipa"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Gráfico 1: Vazão
        axes[0].plot(df['Tempo'], df['Vazao'], 'b-', linewidth=1.5, label='Vazão')
        axes[0].axvline(x=df['Tempo'].iloc[inicio_vazamento], color='red', linestyle='--', 
                       label=f'Início do Vazamento ({df["Tempo"].iloc[inicio_vazamento].strftime("%d/%m %H:%M")})')
        axes[0].set_ylabel('Vazão (m³/h)')
        axes[0].set_title('Série Temporal - Sistema Coleipa: Vazão')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gráfico 2: Pressão
        axes[1].plot(df['Tempo'], df['Pressao'], 'r-', linewidth=1.5, label='Pressão')
        axes[1].axhline(y=10, color='orange', linestyle=':', label='Mínimo NBR (10 mca)')
        axes[1].axvline(x=df['Tempo'].iloc[inicio_vazamento], color='red', linestyle='--')
        axes[1].set_ylabel('Pressão (mca)')
        axes[1].set_title('Série Temporal - Sistema Coleipa: Pressão')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Gráfico 3: Detecções (se disponível)
        if 'Prob_Hibrida' in df.columns:
            axes[2].plot(df['Tempo'], df['Prob_Hibrida'], 'purple', linewidth=2, label='Detecção Híbrida')
            axes[2].plot(df['Tempo'], df['Risco_Fuzzy'], 'green', alpha=0.7, label='Componente Fuzzy')
            axes[2].plot(df['Tempo'], df['Prob_Bayes'], 'orange', alpha=0.7, label='Componente Bayes')
            axes[2].axhline(y=0.5, color='black', linestyle='-.', label='Limiar de Detecção')
            axes[2].axvline(x=df['Tempo'].iloc[inicio_vazamento], color='red', linestyle='--')
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
        Gera mapas de calor mostrando risco de vazamento para diferentes
        combinações de vazão e pressão, com diferentes valores de IVI baseados
        na classificação do Banco Mundial
        """
        # Verificar se sistema fuzzy está criado
        if self.sistema_fuzzy is None:
            self.criar_sistema_fuzzy()
        
        # IVI atual calculado dinamicamente
        ivi_atual = self.caracteristicas_sistema.get('ivi', 16.33)
        
        # Determinar categoria do IVI atual usando função utilitária segura
        ivi_atual = self.caracteristicas_sistema.get('ivi', 16.33)
        
        # Usar classificação manual para evitar problemas de cache
        if ivi_atual <= 4:
            categoria_atual = "BOM"
            indice_atual = 0
        elif ivi_atual <= 8:
            categoria_atual = "REGULAR"
            indice_atual = 1
        elif ivi_atual <= 16:
            categoria_atual = "RUIM"
            indice_atual = 2
        else:
            categoria_atual = "MUITO RUIM"
            indice_atual = 3
        
        # Valores de IVI baseados na classificação do Banco Mundial
        ivi_valores = [2, 6, 12, 18]  # Valores representativos padrão
        ivi_categorias = ['BOM (2.0)', 'REGULAR (6.0)', 'RUIM (12.0)', 'MUITO RUIM (18.0)']
        ivi_classificacoes = ['Categoria A', 'Categoria B', 'Categoria C', 'Categoria D']
        
        # Substituir o valor na categoria correta pelo IVI atual
        ivi_valores[indice_atual] = ivi_atual
        ivi_categorias[indice_atual] = f'{categoria_atual} ({ivi_atual:.2f})'
        
        # Valores para mapa de calor baseados nos dados de Coleipa
        vazoes = np.linspace(7, 16, resolucao)
        pressoes = np.linspace(2.5, 8, resolucao)
        
        # Configurar figura com subplots 2x2 para mapas + 1 subplot para barra de cores
        fig = plt.figure(figsize=(18, 16))
        
        # Criar grade: 3 linhas, 2 colunas
        # Linha 1: 2 mapas superiores
        # Linha 2: 2 mapas inferiores  
        # Linha 3: barra de cores centralizada
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.15], hspace=0.3, wspace=0.2)
        
        # Criar 4 subplots para os mapas
        axes = [
            fig.add_subplot(gs[0, 0]),  # Superior esquerdo
            fig.add_subplot(gs[0, 1]),  # Superior direito
            fig.add_subplot(gs[1, 0]),  # Inferior esquerdo
            fig.add_subplot(gs[1, 1])   # Inferior direito
        ]
        
        # Subplot para barra de cores (ocupando largura total)
        cbar_ax = fig.add_subplot(gs[2, :])
        
        # Gerar um mapa de calor para cada valor de IVI
        im = None  # Para capturar a última imagem para a barra de cores
        for idx, (ax, ivi_valor, categoria, classificacao) in enumerate(zip(axes, ivi_valores, ivi_categorias, ivi_classificacoes)):
            
            # Criar grade para o mapa
            X, Y = np.meshgrid(vazoes, pressoes)
            Z = np.zeros_like(X)
            
            # Calcular risco para cada ponto da grade
            for ii in range(X.shape[0]):
                for jj in range(X.shape[1]):
                    try:
                        # Garantir que valores estão dentro dos limites
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
                        # Heurística baseada nos padrões de Coleipa
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
                niveis_contorno = [20, 40, 60, 80]
                contornos = ax.contour(X, Y, Z, levels=niveis_contorno, colors='black', alpha=0.4, linewidths=1.5)
                ax.clabel(contornos, inline=True, fontsize=10, fmt='%d%%', 
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
            
            # Rótulos dos conjuntos fuzzy com tamanho reduzido
            ax.text(8, 7.5, 'VAZÃO\nBAIXA', color='navy', fontsize=9, fontweight='bold', 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            ax.text(11.5, 7.5, 'VAZÃO\nNORMAL', color='navy', fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            ax.text(15, 7.5, 'VAZÃO\nALTA', color='navy', fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.9))
            
            # Rótulos de pressão com tamanho reduzido
            ax.text(15.5, 3.5, 'PRESSÃO\nBAIXA', color='darkgreen', fontsize=9, fontweight='bold', 
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            ax.text(15.5, 5.2, 'PRESSÃO\nNORMAL', color='darkgreen', fontsize=9, fontweight='bold',
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            ax.text(15.5, 7, 'PRESSÃO\nALTA', color='darkgreen', fontsize=9, fontweight='bold',
                   ha='center', va='center', rotation=90,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.9))
            
            # Marcar o ponto característico de Coleipa em todos os gráficos
            # Destacar especialmente no mapa que corresponde ao IVI atual
            if idx == indice_atual:  # Mapa correspondente ao IVI atual
                ax.scatter([14.5], [3.5], color='red', s=300, marker='*', 
                          edgecolors='darkred', linewidth=3, label=f'Ponto Coleipa\n(IVI={ivi_atual:.2f})', zorder=10)
                ax.annotate(f'SISTEMA COLEIPA\n(Vazão=14.5, Pressão=3.5)\nIVI={ivi_atual:.2f} - {categoria_atual}', 
                           xy=(14.5, 3.5), xytext=(11, 2.8),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           fontsize=9, fontweight='bold', color='red',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9))
                ax.legend(loc='upper left', fontsize=9)
            else:
                # Marcar o ponto de Coleipa nos outros gráficos também, mas com menos destaque
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
        
        # Adicionar ticks personalizados à barra de cores
        cbar.set_ticks([0, 20, 40, 60, 80, 100])
        cbar.set_ticklabels(['0%\n(Muito Baixo)', '20%\n(Baixo)', '40%\n(Médio)', 
                            '60%\n(Alto)', '80%\n(Muito Alto)', '100%\n(Crítico)'])
        
        # Título principal melhorado com IVI dinâmico e categoria correta
        fig.suptitle(f'Mapas de Risco para diferentes IVIs\nClassificação Banco Mundial - Sistema Coleipa (IVI Atual: {ivi_atual:.2f} - {categoria_atual})', 
                     fontsize=16, fontweight='bold', y=0.96)
        
        return fig, ivi_valores
    
    def gerar_relatorio_coleipa(self):
        """Gera relatório completo baseado nos dados do sistema Coleipa"""
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
        Atualiza características do sistema com novos valores
        
        Parâmetros:
        novas_caracteristicas (dict): Dicionário com novas características
        """
        # Definir tipos esperados para cada característica
        tipos_esperados = {
            'area_territorial': int,
            'populacao': int,
            'numero_ligacoes': int,
            'comprimento_rede': float,
            'densidade_ramais': int,
            'vazao_media_normal': float,
            'pressao_media_normal': float,
            'perdas_reais_media': float,
            'volume_consumido_medio': float,
            'percentual_perdas': float,
            'iprl': float,
            'ipri': float,
            'ivi': float,
            # Novos parâmetros para cálculo de IVI
            'volume_perdido_anual': float,
            'distancia_lote_medidor': float,
            'pressao_operacao_adequada': float,
            'coeficiente_rede': float,
            'coeficiente_ligacoes': float,
            'coeficiente_ramais': float
        }
        
        for chave, valor in novas_caracteristicas.items():
            if chave in self.caracteristicas_sistema:
                # Converter para o tipo correto
                if chave in tipos_esperados:
                    try:
                        valor_convertido = tipos_esperados[chave](valor)
                        self.caracteristicas_sistema[chave] = valor_convertido
                        st.success(f"Característica '{chave}' atualizada para: {valor_convertido}")
                    except (ValueError, TypeError) as e:
                        st.error(f"Erro ao converter '{chave}': {e}")
                else:
                    self.caracteristicas_sistema[chave] = valor
                    st.success(f"Característica '{chave}' atualizada para: {valor}")
            else:
                st.warning(f"Aviso: Característica '{chave}' não existe no sistema")
        
        # Resetar sistema fuzzy para forçar recriação com novos parâmetros
        self.sistema_fuzzy = None


# Configuração da página Streamlit
st.set_page_config(
    page_title="Sistema de Detecção de Vazamentos - Coleipa",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Variável global para armazenar instância do detector
@st.cache_resource
def obter_detector(arquivo_uploaded=None):
    try:
        detector = DetectorVazamentosColeipa(arquivo_uploaded)
        # Garantir compatibilidade com versões anteriores
        detector._garantir_parametros_ivi()
        return detector
    except Exception as e:
        st.error(f"Erro ao inicializar detector: {e}")
        # Retornar detector com dados padrão em caso de erro
        detector_padrao = DetectorVazamentosColeipa()
        detector_padrao._garantir_parametros_ivi()
        return detector_padrao

def obter_detector_seguro(arquivo_uploaded=None):
    """
    Função segura para obter detector, com fallback em caso de problemas de cache
    """
    try:
        detector = obter_detector(arquivo_uploaded)
        # Testar se o detector tem os métodos necessários
        if hasattr(detector, 'classificar_ivi') and hasattr(detector, 'exibir_status_ivi_streamlit'):
            return detector
        else:
            # Se não tem os métodos, limpar cache e recriar
            st.warning("⚠️ Cache desatualizado detectado. Recriando detector...")
            obter_detector.clear()
            detector_novo = DetectorVazamentosColeipa(arquivo_uploaded)
            detector_novo._garantir_parametros_ivi()
            return detector_novo
    except AttributeError as e:
        # Erro específico de atributo faltando
        st.warning(f"⚠️ Problema de cache detectado: {str(e)}. Recriando detector...")
        try:
            obter_detector.clear()
        except:
            pass
        return DetectorVazamentosColeipa(arquivo_uploaded)
    except Exception as e:
        st.error(f"❌ Erro ao obter detector: {e}")
        # Em último caso, criar detector sem cache
        return DetectorVazamentosColeipa(arquivo_uploaded)

# Função para download de arquivos
def botao_download(objeto_para_download, nome_arquivo_download, texto_botao):
    """
    Gera um botão que permite fazer download de um objeto
    """
    if isinstance(objeto_para_download, pd.DataFrame):
        # Se for um DataFrame
        buffer = io.BytesIO()
        
        if nome_arquivo_download.endswith('.csv'):
            objeto_para_download.to_csv(buffer, index=False)
            tipo_mime = "text/csv"
        else:
            objeto_para_download.to_excel(buffer, index=False)
            tipo_mime = "application/vnd.ms-excel"
        
        buffer.seek(0)
        st.download_button(
            label=texto_botao,
            data=buffer,
            file_name=nome_arquivo_download,
            mime=tipo_mime
        )
    else:
        # Se for outro tipo de objeto
        st.warning("Tipo de objeto não suportado para download")


# CORREÇÃO APLICADA: Para resolver problemas de AttributeError relacionados ao cache do Streamlit,
# as funções utilitárias classificar_ivi() e exibir_status_ivi_streamlit() foram mantidas na classe
# mas as chamadas foram substituídas por lógica manual nas funções de interface para evitar
# problemas quando o cache contém versões antigas da classe que não possuem esses métodos.
# Isso garante compatibilidade e estabilidade da aplicação.

def validar_ivi(detector):
    """
    Função para validar e obter IVI do detector de forma segura
    """
    try:
        ivi_atual = detector.caracteristicas_sistema.get('ivi', 16.33)
        return float(ivi_atual)
    except (ValueError, TypeError, AttributeError):
        return 16.33

def classificar_ivi_manual(ivi_valor):
    """
    Função manual para classificar IVI sem depender da classe
    """
    try:
        ivi_valor = float(ivi_valor)
    except (ValueError, TypeError):
        ivi_valor = 16.33
        
    if ivi_valor <= 4:
        return {
            'categoria': 'Categoria A (Eficiente)',
            'cor_st': 'success',
            'emoji': '✅',
            'categoria_simples': 'BOM'
        }
    elif ivi_valor <= 8:
        return {
            'categoria': 'Categoria B (Regular)',
            'cor_st': 'info',
            'emoji': 'ℹ️',
            'categoria_simples': 'REGULAR'
        }
    elif ivi_valor <= 16:
        return {
            'categoria': 'Categoria C (Ruim)',
            'cor_st': 'warning',
            'emoji': '⚠️',
            'categoria_simples': 'RUIM'
        }
    else:
        return {
            'categoria': 'Categoria D (Muito Ruim)',
            'cor_st': 'error',
            'emoji': '🚨',
            'categoria_simples': 'MUITO RUIM'
        }

def exibir_ivi_status(detector, prefixo="IVI ATUAL DO SISTEMA"):
    """
    Função para exibir status do IVI de forma segura
    """
    ivi_atual = validar_ivi(detector)
    classificacao = classificar_ivi_manual(ivi_atual)
    
    if classificacao['cor_st'] == 'success':
        st.success(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    elif classificacao['cor_st'] == 'info':
        st.info(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    elif classificacao['cor_st'] == 'warning':
        st.warning(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    else:
        st.error(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    
    return classificacao

def exibir_ivi_calculado(ivi_valor, prefixo="IVI CALCULADO"):
    """
    Função específica para exibir IVI calculado
    """
    classificacao = classificar_ivi_manual(ivi_valor)
    
    if classificacao['cor_st'] == 'success':
        st.success(f"{classificacao['emoji']} **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
    elif classificacao['cor_st'] == 'info':
        st.info(f"{classificacao['emoji']} **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
    elif classificacao['cor_st'] == 'warning':
        st.warning(f"{classificacao['emoji']} **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
    else:
        st.error(f"{classificacao['emoji']} **{prefixo}: {ivi_valor:.2f}** - {classificacao['categoria']}")
    
    return classificacao
    """
    Função para exibir status do IVI de forma segura
    """
    ivi_atual = validar_ivi(detector)
    classificacao = classificar_ivi_manual(ivi_atual)
    
    if classificacao['cor_st'] == 'success':
        st.success(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    elif classificacao['cor_st'] == 'info':
        st.info(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    elif classificacao['cor_st'] == 'warning':
        st.warning(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    else:
        st.error(f"{classificacao['emoji']} **{prefixo}: {ivi_atual:.2f}** - {classificacao['categoria']}")
    
    return classificacao

def testar_integridade_detector(detector):
    """
    Testa a integridade do detector e corrige problemas automaticamente
    """
    problemas_encontrados = []
    
    # Teste 1: Verificar se o detector tem características básicas
    if not hasattr(detector, 'caracteristicas_sistema'):
        problemas_encontrados.append("Detector não possui características do sistema")
        return False, problemas_encontrados
    
    # Teste 2: Verificar se o IVI é válido
    try:
        ivi = detector.caracteristicas_sistema.get('ivi', None)
        if ivi is None:
            problemas_encontrados.append("IVI não encontrado nas características")
        else:
            ivi_float = float(ivi)
            if ivi_float < 0 or ivi_float > 1000:  # Limite razoável
                problemas_encontrados.append(f"IVI com valor suspeito: {ivi_float}")
    except (ValueError, TypeError):
        problemas_encontrados.append("IVI não é um número válido")
    
    # Teste 3: Verificar se métodos básicos existem
    metodos_essenciais = ['_garantir_parametros_ivi', 'gerar_dados_modelo']
    for metodo in metodos_essenciais:
        if not hasattr(detector, metodo):
            problemas_encontrados.append(f"Método essencial '{metodo}' não encontrado")
    
    # Teste 4: Verificar dados padrão
    if not hasattr(detector, 'dados_coleipa'):
        problemas_encontrados.append("Dados de Coleipa não encontrados")
    
    return len(problemas_encontrados) == 0, problemas_encontrados

def aplicacao_principal():
    """Função principal da aplicação Streamlit"""
    st.title("💧 Sistema de Detecção de Vazamentos - SAAP Coleipa")
    st.markdown("##### Sistema híbrido Fuzzy-Bayes para detecção de vazamentos em redes de abastecimento")
    
    # Barra lateral para navegação
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
    
    # Upload de arquivo na barra lateral
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dados de Entrada")
    arquivo_uploaded = st.sidebar.file_uploader("Carregar dados de monitoramento", type=["xlsx", "csv"])
    
    # Inicializar ou obter detector usando função segura
    detector = obter_detector_seguro(arquivo_uploaded)
    
    # Teste de integridade do detector
    integridade_ok, problemas = testar_integridade_detector(detector)
    
    if not integridade_ok:
        st.sidebar.error("❌ Problemas detectados no sistema")
        with st.sidebar.expander("Ver problemas"):
            for problema in problemas:
                st.sidebar.text(f"• {problema}")
        
        # Tentar recriar detector
        st.sidebar.warning("Tentando recriar detector...")
        try:
            obter_detector.clear()
            detector = DetectorVazamentosColeipa(arquivo_uploaded)
            detector._garantir_parametros_ivi()
            
            # Testar novamente
            integridade_ok2, problemas2 = testar_integridade_detector(detector)
            if integridade_ok2:
                st.sidebar.success("✅ Detector recriado com sucesso")
            else:
                st.sidebar.error("❌ Problemas persistem após recriação")
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao recriar detector: {e}")
    else:
        st.sidebar.success("✅ Sistema funcionando corretamente")
    
    # Modelo para download na barra lateral
    st.sidebar.markdown("---")
    st.sidebar.subheader("Modelo de Dados")
    formato = st.sidebar.radio("Formato:", ["Excel (.xlsx)", "CSV (.csv)"], horizontal=True)
    nome_arquivo = "modelo_dados_coleipa." + ("xlsx" if formato == "Excel (.xlsx)" else "csv")
    df_modelo = detector.gerar_dados_modelo()
    botao_download(df_modelo, nome_arquivo, "⬇️ Baixar Modelo")
    
    # Informações na barra lateral
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sobre o Sistema")
    st.sidebar.info(
        "Sistema baseado em dados reais do SAAP do bairro Coleipa, "
        "Santa Bárbara do Pará - PA. Utiliza lógica fuzzy e modelos Bayesianos "
        "para detecção de vazamentos em redes de abastecimento de água."
    )
    
    # Botão para limpar cache em caso de problemas
    st.sidebar.markdown("---")
    st.sidebar.subheader("Status do Sistema")
    
    # Verificar integridade usando a função de teste
    integridade_ok_sidebar, _ = testar_integridade_detector(detector)
    
    if integridade_ok_sidebar:
        st.sidebar.success("✅ Sistema operacional")
        ivi_atual = validar_ivi(detector)
        classificacao = classificar_ivi_manual(ivi_atual)
        
        status_ivi = f"{classificacao['emoji']} IVI: {classificacao['categoria_simples']}"
        st.sidebar.info(f"{status_ivi} ({ivi_atual:.2f})")
    else:
        st.sidebar.warning("⚠️ Sistema com problemas")
    
    if st.sidebar.button("🔄 Limpar Cache (se houver erros)"):
        try:
            obter_detector.clear()
            st.sidebar.success("Cache limpo! Recarregue a página.")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.warning(f"Erro ao limpar cache: {e}")
            try:
                st.rerun()
            except:
                st.sidebar.info("Recarregue a página manualmente")
    
    # Conteúdo principal baseado na página selecionada
    if pagina_selecionada == "Início":
        mostrar_pagina_inicio(detector)
    
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


def mostrar_pagina_inicio(detector):
    """Página inicial da aplicação"""
    st.header("Bem-vindo ao Sistema de Detecção de Vazamentos")
    
    # Descrição do sistema
    st.markdown("""
    Este sistema utiliza uma abordagem híbrida combinando lógica fuzzy e análise Bayesiana para 
    detectar vazamentos em redes de abastecimento de água baseado em dados de monitoramento.
    """)
    
    # Exibir IVI atual no topo usando função utilitária segura
    exibir_ivi_status(detector, "IVI ATUAL DO SISTEMA")
    
    # Visão geral em 3 colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔍 Análise de Dados")
        st.markdown("""
        - Visualização de dados de monitoramento
        - Análise estatística de vazão e pressão
        - Identificação de padrões críticos
        """)
    
    with col2:
        st.subheader("🧠 Inteligência Híbrida")
        st.markdown("""
        - Sistema fuzzy baseado em conhecimento especialista
        - Modelo Bayesiano para classificação
        - Mapas de calor para análise de risco
        """)
    
    with col3:
        st.subheader("📊 Resultados e Relatórios")
        st.markdown("""
        - Simulação de vazamentos em tempo real
        - Análise de casos específicos
        - Relatórios detalhados com recomendações
        """)
    
    # Sobre o caso Coleipa
    st.markdown("---")
    st.subheader("Sobre o Sistema Coleipa")
    
    col1, col2 = st.columns(2)
    
    # Obter IVI atual para uso em ambas as colunas
    ivi_atual = validar_ivi(detector)
    
    with col1:
        # Obter classificação atual do IVI com função utilitária
        classificacao = classificar_ivi_manual(ivi_atual)
        
        st.markdown(f"""
        O SAAP (Sistema de Abastecimento de Água Potável) do bairro Coleipa, localizado em Santa 
        Bárbara do Pará, apresenta características típicas de sistemas com perdas significativas:
        
        - **IVI (Índice de Vazamentos da Infraestrutura)**: {ivi_atual:.2f} - {classificacao['categoria']}
        - **Perdas reais**: 44.50% do volume distribuído
        - **Pressões**: Consistentemente abaixo do mínimo recomendado (10 mca)
        - **Padrão característico**: Vazões altas com pressões baixas
        
        Este sistema foi desenvolvido a partir da análise detalhada desses dados e busca fornecer ferramentas
        para identificação, análise e gerenciamento de vazamentos em redes similares.
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
    1. Use a barra lateral para navegar entre as diferentes funcionalidades
    2. Carregue seus dados de monitoramento ou use os dados padrão de Coleipa
    3. Explore os gráficos e análises disponíveis em cada seção
    4. Gere relatórios e recomendações para seu sistema específico
    """)
    
    # Rodapé
    st.markdown("---")
    st.caption("Sistema de Detecção de Vazamentos Coleipa | Baseado em técnicas híbridas Fuzzy-Bayes")


def mostrar_pagina_dados(detector):
    """Página de visualização de dados de monitoramento"""
    st.header("📊 Dados de Monitoramento")
    st.markdown("Visualização dos dados reais de monitoramento do Sistema Coleipa")
    
    # Botão para processar dados
    if st.button("Visualizar Dados de Monitoramento"):
        with st.spinner("Processando dados de monitoramento..."):
            resultado = detector.visualizar_dados_coleipa()
            
            if resultado[0] is not None:
                fig, stats, df = resultado
                
                # Exibir gráficos
                st.pyplot(fig)
                
                # Exibir estatísticas em colunas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Estatísticas de Vazão")
                    st.metric("Vazão mínima", f"{stats['vazao_min']:.2f} m³/h", f"Hora {int(stats['vazao_min_hora'])}")
                    st.metric("Vazão máxima", f"{stats['vazao_max']:.2f} m³/h", f"Hora {int(stats['vazao_max_hora'])}")
                    st.metric("Relação min/máx", f"{stats['vazao_ratio']:.1f}%")
                
                with col2:
                    st.subheader("Estatísticas de Pressão")
                    st.metric("Pressão mínima", f"{stats['pressao_min']:.2f} mca", f"Hora {int(stats['pressao_min_hora'])}")
                    st.metric("Pressão máxima", f"{stats['pressao_max']:.2f} mca", f"Hora {int(stats['pressao_max_hora'])}")
                    st.metric("Horas com pressão < 10 mca", f"{stats['horas_pressao_baixa']} de {len(df)}", f"{stats['perc_pressao_baixa']:.1f}%")
                
                # Exibir tabela de dados
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
    O sistema fuzzy utiliza regras baseadas na análise do comportamento hidráulico da rede de Coleipa.
    Algumas das principais regras são:
    
    1. **Vazão ALTA + Pressão BAIXA + IVI MUITO_RUIM → Risco MUITO_ALTO**  
       *Esta é a situação típica de vazamento no sistema Coleipa*
       
    2. **Vazão ALTA + Pressão BAIXA + IVI REGULAR/RUIM → Risco ALTO**  
       *Forte indicação de vazamento mesmo em sistemas com melhores condições*
       
    3. **Vazão NORMAL + Pressão BAIXA + IVI MUITO_RUIM → Risco ALTO**  
       *Sistemas com IVI alto têm maior risco mesmo com vazões normais*
       
    4. **Vazão NORMAL + Pressão ALTA + IVI BOM → Risco MUITO_BAIXO**  
       *Operação normal em sistemas bem mantidos*
    """)
    
    # Teste interativo do sistema fuzzy
    st.subheader("Teste Interativo")
    st.markdown("Ajuste os parâmetros abaixo para testar o comportamento do sistema fuzzy:")
    
    # Mostrar IVI atual usando a função utilitária
    ivi_atual = detector.caracteristicas_sistema.get('ivi', 16.33)
    detector.exibir_status_ivi_streamlit(ivi_atual, "IVI atual do sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vazao_teste = st.slider("Vazão (m³/h)", 7.0, 16.0, 14.5, 0.1)
    
    with col2:
        pressao_teste = st.slider("Pressão (mca)", 0.0, 10.0, 3.5, 0.1)
    
    with col3:
        ivi_teste = st.slider("IVI", 1.0, 25.0, float(ivi_atual), 0.01)
    
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
                ivi_atual = detector.caracteristicas_sistema.get('ivi', 16.33)
                
                # Classificação manual para evitar problemas de cache
                if ivi_atual <= 4:
                    categoria = "Categoria A - Eficiente"
                elif ivi_atual <= 8:
                    categoria = "Categoria B - Regular"
                elif ivi_atual <= 16:
                    categoria = "Categoria C - Ruim"
                else:
                    categoria = "Categoria D - Muito Ruim"
                
                st.markdown(f"""
                - **População**: {detector.caracteristicas_sistema['populacao']} habitantes
                - **Área**: {detector.caracteristicas_sistema['area_territorial']/1000:.1f} km²
                - **Perdas reais**: {detector.caracteristicas_sistema['percentual_perdas']:.1f}%
                - **IVI**: {ivi_atual:.2f} ({categoria})
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
    
    Os dados de treinamento são gerados baseados nas seguintes características:
    
    - **Operação Normal**: 
      - Vazão média menor
      - Pressão média maior
      - IVI médio menor (simulando sistemas mais eficientes)
      
    - **Vazamento**: 
      - Vazão média maior
      - Pressão média menor
      - IVI médio próximo ao de Coleipa (16.33)
    
    O classificador é então treinado para reconhecer esses padrões e identificar situações de vazamento em novos dados.
    """)


def mostrar_pagina_mapa_calor(detector):
    """Página dos mapas de calor IVI"""
    st.header("🔥 Mapas de Calor IVI")
    st.markdown("Análise de risco para diferentes combinações de vazão e pressão, considerando diferentes valores de IVI")
    
    # Verificação de segurança para evitar erros de cache
    try:
        # Mostrar IVI atual calculado no topo da página com função utilitária segura
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            exibir_ivi_status(detector, "IVI ATUAL DO SISTEMA")
    except Exception as e:
        st.error(f"⚠️ Erro ao exibir IVI: {e}")
        st.info("Tentando recarregar dados do detector...")
        
        # Fallback manual
        try:
            ivi_atual = float(detector.caracteristicas_sistema.get('ivi', 16.33))
            if ivi_atual > 16:
                st.error(f"🚨 **IVI ATUAL DO SISTEMA: {ivi_atual:.2f}** - Categoria D (Muito Ruim)")
            elif ivi_atual > 8:
                st.warning(f"⚠️ **IVI ATUAL DO SISTEMA: {ivi_atual:.2f}** - Categoria C (Ruim)")
            elif ivi_atual > 4:
                st.info(f"ℹ️ **IVI ATUAL DO SISTEMA: {ivi_atual:.2f}** - Categoria B (Regular)")
            else:
                st.success(f"✅ **IVI ATUAL DO SISTEMA: {ivi_atual:.2f}** - Categoria A (Bom)")
        except Exception as e2:
            st.error(f"❌ Erro crítico ao acessar dados: {e2}")
            st.stop()
    
    # Configuração do mapa de calor
    st.subheader("Configuração")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        resolucao = st.slider("Resolução do mapa", 10, 50, 30, 5, 
                             help="Valores maiores geram mapas mais detalhados, mas aumentam o tempo de processamento")
    
    with col2:
        st.markdown("##### Atualizar IVI")
        if st.button("🔄 Recalcular IVI", help="Recalcula o IVI com os parâmetros atuais"):
            with st.spinner("Recalculando IVI..."):
                novo_ivi, _ = detector.calcular_ivi_automatico()
                st.success(f"IVI atualizado: {novo_ivi:.2f}")
                st.info("Atualize a página para ver as mudanças nos mapas.")
    
    # Botão para gerar mapas de calor
    if st.button("Gerar Mapas de Calor"):
        with st.spinner("Gerando mapas de calor IVI... Isso pode levar alguns segundos."):
            fig, ivi_valores = detector.gerar_mapa_calor_ivi(resolucao)
            st.pyplot(fig)
    
    # Análise detalhada do IVI
    st.markdown("---")
    st.subheader("Análise Detalhada do IVI - Sistema Coleipa")
    
    # Usar função utilitária para obter classificação
    classificacao_ivi = detector.classificar_ivi(ivi_atual)
    
    st.markdown(f"""
    ##### 🔍 IVI Calculado: {ivi_atual:.2f}
    ##### 📊 Classificação: {classificacao_ivi['categoria']}
    ##### ⚠️ Interpretação: {classificacao_ivi['interpretacao']}
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📈 Comparação com outras categorias:")
        st.markdown("""
        - 🟢 **Categoria A (IVI 1-4)**: Sistema eficiente, perdas próximas às inevitáveis
        - 🟡 **Categoria B (IVI 4-8)**: Sistema regular, melhorias recomendadas
        - 🟠 **Categoria C (IVI 8-16)**: Sistema ruim, ações urgentes necessárias
        - 🔴 **Categoria D (IVI >16)**: Sistema muito ruim, intervenção imediata
        """)
    
    with col2:
        st.markdown(f"##### 🎯 Análise específica Coleipa (IVI = {ivi_atual:.2f}):")
        
        # Definir interpretação baseada na categoria
        if classificacao_ivi['categoria_simples'] == 'BOM':
            perdas_interpretacao = "Perdas próximas às inevitáveis"
            zona_mapa = "zona verde (baixo risco)"
            prioridade = "manutenção preventiva"
        elif classificacao_ivi['categoria_simples'] == 'REGULAR':
            perdas_interpretacao = f"Perdas {ivi_atual:.1f}x maiores que as inevitáveis"
            zona_mapa = "zona amarela (risco moderado)"
            prioridade = "melhorias graduais"
        elif classificacao_ivi['categoria_simples'] == 'RUIM':
            perdas_interpretacao = f"Perdas {ivi_atual:.1f}x maiores que as inevitáveis"
            zona_mapa = "zona laranja (risco alto)"
            prioridade = "ações urgentes necessárias"
        else:  # MUITO RUIM
            perdas_interpretacao = f"Perdas {ivi_atual:.1f}x maiores que as inevitáveis"
            zona_mapa = "zona vermelha (risco crítico)"
            prioridade = "detecção e reparo imediato de vazamentos"
        
        st.markdown(f"""
        - {perdas_interpretacao}
        - Potencial de redução de perdas > 400 L/ligação.dia
        - Localização no mapa: {zona_mapa}
        - Combinação crítica: Vazão ALTA + Pressão BAIXA
        - Prioridade máxima: {prioridade}
        """)
    
    st.markdown("##### 🔧 Impacto visual nos mapas:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**IVI BOM (2.0):**  \nPredominantemente verde (baixo risco)")
    
    with col2:
        st.markdown("**IVI REGULAR (6.0):**  \nVerde-amarelo (risco moderado)")
    
    with col3:
        st.markdown("**IVI RUIM (12.0):**  \nAmarelo-laranja (risco alto)")
    
    with col4:
        # Usar classificação dinâmica para o IVI atual
        categoria_cor = "Vermelho intenso" if classificacao_ivi['categoria_simples'] == 'MUITO RUIM' else "Variável conforme categoria"
        st.markdown(f"**IVI {classificacao_ivi['categoria_simples']} ({ivi_atual:.2f}):**  \n{categoria_cor} (risco conforme categoria)")


def mostrar_pagina_simulacao(detector):
    """Página de simulação temporal"""
    st.header("⏱️ Simulação Temporal")
    st.markdown("Simulação de série temporal com detecção de vazamentos")
    
    # Verificar se modelo Bayes está treinado
    if detector.modelo_bayes is None:
        st.warning("Modelo Bayesiano não está treinado. Treinando modelo com parâmetros padrão...")
        with st.spinner("Treinando modelo..."):
            X, y, _ = detector.gerar_dados_baseados_coleipa()
            detector.treinar_modelo_bayesiano(X, y)
    
    # Botão para executar simulação
    if st.button("Executar Simulação"):
        with st.spinner("Simulando série temporal... Isso pode levar alguns segundos."):
            fig, df = detector.simular_serie_temporal_coleipa()
            st.pyplot(fig)
            
            # Mostrar dados da simulação
            with st.expander("Ver dados da simulação"):
                # Formatar coluna de tempo para exibição
                df_exibicao = df.copy()
                df_exibicao['Tempo'] = df_exibicao['Tempo'].dt.strftime('%d/%m %H:%M')
                
                # Selecionar colunas relevantes
                if 'Prob_Hibrida' in df.columns:
                    df_exibicao = df_exibicao[['Tempo', 'Vazao', 'Pressao', 'IVI', 'Vazamento_Real', 'Prob_Hibrida']]
                else:
                    df_exibicao = df_exibicao[['Tempo', 'Vazao', 'Pressao', 'IVI', 'Vazamento_Real']]
                
                st.dataframe(df_exibicao)
    
    # Explicação da simulação
    st.markdown("---")
    st.subheader("Sobre a Simulação Temporal")
    st.markdown("""
    A simulação temporal representa o comportamento do sistema ao longo de 3 dias completos, com um vazamento simulado
    iniciando no segundo dia às 14h. Características da simulação:
    
    #### Comportamento Normal
    - Vazão e pressão seguem os padrões observados no sistema Coleipa
    - Pequenas variações aleatórias são adicionadas para simular flutuações naturais
    - Comportamento cíclico diário com picos de consumo durante o dia e vales durante a noite
    
    #### Vazamento Simulado
    - Inicia no segundo dia às 14h
    - Progressão gradual ao longo de várias horas (simulando vazamento em crescimento)
    - Causa aumento simultâneo de vazão e diminuição de pressão
    
    #### Sistema de Detecção
    - Componente Fuzzy: Avalia risco baseado nas regras definidas
    - Componente Bayes: Calcula probabilidade baseada nos dados aprendidos
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
    """Página de análise de caso"""
    st.header("🔬 Análise de Caso Específico")
    st.markdown("Analise um caso específico de operação baseado nos parâmetros fornecidos")
    
    # Verificar se modelo Bayes está treinado
    if detector.modelo_bayes is None:
        st.warning("Modelo Bayesiano não está treinado. Alguns resultados serão limitados apenas à análise fuzzy.")
        usar_bayes = st.checkbox("Treinar modelo Bayesiano agora", value=True)
        if usar_bayes:
            with st.spinner("Treinando modelo..."):
                X, y, _ = detector.gerar_dados_baseados_coleipa()
                detector.treinar_modelo_bayesiano(X, y)
    
    # Formulário para entrada de dados
    st.subheader("Parâmetros do Sistema")
    
    # Mostrar IVI atual sendo usado com função utilitária segura
    exibir_ivi_status(detector, "IVI atual do sistema")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vazao = st.number_input("Vazão (m³/h)", min_value=7.0, max_value=16.0, value=14.5, step=0.1,
                              help="Valor típico para Coleipa: 14.5 m³/h")
    
    with col2:
        pressao = st.number_input("Pressão (mca)", min_value=0.0, max_value=10.0, value=3.5, step=0.1,
                                help="Valor típico para Coleipa: 3.5 mca")
    
    with col3:
        ivi = st.number_input("IVI", min_value=1.0, max_value=25.0, value=float(ivi_atual), step=0.01,
                            help=f"IVI atual do sistema Coleipa: {ivi_atual:.2f}")
    
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
                - **IVI real**: {resultado['ivi_real']:.2f}
                - **Prioridade recomendada**: Detecção de vazamentos
                """)
    
    # Explicação sobre análise de caso
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
    - **ALTA (boa)**: Valores acima de 8 mca, próximos à recomendação NBR
    
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
            
            # Obter IVI atual calculado
            ivi_atual = detector.caracteristicas_sistema.get('ivi', 16.33)
            
            # Cabeçalho do relatório
            st.markdown("---")
            st.subheader("RELATÓRIO DE ANÁLISE - SISTEMA COLEIPA")
            
            # Mostrar IVI atual no topo com função utilitária segura
            exibir_ivi_status(detector, "IVI ATUAL")
            
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
                st.metric("IPRL", f"{relatorio['indicadores']['iprl']:.3f} m³/lig.dia", "Perdas Reais por Ligação")
            
            with col2:
                st.metric("IPRI", f"{relatorio['indicadores']['ipri']:.3f} m³/lig.dia", "Perdas Reais Inevitáveis")
            
            with col3:
                st.metric("IVI", f"{relatorio['indicadores']['ivi']:.2f}", "Índice de Vazamentos da Infraestrutura")
            
            # 4. Classificação usando função utilitária segura
            st.subheader("4. CLASSIFICAÇÃO (Banco Mundial)")
            
            classificacao_ivi = classificar_ivi_manual(ivi_atual)
            
            if classificacao_ivi['categoria_simples'] == 'BOM':
                interpretacao = "Sistema eficiente com perdas próximas às inevitáveis"
                recomendacao = "Manter práticas atuais de gestão"
            elif classificacao_ivi['categoria_simples'] == 'REGULAR':
                interpretacao = "Sistema regular, melhorias recomendadas"
                recomendacao = "Implementar melhorias graduais no sistema"
            elif classificacao_ivi['categoria_simples'] == 'RUIM':
                interpretacao = "Sistema ruim, ações urgentes necessárias"
                recomendacao = "Implementar programa de redução de perdas urgente"
            else:
                interpretacao = "Sistema muito ruim, intervenção imediata necessária"
                recomendacao = "Programas de redução de perdas são imperiosos e prioritários"
            
            st.markdown(f"""
            - **Categoria**: {classificacao_ivi['categoria']}
            - **Interpretação**: {interpretacao}
            - **Recomendação**: {recomendacao}
            """)
            
            # 5. Metodologia NPR - Priorização de Ações
            st.subheader("5. METODOLOGIA NPR - PRIORIZAÇÃO DE AÇÕES")
            
            # Criar tabela de prioridades
            df_prioridades = pd.DataFrame(relatorio['prioridades'])
            df_prioridades.columns = ["Ordem", "Ação", "Resultado"]
            
            # Gráfico de barras para prioridades
            fig, ax = plt.subplots(figsize=(10, 5))
            barras = ax.barh(
                [p['acao'] for p in relatorio['prioridades']], 
                [p['resultado'] for p in relatorio['prioridades']],
                color=['#3498db', '#2980b9', '#1f618d', '#154360']
            )
            ax.set_xlabel('Resultado NPR')
            ax.set_title('Priorização de Ações (Metodologia NPR)')
            
            # Adicionar valores nas barras
            for barra in barras:
                largura = barra.get_width()
                pos_x_rotulo = largura + 1
                ax.text(pos_x_rotulo, barra.get_y() + barra.get_height()/2, f'{largura}', 
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
            
            # 8. Análise de Impacto Econômico
            st.subheader("8. ANÁLISE DE IMPACTO ECONÔMICO")
            
            # Estimar perda anual de água
            perda_anual_m3 = relatorio['monitoramento']['perdas_reais'] * 365  # m³/ano
            
            # Valores de referência para custos
            custo_agua_tratada = 1.50  # R$/m³ (valor médio para água tratada)
            custo_energia = 0.80  # R$/m³ (custo de energia para bombeamento)
            custo_manutencao = 0.50  # R$/m³ (custo de manutenção relacionado às perdas)
            
            # Cálculo de custos
            custo_anual_agua = perda_anual_m3 * custo_agua_tratada
            custo_anual_energia = perda_anual_m3 * custo_energia
            custo_anual_manutencao = perda_anual_m3 * custo_manutencao
            custo_anual_total = custo_anual_agua + custo_anual_energia + custo_anual_manutencao
            
            # Economia estimada com redução do IVI
            ivi_atual = relatorio['indicadores']['ivi']
            ivi_alvo = 8.0  # Meta: redução para Categoria B
            reducao_percentual = max(0, (ivi_atual - ivi_alvo) / ivi_atual * 100)
            economia_potencial = custo_anual_total * (reducao_percentual / 100)
            
            # Exibir resultados econômicos em colunas
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Perda anual estimada", f"{perda_anual_m3:.0f} m³/ano")
                st.metric("Custo anual água tratada", f"R$ {custo_anual_agua:.2f}")
                st.metric("Custo anual energia", f"R$ {custo_anual_energia:.2f}")
                st.metric("Custo anual manutenção", f"R$ {custo_anual_manutencao:.2f}")
            
            with col2:
                st.metric("Custo anual total", f"R$ {custo_anual_total:.2f}")
                st.metric("Meta de redução IVI", f"{ivi_atual:.2f} → {ivi_alvo:.2f} ({reducao_percentual:.1f}%)")
                st.metric("Economia potencial anual", f"R$ {economia_potencial:.2f}")
                payback_anos = 100000 / economia_potencial if economia_potencial > 0 else float('inf')
                st.metric("Payback estimado (investimento R$ 100.000)", f"{payback_anos:.1f} anos")
            
            # Gráfico de composição dos custos
            fig_custos, ax_custos = plt.subplots(figsize=(10, 6))
            custos = [custo_anual_agua, custo_anual_energia, custo_anual_manutencao]
            rotulos = ['Água Tratada', 'Energia', 'Manutenção']
            cores = ['#3498db', '#2ecc71', '#e74c3c']
            
            ax_custos.pie(custos, labels=rotulos, autopct='%1.1f%%', startangle=90, colors=cores,
                         wedgeprops=dict(width=0.5, edgecolor='w'))
            ax_custos.axis('equal')
            ax_custos.set_title('Composição dos Custos Relacionados às Perdas')
            
            st.pyplot(fig_custos)
            
            # 9. Plano de Ação
            st.subheader("9. PLANO DE AÇÃO")
            
            # Tabela do plano de ação
            plano_acao = [
                {
                    "Etapa": "Curto Prazo (0-6 meses)",
                    "Ação": "Pesquisa de vazamentos não visíveis na rede",
                    "Custo Estimado": "R$ 25.000,00",
                    "Impacto Esperado": "20% redução nas perdas"
                },
                {
                    "Etapa": "Curto Prazo (0-6 meses)",
                    "Ação": "Melhoria do tempo de reparo de vazamentos visíveis",
                    "Custo Estimado": "R$ 10.000,00",
                    "Impacto Esperado": "5% redução nas perdas"
                },
                {
                    "Etapa": "Médio Prazo (6-18 meses)",
                    "Ação": "Instalação de VRPs em pontos críticos",
                    "Custo Estimado": "R$ 40.000,00",
                    "Impacto Esperado": "15% redução nas perdas"
                },
                {
                    "Etapa": "Médio Prazo (6-18 meses)",
                    "Ação": "Setorização da rede de distribuição",
                    "Custo Estimado": "R$ 60.000,00",
                    "Impacto Esperado": "20% redução nas perdas"
                },
                {
                    "Etapa": "Longo Prazo (18-36 meses)",
                    "Ação": "Substituição de trechos críticos da rede",
                    "Custo Estimado": "R$ 120.000,00",
                    "Impacto Esperado": "25% redução nas perdas"
                }
            ]
            
            df_plano = pd.DataFrame(plano_acao)
            st.dataframe(df_plano, use_container_width=True)
            
            # Gráfico de Gantt para cronograma
            fig_gantt, ax_gantt = plt.subplots(figsize=(12, 5))
            
            # Dados do Gantt
            etapas = ['Detecção vazamentos', 'Melhorar tempo reparo', 'Instalar VRPs', 
                    'Setorização rede', 'Substituir trechos críticos']
            inicio = [0, 0, 6, 8, 18]
            duracao = [6, 3, 6, 10, 18]
            cores = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
            
            # Plotar barras
            for i, (etapa, start, dur, cor) in enumerate(zip(etapas, inicio, duracao, cores)):
                ax_gantt.barh(i, dur, left=start, color=cor, alpha=0.8)
                # Adicionar texto na barra
                ax_gantt.text(start + dur/2, i, etapa, ha='center', va='center', fontsize=9, fontweight='bold')
            
            # Configurações dos eixos
            ax_gantt.set_yticks([])
            ax_gantt.set_xlabel('Meses')
            ax_gantt.set_title('Cronograma de Implementação')
            ax_gantt.grid(axis='x', alpha=0.3)
            ax_gantt.set_axisbelow(True)
            
            # Adicionar marcadores de tempo
            for i in range(0, 37, 6):
                ax_gantt.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
                ax_gantt.text(i, -0.5, f'{i}m', ha='center', va='top')
            
            st.pyplot(fig_gantt)
            
            # 10. Considerações Finais
            st.subheader("10. CONSIDERAÇÕES FINAIS")
            st.markdown("""
            A análise detalhada do Sistema de Abastecimento de Água Potável (SAAP) do bairro Coleipa revela
            uma condição crítica em relação às perdas de água, com classificação D (muito ruim) segundo critérios do Banco Mundial.
            Esta condição resulta em desperdício significativo de água e recursos financeiros.
            
            A implementação das ações recomendadas neste relatório tem potencial para:
            
            1. **Reduzir o IVI** de {:.2f} para valores abaixo de 8 (Categoria B)
            2. **Economizar aproximadamente R$ {:.2f} por ano** em custos operacionais
            3. **Postergar investimentos** em ampliação do sistema de produção
            4. **Melhorar a pressão e continuidade** do abastecimento para os usuários
            
            Recomenda-se fortemente a adoção imediata das medidas de curto prazo, com foco especial na detecção
            de vazamentos não visíveis, que constitui a ação de maior impacto imediato segundo a Metodologia NPR.
            
            **Nota importante:** O sucesso do programa de redução de perdas está diretamente vinculado ao
            comprometimento da gestão e à alocação dos recursos necessários para sua implementação.
            """.format(relatorio['indicadores']['ivi'], economia_potencial))
            
            # Assinatura e data
            st.markdown("---")
            data_atual = datetime.now().strftime("%d/%m/%Y")
            st.markdown(f"""
            **Relatório gerado em:** {data_atual}
            
            **Sistema de Detecção de Vazamentos - SAAP Coleipa**  
            *Baseado em técnicas híbridas Fuzzy-Bayes e análise de dados reais de monitoramento*
            """)
            
            st.markdown("---")
            st.success("Relatório completo gerado com sucesso!")


def mostrar_pagina_configuracoes(detector):
    """Página de configurações"""
    st.header("⚙️ Configurações")
    st.markdown("Configurar parâmetros do sistema")
    
    # Garantir que todos os parâmetros existem (segurança adicional)
    detector._garantir_parametros_ivi()
    
    # Verificar se todos os parâmetros necessários estão presentes
    parametros_necessarios = [
        'volume_perdido_anual', 'distancia_lote_medidor', 'pressao_operacao_adequada',
        'coeficiente_rede', 'coeficiente_ligacoes', 'coeficiente_ramais'
    ]
    
    parametros_faltando = [p for p in parametros_necessarios if p not in detector.caracteristicas_sistema]
    
    if parametros_faltando:
        st.warning(f"Parâmetros faltando detectados: {parametros_faltando}. Aplicando valores padrão.")
        # Forçar atualização
        detector._garantir_parametros_ivi()
    
    # Botão de diagnóstico
    if st.button("🔍 Executar Diagnóstico do Sistema"):
        st.subheader("Diagnóstico do Sistema")
        
        # Verificar todos os parâmetros
        todos_parametros = [
            'area_territorial', 'populacao', 'numero_ligacoes', 'comprimento_rede',
            'densidade_ramais', 'vazao_media_normal', 'pressao_media_normal',
            'perdas_reais_media', 'volume_consumido_medio', 'percentual_perdas',
            'iprl', 'ipri', 'ivi', 'volume_perdido_anual', 'distancia_lote_medidor',
            'pressao_operacao_adequada', 'coeficiente_rede', 'coeficiente_ligacoes',
            'coeficiente_ramais'
        ]
        
        parametros_presentes = []
        parametros_ausentes = []
        
        for param in todos_parametros:
            if param in detector.caracteristicas_sistema:
                parametros_presentes.append(param)
            else:
                parametros_ausentes.append(param)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"✅ Parâmetros presentes: {len(parametros_presentes)}/{len(todos_parametros)}")
            if parametros_presentes:
                with st.expander("Ver parâmetros presentes"):
                    for param in parametros_presentes:
                        valor = detector.caracteristicas_sistema.get(param, "N/A")
                        st.text(f"{param}: {valor}")
        
        with col2:
            if parametros_ausentes:
                st.error(f"❌ Parâmetros ausentes: {len(parametros_ausentes)}")
                with st.expander("Ver parâmetros ausentes"):
                    for param in parametros_ausentes:
                        st.text(param)
                
                if st.button("🔧 Corrigir Parâmetros Ausentes"):
                    detector._garantir_parametros_ivi()
                    st.success("Parâmetros corrigidos! Atualize a página para ver as mudanças.")
            else:
                st.success("✅ Todos os parâmetros estão presentes")
    
    # Formulário de características do sistema
    st.subheader("Características do Sistema")
    
    # Criar duas colunas para organização do formulário
    col1, col2 = st.columns(2)
    
    with col1:
        area_territorial = st.number_input("Área Territorial (m²)", 
                                           value=int(detector.caracteristicas_sistema['area_territorial']),
                                           step=1000,
                                           min_value=0)
        
        populacao = st.number_input("População", 
                                    value=int(detector.caracteristicas_sistema['populacao']),
                                    step=100,
                                    min_value=0)
        
        numero_ligacoes = st.number_input("Número de Ligações", 
                                          value=int(detector.caracteristicas_sistema['numero_ligacoes']),
                                          step=10,
                                          min_value=0)
        
        comprimento_rede = st.number_input("Comprimento da Rede (km)", 
                                           value=float(detector.caracteristicas_sistema['comprimento_rede']),
                                           step=0.1,
                                           min_value=0.0)
        
        densidade_ramais = st.number_input("Densidade de Ramais (ramais/km)", 
                                           value=int(detector.caracteristicas_sistema['densidade_ramais']),
                                           step=10,
                                           min_value=0)
    
    with col2:
        vazao_media_normal = st.number_input("Vazão Média Normal (l/s)", 
                                             value=float(detector.caracteristicas_sistema['vazao_media_normal']),
                                             step=0.01,
                                             min_value=0.0)
        
        pressao_media_normal = st.number_input("Pressão Média Normal (mca)", 
                                               value=float(detector.caracteristicas_sistema['pressao_media_normal']),
                                               step=0.01,
                                               min_value=0.0)
        
        perdas_reais_media = st.number_input("Perdas Reais Médias (m³/dia)", 
                                             value=float(detector.caracteristicas_sistema['perdas_reais_media']),
                                             step=0.1,
                                             min_value=0.0)
        
        volume_consumido_medio = st.number_input("Volume Consumido Médio (m³/dia)", 
                                                 value=float(detector.caracteristicas_sistema['volume_consumido_medio']),
                                                 step=0.1,
                                                 min_value=0.0)
        
        percentual_perdas = st.number_input("Percentual de Perdas (%)", 
                                             value=float(detector.caracteristicas_sistema['percentual_perdas']),
                                             step=0.1,
                                             min_value=0.0,
                                             max_value=100.0)
    
    # Botão para atualizar características
    if st.button("Atualizar Características do Sistema"):
        novas_caracteristicas = {
            'area_territorial': int(area_territorial),
            'populacao': int(populacao),
            'numero_ligacoes': int(numero_ligacoes),
            'comprimento_rede': float(comprimento_rede),
            'densidade_ramais': int(densidade_ramais),
            'vazao_media_normal': float(vazao_media_normal),
            'pressao_media_normal': float(pressao_media_normal),
            'perdas_reais_media': float(perdas_reais_media),
            'volume_consumido_medio': float(volume_consumido_medio),
            'percentual_perdas': float(percentual_perdas)
        }
        
        detector.atualizar_caracteristicas_sistema(novas_caracteristicas)
        st.success("Características do sistema atualizadas com sucesso!")
        
        # Sugerir recálculo do IVI se parâmetros relacionados foram alterados
        parametros_ivi_relacionados = ['numero_ligacoes', 'comprimento_rede']
        if any(param in novas_caracteristicas for param in parametros_ivi_relacionados):
            st.info("💡 Parâmetros relacionados ao IVI foram alterados. Considere recalcular o IVI na seção específica abaixo.")
    
    # Cálculo de IVI
    st.markdown("---")
    st.subheader("Parâmetros para Cálculo de IVI")
    st.markdown("Configure os parâmetros específicos para o cálculo do Índice de Vazamentos da Infraestrutura")
    
    # Criar seção expansível para parâmetros de IVI
    with st.expander("📊 Configurar Parâmetros do Cálculo de IVI", expanded=False):
        st.info("💡 **Dica:** Estes parâmetros são baseados no estudo de caso do sistema Coleipa. Ajuste-os conforme as características do seu sistema.")
        
        st.markdown("##### Parâmetros Principais")
        
        col1, col2 = st.columns(2)
        
        with col1:
            volume_perdido_anual = st.number_input("Volume Perdido Anual (Vp) [m³/ano]", 
                                                   value=float(detector.caracteristicas_sistema.get('volume_perdido_anual', 37547.55)),
                                                   step=100.0,
                                                   min_value=0.0,
                                                   help="Volume total de água perdido por ano")
            
            distancia_lote_medidor = st.number_input("Distância Lote-Medidor (Lp) [km]", 
                                                     value=float(detector.caracteristicas_sistema.get('distancia_lote_medidor', 0.001)),
                                                     step=0.001,
                                                     min_value=0.0,
                                                     format="%.3f",
                                                     help="Distância média entre limite do lote e medidor hidrômetro")
            
            pressao_operacao_adequada = st.number_input("Pressão de Operação Adequada (P) [mca]", 
                                                        value=float(detector.caracteristicas_sistema.get('pressao_operacao_adequada', 20.0)),
                                                        step=1.0,
                                                        min_value=0.0,
                                                        help="Pressão média de operação adequada do sistema")
        
        with col2:
            st.markdown("##### Coeficientes da Fórmula IPRI")
            st.markdown("*IPRI = (C₁×Lm + C₂×Nc + C₃×Lp×Nc) × P / (Nc×1000)*")
            
            coeficiente_rede = st.number_input("Coeficiente da Rede (C₁)", 
                                               value=float(detector.caracteristicas_sistema.get('coeficiente_rede', 18.0)),
                                               step=0.1,
                                               min_value=0.0,
                                               help="Coeficiente para comprimento da rede (padrão: 18)")
            
            coeficiente_ligacoes = st.number_input("Coeficiente das Ligações (C₂)", 
                                                   value=float(detector.caracteristicas_sistema.get('coeficiente_ligacoes', 0.8)),
                                                   step=0.1,
                                                   min_value=0.0,
                                                   help="Coeficiente para número de ligações (padrão: 0.8)")
            
            coeficiente_ramais = st.number_input("Coeficiente dos Ramais (C₃)", 
                                                 value=float(detector.caracteristicas_sistema.get('coeficiente_ramais', 25.0)),
                                                 step=1.0,
                                                 min_value=0.0,
                                                 help="Coeficiente para distância dos ramais (padrão: 25)")
        
        # Botão para atualizar parâmetros de IVI
        if st.button("Atualizar Parâmetros de IVI"):
            novos_parametros_ivi = {
                'volume_perdido_anual': float(volume_perdido_anual),
                'distancia_lote_medidor': float(distancia_lote_medidor),
                'pressao_operacao_adequada': float(pressao_operacao_adequada),
                'coeficiente_rede': float(coeficiente_rede),
                'coeficiente_ligacoes': float(coeficiente_ligacoes),
                'coeficiente_ramais': float(coeficiente_ramais)
            }
            
            detector.atualizar_caracteristicas_sistema(novos_parametros_ivi)
            st.success("Parâmetros de IVI atualizados com sucesso!")
            
            # Recalcular IVI automaticamente
            with st.spinner("Recalculando IVI com novos parâmetros..."):
                ivi_novo, resultados_novo = detector.calcular_ivi_automatico()
                st.info(f"Novo IVI calculado: {ivi_novo:.2f}")
        
        # Mostrar fórmulas de referência
        st.markdown("---")
        st.markdown("##### 📐 Fórmulas de Referência (Atualizadas)")
        st.markdown("""
        **Equação 3 - IPRL (Índice de Perdas Reais por Ligação):**  
        `IPRL = Vp / (Nc × 365)`
        
        **Equação 4 - IPRI (Índice de Perdas Reais Inevitáveis) - NOVA FÓRMULA:**  
        `IPRI = (18 × Lm + 0,8 × Nc + 25 × Lp × Nc) × P / (Nc × 1000)`
        
        **Equação 5 - IVI (Índice de Vazamentos na Infraestrutura):**  
        `IVI = IPRL / IPRI`
        
        **Onde:**
        - Vp = Volume perdido anual (m³/ano)
        - Nc = Número de ligações
        - Lm = Comprimento da rede (km)
        - Lp = Distância lote-medidor (km)
        - P = Pressão de operação adequada (mca)
        
        **Mudanças na Nova Fórmula:**
        - Coeficiente da rede: 8 → 18
        - Termo dos ramais: 25 × Lp → 25 × Lp × Nc  
        - Denominador: Nc → Nc × 1000
        """)
        
        # Seção de ajuda atualizada
        st.markdown("---")
        st.markdown("##### 💡 Ajuda - Nova Fórmula IPRI")
        
        with st.expander("📖 Guia da Nova Fórmula IPRI"):
            st.markdown("""
            **Nova Fórmula IPRI:**
            `IPRI = (18 × Lm + 0,8 × Nc + 25 × Lp × Nc) × P / (Nc × 1000)`
            
            **Principais Mudanças:**
            
            1. **Coeficiente da Rede (18):**
               - Valor fixo aumentado de 8 para 18
               - Reflete maior impacto do comprimento da rede nas perdas inevitáveis
            
            2. **Termo dos Ramais (25 × Lp × Nc):**
               - Agora multiplicado pelo número de ligações (Nc)
               - Considera que cada ligação tem sua própria distância lote-medidor
               - Impacto proporcional ao número total de ligações
            
            3. **Denominador (Nc × 1000):**
               - Fator 1000 para conversão de unidades
               - Resulta em valores IPRI menores
               - Melhor adequação às escalas típicas de sistemas
            
            **Exemplo de Cálculo (Coleipa):**
            - Lm = 3 km, Nc = 300, Lp = 0,001 km, P = 20 mca
            - Numerador: (18×3 + 0,8×300 + 25×0,001×300) × 20
            - Numerador: (54 + 240 + 7,5) × 20 = 6.030
            - Denominador: 300 × 1000 = 300.000
            - IPRI = 6.030 / 300.000 = 0,0201 m³/lig.dia
            
            **Vantagens da Nova Fórmula:**
            - Melhor representação do impacto dos ramais
            - Valores mais realistas para IPRI
            - Maior sensibilidade ao número de ligações
            - Adequação a diferentes portes de sistema
            """)
    
    # Cálculo automático de IVI com parâmetros atuais
    st.markdown("---")
    st.subheader("Cálculo Automático de IVI")
    st.markdown("Calcular IVI baseado nos parâmetros atuais do sistema com a nova fórmula")
    
    if st.button("Calcular IVI"):
        with st.spinner("Calculando IVI..."):
            ivi, resultados = detector.calcular_ivi_automatico()
            
            st.success(f"IVI calculado com sucesso: {ivi:.2f}")
            
            # Exibir classificação com função específica para IVI calculado
            classificacao_calculada = exibir_ivi_calculado(ivi, "IVI CALCULADO")
            
            # Exibir resultados detalhados conforme as imagens
            st.subheader("Detalhes do Cálculo - Conforme Documentação")
            
            # Mostrar fórmulas e cálculos
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📐 Fórmulas Utilizadas (Atualizadas)")
                st.markdown("""
                **Equação 3 - IPRL:**  
                `IPRL = Vp / (Nc × 365)`
                
                **Equação 4 - IPRI (Nova Fórmula):**  
                `IPRI = (18 × Lm + 0,8 × Nc + 25 × Lp × Nc) × P / (Nc × 1000)`
                
                **Equação 5 - IVI:**  
                `IVI = IPRL / IPRI`
                """)
                
                st.markdown("##### 📊 Parâmetros do Sistema")
                st.text(f"Vp (Volume perdido anual): {resultados['volume_perdido_anual']:.2f} m³/ano")
                st.text(f"Nc (Número de ligações): {resultados['numero_ligacoes']}")
                st.text(f"Lm (Comprimento da rede): {resultados['comprimento_rede']} km")
                st.text(f"Lp (Distância lote-medidor): {resultados['distancia_lote_medidor']} km")
                st.text(f"P (Pressão de operação): {resultados['pressao_operacao']} mca")
                
                st.markdown("##### ⚙️ Coeficientes IPRI (Fixos na Nova Fórmula)")
                st.text(f"C₁ (Coef. rede): {resultados['coeficiente_rede']} (fixo)")
                st.text(f"C₂ (Coef. ligações): {resultados['coeficiente_ligacoes']} (fixo)")
                st.text(f"C₃ (Coef. ramais): {resultados['coeficiente_ramais']} (fixo)")
            
            with col2:
                st.markdown("##### 🧮 Cálculos Detalhados (Nova Fórmula)")
                st.markdown(f"""
                **IPRL Calculation:**  
                {resultados['calculo_iprl']}
                
                **IPRI Calculation (Nova Fórmula):**  
                {resultados['calculo_ipri']}
                
                **IVI Calculation:**  
                {resultados['calculo_ivi']}
                """)
                
                st.markdown("##### 📈 Resultados Finais")
                st.metric("IPRL", f"{resultados['iprl']:.3f} m³/lig.dia", "Perdas Reais por Ligação")
                st.metric("IPRI", f"{resultados['ipri']:.6f} m³/lig.dia", "Perdas Reais Inevitáveis (Nova Fórmula)")
                st.metric("IVI", f"{resultados['ivi']:.2f}", "Índice de Vazamentos da Infraestrutura")
                
                # Destacar que está usando nova fórmula
                st.info("💡 **Cálculo realizado com a nova fórmula IPRI**")
            
            # Classificação detalhada com função utilitária
            st.markdown("---")
            st.subheader("Classificação do IVI (Banco Mundial)")
            
            classificacao_ivi = classificar_ivi_manual(ivi)
            
            if classificacao_ivi['categoria_simples'] == 'BOM':
                interpretacao = "Sistema eficiente com perdas próximas às inevitáveis"
                recomendacao = "Manter práticas atuais de gestão"
            elif classificacao_ivi['categoria_simples'] == 'REGULAR':
                interpretacao = "Sistema regular, melhorias recomendadas"
                recomendacao = "Implementar melhorias graduais no sistema"
            elif classificacao_ivi['categoria_simples'] == 'RUIM':
                interpretacao = "Sistema ruim, ações urgentes necessárias"
                recomendacao = "Implementar programa de redução de perdas urgente"
            else:
                interpretacao = "Sistema muito ruim, intervenção imediata necessária"
                recomendacao = "Programas de redução de perdas são imperiosos e prioritários"
            
            st.markdown(f"""
            ### {classificacao_ivi['emoji']} {classificacao_ivi['categoria']}
            **IVI: {ivi:.2f}**  
            *{interpretacao}*
            
            **Recomendação:** {recomendacao}
            
            O sistema Coleipa apresenta IVI = {ivi:.2f}, indicando que as perdas reais são 
            {ivi:.2f} vezes maiores que as perdas inevitáveis.
            
            **Nota:** Com a nova fórmula IPRI, o resultado pode diferir ligeiramente do valor original 
            devido às mudanças nos coeficientes e estrutura da equação.
            """)
    
    # Opções avançadas
    st.markdown("---")
    st.subheader("Opções Avançadas")
    
    # Configuração do sistema fuzzy
    st.markdown("##### Configuração do Sistema Fuzzy")
    
    with st.expander("Configurar Parâmetros do Sistema Fuzzy"):
        st.info("Configure os pontos centrais dos conjuntos fuzzy. As faixas serão calculadas automaticamente.")
        
        # Parâmetros de vazão
        st.markdown("**Parâmetros de Vazão (m³/h)**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vazao_baixa_centro = st.number_input("Centro Vazão BAIXA", 
                                                min_value=5.0, max_value=12.0, 
                                                value=float(detector.param_vazao['BAIXA']['faixa'][1]), 
                                                step=0.1)
        
        with col2:
            vazao_normal_centro = st.number_input("Centro Vazão NORMAL", 
                                                 min_value=8.0, max_value=15.0, 
                                                 value=float(detector.param_vazao['NORMAL']['faixa'][1]), 
                                                 step=0.1)
        
        with col3:
            vazao_alta_centro = st.number_input("Centro Vazão ALTA", 
                                               min_value=12.0, max_value=18.0, 
                                               value=float(detector.param_vazao['ALTA']['faixa'][1]), 
                                               step=0.1)
        
        # Parâmetros de pressão
        st.markdown("**Parâmetros de Pressão (mca)**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pressao_baixa_centro = st.number_input("Centro Pressão BAIXA", 
                                                  min_value=0.0, max_value=6.0, 
                                                  value=float(detector.param_pressao['BAIXA']['faixa'][1]), 
                                                  step=0.1)
        
        with col2:
            pressao_media_centro = st.number_input("Centro Pressão MÉDIA", 
                                                  min_value=3.0, max_value=9.0, 
                                                  value=float(detector.param_pressao['MEDIA']['faixa'][1]), 
                                                  step=0.1)
        
        with col3:
            pressao_alta_centro = st.number_input("Centro Pressão ALTA", 
                                                 min_value=6.0, max_value=12.0, 
                                                 value=float(detector.param_pressao['ALTA']['faixa'][1]), 
                                                 step=0.1)
        
        if st.button("Atualizar Parâmetros Fuzzy"):
            # Calcular faixas automaticamente baseadas nos centros
            # Para conjuntos triangulares: [min, centro, max]
            vazao_baixa_faixa = [max(5.0, vazao_baixa_centro - 2), vazao_baixa_centro, min(12.0, vazao_baixa_centro + 2)]
            vazao_normal_faixa = [max(8.0, vazao_normal_centro - 2.5), vazao_normal_centro, min(15.0, vazao_normal_centro + 2.5)]
            vazao_alta_faixa = [max(12.0, vazao_alta_centro - 2), vazao_alta_centro, min(18.0, vazao_alta_centro + 2)]
            
            pressao_baixa_faixa = [max(0.0, pressao_baixa_centro - 2), pressao_baixa_centro, min(6.0, pressao_baixa_centro + 2)]
            pressao_media_faixa = [max(3.0, pressao_media_centro - 2), pressao_media_centro, min(9.0, pressao_media_centro + 2)]
            pressao_alta_faixa = [max(6.0, pressao_alta_centro - 2), pressao_alta_centro, min(12.0, pressao_alta_centro + 2)]
            
            # Atualizar parâmetros fuzzy
            detector.param_vazao = {
                'BAIXA': {'faixa': vazao_baixa_faixa},
                'NORMAL': {'faixa': vazao_normal_faixa},
                'ALTA': {'faixa': vazao_alta_faixa}
            }
            
            detector.param_pressao = {
                'BAIXA': {'faixa': pressao_baixa_faixa},
                'MEDIA': {'faixa': pressao_media_faixa},
                'ALTA': {'faixa': pressao_alta_faixa}
            }
            
            # Resetar sistema fuzzy para forçar recriação com novos parâmetros
            detector.sistema_fuzzy = None
            
            st.success("Parâmetros fuzzy atualizados com sucesso!")
            
            # Mostrar as faixas calculadas
            st.subheader("Faixas Calculadas")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Vazão:**")
                st.text(f"BAIXA: {vazao_baixa_faixa}")
                st.text(f"NORMAL: {vazao_normal_faixa}")
                st.text(f"ALTA: {vazao_alta_faixa}")
            
            with col2:
                st.markdown("**Pressão:**")
                st.text(f"BAIXA: {pressao_baixa_faixa}")
                st.text(f"MÉDIA: {pressao_media_faixa}")
                st.text(f"ALTA: {pressao_alta_faixa}")
    
    # Presets de configuração
    st.markdown("---")
    st.subheader("Presets de Configuração")
    st.markdown("Salve e carregue configurações predefinidas para diferentes sistemas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Carregar Preset Coleipa", use_container_width=True):
            # Valores originais de Coleipa com nova fórmula
            preset_coleipa = {
                'area_territorial': 319000,
                'populacao': 1200,
                'numero_ligacoes': 300,
                'comprimento_rede': 3.0,
                'densidade_ramais': 100,
                'vazao_media_normal': 3.17,
                'pressao_media_normal': 5.22,
                'perdas_reais_media': 102.87,
                'volume_consumido_medio': 128.29,
                'percentual_perdas': 44.50,
                'volume_perdido_anual': 37547.55,
                'distancia_lote_medidor': 0.001,
                'pressao_operacao_adequada': 20.0,
                'coeficiente_rede': 18.0,      # Novo valor
                'coeficiente_ligacoes': 0.8,
                'coeficiente_ramais': 25.0
            }
            detector.atualizar_caracteristicas_sistema(preset_coleipa)
            st.success("Preset Coleipa carregado com nova fórmula!")
    
    with col2:
        if st.button("📋 Exportar Configuração Atual", use_container_width=True):
            # Criar DataFrame com configuração atual
            config_atual = pd.DataFrame.from_dict(detector.caracteristicas_sistema, orient='index', columns=['Valor'])
            config_atual.index.name = 'Parâmetro'
            
            # Gerar download
            buffer = io.BytesIO()
            config_atual.to_excel(buffer, index=True)
            buffer.seek(0)
            
            st.download_button(
                label="⬇️ Baixar Configuração (Excel)",
                data=buffer,
                file_name=f"configuracao_sistema_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.ms-excel",
                use_container_width=True
            )
    
    with col3:
        arquivo_config = st.file_uploader("📂 Importar Configuração", type=["xlsx", "csv"], key="config_upload")
        if arquivo_config and st.button("🔼 Carregar Configuração", use_container_width=True):
            try:
                if arquivo_config.name.endswith('.xlsx'):
                    df_config = pd.read_excel(arquivo_config, index_col=0)
                else:
                    df_config = pd.read_csv(arquivo_config, index_col=0)
                
                # Converter para dicionário
                nova_config = df_config['Valor'].to_dict()
                
                # Atualizar sistema
                detector.atualizar_caracteristicas_sistema(nova_config)
                st.success("Configuração importada com sucesso!")
                
            except Exception as e:
                st.error(f"Erro ao importar configuração: {e}")
    
    # Resumo da configuração atual
    st.markdown("---")
    st.subheader("Resumo da Configuração Atual")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 🏗️ Sistema")
        st.text(f"População: {detector.caracteristicas_sistema.get('populacao', 1200):,}")
        st.text(f"Ligações: {detector.caracteristicas_sistema.get('numero_ligacoes', 300):,}")
        st.text(f"Rede: {detector.caracteristicas_sistema.get('comprimento_rede', 3.0):.1f} km")
        st.text(f"Área: {detector.caracteristicas_sistema.get('area_territorial', 319000)/1000:.1f} km²")
    
    with col2:
        st.markdown("##### 📊 IVI")
        ivi_atual = validar_ivi(detector)
        classificacao_ivi = classificar_ivi_manual(ivi_atual)
            
        st.text(f"IVI Atual: {ivi_atual:.2f}")
        st.text(f"Categoria: {classificacao_ivi['categoria_simples']}")
        st.text(f"IPRL: {detector.caracteristicas_sistema.get('iprl', 0.343):.3f}")
        st.text(f"IPRI: {detector.caracteristicas_sistema.get('ipri', 0.021):.3f}")
    
    with col3:
        st.markdown("##### ⚙️ Operação")
        st.text(f"Vazão Média: {detector.caracteristicas_sistema.get('vazao_media_normal', 3.17):.2f} l/s")
        st.text(f"Pressão Média: {detector.caracteristicas_sistema.get('pressao_media_normal', 5.22):.2f} mca")
        st.text(f"Perdas: {detector.caracteristicas_sistema.get('percentual_perdas', 44.50):.1f}%")
        st.text(f"Densidade Ramais: {detector.caracteristicas_sistema.get('densidade_ramais', 100)} ramais/km")
    
    # Resetar sistema para valores padrão
    st.markdown("---")
    st.markdown("##### ⚠️ Resetar Sistema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Limpar Cache", use_container_width=True, help="Limpa o cache para resolver problemas de compatibilidade"):
            try:
                obter_detector.clear()
                st.success("Cache limpo com sucesso!")
                st.info("Atualize a página para ver as mudanças.")
            except Exception as e:
                st.warning(f"Erro ao limpar cache: {e}")
    
    with col2:
        if st.button("🔧 Resetar Sistema Completo", type="primary", use_container_width=True):
            # Limpar cache e forçar recriação
            try:
                obter_detector.clear()
            except:
                pass
            
            # Recriar detector com valores padrão
            detector_novo = DetectorVazamentosColeipa()
            detector_novo._garantir_parametros_ivi()
            
            st.success("Sistema resetado completamente!")
            st.info("Atualize a página manualmente para ver as mudanças.")


# Executar a aplicação
if __name__ == "__main__":
    aplicacao_principal()