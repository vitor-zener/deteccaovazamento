# 💧 Sistema de Detecção de Vazamentos - Coleipa

Sistema híbrido **Fuzzy-Bayesiano** para detecção de vazamentos em redes de abastecimento de água, desenvolvido com base nos dados reais do bairro **Coleipa**, localizado em Santa Bárbara do Pará-PA.

---

## 📌 Descrição

Este projeto implementa um sistema de apoio à tomada de decisão que:

- Analisa séries temporais de vazão e pressão;
- Emprega lógica fuzzy para inferência de risco de vazamento;
- Treina um classificador Bayesiano para reconhecimento de padrões;
- Gera mapas de calor com base no IVI (Índice de Vazamento na Infraestrutura);
- Simula comportamento de rede com e sem vazamento;
- Exibe relatórios técnicos detalhados conforme metodologia do Banco Mundial.

---

## 🧠 Tecnologias Utilizadas

- [Python 3.8+](https://www.python.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Scikit-fuzzy](https://pythonhosted.org/scikit-fuzzy/)
- [Seaborn](https://seaborn.pydata.org/)

---

## ⚙️ Instalação

Clone o repositório e instale os pacotes:

```bash
git clone https://github.com/seuusuario/detector-vazamentos-coleipa.git
cd detector-vazamentos-coleipa
pip install -r requirements.txt
```

Ou instale os pacotes individualmente:

```bash
pip install numpy pandas matplotlib scikit-learn scikit-fuzzy seaborn openpyxl
```

---

## ▶️ Execução

Basta executar o script principal:

```bash
python detector_coleipa.py
```

Será apresentado um menu com as seguintes opções:

1. Visualizar dados do monitoramento  
2. Criar e visualizar sistema fuzzy  
3. Treinar modelo Bayesiano  
4. Gerar mapa de calor IVI  
5. Simular série temporal  
6. Analisar caso específico  
7. Gerar relatório completo  
8. Análise completa  
9. Salvar template de dados (.csv/.xlsx)  
10. Atualizar características do sistema  

---

## 📁 Estrutura Esperada de Arquivo de Entrada

Se desejar usar seus próprios dados, forneça um arquivo `.xlsx` ou `.csv` com as colunas:

```
hora, vazao_dia1, pressao_dia1, vazao_dia2, pressao_dia2, vazao_dia3, pressao_dia3
```

Você pode gerar esse modelo com a opção `9` do menu: **Salvar template de dados**.

---

## 📊 Exemplos de Análise

- **Mapa de calor IVI** com risco de vazamento para diferentes combinações de vazão e pressão.
- **Série temporal** simulando vazamento progressivo.
- **Matriz de confusão** do classificador Bayesiano.
- **Relatório técnico automatizado** com diagnósticos e recomendações.

---

## 📍 Local de Estudo

📌 Bairro Coleipa  
📍 Município de Santa Bárbara do Pará – PA  
🧪 Dados extraídos de estudo de campo e literatura técnica especializada.

---

## 📄 Licença

Este projeto é de uso acadêmico e educacional. Consulte a instituição responsável antes de qualquer uso comercial.

---

## ✍️ Autor

Desenvolvido por Vitor Hugo Pereira de Souza
Mestrando em Engenharia Industrial – UFPA  
Contato: [zener.vitor@gmail.com](mailto:zener.vitor@gmail.com)
