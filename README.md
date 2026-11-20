# Desafio — Ensemble Learning (classificação e regressão)

Aplicação de Bagging, Boosting, Voting e Stacking com scikit-learn.

- `classificacao/` — **Airline Passenger Satisfaction** (concluído)
- `regressao/` — Taxi Price Prediction (a adicionar)

## Classificação: Airline Passenger Satisfaction

### Estrutura
| Arquivo | O que faz |
|---|---|
| `01_eda_classificacao.py` | EDA: tipos, nulos, outliers, alvo, correlações |
| `02_preprocessamento_classificacao.py` | Limpeza, codificação e escalonamento; salva `data/airline_*_clean.csv.gz` |
| `03_bagging.py` | Bagging, Random Forest, Extra Trees |
| `04_boosting.py` | AdaBoost, Gradient Boosting, HistGradientBoosting |
| `05_voting.py` | Voting hard, soft e soft com pesos |
| `06_stacking.py` | Stacking com meta-modelo LR e HGB |
| `07_comparacao.py` | Tabela e gráfico finais (`comparacao.png`) |
| `utils.py` | Carga dos dados, métricas e treino |

Os arquivos são scripts com células `# %%`: no VS Code dá para rodar célula a célula ou converter em notebook.

### Como executar
```bash
pip install -r requirements.txt
cd classificacao
python 01_eda_classificacao.py
python 02_preprocessamento_classificacao.py
python 03_bagging.py && python 04_boosting.py && python 05_voting.py && python 06_stacking.py
python 07_comparacao.py
```
Os modelos usam o treino completo (103.904 linhas). Se a sua máquina for lenta, ajuste `N_AMOSTRA` em `utils.py` (por exemplo, 30000 para uma amostra estratificada). O Stacking é o mais demorado (~100 s na minha execução).

### Dados e pré-processamento
- Treino com 103.904 linhas e teste com 25.976; 22 variáveis de entrada; sem duplicados. Alvo: 43,3% satisfeitos (teste: 43,9%).
- Único nulo: `Arrival Delay in Minutes` (310 linhas, 0,3%), preenchido com o atraso na partida (correlação de 0,965 entre os dois).
- `Gender`, `Customer Type` e `Type of Travel` viraram 0/1; `Class` é ordinal (Eco < Eco Plus < Business); todas as entradas foram padronizadas com estatísticas do treino (sem vazamento para o teste).
- Os atrasos têm cauda longa (~14% das linhas acima do limite de 1,5×IQR, máximo de ~1.600 min); foram mantidos, pois os modelos de árvore não são sensíveis a isso.
- No EDA, os maiores sinais de satisfação são `Online boarding` (correlação 0,50), `Inflight entertainment` (0,40), classe (Business: 69% satisfeitos contra 19% na Eco) e tipo de viagem (Business: 58% contra 10% em viagens pessoais). Atrasos têm correlação fraca (≈ -0,05).

### Resultados (conjunto de teste completo, 25.976 linhas)
| Modelo | Técnica | Acurácia | F1 | AUC | Treino (s) |
|---|---|---|---|---|---|
| Stacking (meta = HGB) | Stacking | 0,9659 | 0,9605 | 0,9955 | 98,5 |
| Stacking (meta = LR) | Stacking | 0,9652 | 0,9599 | 0,9946 | 98,1 |
| HistGradientBoosting | Boosting | 0,9644 | 0,9588 | 0,9955 | 3,1 |
| Voting soft (pesos 1:2:2) | Voting | 0,9641 | 0,9585 | 0,9941 | 21,7 |
| Random Forest (200) | Bagging | 0,9638 | 0,9581 | 0,9940 | 19,2 |
| Voting hard | Voting | 0,9631 | 0,9573 | — | 21,1 |
| Voting soft | Voting | 0,9630 | 0,9573 | 0,9922 | 20,7 |
| Extra Trees (200) | Bagging | 0,9617 | 0,9556 | 0,9934 | 13,6 |
| Bagging (100 árvores) | Bagging | 0,9612 | 0,9552 | 0,9935 | 37,3 |
| Gradient Boosting (200) | Boosting | 0,9533 | 0,9460 | 0,9914 | 29,8 |
| Árvore única (base) | — | 0,9469 | 0,9396 | 0,9462 | 0,6 |
| AdaBoost (200, depth=2) | Boosting | 0,9313 | 0,9209 | 0,9836 | 23,0 |
| Regressão logística (base) | — | 0,8715 | 0,8507 | 0,9255 | 0,2 |

![Comparação](classificacao/comparacao.png)

### Análise
- **Ensembles valem a pena frente a um modelo único:** a árvore única chega a 94,7% de acurácia (AUC 0,946) e os melhores ensembles a ~96,4–96,6% (AUC ≈ 0,994–0,996). A regressão logística, linear, fica em 87,2%: há interações e não linearidades importantes entre as notas de serviço, classe e tipo de viagem.
- **Bagging:** reduz a variância da árvore (+1,4 a 1,7 ponto de acurácia e AUC de 0,946 para ≈ 0,994). Random Forest e Extra Trees foram mais rápidos que o Bagging simples e iguais em desempenho, porque sorteiam também as variáveis em cada divisão.
- **Boosting:** o HistGradientBoosting foi o melhor modelo individual (96,44%) e o mais barato (3 s). Já AdaBoost com árvores rasas (93,1%) e Gradient Boosting com 200 árvores de profundidade 3 (95,3%) ficaram atrás: o problema pede mais capacidade (mais árvores/folhas), que o HGB tem por padrão. Não ajustei hiperparâmetros, então o resultado reflete essas configurações.
- **Voting:** o soft com pesos (96,41%) ficou praticamente igual ao melhor membro (HGB, 96,44%); o voting simples (96,30%) até perdeu um pouco, porque a regressão logística, bem mais fraca, entra com o mesmo peso. Voting ajuda quando os modelos são parecidos em qualidade e erram de formas diferentes, o que aqui só vale para RF e HGB.
- **Stacking:** foi o melhor (96,59% de acurácia, F1 0,9605), mas por pouco e com custo alto: ~100 s contra 3 s do HGB.
- **Cuidado com a interpretação:** com 25.976 exemplos de teste, o erro-padrão da acurácia é de cerca de 0,11 ponto percentual. Os seis primeiros modelos ficam entre 96,30% e 96,59%, então a diferença entre eles **não é estatisticamente clara** (uso de um único split, sem repetições com várias sementes ou validação cruzada). O que é robusto é a distância para a árvore única e a regressão logística. Em produção, o HistGradientBoosting seria a escolha custo-benefício.
- **Variáveis mais importantes (Random Forest):** `Online boarding` (17,7%), `Inflight wifi service` (14,1%), `Class` (9,6%) e `Type of Travel` (9,4%). É importância preditiva, não causa: nota alta de embarque online pode refletir o perfil do passageiro.
- **Limitações:** sem ajuste de hiperparâmetros, sem validação cruzada e com um único split; a nota 0 ("não se aplica") foi tratada como valor numérico.
