# Diário de Pesquisa - Mestrado UFPR (Mateus Ferro)

Este documento registra achados importantes e decisões estratégicas obtidos durante o desenvolvimento do projeto.

## [2025-12-09] Análise Preliminar de Feature Engineering

**Contexto:** Execução do Step 03 com *Polynomial Features* e *Wrapper Selection*.
**Nota Importante:** Nesta rodada, a variável `DSM_IV_TR` ainda estava presente, o que pode causar futuramente um *Data Leakage*. Contudo, os padrões observados nas demais variáveis revelam dinâmicas fenotípicas importantes.

### Achados Principais:

1.  **Não-Linearidade do Fenótipo**
    * **Observação:** Das 15 features selecionadas, 6 eram polinomiais ou interações (ex: `SRS_RAW_TOTAL AGE_AT_SCAN`).
    * **Interpretação Clínica:** A relação entre sintomas e diagnóstico não é puramente linear. A sobrevivência da interação `Responsividade Social * Idade` sugere que a severidade dos sintomas sociais escala ou muda de peso diagnóstico conforme o indivíduo envelhece.
    * **Impacto na Dissertação:** Justificativa robusta para o uso de Engenharia de Features Polinomiais em vez de apenas dados brutos.

2.  **Predominância do Domínio Clínico**
    * **Observação:** Quase todas as features sobreviventes derivam de ADOS, ADI-R e SRS.
    * **Interpretação:** O modelo tende a aprender/replicar as "regras de pontuação" dos testes padrão-ouro.
    * **Conclusão:** Os dados fenotípicos comportamentais possuem sinal muito mais forte que os demográficos isolados.

3.  **Irrelevância da Anatomia e QI Isolados**
    * **Observação:** Nenhuma métrica de qualidade anatômica (`anat_cnr`, etc.) ou funcional sobreviveu ao filtro. O `FIQ` (QI Total) também foi descartado como preditor isolado.
    * **Interpretação:** Para classificação neste dataset específico, a qualidade da imagem ou o QI sozinho não discriminam TEA de Controle com a mesma eficácia que os escores comportamentais. O QI serve mais como modulador (em interações) do que como preditor direto.

---

## [2025-12-10] Estudo sobre Modelagem (Step 4) e Otimização (Step 5)

Como decidimos separar a Otimização (Step 5) da Modelagem (Step 4), neste passo implementaremos uma Validação Cruzada Estratificada (Stratified K-Fold). Isso funcionará como o "Loop Externo" do NCV. No Step 5, injetaremos o Optuna dentro desse processo para completar o NCV.

---

## [2026-01-02] Benchmarking "Big Six" (Step 04)

**Contexto:** Comparação sistemática de 7 algoritmos (+ Baseline) usando Nested CV (k=10).
**Features:** 14 variáveis selecionadas (baseadas em ADOS/ADI-R).

### Resultados Principais:
1.  **Liderança do Gradient Boosting:**
    * **LightGBM:** F1 = 99.44% (AUC = 0.9996)
    * **XGBoost:** F1 = 99.35% (AUC = 0.9991)
    * **Interpretação:** A natureza hierárquica e baseada em regras de corte (thresholds) dos algoritmos de árvore se alinha perfeitamente com a lógica de pontuação dos testes clínicos (ex: ADOS cutoff).

2.  **KNN superou SVM:**
    * O KNN (k=5) obteve F1=99.07%, superando o SVM RBF (98.79%). Isso indica que a topologia dos dados favorece abordagens baseadas em densidade local/vizinhança em vez de margens globais.

3.  **Linearidade Imperfeita:**
    * O SVM Linear teve o menor desempenho (98.50%) entre os modelos inteligentes, confirmando que a fronteira de decisão possui não-linearidades sutis que justificam o uso de modelos mais complexos.

### Decisão Estratégica para Otimização (Step 5):
Selecionamos o **LightGBM** e o **XGBoost** para a etapa de Otimização de Hiperparâmetros (Optuna).
* **Motivo:** Maior performance bruta.
* **XAI:** Ambos são compatíveis com TreeSHAP, permitindo explicabilidade detalhada no Step 6.
* **Plano:** Faremos um ajuste fino para tentar extrair o "último milésimo" de performance e garantir estabilidade, embora o ganho marginal esperado seja baixo dado o teto de 99.4%.

---

## [2026-01-03] Otimização de Hiperparâmetros (Optuna)

**Contexto:** Refinamento dos modelos LightGBM e XGBoost (Top 2 do Step 4).
**Resultado:** Otimização concluída com sucesso.

### O Campeão: LightGBM
* **F1-Score Final:** 99.54% (+0.10% sobre o baseline).
* **Configuração Vencedora:** Árvores rasas (`max_depth=4`) com regularização balanceada (`reg_alpha=0.42`, `reg_lambda=0.61`).
* **Significado:** O modelo atingiu o teto de performance possível com este conjunto de features, priorizando generalização (árvores curtas) em vez de memorização.

**Próximo Passo:** Utilizar este modelo configurado para gerar explicações globais e locais via SHAP (Step 6).

---

## [2026-01-06] Análise de Explicabilidade (SHAP) - Parte 1

**Contexto:** Interpretação dos gráficos *Beeswarm* e *Bar Plot* do modelo LightGBM.

### Achados Críticos:

1.  **Validação da Hipótese Polinomial:**
    * A feature mais impactante do modelo é `SRS_RAW_TOTAL^2` (SRS ao quadrado), superando o próprio `ADOS_TOTAL`.
    * **Significado:** A relação entre responsividade social e diagnóstico é exponencial. Valores extremos de SRS pesam desproporcionalmente mais para a classificação de TEA do que valores medianos. O uso de features polinomiais foi determinante para capturar essa nuance.

2.  **Viés de Processo (Reliability):**
    * A variável `ADOS_RSRCH_RELIABLE` aparece como 3ª mais importante. Valores altos (confiável) predizem autismo.
    * **Interpretação:** Provável artefato do dataset ABIDE, onde casos positivos passaram por validação mais rigorosa que controles. Isso deve ser discutido como limitação de "Data Quality" na dissertação, e não como achado fenotípico.

3.  **Confirmação do Padrão-Ouro:**
    * `ADOS_TOTAL` apresenta uma separação linear clara (Azul/Esquerda vs Vermelho/Direita), atuando como o "corte" principal de decisão.

4.  **Assinatura de Alto Funcionamento:**
    * `ADOS_MODULE` (Módulos 3/4 para fluentes) tem correlação positiva com o diagnóstico no modelo. Isso sugere que o classificador é particularmente sensível ao fenótipo de autismo com fluência verbal (antigo Asperger/Alto Funcionamento) presente no ABIDE.

---

## [2026-01-06] Análise de Explicabilidade (SHAP) - Parte 2 (Dependence Plots)

**Contexto:** Análise detalhada das interações não-lineares.

### Achados Fenomenológicos:

1.  **O Efeito de Saturação do SRS (`SRS_RAW_TOTAL^2`):**
    * A curva de dependência revela um comportamento sigmoide. Scores baixos protegem, scores médios (50-70) causam um salto abrupto no risco, e scores altos (>100) atingem um platô.
    * Isso confirma que a decisão de usar polinômios quadráticos foi vital para capturar esse comportamento de "gatilho" que modelos lineares falhariam em modelar.

2.  **A Zona Limítrofe do ADOS (Interação com ADI-R):**
    * Identificou-se um "cutoff" claro ao redor do score 11 no `ADOS_TOTAL`.
    * **Desempate Inteligente:** Em scores limítrofes (ex: 11), observa-se uma dispersão vertical explicada pela interação com o `ADI-R`. O modelo utiliza o histórico social do paciente (ADI-R) para decidir a classificação quando a observação direta (ADOS) está na zona cinzenta, mimetizando o raciocínio clínico multi-instrumental.

3.  **Auditoria de Viés (`ADOS_RSRCH_RELIABLE`):**
    * Confirmou-se que a flag de confiabilidade atua como um proxy forte para o diagnóstico, indicando um viés procedimental na coleta do ABIDE (pacientes TEA são mais auditados que controles). Este achado será reportado como limitação do dataset.
