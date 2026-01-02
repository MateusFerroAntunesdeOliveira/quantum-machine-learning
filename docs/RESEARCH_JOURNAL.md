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

