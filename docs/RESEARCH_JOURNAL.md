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

## [2026-01-02] Análise de Desempenho (Baseline vs SVM)

**Contexto:** Execução do Step 04 (Modelagem) com SVM Linear e RBF.
**Resultados:** F1-Score ~98.8% e AUC ~0.997.

### Interpretação Crítica dos Resultados:
1.  **Alta Performance Esperada:** A precisão quase perfeita deve-se à presença das variáveis `ADOS` e `ADI-R` no conjunto de treino. Como estes instrumentos constituem a base do critério diagnóstico (Ground Truth), o modelo aprendeu a "regra de corte" clínica.
2.  **Linearidade:** A proximidade entre o desempenho do SVM Linear e RBF indica que a fronteira de decisão é, em grande parte, linear.
3.  **Validação do Pipeline:** O Baseline (Dummy) obteve F1=0.0, confirmando que a performance do SVM é real e discriminativa, e não fruto de artefatos de classe majoritária.

**Decisão Estratégica:** Manteremos o foco em "Replicabilidade Computacional do Diagnóstico Clínico" e "Biomarcadores Fenotípicos". Não tentaremos remover ADOS/ADI-R, pois isso resultaria em perda total de poder preditivo (visto que MRI/QI foram descartados no Step 3).
