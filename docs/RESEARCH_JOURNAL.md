# Diário de Pesquisa - Mestrado UFPR (Mateus Ferro)

Este documento registra insights, decisões estratégicas e achados preliminares obtidos durante o desenvolvimento do pipeline computacional.

## [2025-12-09] Análise Preliminar de Feature Engineering (Rodada 1)

**Contexto:** Execução do Step 03 com *Polynomial Features* e *Wrapper Selection*.
**Nota Importante:** Nesta rodada, a variável `DSM_IV_TR` ainda estava presente, o que causou *Data Leakage*. Contudo, os padrões observados nas demais variáveis revelam dinâmicas fenotípicas importantes.

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