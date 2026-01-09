
# * Used to manage translations for plot labels in Portuguese (Brazil)

# * Belongs to Step 2 - Used for grouping features in EDA plots (step 2), ordering by missingness
# Dictionary: { 'Substring to find': 'Readable Group Name' }
# Order matters. More specific patterns should come first if needed.
GROUP_PATTERNS = {
    # Families groups
    'WISC_IV': 'Bateria WISC-IV (QI)',                              # .1
    'VINELAND': 'Escalas Vineland (Comportamento Adaptativo)',      # .10
    'SRS_': 'Sub-escalas SRS (Responsividade Social)',              # .6
    'ADI_R_': 'Sub-escores ADI-R',                                  # .14
    'ADI_RRB': 'Comportamento Repetitivo ADI-R',                    # .15
    'ADOS_': 'Sub-escores ADOS',                                    # .13
    'SCQ_': 'Questionário de Comunicação Social (SCQ)',             # .7

    # Quality Control (QC) Metrics
    'qc_.*notes': 'Anotações de Controle de Qualidade (Texto)',     # .8
    'qc_.*rater': 'Avaliadores de Qualidade (QC Rater)',
    'anat_': 'Métricas Anatômicas (MRI)',
    'func_': 'Métricas Funcionais (fMRI)',

    # Individuals (Direct Translation)
    'AQ_TOTAL': 'Quociente de Autismo Total (AQ)',                  # .2
    'COMORBIDITY': 'Indicadores de Comorbidade',                    # .3
    'AGE_AT_MPRAGE': 'Idade na Ressonância',                        # .4
    'OFF_STIMULANTS_AT_SCAN': 'Sem Estimulantes no Scan',           # .5
    'MEDICATION_NAME': 'Nome da Medicação',                         # .9
    'BMI': 'Índice de Massa Corporal (IMC)',                        # .11
    'HANDEDNESS_SCORES': 'Escores de Lateralidade',                 # .12
    'HANDEDNESS_CATEGORY': 'Categoria de Lateralidade',             # .16
    'CURRENT_MED_STATUS': 'Status de Medicação Atual',              # .17
    'VIQ_TEST_TYPE': 'Tipo de Teste QI Verbal',                     # .18
    'PIQ_TEST_TYPE': 'Tipo de Teste QI Performance',                # .19
    'VIQ': 'QI Verbal',                                             # .20
}

# * Plot labels translations
PLOT_LABELS = {
    'missing_title': 'Top 20 Grupos de Features com Dados Ausentes',
    'missing_xlabel': 'Porcentagem Média de Valores Ausentes (%)',

    'class_balance_title': 'Distribuição das Classes (Diagnóstico)',
    'class_balance_ylabel': 'Porcentagem do Dataset (%)',
    'class_labels': {1: 'Autismo (TEA)', 0: 'Controle (CT)', 2: 'Controle (CT)'}, # Handle 1/0 or 1/2

    'stratified_title': 'Dados Ausentes Estratificados por Diagnóstico',
    'stratified_ylabel': 'Porcentagem de Ausência (%)',

    'corr_pearson_title': 'Matriz de Correlação de Pearson',
    'corr_spearman_title': 'Matriz de Correlação de Spearman',

    'pps_full_title': 'Matriz de Poder Preditivo (PPS) - Completa',
    'pps_target_title': 'Top Preditores para {}', # Format with target name
    'pps_xlabel': 'Target (Alvo)',
    'pps_ylabel': 'Feature (Variável)',
    'pps_cbar': 'Predictive Power Score (PPS)',

    'final_features_corr_title': 'Correlação entre Features Selecionadas (Final)',
    
    'benchmark_title': 'Comparação de Desempenho dos Modelos (F1-Score Médio)',
    'benchmark_xlabel': 'Média do F1-Score (Validação Cruzada 10-Fold)',
}
