
# * Used to manage translations for plot labels in Portuguese (Brazil)

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
    'pps_cbar': 'Predictive Power Score (PPS)'
}
