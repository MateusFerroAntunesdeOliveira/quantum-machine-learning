import os

import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import get_data_adhd as adhd

base_data_path = "../data/download_folder/NeuroIMAGE"

def main():
    free_control_csv, free_ad_hd_csv = adhd.get_data()
    free_participants = pd.read_csv(free_control_csv)

    for idx, row in free_participants.iterrows():
        participant_id = row['participant_id']

        # Formatting participant_id to match folder structure (e.g., sub-0000001)
        participant_str = str(participant_id).zfill(7)
        participant_folder = f"sub-{participant_str}"
        participant_path = os.path.join(base_data_path, participant_folder)

        if not os.path.isdir(participant_path):
            print(f"Folder for participant {participant_folder} not found. Skipping...")
            continue

        print(f"\nProcessing participant folder: {participant_path}")
        anat_path = os.path.join(participant_path, "ses-1", "anat", f"{participant_folder}_ses-1_run-1_T1w.nii.gz")
        print(f"Loading T1w image from {anat_path}")

        if os.path.isfile(anat_path):
            nii_img = nib.load(anat_path)
            nii_data = nii_img.get_fdata()
            print(f"Participant {participant_folder}: Image shape {nii_data.shape}")

            # preprocessed_data = preprocess(nii_data)

            slice_index = nii_data.shape[2] // 2
            plt.imshow(nii_data[:, :, slice_index], cmap="gray")
            plt.title(f"{participant_folder} - Axial Slice")
            plt.axis("off")
            plt.show()
        else:
            print(f"T1w file not found for participant {participant_folder}.")

        if idx >= 1:
            break


if __name__ == "__main__":
    main()


# =========================================================
# Próximos passos e ideias para o pipeline de treinamento:
# =========================================================
#
# 1. Data Labeling:
#    - Definir os rótulos (labels) para o modelo.
#      Exemplo: 1 para participantes com ADHD (tanto 'ADHD-Combined' quanto 'ADHD-Hyperactive/Impulsive')
#               0 para controles ('Typically Developing Children').
#
# 2. Pré-processamento das Imagens:
#    - Realizar procedimentos como normalização, remoção de ruído e possivelmente segmentação.
#    - Considerar técnicas de skull-stripping (brain extraction) para isolar o cérebro.
#       - "It's a crucial step in brain image analysis and is used in many neuroimaging pipeline."
#    - Redimensionamento das imagens para um formato padrão, se necessário.
#
# 3. Extração de Features:
#    - Abordagem 1: Utilizar técnicas de deep learning (por exemplo, redes convolucionais) para extrair features relevantes.
#    - Abordagem 2: Aplicar técnicas clássicas, como PCA, para redução da dimensionalidade, mantendo as variâncias mais importantes.
#    - Abordagem 3: Combinar métodos de extração manual de características baseadas em conhecimento neurocientífico.
#
# 4. Treinamento e Validação do Modelo:
#    - Dividir os dados em conjuntos de treinamento e teste.
#    - Experimentar modelos de Machine Learning (SVM, Random Forest, etc.) e/ou redes neurais profundas.
#    - Avaliar a performance utilizando métricas como acurácia, F1-score, ROC-AUC, etc.
#
# 5. Abordagem Quantum Machine Learning (QML):
#    - Após a implementação do pipeline clássico, tentar utilizar frameworks como Qiskit para aplicar QML aos dados.
#
# 6. Interpretação e Validação:
#    - Analisar os resultados e identificar quais características anatômicas podem estar correlacionadas com o diagnóstico.
#    - Considerar a utilização de técnicas de visualização de features e análise estatística para interpretar os achados.
