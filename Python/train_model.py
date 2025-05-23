import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Exemplo de modelo, pode ser outro

# --- CONFIGURAÇÕES DE CAMINHOS ---
# BASE_DIR é o diretório onde este script (train_model.py) está localizado (ex: Hackathon/Python/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# O arquivo CSV filtrado está em 'Hackathon/data/'
# Então, a partir de BASE_DIR (Hackathon/Python/), subimos um nível (..) e entramos em 'data'
FILTERED_CSV_FILE_PATH = os.path.join(BASE_DIR, '..', 'data', 'Pacientes_Caninos_Filtrado.csv')

# Os modelos e encoders serão salvos na mesma pasta deste script (Hackathon/Python/)
MODEL_SAVE_DIR = BASE_DIR
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'modelo_risco_animais.joblib')
ENCODER_RISCO_PATH = os.path.join(MODEL_SAVE_DIR, 'label_encoder_risco.joblib')

print(f"DEBUG: Caminho do CSV filtrado esperado: {FILTERED_CSV_FILE_PATH}")
print(f"DEBUG: Caminho de salvamento do modelo: {MODEL_PATH}")

# --- 1. CARREGAR DADOS DE TREINAMENTO DO ARQUIVO FILTRADO ---
try:
    # Use pd.read_csv para carregar o arquivo CSV filtrado
    # Mantenha o separador (sep=';') e a codificação (encoding='utf-8')
    # que você usou ao SALVAR o arquivo filtrado com o script 'exl.py'.
    df = pd.read_csv(FILTERED_CSV_FILE_PATH, sep=';', encoding='utf-8')
    print(f"Dados carregados com sucesso de: {FILTERED_CSV_FILE_PATH}")
    print("\nPrimeiras 5 linhas do DataFrame filtrado (agora input para o treino):")
    print(df.head())
    print("\nTipos de dados:")
    print(df.dtypes)
    print(f"Classes únicas de risco: {df['risco'].unique()}")

except FileNotFoundError:
    print(f"\nERRO FATAL: Arquivo CSV filtrado NÃO ENCONTRADO em: {FILTERED_CSV_FILE_PATH}")
    print("Por favor, certifique-se de que o arquivo 'Pacientes_Caninos_Filtrado.csv' foi gerado corretamente")
    print("pelo script de filtro (ex: exl.py) e está na pasta 'Hackathon/data/'.")
    exit(1) # Sai do script porque sem os dados não podemos treinar
except Exception as e:
    print(f"\nERRO FATAL ao carregar o arquivo CSV filtrado: {e}")
    print("Verifique o separador (sep=';') e a codificação (encoding='utf-8') na linha pd.read_csv().")
    exit(1) # Sai do script em caso de outro erro de carregamento

# --- VALIDAÇÃO BÁSICA DOS DADOS ---
# Lista das colunas que seu modelo espera como features (input)
# E a coluna 'risco' que é o target (output)
# AJUSTE ESTA LISTA PARA AS COLUNAS REAIS QUE SEU MODELO VAI USAR DO SEU CSV
expected_columns = [
    'raca', 'sexo', 'peso', 'idade', 'especie', 'temperamento',
    'historico_doencas', 'estado_vacinal', 'alergias', 'risco'
]
for col in expected_columns:
    if col not in df.columns:
        print(f"\nERRO FATAL: Coluna '{col}' esperada para o treino NÃO ENCONTRADA no arquivo CSV.")
        print("As colunas encontradas são:", df.columns.tolist())
        print("Por favor, verifique os nomes das colunas no seu arquivo CSV e ajuste a lista 'expected_columns' neste script.")
        exit(1)

# Verifique se a coluna 'risco' (seu target) tem valores
if df['risco'].isnull().any():
    print("\nAVISO: A coluna 'risco' contém valores ausentes. Removendo linhas com risco ausente.")
    df.dropna(subset=['risco'], inplace=True)
    if df.empty:
        print("\nERRO FATAL: Não há dados suficientes após remover linhas com 'risco' ausente para treinar o modelo.")
        exit(1)

# --- 2. SEPARAR FEATURES (X) E TARGET (y) ---
X = df.drop('risco', axis=1) # Todas as colunas, exceto 'risco', são features
y = df['risco'] # 'risco' é o alvo

# --- 3. DEFINIR COLUNAS CATEGÓRICAS E NUMÉRICAS ---
# Estas listas devem conter as colunas do seu DataFrame 'X' que serão processadas
# Confirme que são essas as colunas e os tipos (categórica/numérica)
categorical_features = [
    'raca', 'sexo', 'especie', 'temperamento',
    'historico_doencas', 'estado_vacinal', 'alergias'
]
numeric_features = ['peso', 'idade']

# Validação extra: Remover features que não estão no DataFrame 'X' (se houver alguma inconsistência)
categorical_features = [f for f in categorical_features if f in X.columns]
numeric_features = [f for f in numeric_features if f in X.columns]
print(f"\nFeatures categóricas para processamento: {categorical_features}")
print(f"Features numéricas para processamento: {numeric_features}")

# --- 4. PRÉ-PROCESSAMENTO: ColumnTransformer e Pipeline ---

# Pré-processador para as features de entrada (X)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features), # Mantém as colunas numéricas como estão
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # Codifica categóricas
    ],
    remainder='drop' # Descarta colunas em X que não foram especificadas acima
)

# LabelEncoder para a variável alvo (y), pois ela é categórica
label_encoder_risco = LabelEncoder()
y_encoded = label_encoder_risco.fit_transform(y)

# Validação: Verificar se há classes suficientes no target para estratificação
if len(np.unique(y_encoded)) < 2:
    print(f"\nERRO FATAL: A coluna 'risco' possui apenas {len(np.unique(y_encoded))} classe(s) única(s).")
    print("Um modelo de classificação requer pelo menos duas classes distintas (ex: 'Alto', 'Baixo').")
    exit(1)

# Lógica para ajustar test_size dinamicamente para a divisão treino/teste
min_samples_per_class = np.bincount(y_encoded).min()
num_classes = len(np.unique(y_encoded))
optimal_test_size = 0.2 # Valor padrão para test_size

if min_samples_per_class < 2:
    print(f"\nAVISO: A menor classe de risco tem apenas {min_samples_per_class} amostra(s).")
    print("A estratificação pode ser problemática. Considere adicionar mais dados ou remover 'stratify'.")
    # Ajusta test_size para garantir que o test_set tenha pelo menos uma amostra de cada classe,
    # mas não mais que a metade do dataset.
    calculated_test_size = (num_classes + 1) / len(df) if len(df) > 0 else 0.2
    optimal_test_size = min(max(0.2, calculated_test_size), 0.5) # Limita a 50% do dataset
    print(f"Ajustando test_size para {optimal_test_size:.2f} para tentar garantir estratificação e evitar erros.")
elif len(df) * optimal_test_size < num_classes:
    calculated_test_size = (num_classes + 1) / len(df) if len(df) > 0 else 0.2
    optimal_test_size = min(max(0.2, calculated_test_size), 0.5)
    print(f"Ajustando test_size para {optimal_test_size:.2f} para garantir estratificação em dataset pequeno.")


# --- 5. CRIAR E TREINAR O PIPELINE DO MODELO ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000)) # Seu modelo de classificação
])

# Dividir os dados em treino e teste
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X, y_encoded, test_size=optimal_test_size, random_state=42, stratify=y_encoded
)

print("\nIniciando o treinamento do modelo...")
model_pipeline.fit(X_train, y_train_encoded)
print("Treinamento concluído!")

# Avaliação básica
accuracy = model_pipeline.score(X_test, y_test_encoded)
print(f"Acurácia do modelo no conjunto de teste: {accuracy:.2f}")

# --- 6. SALVAR O MODELO E O LABEL ENCODER DO RISCO ---
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Modelo (pipeline completo) salvo em: {MODEL_PATH}")

joblib.dump(label_encoder_risco, ENCODER_RISCO_PATH)
print(f"Label Encoder do risco salvo em: {ENCODER_RISCO_PATH}")

print("\nProcesso de treino e salvamento concluído com sucesso!")
print("Agora você pode executar seu app.py para usar o modelo treinado com dados de caninos.")
