import os
from flask import Flask, request, redirect, render_template, url_for
import sqlite3
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURAÇÕES DE CAMINHOS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, '..', 'templates')
STATIC_FOLDER = os.path.join(BASE_DIR, '..', 'static')
DB_PATH = os.path.join(BASE_DIR, '..', 'SQL', 'veterinaria.db')

# Caminhos para o modelo e o LabelEncoder do risco
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_risco_animais.joblib')
ENCODER_RISCO_PATH = os.path.join(BASE_DIR, 'label_encoder_risco.joblib')


# --- INICIALIZAÇÃO DO APP ---
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)

# --- CARREGAMENTO DO MODELO E ENCODER DO RISCO ---
# Use try-except para lidar com erros de carregamento e avisar o usuário
modelo_pipeline = None # Inicializa como None
label_encoder_risco = None # Inicializa como None

try:
    modelo_pipeline = joblib.load(MODEL_PATH)
    print(f"Modelo (pipeline completo) carregado de: {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERRO: Arquivo do modelo não encontrado em: {MODEL_PATH}")
    print("Por favor, execute o script 'train_model.py' para gerar o modelo antes de iniciar o aplicativo.")
    exit(1) # Sai do aplicativo se o modelo principal não for encontrado
except EOFError:
    print(f"ERRO: Arquivo do modelo '{MODEL_PATH}' corrompido ou incompleto (EOFError).")
    print("Por favor, execute o script 'train_model.py' para regenerar o modelo.")
    exit(1)
except Exception as e:
    print(f"ERRO inesperado ao carregar o modelo: {e}")
    exit(1)

try:
    label_encoder_risco = joblib.load(ENCODER_RISCO_PATH)
    print(f"Label Encoder para o risco carregado de: {ENCODER_RISCO_PATH}")
except FileNotFoundError:
    print(f"AVISO: Label Encoder para o risco não encontrado em: {ENCODER_RISCO_PATH}")
    print("As previsões de risco serão exibidas como valores numéricos (0, 1, 2...) em vez de rótulos de texto.")
    # Permite que o app continue, mas a previsão será um número
except EOFError:
    print(f"ERRO: Arquivo do Label Encoder de risco '{ENCODER_RISCO_PATH}' corrompido ou incompleto (EOFError).")
    print("As previsões de risco serão exibidas como valores numéricos.")
except Exception as e:
    print(f"ERRO inesperado ao carregar o Label Encoder de risco: {e}")
    print("As previsões de risco serão exibidas como valores numéricos.")


# --- BANCO DE DADOS ---
def criar_tabela():
    conexao = sqlite3.connect(DB_PATH)
    cursor = conexao.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            raca TEXT NOT NULL,
            sexo TEXT NOT NULL,
            peso REAL NOT NULL,
            idade INTEGER NOT NULL,
            id_veterinario TEXT,
            especie TEXT NOT NULL DEFAULT 'Não Informada',
            data_nascimento TEXT,
            temperamento TEXT,
            historico_doencas TEXT,
            estado_vacinal TEXT DEFAULT 'Não Informado',
            alergias TEXT
        )
    ''')
    conexao.commit()
    conexao.close()

criar_tabela()

# --- ROTAS DO APP ---

# Página principal: lista de pacientes
@app.route('/')
def listar_pacientes_principal():
    conexao = sqlite3.connect(DB_PATH)
    cursor = conexao.cursor()
    cursor.execute('SELECT * FROM pacientes')
    pacientes = cursor.fetchall()
    conexao.close()
    return render_template('Tables_List.html', pacientes=pacientes)

# Formulário para adicionar paciente
@app.route('/adicionar_form')
def exibir_form_adicionar():
    return render_template('Tables.html')

# Adicionar paciente
@app.route('/adicionar', methods=['POST'])
def adicionar():
    dados = coletar_dados_formulario(request)
    inserir_paciente(dados)
    return redirect(url_for('listar_pacientes_principal'))

# Editar paciente
@app.route('/editar/<int:paciente_id>')
def editar_paciente(paciente_id):
    paciente = buscar_paciente_por_id(paciente_id)
    if paciente is None:
        return redirect(url_for('listar_pacientes_principal'))
    return render_template('Edit_Patient.html', paciente=paciente)

# Atualizar paciente
@app.route('/atualizar/<int:paciente_id>', methods=['POST'])
def atualizar_paciente(paciente_id):
    dados = coletar_dados_formulario(request)
    atualizar_paciente_db(paciente_id, dados)
    return redirect(url_for('listar_pacientes_principal'))

# 🔥 Previsão de risco
@app.route('/prever/<int:paciente_id>')
def prever(paciente_id):
    if modelo_pipeline is None:
        return "Erro: Modelo de previsão não foi carregado. Verifique os logs do servidor para detalhes."

    paciente = buscar_paciente_por_id(paciente_id)
    if paciente is None:
        return f'Paciente com ID {paciente_id} não encontrado.'

    # Mapeie os dados do paciente do banco de dados para um dicionário,
    # usando os nomes das colunas como chaves.
    # A ordem das colunas do banco de dados: id (0), nome (1), raca (2), sexo (3), peso (4), idade (5),
    # id_veterinario (6), especie (7), data_nascimento (8), temperamento (9),
    # historico_doencas (10), estado_vacinal (11), alergias (12)
    paciente_dados_dict = {
        'raca': paciente[2],
        'sexo': paciente[3],
        'peso': paciente[4],
        'idade': paciente[5],
        'especie': paciente[7],
        'temperamento': paciente[9],
        'historico_doencas': paciente[10],
        'estado_vacinal': paciente[11],
        'alergias': paciente[12]
    }

    # Crie um DataFrame com as features do paciente.
    # As chaves do dicionário devem corresponder aos nomes das colunas
    # que o ColumnTransformer (dentro do pipeline) espera como entrada.
    df_paciente_input = pd.DataFrame([paciente_dados_dict])

    try:
        # O pipeline (modelo_pipeline) já cuida do pré-processamento e da previsão.
        # Ele transformará as colunas categóricas e numéricas automaticamente.
        risco_codificado = modelo_pipeline.predict(df_paciente_input)[0]

        # Descodificar o risco se o LabelEncoder para o risco foi carregado com sucesso.
        if label_encoder_risco is not None:
            risco = label_encoder_risco.inverse_transform([risco_codificado])[0]
        else:
            risco = str(risco_codificado) # Retorna o valor numérico como string se o encoder não foi carregado

        return f'O risco predito para {paciente[1]} é: {risco}'
    except Exception as e:
        return f"Erro ao fazer a previsão do risco para {paciente[1]}: {e}. Verifique a consistência dos dados de entrada com o que o modelo espera."


# --- FUNÇÕES AUXILIARES (inalteradas) ---

def coletar_dados_formulario(req):
    # Garante que os valores numéricos são convertidos e trata campos opcionais.
    # Usa .get() com valor padrão para campos que podem não estar presentes no formulário,
    # e trata a conversão de string vazia para numérico.
    
    peso_str = req.form.get('peso')
    peso = float(peso_str) if peso_str and peso_str.replace('.', '', 1).isdigit() else 0.0

    idade_str = req.form.get('idade')
    idade = int(idade_str) if idade_str and idade_str.isdigit() else 0

    return (
        req.form['nome'],
        req.form['raca'],
        req.form['sexo'],
        peso,
        idade,
        req.form.get('id_veterinario') or None,
        req.form['especie'],
        req.form.get('data_nascimento') or None,
        req.form.get('temperamento') or None,
        req.form.get('historico_doencas') or None,
        req.form.get('estado_vacinal') or None,
        req.form.get('alergias') or None
    )


def inserir_paciente(dados):
    conexao = sqlite3.connect(DB_PATH)
    cursor = conexao.cursor()
    cursor.execute('''
        INSERT INTO pacientes (
            nome, raca, sexo, peso, idade, id_veterinario,
            especie, data_nascimento, temperamento, historico_doencas, estado_vacinal, alergias
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', dados)
    conexao.commit()
    conexao.close()

def buscar_paciente_por_id(paciente_id):
    conexao = sqlite3.connect(DB_PATH)
    cursor = conexao.cursor()
    cursor.execute('SELECT * FROM pacientes WHERE id = ?', (paciente_id,))
    paciente = cursor.fetchone()
    conexao.close()
    return paciente

def atualizar_paciente_db(paciente_id, dados):
    conexao = sqlite3.connect(DB_PATH)
    cursor = conexao.cursor()
    cursor.execute('''
        UPDATE pacientes SET
            nome = ?, raca = ?, sexo = ?, peso = ?, idade = ?, id_veterinario = ?,
            especie = ?, data_nascimento = ?, temperamento = ?, historico_doencas = ?,
            estado_vacinal = ?, alergias = ?
        WHERE id = ?
    ''', dados + (paciente_id,))
    conexao.commit()
    conexao.close()

# --- EXECUTAR O APP ---
if __name__ == '__main__':
    app.run(debug=True)