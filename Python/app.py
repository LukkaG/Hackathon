import os
from flask import Flask, request, redirect, render_template, url_for
import sqlite3

# --- DEFINIÇÕES DE CAMINHOS ---
# Define o caminho base do projeto (onde app.py está sendo executado)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define o caminho para a pasta 'templates' (um nível acima da pasta 'Python')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, '..', 'templates')

# Define o caminho para a pasta 'static' (um nível acima da pasta 'Python')
STATIC_FOLDER = os.path.join(BASE_DIR, '..', 'static')

# Define o caminho completo para o arquivo do banco de dados SQLite
# (um nível acima da pasta 'Python', dentro da pasta 'SQL')
DB_PATH = os.path.join(BASE_DIR, '..', 'SQL', 'veterinaria.db')

# Imprime o caminho do banco de dados no console para verificação
print(f"Caminho do Banco de Dados: {DB_PATH}")

# --- INICIALIZAÇÃO DA APLICAÇÃO FLASK ---
# Cria a instância da aplicação Flask, especificando os caminhos para templates e arquivos estáticos
app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)

# --- FUNÇÃO PARA CRIAR A TABELA NO BANCO DE DADOS ---
def criar_tabela():
    conexao = sqlite3.connect(DB_PATH) # Conecta-se ao banco de dados
    cursor = conexao.cursor()          # Cria um cursor para executar comandos SQL

    # ATENÇÃO: Esta linha foi usada para depuração.
    # Remova-a ou comente-a para que os dados persistam entre as reinicializações do servidor.
    # cursor.execute('DROP TABLE IF EXISTS pacientes')

    # Cria a tabela 'pacientes' se ela ainda não existir
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            raca TEXT NOT NULL,
            sexo TEXT NOT NULL,
            peso REAL NOT NULL,
            idade INTEGER NOT NULL,
            id_veterinario TEXT
        )
    ''')
    conexao.commit() # Confirma as alterações no banco de dados
    conexao.close()  # Fecha a conexão com o banco de dados

# Chama a função 'criar_tabela' assim que o script é executado.
# Isso garante que a tabela 'pacientes' exista antes de qualquer operação de leitura ou escrita.
criar_tabela()

# --- ROTAS DA APLICAÇÃO WEB ---

@app.route('/')
def listar_pacientes_principal():
    """
    Rota principal que agora renderiza a lista de pacientes.
    """
    conexao = sqlite3.connect(DB_PATH)
    cursor = conexao.cursor()
    cursor.execute('SELECT * FROM pacientes')
    pacientes = cursor.fetchall()
    conexao.close()
    return render_template('Tables_List.html', pacientes=pacientes)

@app.route('/adicionar_form')
def exibir_form_adicionar():
    """
    Rota para exibir o formulário de adição de novos pacientes.
    """
    return render_template('Tables.html')

@app.route('/adicionar', methods=['POST'])
def adicionar():
    """
    Rota que processa os dados do formulário de adição de pacientes (método POST).
    Insere um novo paciente no banco de dados.
    """
    nome = request.form['nome']
    raca = request.form['raca']
    sexo = request.form['sexo']
    peso = float(request.form['peso'])
    idade = int(request.form['idade'])
    id_veterinario = request.form.get('id_veterinario') or None

    conexao = sqlite3.connect(DB_PATH)
    cursor = conexao.cursor()
    # Ordem das colunas na query: (nome, raca, sexo, peso, idade, id_veterinario)
    # Ordem dos valores no tuple: (nome, raca, sexo, peso, idade, id_veterinario)
    cursor.execute('''
        INSERT INTO pacientes (nome, raca, sexo, peso, idade, id_veterinario)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (nome, raca, sexo, peso, idade, id_veterinario))
    conexao.commit()
    conexao.close()

    # Após adicionar, redireciona para a rota principal (lista de pacientes)
    return redirect(url_for('listar_pacientes_principal'))

# --- INÍCIO DA APLICAÇÃO QUANDO O SCRIPT É EXECUTADO DIRETAMENTE ---
if __name__ == '__main__':
    # Inicia o servidor de desenvolvimento do Flask.
    # 'debug=True' habilita o modo de depuração, que recarrega o servidor automaticamente
    # a cada alteração no código e fornece mensagens de erro detalhadas.
    app.run(debug=True)