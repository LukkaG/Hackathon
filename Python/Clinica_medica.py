import sqlite3
import pandas as pd
import os

current_dir = os.path.dirname(__file__)
# Este é o caminho COMPLETO e CORRETO para o seu arquivo SQL
sql_file_path = os.path.join(current_dir, '..', 'SQL', 'paciente_sqlite.sql')

def gerenciar_db_paciente(sql_script_path, db_file_name="meus_pacientes.db"): # Renomeei o parâmetro para clareza
    """
    Cria ou conecta a um banco de dados SQLite e executa um script SQL.
    Em seguida, visualiza os dados da tabela Paciente.

    Args:
        sql_script_path (str): Caminho COMPLETO para o arquivo SQL (limpo para SQLite).
        db_file_name (str): Nome do arquivo do banco de dados SQLite a ser criado/conectado.
    """
    conn = None
    try:
        # Conecta ao banco de dados SQLite. Se o arquivo não existir, ele será criado.
        conn = sqlite3.connect(db_file_name)
        cursor = conn.cursor()

        # Lê o script SQL do arquivo
        with open(sql_script_path, 'r', encoding='utf-8') as f: # Usa o caminho COMPLETO aqui
            sql_script = f.read()

        # Divide o script em instruções individuais e as executa
        cursor.executescript(sql_script)
        conn.commit()
        print(f"Banco de dados '{db_file_name}' criado/atualizado com sucesso usando '{sql_script_path}'.")

        # --- Visualização dos dados ---
        print("\nVisualizando os primeiros 5 registros da tabela 'Paciente':")
        df_pacientes = pd.read_sql_query("SELECT * FROM Paciente LIMIT 5;", conn)
        print(df_pacientes.to_string())

        print("\nNúmero total de registros na tabela 'Paciente':")
        cursor.execute("SELECT COUNT(*) FROM Paciente;")
        count = cursor.fetchone()[0]
        print(f"Total: {count} registros.")

        # Exemplo de consulta específica: pacientes caninos
        print("\nPacientes da espécie 'Canina':")
        df_caninos = pd.read_sql_query("SELECT NM_Animal, DS_Raca, DS_Desfecho_Inferido FROM Paciente WHERE DS_Especie = 'Canina';", conn)
        print(df_caninos.to_string())

    except sqlite3.Error as e:
        print(f"Erro ao interagir com o banco de dados: {e}")
    except FileNotFoundError:
        print(f"Erro: O arquivo SQL '{sql_script_path}' não foi encontrado. Verifique o caminho.") # Mensagem mais útil
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
    finally:
        if conn:
            conn.close()
            print(f"\nConexão com '{db_file_name}' fechada.")

# Não precisamos mais de 'sql_limpo_file' com apenas o nome,
# pois 'sql_file_path' já tem o caminho completo.

# Nome do arquivo do banco de dados SQLite que será criado
db_file = 'meus_pacientes.db'

# Executar a função, passando o caminho COMPLETO
gerenciar_db_paciente(sql_file_path, db_file)