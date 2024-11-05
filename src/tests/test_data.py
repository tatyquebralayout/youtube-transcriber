import os
from pathlib import Path

def create_test_data():
    """Cria dados de teste para treinar o modelo"""
    
    # Define o caminho base do projeto
    base_path = Path(__file__).parent.parent.parent
    
    # Cria pasta de transcrições se não existir
    transcricoes_path = base_path / "transcricoes"
    transcricoes_path.mkdir(exist_ok=True)
    
    # Dados de exemplo
    test_data = [
        {
            "filename": "python_basics.txt",
            "content": """
            Python é uma linguagem de programação versátil e poderosa
            Desenvolvimento de software usando Python é muito produtivo
            Programação orientada a objetos em Python
            Frameworks populares incluem Django e Flask
            Python para ciência de dados e machine learning
            """
        },
        {
            "filename": "web_dev.txt",
            "content": """
            Desenvolvimento web fullstack
            Frontend com HTML, CSS e JavaScript
            Backend com Python e Django
            APIs RESTful e microsserviços
            Banco de dados SQL e NoSQL
            """
        },
        {
            "filename": "data_science.txt",
            "content": """
            Análise de dados com Python
            Pandas para manipulação de dados
            Visualização com matplotlib e seaborn
            Machine learning com scikit-learn
            Deep learning e redes neurais
            """
        }
    ]
    
    # Criar arquivos de teste
    files_created = []
    for data in test_data:
        file_path = transcricoes_path / data["filename"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data["content"])
        files_created.append(str(file_path))
    
    return f"Criados {len(files_created)} arquivos em {transcricoes_path}"

def clean_test_data():
    """Limpa dados de teste"""
    base_path = Path(__file__).parent.parent.parent
    transcricoes_path = base_path / "transcricoes"
    
    if transcricoes_path.exists():
        for file in transcricoes_path.glob("*.txt"):
            file.unlink()
        return "Arquivos de teste removidos"
    
    return "Pasta de transcrições não encontrada"

if __name__ == "__main__":
    # Criar dados de teste
    result = create_test_data()
    print(result)
    
    # Para limpar, descomente a linha abaixo
    # print(clean_test_data())