from pathlib import Path
import sys

# Adiciona o diretório pai ao path para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from word2vec_utils import Word2VecUtils
from tests.test_data import create_test_data, clean_test_data

def test_word2vec_model():
    """Testa o modelo Word2Vec"""
    
    # Cria dados de teste
    print("Criando dados de teste...")
    create_test_data()
    
    # Inicializa Word2Vec
    print("\nInicializando Word2Vec...")
    w2v = Word2VecUtils(obsidian_vault="F:/pasta estudos/Code Brain")
    
    # Palavras para testar
    test_words = [
        "python",
        "programação",
        "desenvolvimento",
        "web",
        "dados",
        "machine",
        "learning"
    ]
    
    # Testa cada palavra
    print("\nTestando palavras...")
    for word in test_words:
        print(f"\n🔍 Palavra: {word}")
        if w2v.existe_palavra(word):
            similares = w2v.palavras_similares(word)
            print("Palavras similares:")
            for similar in similares:
                print(f"  - {similar['palavra']}: {similar['score']:.4f}")
        else:
            print(f"❌ Palavra não encontrada no modelo")
    
    # Limpa dados de teste (opcional)
    # print("\nLimpando dados de teste...")
    # clean_test_data()

if __name__ == "__main__":
    test_word2vec_model()