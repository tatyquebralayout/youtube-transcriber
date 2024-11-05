import os
from pathlib import Path
from typing import List, Optional, Dict
from gensim.models import Word2Vec
from rich.console import Console
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import logging

# Download necessário do NLTK
nltk.download('punkt')

console = Console()
logger = logging.getLogger(__name__)

class Word2VecUtils:
    def __init__(self, 
                 obsidian_vault: str = "F:/pasta estudos/Code Brain",
                 model_dir: str = "models/word2vec",
                 model_name: str = "model.bin",
                 vector_size: int = 100):
        # Configura caminhos
        self.vault_path = Path(obsidian_vault)
        self.transcricoes_path = self.vault_path / "Transcrições"
        self.base_path = Path(__file__).parent.parent
        self.model_path = self.base_path / model_dir / model_name
        
        self.vector_size = vector_size
        self.model = None
        
        console.print(f"📂 Vault Obsidian: {self.vault_path}")
        console.print(f"📂 Pasta de Transcrições: {self.transcricoes_path}")
        
        # Força treinamento inicial
        self.train_initial_model()
        
    def prepare_sentences(self) -> List[List[str]]:
        """Prepara sentenças das transcrições do Obsidian"""
        sentences = []
        
        if not self.transcricoes_path.exists():
            console.print("❌ Pasta de transcrições não encontrada!", style="red")
            return sentences
        
        # Lista todos os arquivos .md e .txt
        files = list(self.transcricoes_path.glob("*.md"))
        files.extend(self.transcricoes_path.glob("*.txt"))
        
        if not files:
            console.print("❌ Nenhuma transcrição encontrada!", style="red")
            return sentences
        
        # Processa cada arquivo
        for file in files:
            console.print(f"📄 Processando: {file.name}")
            try:
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()
                    sentences.extend([sentence.split() for sentence in text.split("\n")])
            except Exception as e:
                console.print(f"❌ Erro ao ler {file.name}: {e}", style="red")
        
        return sentences

    def train_initial_model(self):
        """Treina o modelo inicial de Word2Vec"""
        sentences = self.prepare_sentences()
        if not sentences:
            console.print("❌ Nenhuma sentença preparada para treinamento!", style="red")
            return
        
        try:
            self.model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=5, min_count=1, workers=4)
            self.model.save(self.model_path)
            console.print("✅ Modelo Word2Vec treinado e salvo!", style="green")
        except Exception as e:
            console.print(f"❌ Erro ao treinar o modelo Word2Vec: {e}", style="red")

    def treinar_word2vec(self):
        """Treina o modelo Word2Vec e analisa as transcrições."""
        console.print("\n🔍 Analisando transcrições...", style="blue")
        
        # Carregar as transcrições
        transcricoes = []
        palavras_frequentes = Counter()
        temas = set()
        
        for arquivo in os.listdir(self.transcricoes_path):
            if arquivo.endswith(".txt"):
                try:
                    with open(os.path.join(self.transcricoes_path, arquivo), "r", encoding="utf-8") as f:
                        texto = f.read()
                        transcricoes.append(texto)
                        
                        # Análise do conteúdo
                        palavras = texto.lower().split()
                        palavras_frequentes.update(palavras)
                        
                        # Identifica possíveis temas
                        for palavra, freq in Counter(palavras).items():
                            if freq > 3 and len(palavra) > 3:  # Palavras significativas
                                temas.add(palavra)
                                
                    console.print(f"✅ Analisado: {arquivo}", style="green")
                except Exception as e:
                    console.print(f"❌ Erro ao ler {arquivo}: {e}", style="red")

        # Tokenizar as transcrições
        sentencas = []
        for transcricao in transcricoes:
            sentencas.extend([sentenca.split() for sentenca in transcricao.split("\n")])

        # Criar e treinar o modelo
        try:
            model = Word2Vec(
                sentences=sentencas, 
                vector_size=100, 
                window=5, 
                min_count=1, 
                workers=4
            )
            
            # Salvar o modelo
            model.save("word2vec.model")
            console.print("\n✅ Modelo treinado e salvo!", style="green")
            
            # Mostrar análise
            console.print("\n📊 Análise do Conteúdo:", style="bold blue")
            
            # Palavras mais frequentes
            console.print("\n🔤 Palavras mais frequentes:", style="cyan")
            for palavra, freq in palavras_frequentes.most_common(20):
                if len(palavra) > 3:  # Ignorar palavras muito curtas
                    console.print(f"  • {palavra}: {freq}", style="green")
            
            # Temas identificados
            console.print("\n🏷️ Possíveis temas:", style="cyan")
            for tema in sorted(list(temas))[:15]:
                console.print(f"  • {tema}", style="green")
            
            # Exemplos de relações
            console.print("\n🔗 Exemplos de palavras relacionadas:", style="cyan")
            for palavra in sorted(list(temas))[:5]:
                try:
                    similares = model.wv.most_similar(palavra)
                    console.print(f"\n  Relacionadas a '{palavra}':", style="yellow")
                    for similar, score in similares[:3]:
                        console.print(f"   - {similar}: {score:.2f}", style="green")
                except:
                    continue
                    
        except Exception as e:
            console.print(f"\n❌ Erro ao treinar modelo: {e}", style="red")
            return None
        
        return model

    def load_model(self) -> bool:
        """Carrega o modelo Word2Vec existente"""
        try:
            if self.model_path.exists():
                self.model = Word2Vec.load(str(self.model_path))
                console.print(f"✅ Modelo carregado de: {self.model_path}", style="green")
                return True
            return False
        except Exception as e:
            console.print(f"❌ Erro ao carregar modelo: {e}", style="red")
            return False

    def obter_vetor_palavra(self, palavra):
        """Obtém o vetor de uma palavra usando o modelo Word2Vec."""
        model = Word2Vec.load("word2vec.model")
        return model.wv[palavra]

    def palavras_similares(self, palavra: str, n: int = 5) -> List[Dict[str, float]]:
        """
        Encontra palavras similares
        
        Args:
            palavra: Palavra de referência
            n: Número de palavras similares
            
        Returns:
            Lista de dicionários com palavras e scores
        """
        if self.model is None:
            if not self.load_model():
                return []
        
        try:
            similares = self.model.wv.most_similar(palavra.lower(), topn=n)
            return [{"palavra": p, "score": float(s)} for p, s in similares]
        except KeyError:
            return []

    def calcular_similaridade(self, palavra1: str, palavra2: str) -> Optional[float]:
        """
        Calcula similaridade entre duas palavras
        
        Args:
            palavra1: Primeira palavra
            palavra2: Segunda palavra
            
        Returns:
            Score de similaridade ou None se erro
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        try:
            return float(self.model.wv.similarity(palavra1.lower(), palavra2.lower()))
        except KeyError:
            return None

    def existe_palavra(self, palavra: str) -> bool:
        """Verifica se uma palavra existe no vocabulário"""
        if self.model is None:
            if not self.load_model():
                return False
        return palavra.lower() in self.model.wv

def preprocess_text(text: str) -> List[List[str]]:
    """
    Pré-processa o texto para treinamento do Word2Vec.
    
    Args:
        text: Texto para processar
        
    Returns:
        Lista de sentenças tokenizadas
    """
    try:
        # Divide em sentenças
        sentences = sent_tokenize(text.lower())
        
        # Tokeniza cada sentença
        return [
            word_tokenize(sentence)
            for sentence in sentences
        ]
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        return []

def train_word2vec_model(
    texts: List[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    epochs: int = 10
) -> Word2Vec:
    """
    Treina um novo modelo Word2Vec.
    
    Args:
        texts: Lista de textos para treinamento
        vector_size: Dimensão dos vetores
        window: Tamanho da janela de contexto
        min_count: Frequência mínima das palavras
        workers: Número de threads
        epochs: Número de épocas de treinamento
        
    Returns:
        Modelo Word2Vec treinado
    """
    try:
        # Pré-processa todos os textos
        sentences = []
        for text in texts:
            sentences.extend(preprocess_text(text))
            
        if not sentences:
            raise ValueError("Nenhuma sentença válida para treinamento")
            
        # Treina o modelo
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Erro no treinamento do modelo: {e}")
        raise

def load_word2vec_model(path: str) -> Word2Vec:
    """
    Carrega um modelo Word2Vec salvo.
    
    Args:
        path: Caminho do arquivo do modelo
        
    Returns:
        Modelo Word2Vec carregado
    """
    try:
        model = Word2Vec.load(path)
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise

def update_word2vec_model(
    model: Word2Vec,
    new_texts: List[str],
    epochs: int = 1
) -> Word2Vec:
    """
    Atualiza um modelo existente com novos textos.
    
    Args:
        model: Modelo Word2Vec existente
        new_texts: Novos textos para treinamento
        epochs: Número de épocas para atualização
        
    Returns:
        Modelo Word2Vec atualizado
    """
    try:
        # Pré-processa novos textos
        new_sentences = []
        for text in new_texts:
            new_sentences.extend(preprocess_text(text))
            
        if not new_sentences:
            return model
            
        # Atualiza o modelo
        model.build_vocab(new_sentences, update=True)
        model.train(
            new_sentences,
            total_examples=len(new_sentences),
            epochs=epochs
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Erro na atualização do modelo: {e}")
        raise

def main():
    """Testa a utilidade"""
    # Inicializa com o caminho do seu vault
    w2v = Word2VecUtils(obsidian_vault="F:/pasta estudos/Code Brain")
    
    # Palavras para testar
    test_words = [
        "python",
        "programação",
        "desenvolvimento",
        "web",
        "dados"
    ]
    
    # Testa cada palavra
    for palavra in test_words:
        console.print(f"\n🔍 Testando: {palavra}")
        
        if w2v.existe_palavra(palavra):
            similares = w2v.palavras_similares(palavra)
            console.print("Palavras similares:", style="green")
            for similar in similares:
                console.print(f"  - {similar['palavra']}: {similar['score']:.4f}")
        else:
            console.print(f"❌ Palavra '{palavra}' não encontrada", style="yellow")

if __name__ == "__main__":
    main()
    # Chama a função para analisar o conteúdo do vault