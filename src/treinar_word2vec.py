import os
from pathlib import Path
from typing import List
from gensim.models import Word2Vec
from rich.console import Console
from rich.progress import track
from collections import Counter

console = Console()

class WordVecTrainer:
    def __init__(
        self,
        transcriptions_path: str = "transcricoes",
        model_path: str = "models/word2vec/model.bin",
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4
    ):
        # Ajusta os caminhos para o projeto
        base_path = Path(__file__).parent
        self.transcriptions_path = base_path / transcriptions_path
        self.model_path = base_path / model_path
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

        console.print(f"📂 Pasta de transcrições: {self.transcriptions_path}")
        console.print(f"📂 Pasta do modelo: {self.model_path}")

    def load_transcriptions(self) -> List[str]:
        """Carrega as transcrições"""
        console.print("\n📚 Carregando transcrições...", style="bold blue")
        transcriptions = []
        
        if not self.transcriptions_path.exists():
            console.print(f"❌ Pasta não encontrada: {self.transcriptions_path}", style="bold red")
            return transcriptions

        # Procura por arquivos .txt e .md
        files = list(self.transcriptions_path.glob("*.txt"))
        files.extend(self.transcriptions_path.glob("*.md"))
        
        if not files:
            console.print("❌ Nenhuma transcrição encontrada!", style="bold red")
            return transcriptions

        for file in track(files, description="Lendo arquivos"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        transcriptions.append(content)
                        console.print(f"✅ Lido: {file.name}", style="green")
            except Exception as e:
                console.print(f"❌ Erro ao ler {file.name}: {str(e)}", style="red")

        return transcriptions

    def prepare_sentences(self, transcriptions: List[str]) -> List[List[str]]:
        """Prepara as sentenças para treinamento"""
        console.print("\n🔄 Preparando texto...", style="bold blue")
        sentences = []
        
        for text in track(transcriptions, description="Processando"):
            # Divide em linhas e palavras
            lines = text.lower().split("\n")
            for line in lines:
                if line.strip():
                    words = line.split()
                    if words:
                        sentences.append(words)

        console.print(f"✅ Preparadas {len(sentences)} sentenças", style="bold green")
        return sentences

    def train_model(self):
        """Treina o modelo Word2Vec"""
        console.print("\n🚀 Iniciando treinamento...", style="bold blue")

        # Carrega e prepara dados
        transcriptions = self.load_transcriptions()
        if not transcriptions:
            return False

        sentences = self.prepare_sentences(transcriptions)
        if not sentences:
            console.print("❌ Sem dados para treinar!", style="bold red")
            return False

        try:
            # Treina o modelo
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers
            )

            # Cria pasta para o modelo
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            # Salva o modelo
            self.model.save(str(self.model_path))
            
            console.print(f"✅ Modelo salvo em: {self.model_path}", style="bold green")
            return True

        except Exception as e:
            console.print(f"❌ Erro no treinamento: {str(e)}", style="bold red")
            return False

    def test_model(self, test_words: List[str] = None):
        """Testa o modelo"""
        if not test_words:
            test_words = ["python", "código", "programação"]

        if not self.model:
            try:
                self.model = Word2Vec.load(str(self.model_path))
            except:
                console.print("❌ Modelo não encontrado!", style="bold red")
                return

        console.print("\n🔍 Testando modelo:", style="bold blue")
        for word in test_words:
            try:
                similar = self.model.wv.most_similar(word)
                console.print(f"\nSimilares a '{word}':", style="bold cyan")
                for w, score in similar[:5]:
                    console.print(f"  - {w}: {score:.4f}")
            except KeyError:
                console.print(f"⚠️  Palavra '{word}' não encontrada", style="yellow")

def main():
    trainer = WordVecTrainer(
        transcriptions_path="transcricoes",
        model_path="models/word2vec/model.bin"
    )
    
    if trainer.train_model():
        trainer.test_model([
            "python",
            "código",
            "programação",
            "web",
            "dados"
        ])

if __name__ == "__main__":
    main()
    def treinar_word2vec(self):
        """Treina o modelo Word2Vec e analisa as transcrições."""
        console.print("\n🔍 Analisando transcrições...", style="blue")
        
        # Carregar as transcrições
        transcricoes = []
        palavras_frequentes = Counter()
        temas = set()
        
        for arquivo in os.listdir(self.transcriptions_path):
            if arquivo.endswith(".txt") or arquivo.endswith(".md"):
                try:
                    with open(os.path.join(self.transcriptions_path, arquivo), "r", encoding="utf-8") as f:
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
                vector_size=self.vector_size, 
                window=self.window, 
                min_count=self.min_count, 
                workers=self.workers
            )
            
            # Salvar o modelo
            model.save(str(self.model_path))
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