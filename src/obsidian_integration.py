import re
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Tuple
from dataclasses import dataclass

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
import markdown

# obsidian_integration.py

from .model_manager import ModelManager
from .keyword_extractor import extract_keywords, get_stop_words
from .word2vec_utils import train_word2vec_model, load_word2vec_model

# Configuração de logging
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class VideoMetadata:
    """Metadados do vídeo para transcrição"""
    title: str
    url: str
    channel: str
    channel_url: str
    duration: str
    tags: List[str]
    view_count: int
    like_count: int
    upload_date: str
    transcript_date: str = None
    language: str = None
    confidence: float = None

@dataclass
class TranscriptionResult:
    """Resultado da transcrição"""
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    metadata: VideoMetadata
    path: Path

class ObsidianIntegration:
    def __init__(self, vault_path: str, template_path: Optional[str] = None, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.model_manager = None  # Ensure the attribute exists even if initialization fails
        self.logger.debug("Iniciando ObsidianIntegration.__init__")
        print("ObsidianIntegration.__init__ iniciado")

        # Verificação se o vault_path existe
        if not Path(vault_path).exists():
            self.logger.error(f"Vault path '{vault_path}' does not exist or is not accessible.")
            print(f"Vault path '{vault_path}' does not exist or is not accessible.")
            raise ValueError(f"Vault path '{vault_path}' does not exist or is not accessible.")
        else:
            self.logger.debug(f"Vault path '{vault_path}' existe.")
            print(f"Vault path '{vault_path}' existe.")

        # Continue com a inicialização
        self.vault_path = Path(vault_path)
        self.template_path = Path(template_path) if template_path else None
        self.config = config or {}
        self.logger.debug("Configurações iniciais definidas")
        print("Configurações iniciais definidas")

        # Validação do vault
        if not self.vault_path.exists():
            self.logger.error(f"Vault path '{vault_path}' does not exist")
            raise ValueError(f"Vault path '{vault_path}' does not exist")

        self.logger.debug("Vault path existe")
        print("Vault path existe")

        # Configuração de diretórios
        self._setup_directories()
        self.logger.debug("Diretórios configurados")
        print("Diretórios configurados")

        # Carrega o mapeamento de palavras-chave para pastas
        self.mapping = self.load_mapping()
        self.logger.debug("Mapping carregado")
        print("Mapping carregado")

        # Inicialização do ModelManager
        try:
            self.model_manager = ModelManager()
            self.logger.info("ModelManager inicializado com sucesso")
            print("ModelManager inicializado com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao inicializar ModelManager: {str(e)}")
            print(f"Erro ao inicializar ModelManager: {str(e)}")
            self.model_manager = None

        # Carrega ou treina modelo Word2Vec
        try:
            self._setup_word2vec()
            self.logger.debug("Word2Vec configurado")
            print("Word2Vec configurado")
        except Exception as e:
            self.logger.error(f"Erro ao configurar Word2Vec: {e}")
            print(f"Erro ao configurar Word2Vec: {e}")
            self.word2vec = None

        print("ObsidianIntegration.__init__ concluído")

    def process_transcription(self, video_data: Dict[str, Any], transcription: str) -> TranscriptionResult:
        """
        Cria uma nota no Obsidian com a transcrição do vídeo.
        
        Args:
            video_data: Dicionário contendo metadados do vídeo
            transcription: Texto da transcrição

        Returns:
            TranscriptionResult: Resultado da transcrição com metadados e caminhos

        Raises:
            ValueError: Se os dados do vídeo estiverem incompletos
            RuntimeError: Se houver erro no processamento
        """
        if not all(k in video_data for k in ["title", "url", "channel", "channel_url", "duration", "tags"]):
            raise ValueError("Dados do vídeo incompletos")

        if self.model_manager is None:
            self.logger.warning("ModelManager não foi inicializado corretamente, continuando sem ele.")
            print("ModelManager não foi inicializado corretamente, continuando sem ele.")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                progress_tasks = {}
                
                # Criação dos metadados
                progress_tasks["metadata"] = progress.add_task(
                    "✍️ Processando metadados...", 
                    total=None
                )
                metadata = self._create_metadata(video_data)

                # Formatação do conteúdo
                progress_tasks["content"] = progress.add_task(
                    "📝 Formatando conteúdo...", 
                    total=None
                )
                note_content = self._format_note_content(metadata, transcription)

                # Criação do arquivo
                progress_tasks["file"] = progress.add_task(
                    "💾 Criando arquivo...", 
                    total=None
                )
                note_path = self._create_note_file(metadata, note_content)
                
                # Atualização do índice
                progress_tasks["index"] = progress.add_task(
                    "📑 Atualizando índice...", 
                    total=None
                )
                self.create_index_note()

                # Atualiza o modelo Word2Vec
                if self.word2vec is not None:
                    self._update_word2vec(transcription)

                self.logger.info(f"Created Obsidian note: {note_path}")
                
                return TranscriptionResult(
                    text=transcription,
                    language=metadata.language,
                    confidence=metadata.confidence,
                    segments=video_data.get("segments", []),
                    metadata=metadata,
                    path=note_path
                )

        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")
            raise RuntimeError(f"Falha no processamento da transcrição: {str(e)}")

    # Outros métodos da classe permanecem inalterados

    def _setup_directories(self) -> None:
        """
        Configura os diretórios necessários para o funcionamento do sistema.
        Cria diretórios se não existirem.
        """
        directories = {
            "transcriptions": self.vault_path / "Transcrições",
            "models": self.vault_path / ".models",
            "config": self.vault_path / ".config",
            "backup": self.vault_path / "backups"
        }

        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)

    def load_mapping(self) -> Dict:
        """
        Carrega o mapeamento de palavras-chave para pastas.
        
        Returns:
            Dicionário com o mapeamento palavra-chave -> pasta
        """
        try:
            mapping_path = self.vault_path / "mapping.json"
            if mapping_path.exists():
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Erro ao carregar mapeamento: {e}")
            return {}

    def _setup_word2vec(self) -> None:
        """
        Configura o modelo Word2Vec, carregando um existente ou treinando um novo.
        O modelo é usado para análise semântica das transcrições.
        """
        model_path = self.vault_path / ".models" / "word2vec.model"
        
        try:
            if model_path.exists():
                self.word2vec = load_word2vec_model(model_path)
            else:
                texts = self._collect_training_texts()
                self.word2vec = train_word2vec_model(texts, model_path)
        except Exception as e:
            self.logger.error(f"Erro ao configurar Word2Vec: {e}")
            self.word2vec = None

    def _collect_training_texts(self) -> List[str]:
        """
        Coleta textos das transcrições existentes para treinar o modelo Word2Vec.
        
        Returns:
            Lista de textos das transcrições
        """
        texts = []
        try:
            for note_path in self.vault_path.glob("Transcrições/*.md"):
                with open(note_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        except Exception as e:
            self.logger.error(f"Erro ao coletar textos para treinamento: {e}")
        
        return texts

    def _create_metadata(self, video_data: Dict[str, Any]) -> VideoMetadata:
        """
        Cria objeto VideoMetadata a partir dos dados do vídeo.
        
        Args:
            video_data: Dicionário com dados do vídeo
            
        Returns:
            VideoMetadata: Objeto com metadados formatados
        """
        try:
            return VideoMetadata(
                title=video_data["title"],
                url=video_data["url"],
                channel=video_data["channel"],
                channel_url=video_data["channel_url"],
                duration=video_data["duration"],
                tags=video_data["tags"],
                view_count=video_data.get("view_count", 0),
                like_count=video_data.get("like_count", 0),
                upload_date=video_data.get("upload_date", ""),
                transcript_date=datetime.now().strftime("%Y-%m-%d"),
                language=video_data.get("language", ""),
                confidence=video_data.get("confidence", 0.0)
            )
        except Exception as e:
            self.logger.error(f"Erro ao criar metadados: {e}")
            raise

    def _format_note_content(self, metadata: VideoMetadata, transcription: str) -> str:
        """Formata o conteúdo da nota usando YAML frontmatter e markdown."""
        # YAML frontmatter
        frontmatter = {
            "title": metadata.title,
            "url": metadata.url,
            "channel": metadata.channel,
            "channel_url": metadata.channel_url,
            "duration": metadata.duration,
            "view_count": metadata.view_count,
            "like_count": metadata.like_count,
            "upload_date": metadata.upload_date,
            "transcript_date": metadata.transcript_date,
            "language": metadata.language,
            "confidence": metadata.confidence,
            "tags": ["transcrição", "youtube"] + metadata.tags,
        }

        yaml_content = yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)

        # Extrai palavras-chave
        try:
            keywords = extract_keywords(transcription, metadata.language)
        except Exception as e:
            self.logger.error(f"Erro ao extrair palavras-chave: {e}")
            keywords = []

        # Formata links e cria pastas
        content = []
        for keyword in keywords:
            folder = self.mapping.get(keyword, "Outros")
            content.append(f"[{keyword}]({folder})")

        # Template personalizado ou formato padrão
        if self.template_path and self.template_path.exists():
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            note_content = template.format(
                frontmatter=yaml_content,
                transcription=transcription,
                keywords=", ".join(content)
            )
        else:
            note_content = (
                f"---\n{yaml_content}---\n\n"
                f"# {metadata.title}\n\n"
                f"## Metadados\n"
                f"- 📺 **Canal:** [{metadata.channel}]({metadata.channel_url})\n"
                f"- ⏱️ **Duração:** {metadata.duration}\n"
                f"- 👀 **Visualizações:** {metadata.view_count:,}\n"
                f"- 👍 **Likes:** {metadata.like_count:,}\n"
                f"- 📅 **Data de Upload:** {metadata.upload_date}\n"
                f"- 🔄 **Data da Transcrição:** {metadata.transcript_date}\n\n"
                f"## Palavras-chave\n\n"
                f"{', '.join(content)}\n\n"
                f"## Transcrição\n\n"
                f"{transcription}\n"
            )

        return note_content

    def _create_note_file(self, metadata: VideoMetadata, content: str) -> Path:
        """Cria o arquivo da nota no vault do Obsidian."""
        safe_title = self._sanitize_filename(metadata.title)
        date_prefix = datetime.now().strftime("%Y%m%d")
        filename = f"{date_prefix} - {safe_title}.md"
        note_path = self.vault_path / "Transcrições" / filename
        note_path.write_text(content, encoding="utf-8")
        return note_path

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitiza o nome do arquivo removendo caracteres inválidos."""
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        sanitized = re.sub(r'\s+', ' ', sanitized)
        return sanitized[:100].strip()

    def create_index_note(self) -> None:
        """Cria ou atualiza uma nota índice com links para todas as transcrições."""
        index_content = [
            "# 📝 Índice de Transcrições",
            "",
            "Este é um índice automático de todas as transcrições de vídeos.",
            "",
            "## Transcrições Recentes",
            "",
        ]

        transcriptions = sorted(
            (self.vault_path / "Transcrições").glob("*.md"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for note_path in transcriptions:
            index_content.append(f"- [{note_path.stem}]({note_path.name})")

        index_path = self.vault_path / "Transcrições" / "Índice de Transcrições.md"
        index_path.write_text("\n".join(index_content), encoding="utf-8")

    def cleanup(self) -> None:
        """Limpa recursos e salva estados finais."""
        try:
            if self.model_manager:
                self.model_manager.cleanup()
        except Exception as e:
            self.logger.error(f"Erro ao limpar recursos do ModelManager: {e}")
        finally:
            if self.word2vec:
                self.word2vec.save(self.vault_path / ".models" / "word2vec.model")

    def __enter__(self) -> 'ObsidianIntegration':
        """Suporte para uso com context manager (with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup automático ao sair do context manager."""
        self.cleanup()

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso."""
        stats = {
            "total_transcriptions": len(list((self.vault_path / "Transcrições").glob("*.md"))),
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if self.model_manager is not None:
            stats.update(self.model_manager.get_stats())

        if self.word2vec is not None:
            stats["word2vec_vocab_size"] = len(self.word2vec.wv)

        return stats

    def update_mapping(self, new_mappings: Dict[str, str]) -> None:
        """
        Atualiza o mapeamento de palavras-chave para pastas.
        
        Args:
            new_mappings: Dicionário com novos mapeamentos
        """
        try:
            self.mapping.update(new_mappings)
            mapping_path = self.vault_path / "mapping.json"
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.error(f"Erro ao atualizar mapeamento: {e}")

    def search_transcriptions(self, query: str) -> List[Dict[str, Any]]:
        """
        Pesquisa nas transcrições existentes.
        
        Args:
            query: Termo de busca
            
        Returns:
            Lista de resultados encontrados
        """
        results = []
        try:
            for note_path in (self.vault_path / "Transcrições").glob("*.md"):
                with open(note_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        results.append({
                            "path": note_path,
                            "content": content
                        })
        except Exception as e:
            self.logger.error(f"Erro ao pesquisar transcrições: {e}")

        return results

    def export_transcription(self, note_path: Path, format: str = "md") -> Optional[Path]:
        """
        Exporta uma transcrição para outro formato.
        
        Args:
            note_path: Caminho da nota
            format: Formato de exportação ('md', 'txt', 'json', 'html')
            
        Returns:
            Path do arquivo exportado ou None se houver erro
        """
        try:
            content = note_path.read_text(encoding='utf-8')
            export_path = note_path.with_suffix(f".{format}")

            if format == "txt":
                export_path.write_text(content, encoding='utf-8')
            elif format == "json":
                data = {
                    "content": content,
                    "metadata": {
                        "title": note_path.stem,
                        "path": str(note_path)
                    }
                }
                export_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding='utf-8')
            elif format == "html":
                html_content = markdown.markdown(content)
                export_path.write_text(html_content, encoding='utf-8')
            else:
                export_path.write_text(content, encoding='utf-8')

            return export_path
        except Exception as e:
            self.logger.error(f"Erro ao exportar transcrição: {e}")
            return None
