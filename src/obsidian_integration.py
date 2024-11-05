import os
from pathlib import Path
from typing import Dict, Any, Optional
import re
from datetime import datetime
import yaml
from rich.console import Console
import logging
from dataclasses import dataclass

import whisper
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Importações necessárias para stop words em múltiplos idiomas
from nltk.corpus import stopwords
import nltk

console = Console()


@dataclass
class VideoMetadata:
    title: str
    url: str
    channel: str
    channel_url: str
    duration: str
    tags: list[str]
    view_count: int
    like_count: int
    upload_date: str
    transcript_date: str = None


class ObsidianIntegration:
    def __init__(self, vault_path: str, template_path: Optional[str] = None):
        """
        Inicializa a integração com o Obsidian.

        Args:
            vault_path: Caminho para o vault do Obsidian
            template_path: Caminho opcional para um template personalizado
        """
        self.vault_path = Path(vault_path)
        self.template_path = Path(template_path) if template_path else None
        self.console = Console()
        self.logger = logging.getLogger(__name__)

        if not self.vault_path.exists():
            raise ValueError(f"Vault path '{vault_path}' does not exist")

        # Cria diretório de transcrições se não existir
        self.transcriptions_dir = self.vault_path / "Transcrições"
        self.transcriptions_dir.mkdir(exist_ok=True)

        # Carrega o mapeamento de palavras-chave para pastas
        self.mapping = self.load_mapping()

    def load_mapping(self):
        """Carrega o mapeamento de palavras-chave para pastas."""
        try:
            with open("mapping.yaml", "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Cria um dicionário vazio caso o arquivo não seja encontrado
            logging.warning("Arquivo mapping.yaml não encontrado. Criando um novo mapeamento.")  
            return {}
        except Exception as e:
            self.logger.error(f"Erro ao carregar o arquivo mapping.yaml: {e}")
            return {}

    def transcribe_with_language_detection(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcreve o áudio e detecta o idioma.
        """
        # Implementação fictícia
        return {
            "segments": ["example segment"],
            "detected_language": "en"
        }

    def extract_keywords(self, segments: list, language_code: str) -> list:
        """
        Extrai palavras-chave dos segmentos de transcrição.
        """
        # Implementação fictícia
        return ["keyword1", "keyword2"]

    def suggest_folders(self, keywords, n_clusters=5):
        """Sugere pastas com base no agrupamento de palavras-chave."""
        if len(keywords) < 2:
            return {}

        # Carregar modelo Word2Vec (ou treinar um novo se necessário)
        model = Word2Vec.load("word2vec.model")  # Substitua pelo caminho do seu modelo

        # Obter embeddings das palavras-chave
        vectors = [model.wv[word] for word in keywords if word in model.wv]
        if len(vectors) < 2:
            return {}

        # Aplicar k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(vectors)

        # Criar sugestões de pastas
        suggestions = {}
        for i, label in enumerate(kmeans.labels_):
            cluster_name = f"Cluster {i+1}"  # Nome da pasta (pode ser melhorado)
            suggestions.setdefault(cluster_name, []).append(keywords[i])
        return suggestions

    def process_transcription(self, video_data: Dict[str, Any], transcription: str) -> Path:
        """
        Cria uma nota no Obsidian com a transcrição do vídeo.

        Args:
            video_data: Dicionário contendo metadados do vídeo
            transcription: Texto da transcrição

        Returns:
            Path: Caminho da nota criada
        """
        try:
            # Converte dados do vídeo para VideoMetadata
            metadata = VideoMetadata(
                title=video_data["title"],
                url=video_data["url"],
                channel=video_data["channel"],
                channel_url=video_data["channel_url"],
                duration=video_data["duration"],
                tags=video_data["tags"],
                view_count=video_data["view_count"],
                like_count=video_data["like_count"],
                upload_date=video_data["upload_date"],
                transcript_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Formata o conteúdo da nota
            note_content = self._format_note_content(metadata, transcription)

            # Cria o arquivo da nota
            note_path = self._create_note_file(metadata, note_content)

            self.logger.info(f"Created Obsidian note: {note_path}")
            return note_path

        except Exception as e:
            self.logger.error(f"Error processing transcription: {e}")
            raise

    def _format_note_content(self, metadata: VideoMetadata, transcription: str) -> str:
        """
        Formata o conteúdo da nota usando YAML frontmatter e markdown.
        """
        # Defina o frontmatter
        frontmatter = {
            "title": metadata.title,
            "url": metadata.url,
            "channel": metadata.channel,
            "channel_url": metadata.channel_url,
            "duration": metadata.duration,
            "tags": metadata.tags,
            "view_count": metadata.view_count,
            "like_count": metadata.like_count,
            "upload_date": metadata.upload_date,
            "transcript_date": metadata.transcript_date,
        }

        # Extrair palavras-chave
        keywords = self.extract_keywords(
            transcription.split(),  # Passar a transcrição diretamente
            language_code="en",  # Substitua pelo código de idioma correto
        )

        # Obter sugestões de pastas
        suggestions = self.suggest_folders(keywords)

        # Combinar mapeamento manual com sugestões
        content = []
        for keyword in keywords:
            folder = self.mapping.get(keyword.lower())
            if folder is None and keyword.lower() in suggestions:
                folder = suggestions[keyword.lower()][0]  # Pega a primeira sugestão
            if folder:
                try:
                    # Criar pasta se não existir
                    folder_path = self.vault_path / folder
                    if not folder_path.exists():
                        folder_path.mkdir(parents=True)

                    # Adicionar link no conteúdo
                    content.append(f"[[{folder}|{keyword}]]")
                except Exception as e:
                    self.logger.warning(f"Erro ao criar pasta ou link: {e}")

        # Converte frontmatter para YAML
        yaml_content = yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)

        # Extrair palavras-chave
        transcription_result = self.transcribe_with_language_detection("audio.mp3")  # Substitua pelo caminho do seu áudio
        keywords = self.extract_keywords(
            transcription_result["segments"],
            language_code=transcription_result["detected_language"],
        )

        # Obter sugestões de pastas
        suggestions = self.suggest_folders(keywords)

        # Combinar mapeamento manual com sugestões
        content = []
        for keyword in keywords:
            folder = self.mapping.get(keyword.lower())
            if folder is None and keyword.lower() in suggestions:
                folder = suggestions[keyword.lower()][0]  # Pega a primeira sugestão
            if folder:
                try:
                    # Criar pasta se não existir
                    folder_path = self.vault_path / folder
                    if not folder_path.exists():
                        folder_path.mkdir(parents=True)

                    # Adicionar link no conteúdo
                    content.append(f"[[{folder}|{keyword}]]")
                except Exception as e:
                    self.logger.warning(f"Erro ao criar pasta ou link: {e}")

        # Formata o conteúdo da nota
        note_content = [
            "---",
            yaml_content,
            "---",
            "",
            f"# {metadata.title}",
            "",
            "## Metadados",
            f"- 📺 **Canal:** [{metadata.channel}]({metadata.channel_url})",
            f"- ⏱️ **Duração:** {metadata.duration}",
            f"- 👀 **Visualizações:** {metadata.view_count:,}",
            f"- 👍 **Likes:** {metadata.like_count:,}",
            f"- 📅 **Data de Upload:** {metadata.upload_date}",
            f"- 🔄 **Data da Transcrição:** {metadata.transcript_date}",
            "",
            "## Palavras-chave",
            "",
            " ".join(content),
            "",
            "## Transcrição",
            "",
            transcription,
        ]

        return "\n".join(note_content)

    def _create_note_file(self, metadata: VideoMetadata, content: str) -> Path:
        """
        Cria o arquivo da nota no vault do Obsidian.
        """
        # Sanitiza o título para usar como nome do arquivo
        safe_title = self._sanitize_filename(metadata.title)

        # Adiciona a data da transcrição ao nome do arquivo
        date_prefix = datetime.now().strftime("%Y%m%d")
        filename = f"{date_prefix} - {safe_title}.md"

        # Caminho completo da nota
        note_path = self.transcriptions_dir / filename

        # Escreve o conteúdo no arquivo
        note_path.write_text(content, encoding="utf-8")

        return note_path

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitiza o nome do arquivo removendo caracteres inválidos.
        """
        # Remove caracteres inválidos
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Remove espaços múltiplos
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # Limita o tamanho do nome do arquivo
        sanitized = sanitized[:100].strip()

        return sanitized

    def get_template(self) -> Optional[str]:
        """
        Carrega o template personalizado se existir.
        """
        if self.template_path and self.template_path.exists():
            return self.template_path.read_text(encoding="utf-8")
        return None

    def create_index_note(self) -> None:
        """
        Cria ou atualiza uma nota índice com links para todas as transcrições.
        """
        index_content = [
            "# 📝 Índice de Transcrições",
            "",
            "Este é um índice automático de todas as transcrições de vídeos.",
            "",
            "## Transcrições Recentes",
            "",
        ]

        # Lista todas as transcrições e ordena por data
        transcriptions = sorted(
            self.transcriptions_dir.glob("*.md"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        # Adiciona links para cada transcrição
        for note_path in transcriptions:
            if note_path.name != "Índice de Transcrições.md":
                link_name = note_path.stem.split(" - ", 1)[1] if " - " in note_path.stem else note_path.stem
                date = datetime.fromtimestamp(note_path.stat().st_mtime).strftime("%d/%m/%Y")
                index_content.append(f"- {date} - [[{note_path.stem}|{link_name}]]")

        # Salva o índice
        index_path = self.transcriptions_dir / "Índice de Transcrições.md"
        index_path.write_text("\n".join(index_content), encoding="utf-8")
