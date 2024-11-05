# src/obsidian_integration.py

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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

console = Console()

# Importações necessárias para stop words em múltiplos idiomas
from nltk.corpus import stopwords
import nltk

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
                title=video_data['title'],
                url=video_data['url'],
                channel=video_data['channel'],
                channel_url=video_data['channel_url'],
                duration=video_data['duration'],
                tags=video_data['tags'],
                view_count=video_data['view_count'],
                like_count=video_data['like_count'],
                upload_date=video_data['upload_date'],
                transcript_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        # YAML frontmatter
        frontmatter = {
            'title': metadata.title,
            'url': metadata.url,
            'channel': metadata.channel,
            'channel_url': metadata.channel_url,
            'duration': metadata.duration,
            'view_count': metadata.view_count,
            'like_count': metadata.like_count,
            'upload_date': metadata.upload_date,
            'transcript_date': metadata.transcript_date,
            'tags': ['transcrição', 'youtube'] + metadata.tags
        }

        # Converte frontmatter para YAML
        yaml_content = yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)

        # Formata o conteúdo da nota
        content = [
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
            "## Transcrição",
            "",
            transcription
        ]

        return "\n".join(content)

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
        note_path.write_text(content, encoding='utf-8')
        
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
            return self.template_path.read_text(encoding='utf-8')
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
            ""
        ]

        # Lista todas as transcrições e ordena por data
        transcriptions = sorted(
            self.transcriptions_dir.glob("*.md"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # Adiciona links para cada transcrição
        for note_path in transcriptions:
            if note_path.name != "Índice de Transcriç��es.md":
                link_name = note_path.stem.split(" - ", 1)[1] if " - " in note_path.stem else note_path.stem
                date = datetime.fromtimestamp(note_path.stat().st_mtime).strftime("%d/%m/%Y")
                index_content.append(f"- {date} - [[{note_path.stem}|{link_name}]]")

        # Salva o índice
        index_path = self.transcriptions_dir / "Índice de Transcrições.md"
        index_path.write_text("\n".join(index_content), encoding='utf-8')

def get_stop_words(language_code):
    LANGUAGE_CODE_MAP = {
        'en': 'english',
        'pt': 'portuguese',
        'es': 'spanish',
        # Adicionar mais mapeamentos conforme necessário
    }
    language = LANGUAGE_CODE_MAP.get(language_code, 'english')
    try:
        stop_words = stopwords.words(language)
    except LookupError:
        # Baixa as stop words se não estiverem disponíveis
        nltk.download('stopwords')
        stop_words = stopwords.words(language)
    return stop_words

def transcribe_with_language_detection(audio_path, model_size="base"):
    """
    Detecta o idioma e transcreve o áudio automaticamente.
    """
    try:
        model = whisper.load_model(model_size)
        
        # Primeiro detecta o idioma
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        
        detected_language = max(probs, key=probs.get)
        
        # Mapeia o initial_prompt de acordo com o idioma detectado
        INITIAL_PROMPTS = {
            'en': '[Music]',
            'pt': '[Música]',
            'es': '[Música]',
            # Adicionar mais conforme necessário
        }
        initial_prompt = INITIAL_PROMPTS.get(detected_language, '[Music]')
        
        # Configura para formatos específicos de legendas
        result = model.transcribe(
            audio_path,
            language=detected_language,
            task="translate" if detected_language != "en" else "transcribe",
            word_timestamps=True,  # Habilita timestamps por palavra
            initial_prompt=initial_prompt  # Ajuda a identificar partes não-faladas
        )
        
        return {
            'detected_language': detected_language,
            'confidence': probs[detected_language],
            'segments': result["segments"],
            'language_info': {
                'is_translated': detected_language != "en",
                'original_language': detected_language,
                'target_language': "pt"
            }
        }
    except FileNotFoundError as e:
        console.print(f"Arquivo de áudio não encontrado: {str(e)}")
        return None
    except ValueError as e:
        console.print(f"Valor inválido: {str(e)}")
        return None
    except Exception as e:
        console.print(f"Erro inesperado na transcrição: {str(e)}")
        raise

def format_timestamp_srt(seconds):
    """
    Formata o timestamp para o formato SRT.
    """
    from datetime import timedelta
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{remaining_seconds:02},{milliseconds:03}"

def format_timestamp_vtt(seconds):
    """
    Formata o timestamp para o formato VTT.
    """
    from datetime import timedelta
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{remaining_seconds:02}.{milliseconds:03}"

def format_as_srt(segments):
    """
    Formata os segmentos como um arquivo SRT.
    """
    srt_content = ""
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp_srt(segment['start'])
        end = format_timestamp_srt(segment['end'])
        text = segment['text'].strip()
        srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
    return srt_content

def format_as_vtt(segments):
    """
    Formata os segmentos como um arquivo VTT.
    """
    vtt_content = "WEBVTT\n\n"
    for segment in segments:
        start = format_timestamp_vtt(segment['start'])
        end = format_timestamp_vtt(segment['end'])
        text = segment['text'].strip()
        vtt_content += f"{start} --> {end}\n{text}\n\n"
    return vtt_content

def format_as_text(segments):
    """
    Formata os segmentos como texto simples.
    """
    return "\n".join(segment['text'].strip() for segment in segments)

def format_as_json(segments):
    """
    Formata os segmentos como JSON.
    """
    import json
    return json.dumps(segments, ensure_ascii=False, indent=4)

def save_multiple_formats(transcription_result, base_filename):
    """
    Salva a transcrição em múltiplos formatos úteis.
    """
    formats = {
        "srt": format_as_srt,
        "vtt": format_as_vtt,
        "txt": format_as_text,
        "json": format_as_json,
    }
    
    saved_files = {}
    
    for fmt, formatter in formats.items():
        output_file = f"{base_filename}.{fmt}"
        try:
            content = formatter(transcription_result['segments'])
        except Exception as e:
            console.print(f"Erro ao formatar as legendas: {str(e)}")
            continue
        if content:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            saved_files[fmt] = output_file
        else:
            console.print(f"Não foi possível salvar o arquivo {output_file}")
    
    return saved_files

def extract_keywords(segments, language_code='en'):
    """
    Extrai palavras-chave dos segmentos usando TF-IDF.
    """
    texts = [segment['text'] for segment in segments]
    stop_words = get_stop_words(language_code)
    vectorizer = TfidfVectorizer(max_features=20, stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    return keywords

def extract_topics(segments, language_code='en'):
    """
    Extrai tópicos principais dos segmentos usando LDA.
    """
    texts = [segment['text'] for segment in segments]
    stop_words = get_stop_words(language_code)
    count_vectorizer = CountVectorizer(stop_words=stop_words)
    count_data = count_vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(count_data)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topic_terms = [count_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-5 - 1:-1]]
        topics.append(", ".join(topic_terms))
    return topics

def is_important_segment(segment):
    """
    Determina se um segmento é importante com base na confiança.
    """
    return segment.get('confidence', 1.0) > 0.8

def generate_youtube_description(transcription_result):
    """
    Gera uma descrição otimizada para YouTube.
    """
    detected_language = transcription_result.get('detected_language', 'en')
    # Extrai palavras-chave e tópicos principais
    keywords = extract_keywords(transcription_result['segments'], language_code=detected_language)
    topics = extract_topics(transcription_result['segments'], language_code=detected_language)
    
    description = [
        "🎥 **Conteúdo do Vídeo:**",
        "=" * 30,
        "",
        "⌚ **Timestamps:**",
    ]
    
    # Adiciona timestamps importantes
    for segment in transcription_result['segments']:
        if is_important_segment(segment):
            time = format_timestamp_vtt(segment['start'])
            description.append(f"{time} - {segment['text'][:60]}...")
    
    description.extend([
        "",
        "🏷️ **Tópicos Abordados:**",
        "; ".join(topics),
        "",
        "#️⃣ **Tags:**",
        " ".join(f"#{k}" for k in keywords)
    ])
    
    return "\n".join(description)

def analyze_audio_quality(segments):
    """
    Analisa a qualidade do áudio e sugere melhorias.
    """
    analysis = {
        'unclear_segments': [],
        'noise_segments': [],
    }
    
    for segment in segments:
        # Analisa a confiança da transcrição
        if segment.get('confidence', 1.0) < 0.8:
            analysis['unclear_segments'].append({
                'timestamp': segment['start'],
                'text': segment['text'],
                'confidence': segment['confidence']
            })
            
        # Detecta segmentos com música ou ruído
        if '[Música]' in segment['text'] or '[Ruído]' in segment['text']:
            analysis['noise_segments'].append({
                'timestamp': segment['start'],
                'duration': segment['end'] - segment['start']
            })
    
    return analysis

# Nota: Algumas funções são utilidades que podem ser usadas por outros módulos.
# O módulo inclui funções para transcrição, formatação de legendas, extração de palavras-chave e tópicos,
# geração de descrições para YouTube e análise de qualidade de áudio.
