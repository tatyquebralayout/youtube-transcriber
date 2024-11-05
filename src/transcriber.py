import whisper
import yt_dlp
from pathlib import Path
import torch
import os
import warnings
from datetime import timedelta
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from rich.console import Console
from .obsidian_integration import ObsidianIntegration  # Adicionado
from werkzeug.utils import secure_filename  # Adicionado
from yt_dlp.utils import DownloadError  # Adicionado

# Configuração de logging
logger = logging.getLogger(__name__)
console = Console()

# Supressão de avisos
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

@dataclass
class VideoMetadata:
    """Estrutura de dados para metadados do vídeo"""
    title: str
    channel: str
    channel_url: str
    duration: str
    description: str
    tags: list
    view_count: int
    like_count: int
    upload_date: str

class Transcriber:
    """Classe responsável pela transcrição de vídeos do YouTube"""

    def __init__(
        self,
        model_size: str = "base",
        output_dir: str = "transcricoes",
        downloads_dir: str = "downloads",
        keep_audio: bool = False,
        device: Optional[str] = None,
        obsidian_vault: Optional[str] = None  # Adicionado
    ):
        """
        Inicializa o transcritor.

        Args:
            model_size: Tamanho do modelo Whisper ("tiny", "base", "small", "medium", "large")
            output_dir: Diretório para salvar as transcrições
            downloads_dir: Diretório para downloads temporários
            keep_audio: Se deve manter o arquivo de áudio após a transcrição
            device: Dispositivo para processamento ("cpu" ou "cuda")
            obsidian_vault: Caminho para o vault do Obsidian  # Adicionado
        """
        self.model_size = model_size
        self.output_dir = Path(output_dir)
        self.downloads_dir = Path(downloads_dir)
        self.keep_audio = keep_audio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obsidian_vault = obsidian_vault  # Adicionado
        logger.info(f"Using device: {self.device}")
        
        # Verificação do caminho do vault
        if self.obsidian_vault:
            if not os.path.exists(self.obsidian_vault):
                raise ValueError(f"O caminho do vault '{self.obsidian_vault}' não existe. Verifique se o caminho está correto.")
            else:
                logger.info("Caminho do vault verificado com sucesso.")
        
        # Carregamento do modelo
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info(f"Loaded Whisper model: {model_size}")
        except Exception as e:  # Captura qualquer exceção durante o carregamento do modelo
            logger.error(f"Error loading Whisper model: {e}")
            raise
        
        # Criação de diretórios
        for directory in [self.output_dir, self.downloads_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Converte segundos para formato HH:MM:SS"""
        return str(timedelta(seconds=round(seconds)))

    @staticmethod
    def format_duration(seconds: int) -> str:
        """
        Formata a duração do vídeo em formato legível.
        
        Args:
            seconds: Duração em segundos
            
        Returns:
            String formatada (ex: "02h 30min 45s")
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        
        return f"{hours:02d}h {minutes:02d}min {remaining_seconds:02d}s"

    def get_video_metadata(self, url: str) -> Optional[VideoMetadata]:
        """
        Obtém metadados do vídeo do YouTube.
        
        Args:
            url: URL do vídeo
            
        Returns:
            VideoMetadata ou None se houver erro
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return VideoMetadata(
                    title=info.get('title', 'Título não disponível'),
                    channel=info.get('uploader', 'Canal não disponível'),
                    channel_url=info.get('uploader_url', ''),
                    duration=self.format_duration(info.get('duration', 0)),
                    description=info.get('description', 'Descrição não disponível'),
                    tags=info.get('tags', []),
                    view_count=info.get('view_count', 0),
                    like_count=info.get('like_count', 0),
                    upload_date=info.get('upload_date', '')
                )
        except Exception as e:
            logger.error(f"Error fetching video metadata: {e}")
            return None

    def format_header(self, metadata: VideoMetadata, video_url: str) -> str:
        """
        Formata o cabeçalho da transcrição com metadados do vídeo.
        
        Args:
            metadata: Metadados do vídeo
            video_url: URL original do vídeo
            
        Returns:
            Cabeçalho formatado
        """
        return "\n".join([
            f"Título: {metadata.title}",
            f"Canal: {metadata.channel} ({metadata.channel_url})",
            f"Duração: {metadata.duration}",
            f"Visualizações: {metadata.view_count:,}",
            f"Likes: {metadata.like_count:,}",
            f"Data de Upload: {metadata.upload_date}",
            f"URL: {video_url}",
            f"Descrição: {metadata.description}",
            f"Tags: {', '.join(metadata.tags)}",
            "\n--- Transcrição ---\n"
        ])

    def download_youtube_video(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Baixa o vídeo do YouTube e extrai o áudio.
        
        Args:
            url: URL do vídeo
            
        Returns:
            Tupla (caminho_do_audio, titulo_do_video) ou (None, None) se houver erro
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.downloads_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = ydl.prepare_filename(info)
                return audio_path, info.get('title', 'audio')
        except DownloadError as e:
            logger.error(f"Error downloading video: {e}")
            return None, None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcreve o áudio usando o modelo Whisper.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            
        Returns:
            Transcrição formatada ou None se houver erro
        """
        try:
            result = self.model.transcribe(audio_path)
            
            return "\n".join(
                [
                    f"[{self.format_timestamp(segment['start'])} -> "
                    f"{self.format_timestamp(segment['end'])}] {segment['text'].strip()}"
                    for segment in result["segments"]
                ]
            )
        except WhisperError as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def process_video(self, url: str) -> bool:
        """
        Processa um vídeo do YouTube: baixa, extrai áudio e realiza a transcrição.
        
        Args:
            url: URL do vídeo do YouTube
            
        Returns:
            True se o processo for bem-sucedido, False caso contrário
        """
        logger.info(f"Starting video processing: {url}")
        
        try:
            # Obtém metadados
            metadata = self.get_video_metadata(url)
            if not metadata:
                raise ValueError("Failed to fetch video metadata")
            
            # Prepara cabeçalho
            header = self.format_header(metadata, url)
            
            # Download do vídeo
            logger.info("Downloading audio from YouTube...")
            audio_path, video_title = self.download_youtube_video(url)
            
            if not audio_path or not video_title:
                raise ValueError("Failed to download video")
            
            # Processa a transcrição
            safe_title = secure_filename(video_title)
            transcription_path = self.output_dir / f"{safe_title}_transcricao.txt"
            logger.info("Transcribing audio...")
            
            transcription = self.transcribe_audio(audio_path)
            if not transcription:
                raise ValueError("Failed to transcribe audio")
            
            # Salva o resultado
            full_content = f"{header}\n{transcription}"
            transcription_path.write_text(full_content, encoding="utf-8")
            
            # Limpa arquivos temporários
            if not self.keep_audio and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Temporary audio file removed")
            
            logger.info(f"Transcription completed: {transcription_path}")
            
            # Após a transcrição bem-sucedida
            if self.obsidian_vault:
                try:
                    obsidian = ObsidianIntegration(vault_path=self.obsidian_vault)
                    obsidian.process_transcription(
                        video_data={
                            'title': metadata.title,
                            'url': url,
                            'channel': metadata.channel,
                            'channel_url': metadata.channel_url,
                            'duration': metadata.duration,
                            'tags': metadata.tags,
                            'view_count': metadata.view_count,
                            'like_count': metadata.like_count,
                            'upload_date': metadata.upload_date
                        },
                        transcription=transcription
                    )
                    logger.info(f"Created Obsidian note for video: {metadata.title}")
                except Exception as e:
                    logger.warning(f"Failed to create Obsidian note: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False