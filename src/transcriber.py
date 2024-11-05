import os
import warnings
import logging
from pathlib import Path
from datetime import timedelta
from typing import Optional, Tuple
import torch
from dataclasses import dataclass

import yt_dlp
from rich.console import Console
from werkzeug.utils import secure_filename
from yt_dlp.utils import DownloadError

from model_manager import ModelManager, ModelConfig
from obsidian_integration import ObsidianIntegration
from video_metadata import VideoMetadata

# Configuração de logging
logger = logging.getLogger(__name__)
console = Console()

# Suppression of warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

@dataclass
class TranscriberConfig:
    """Transcriber configuration"""
    model_size: str = "base"
    output_dir: str = "transcricoes"
    downloads_dir: str = "downloads"
    keep_audio: bool = False
    device: Optional[str] = None
    obsidian_vault: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 4
    compute_type: Optional[str] = None

class Transcriber:
    """Class responsible for video transcription"""

    def __init__(self, config: Optional[TranscriberConfig] = None):
        """
        Initializes the transcriber.

        Args:
            config: Transcriber configuration
        """
        self.config = config or TranscriberConfig()
        self.output_dir = Path(self.config.output_dir)
        self.downloads_dir = Path(self.config.downloads_dir)
        
        # Configuração do modelo
        model_config = ModelConfig(
            model_size=self.config.model_size,
            device=self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            compute_type=self.config.compute_type,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers
        )
        
        # Inicialização do gerenciador de modelo
        try:
            self.model_manager = ModelManager(config=model_config)
            logger.info(f"Model manager initialized: {model_config.device}, {model_config.model_size}")
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise
        
        # Configuração do Obsidian
        if self.config.obsidian_vault:
            try:
                self.obsidian = ObsidianIntegration(vault_path=self.config.obsidian_vault)
                logger.info("Obsidian integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Obsidian integration: {e}")
                raise
        
        # Criação de diretórios
        for directory in [self.output_dir, self.downloads_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")

    def get_video_metadata(self, url: str) -> Optional[VideoMetadata]:
        """Fetches metadata from the YouTube video."""
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
                    url=url,
                    channel=info.get('uploader', 'Canal não disponível'),
                    channel_url=info.get('uploader_url', ''),
                    duration=self.format_duration(info.get('duration', 0)),
                    tags=info.get('tags', []),
                    view_count=info.get('view_count', 0),
                    like_count=info.get('like_count', 0),
                    upload_date=info.get('upload_date', '')
                )
        except Exception as e:
            logger.error(f"Error fetching video metadata: {e}")
            return None

    def download_youtube_video(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Download YouTube video and extract audio."""
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.downloads_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = os.path.join(
                    self.downloads_dir,
                    secure_filename(f"{info['title']}.wav")
                )
                return audio_path, info.get('title')
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None, None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribes the audio using the ModelManager.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Formatted transcription or None if there is an error
        """
        try:
            # Usa o ModelManager para transcrição
            result = self.model_manager.transcribe(audio_path)
            
            # Formata o resultado
            transcription = "\n".join(
                [
                    f"[{self.format_timestamp(segment['start'])} -> "
                    f"{self.format_timestamp(segment['end'])}] {segment['text'].strip()}"
                    for segment in result["segments"]
                ]
            )
            
            # Adiciona informações sobre o idioma detectado
            language_info = f"\nIdioma detectado: {result['language']} "
            language_info += f"(Confiança: {result['confidence']:.2%})\n"
            
            return language_info + "\n" + transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Converts seconds to HH:MM:SS format"""
        return str(timedelta(seconds=round(seconds)))

    @staticmethod
    def format_duration(seconds: int) -> str:
        """Formats the duration of the video"""
        return f"{seconds//3600:02d}h {(seconds%3600)//60:02d}min {seconds%60:02d}s"

    def process_video(self, url: str) -> bool:
        """
        Processes a YouTube video: downloads, extracts audio, and performs transcription.
        """
        logger.info(f"Starting video processing: {url}")
        
        try:
            # Obtém metadados
            metadata = self.get_video_metadata(url)
            if not metadata:
                raise ValueError("Failed to fetch video metadata")
            
            # Download do vídeo
            logger.info("Downloading audio from YouTube...")
            audio_path, video_title = self.download_youtube_video(url)
            
            if not audio_path or not video_title:
                raise ValueError("Failed to download video")
            
            # Processa a transcrição
            logger.info("Transcribing audio...")
            transcription = self.transcribe_audio(audio_path)
            if not transcription:
                raise ValueError("Failed to transcribe audio")
            
            # Salva a transcrição
            safe_title = secure_filename(video_title)
            transcription_path = self.output_dir / f"{safe_title}_transcricao.txt"
            transcription_path.write_text(transcription, encoding="utf-8")
            
            # Integração com Obsidian
            if hasattr(self, 'obsidian'):
                try:
                    video_data = metadata.__dict__
                    self.obsidian.process_transcription(video_data, transcription)
                    self.obsidian.create_index_note()
                    logger.info(f"Created Obsidian note for video: {metadata.title}")
                except Exception as e:
                    logger.warning(f"Failed to create Obsidian note: {e}")
            
            # Limpa arquivos temporários
            if not self.config.keep_audio and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Temporary audio file removed")
            
            # Exibe estatísticas do modelo
            stats = self.model_manager.get_stats()
            logger.info(f"Model stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return False

    def cleanup(self):
        """Cleans up transcriber resources"""
        if hasattr(self, 'model_manager'):
            self.model_manager.cleanup()