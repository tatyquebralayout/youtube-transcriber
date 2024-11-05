import os
import warnings
import logging
import shutil
import subprocess
from pathlib import Path
from datetime import timedelta
from typing import Optional, Tuple, Dict, Any
import torch
from dataclasses import dataclass

import yt_dlp
from rich.console import Console
from werkzeug.utils import secure_filename
from yt_dlp.utils import DownloadError

from .model_manager import ModelManager, ModelConfig
from .obsidian_integration import ObsidianIntegration
from .video_metadata import VideoMetadata

# Configuração de logging
logger = logging.getLogger(__name__)
console = Console()

# Supressão de avisos
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

def check_ffmpeg() -> bool:
    """
    Verifica se o FFmpeg está instalado e acessível.
    
    Returns:
        bool: True se FFmpeg está disponível, False caso contrário
    """
    if not shutil.which('ffmpeg'):
        logger.error("FFmpeg não encontrado. Por favor, instale o FFmpeg e certifique-se "
                    "de que está no PATH do sistema.")
        return False
    return True

def convert_audio(input_path: str, output_path: str) -> bool:
    """
    Converte áudio para o formato adequado usando FFmpeg.
    
    Args:
        input_path: Caminho do arquivo de entrada
        output_path: Caminho do arquivo de saída
        
    Returns:
        bool: True se a conversão foi bem-sucedida, False caso contrário
    """
    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path
        ], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Erro na conversão de áudio: {e.stderr.decode()}")
        return False

@dataclass
class TranscriberConfig:
    """Configuração do transcritor"""
    model_size: str = "base"
    output_dir: str = "transcricoes"
    downloads_dir: str = "downloads"
    keep_audio: bool = False
    device: Optional[str] = None
    obsidian_vault: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 4
    compute_type: Optional[str] = None
    language: Optional[str] = None

class Transcriber:
    """Classe responsável pela transcrição de vídeos do YouTube"""

    def __init__(self, config: Optional[TranscriberConfig] = None):
        """
        Inicializa o transcritor.

        Args:
            config: Configuração do transcritor
        """
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg é necessário para o funcionamento do transcritor.")
        
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
        """
        Baixa o vídeo do YouTube e extrai o áudio.
        
        Args:
            url: URL do vídeo
            
        Returns:
            Tuple contendo (caminho_do_audio, titulo_do_video) ou (None, None) se houver erro
        """
        try:
            # Primeiro, obtém o título do vídeo
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = secure_filename(info['title'])
            
            # Define os caminhos dos arquivos
            temp_audio = str(self.downloads_dir / f"{video_title}.%(ext)s")
            final_path = str(self.downloads_dir / f"{video_title}.wav")
            
            # Configurações para download
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_audio,
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'prefer_ffmpeg': True,
                'keepvideo': False,
                'writethumbnail': False,
                'postprocessor_args': [
                    '-ar', '16000',
                    '-ac', '1',
                    '-acodec', 'pcm_s16le',
                ],
            }
            
            # Realiza o download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Procura pelo arquivo WAV gerado
            wav_files = list(self.downloads_dir.glob(f"{video_title}.wav"))
            if not wav_files:
                # Se não encontrou WAV, procura por qualquer arquivo de áudio
                audio_files = list(self.downloads_dir.glob(f"{video_title}.*"))
                if not audio_files:
                    raise FileNotFoundError("No audio file found after download")
                
                # Pega o primeiro arquivo encontrado e converte para WAV
                source_audio = str(audio_files[0])
                logger.info(f"Converting {source_audio} to WAV format...")
                
                # Usa FFmpeg para converter para o formato correto
                try:
                    subprocess.run([
                        'ffmpeg', '-y',
                        '-i', source_audio,
                        '-ar', '16000',
                        '-ac', '1',
                        '-acodec', 'pcm_s16le',
                        final_path
                    ], check=True, capture_output=True)
                    
                    # Remove o arquivo original após conversão
                    os.remove(source_audio)
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
                    raise
            else:
                final_path = str(wav_files[0])
            
            # Verifica se o arquivo final existe e não está vazio
            if not os.path.exists(final_path):
                raise FileNotFoundError(f"Final audio file not found: {final_path}")
            
            if os.path.getsize(final_path) == 0:
                raise ValueError("Downloaded audio file is empty")
            
            logger.info(f"Audio downloaded and converted successfully: {final_path}")
            return final_path, video_title
            
        except Exception as e:
            logger.error(f"Error in download_youtube_video: {str(e)}")
            # Limpa arquivos parciais
            try:
                for f in self.downloads_dir.glob(f"{video_title}.*"):
                    os.remove(str(f))
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up partial downloads: {cleanup_error}")
            return None, None

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcreve o áudio usando o ModelManager.
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            
        Returns:
            Transcrição formatada ou None se houver erro
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
            language_info = f"\nIdioma detectado: {result.get('language', 'desconhecido')} "
            if 'confidence' in result:
                language_info += f"(Confiança: {result['confidence']:.2%})"
            
            return language_info + "\n\n" + transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Converte segundos para formato HH:MM:SS"""
        return str(timedelta(seconds=round(seconds)))

    @staticmethod
    def format_duration(seconds: int) -> str:
        """Formata a duração do vídeo"""
        return f"{seconds//3600:02d}h {(seconds%3600)//60:02d}min {seconds%60:02d}s"

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
            # Verificações iniciais
            logger.info("Checking system requirements...")
            if not check_ffmpeg():
                raise RuntimeError("FFmpeg is required but not found")
            
            # Verifica espaço em disco
            total, used, free = shutil.disk_usage("/")
            logger.info(f"Disk space - Total: {total // (2**30)} GiB, Free: {free // (2**30)} GiB")
            
            # Verifica permissões
            logger.info("Checking directory permissions...")
            if not os.access(self.downloads_dir, os.W_OK):
                raise PermissionError(f"No write access to downloads directory: {self.downloads_dir}")
            if not os.access(self.output_dir, os.W_OK):
                raise PermissionError(f"No write access to output directory: {self.output_dir}")
            
            # Obtém metadados
            metadata = self.get_video_metadata(url)
            if not metadata:
                raise ValueError("Failed to fetch video metadata")
            
            # Download do vídeo
            logger.info("Downloading audio from YouTube...")
            audio_path, video_title = self.download_youtube_video(url)
            
            if not audio_path or not video_title:
                raise ValueError("Failed to download video")
                
            # Verifica se o arquivo existe
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found at: {audio_path}")
            
            logger.info(f"Audio file ready at: {audio_path}")
            
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
                    video_data['url'] = url  # Garante que a URL está presente
                    self.obsidian.process_transcription(video_data, transcription)
                    self.obsidian.create_index_note()
                    logger.info(f"Created Obsidian note for video: {metadata.title}")
                except Exception as e:
                    logger.warning(f"Failed to create Obsidian note: {e}")
            
            # Limpa arquivos temporários
            if not self.config.keep_audio and audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info("Temporary audio file removed")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary audio file: {e}")
            
            # Exibe estatísticas do modelo
            stats = self.model_manager.get_stats()
            logger.info(f"Model stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return False

    def cleanup(self):
        """
        Limpa recursos do transcritor.
        Deve ser chamado quando o transcritor não for mais necessário.
        """
        try:
            # Limpa o modelo
            if hasattr(self, 'model_manager'):
                self.model_manager.cleanup()
                logger.info("Model manager cleaned up successfully")

            # Limpa arquivos temporários no diretório de downloads
            if self.downloads_dir.exists():
                for file in self.downloads_dir.glob("*"):
                    try:
                        if file.is_file() and not self.config.keep_audio:
                            file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {file}: {e}")
                logger.info("Temporary files cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def validate_url(self, url: str) -> bool:
        """
        Valida se a URL é do YouTube.
        
        Args:
            url: URL para validar
            
        Returns:
            bool: True se a URL é válida, False caso contrário
        """
        try:
            # Lista de domínios válidos do YouTube
            valid_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
            from urllib.parse import urlparse
            
            # Parse da URL
            parsed_url = urlparse(url)
            
            # Verifica se o domínio é válido
            domain = parsed_url.netloc.lower()
            is_valid = any(domain.endswith(d) for d in valid_domains)
            
            if not is_valid:
                logger.warning(f"Invalid YouTube URL: {url}")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating URL: {e}")
            return False

    def get_available_transcriptions(self) -> list:
        """
        Retorna uma lista de transcrições disponíveis.
        
        Returns:
            list: Lista de dicionários contendo informações sobre as transcrições
        """
        try:
            transcriptions = []
            for file in self.output_dir.glob("*_transcricao.txt"):
                try:
                    stats = file.stat()
                    transcriptions.append({
                        'name': file.name,
                        'size': stats.st_size,
                        'created': stats.st_ctime,
                        'modified': stats.st_mtime,
                        'path': str(file)
                    })
                except Exception as e:
                    logger.warning(f"Error processing transcription file {file}: {e}")
                    continue
                    
            return sorted(transcriptions, key=lambda x: x['modified'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting available transcriptions: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o modelo atual.
        
        Returns:
            Dict contendo informações sobre o modelo
        """
        try:
            return {
                'model_size': self.config.model_size,
                'device': self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                'batch_size': self.config.batch_size,
                'num_workers': self.config.num_workers,
                'compute_type': self.config.compute_type,
                'stats': self.model_manager.get_stats() if hasattr(self, 'model_manager') else None
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

    def get_disk_usage(self) -> Dict[str, int]:
        """
        Retorna informações sobre o uso de disco.
        
        Returns:
            Dict contendo informações sobre uso de disco
        """
        try:
            total, used, free = shutil.disk_usage("/")
            downloads_size = sum(f.stat().st_size for f in self.downloads_dir.glob('*') if f.is_file())
            transcriptions_size = sum(f.stat().st_size for f in self.output_dir.glob('*') if f.is_file())
            
            return {
                'total_space': total,
                'used_space': used,
                'free_space': free,
                'downloads_size': downloads_size,
                'transcriptions_size': transcriptions_size
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return {}

    def estimate_processing_time(self, duration_seconds: int) -> float:
        """
        Estima o tempo de processamento para um vídeo.
        
        Args:
            duration_seconds: Duração do vídeo em segundos
            
        Returns:
            float: Tempo estimado em segundos
        """
        try:
            # Fatores que influenciam o tempo de processamento
            device_factor = 0.5 if torch.cuda.is_available() else 1.0
            model_factors = {
                "tiny": 0.5,
                "base": 1.0,
                "small": 1.5,
                "medium": 2.0,
                "large": 3.0
            }
            
            model_factor = model_factors.get(self.config.model_size, 1.0)
            
            # Estimativa básica: tempo_video * fator_modelo * fator_dispositivo
            estimated_time = duration_seconds * model_factor * device_factor
            
            return estimated_time
            
        except Exception as e:
            logger.error(f"Error estimating processing time: {e}")
            return duration_seconds  # Retorna a própria duração como fallback

    @staticmethod
    def get_version() -> str:
        """
        Retorna a versão atual do Transcriber.
        
        Returns:
            str: Versão atual
        """
        return "1.0.0"