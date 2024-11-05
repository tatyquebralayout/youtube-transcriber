import os
import logging
from pathlib import Path
import yt_dlp
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class YouTubeAudioDownloader:
    """Classe responsável por baixar e processar áudio de vídeos do YouTube"""
    
    def __init__(self, downloads_dir: str):
        """
        Inicializa o downloader.
        
        Args:
            downloads_dir: Diretório para downloads temporários
        """
        self.downloads_dir = Path(downloads_dir)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        
    def download_audio(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Baixa o áudio de um vídeo do YouTube e converte para WAV.
        
        Args:
            url: URL do vídeo do YouTube
            
        Returns:
            Tuple contendo (caminho_do_arquivo_wav, título_do_vídeo) ou None se houver erro
        """
        try:
            # Configurações do yt-dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': str(self.downloads_dir / '%(title)s'),
                'quiet': True,
                'no_warnings': True,
                'extract_audio': True
            }
            
            # Primeiro, obtém as informações do vídeo
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'unknown_title')
                safe_title = "".join(x for x in video_title if (x.isalnum() or x in "._- "))
                
                # Atualiza o template de saída com o título seguro
                ydl_opts['outtmpl'] = str(self.downloads_dir / safe_title)
                
                # Agora faz o download
                logger.info(f"Downloading audio from: {url}")
                ydl.download([url])
                
                # O arquivo WAV terá o mesmo nome que o template, mas com extensão .wav
                wav_path = str(self.downloads_dir / f"{safe_title}.wav")
                
                if not os.path.exists(wav_path):
                    raise FileNotFoundError(f"WAV file not found after download: {wav_path}")
                
                logger.info(f"Audio downloaded successfully: {wav_path}")
                return wav_path, safe_title
                
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            # Limpa arquivos parciais que podem ter sido criados
            self._cleanup_partial_downloads(safe_title)
            return None
            
    def _cleanup_partial_downloads(self, base_name: str):
        """Limpa arquivos parciais de download que podem ter sido criados"""
        try:
            for ext in ['.wav', '.mp3', '.m4a', '.part', '.webm']:
                partial_file = self.downloads_dir / f"{base_name}{ext}"
                if partial_file.exists():
                    partial_file.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning up partial downloads: {str(e)}")