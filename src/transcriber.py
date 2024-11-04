import whisper
import yt_dlp
from pathlib import Path
import torch
import os
import warnings
from datetime import timedelta
from rich.console import Console

warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

console = Console()

def format_timestamp(seconds: float) -> str:
    """Converte segundos para formato HH:MM:SS"""
    return str(timedelta(seconds=round(seconds)))

def format_duration(seconds: int) -> str:
    """Formata a duração do vídeo"""
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if minutes > 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}min {remaining_seconds}s"
    return f"{minutes}min {remaining_seconds}s"

def get_video_metadata(url):
    """Obtém metadados do vídeo do YouTube"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Formata os dados
            metadata = {
                'title': info.get('title', 'Título não disponível'),
                'channel': info.get('uploader', 'Canal não disponível'),
                'channel_url': info.get('uploader_url', ''),
                'duration': format_duration(info.get('duration', 0)),
                'description': info.get('description', 'Descrição não disponível'),
                'tags': info.get('tags', []),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0),
                'upload_date': info.get('upload_date', '')
            }
            
            return metadata
    except Exception as e:
        console.print(f"Erro ao obter metadados: {str(e)}")
        return None

def format_header(metadata, video_url):
    """Formata o cabeçalho da transcrição com os metadados"""
    header = [
        "=" * 80,
        "INFORMAÇÕES DO VÍDEO",
        "=" * 80,
        "",
        f"Título: {metadata['title']}",
        f"Canal: {metadata['channel']}",
        f"Link do Canal: {metadata['channel_url']}",
        f"Link do Vídeo: {video_url}",
        f"Duração: {metadata['duration']}",
        "",
        "Descrição:",
        "-" * 40,
        metadata['description'],
        "",
        "Tags:",
        "-" * 40,
        ", ".join(metadata['tags']) if metadata['tags'] else "Nenhuma tag encontrada",
        "",
        "Estatísticas:",
        "-" * 40,
        f"Visualizações: {metadata['view_count']:,}",
        f"Likes: {metadata['like_count']:,}",
        "",
        "=" * 80,
        "TRANSCRIÇÃO",
        "=" * 80,
        ""
    ]
    
    return "\n".join(header)

def download_youtube_video(url, output_dir):
    """Baixa o áudio de um vídeo do YouTube"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info['title']
            audio_path = os.path.join(output_dir, f"{video_title}.mp3")
            return audio_path, video_title
    except Exception as e:
        console.print(f"Erro ao baixar vídeo: {str(e)}")
        return None, None

def transcribe_audio(audio_path, model_size="base"):
    """Transcreve um arquivo de áudio usando o modelo Whisper"""
    try:
        model = whisper.load_model(model_size)
        console.print("Iniciando transcrição... Isso pode levar alguns minutos.")
        
        result = model.transcribe(
            audio_path,
            language="pt",
            task="transcribe",
            verbose=False
        )
        
        formatted_transcription = []
        for segment in result["segments"]:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            formatted_transcription.append(f"[{start_time} -> {end_time}] {text}")
        
        return "\n".join(formatted_transcription)
    except Exception as e:
        console.print(f"Erro na transcrição: {str(e)}")
        return None

def process_youtube_video(url, output_dir="transcricoes", downloads_dir="downloads", model_size="base", keep_audio=False):
    """Processa um vídeo do YouTube: baixa, extrai áudio e realiza a transcrição"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(downloads_dir).mkdir(parents=True, exist_ok=True)
    
    console.print("Iniciando processamento do vídeo do YouTube")
    
    # Obtém metadados do vídeo
    metadata = get_video_metadata(url)
    if not metadata:
        console.print("Falha ao obter informações do vídeo.")
        return False
    
    # Cria o cabeçalho com os metadados
    header = format_header(metadata, url)
    
    console.print("Baixando áudio do YouTube...")
    audio_path, video_title = download_youtube_video(url, output_dir=downloads_dir)
    
    if not audio_path or not video_title:
        console.print("Falha ao baixar o vídeo.")
        return False
    
    transcription_path = os.path.join(output_dir, f"{video_title}_transcricao.txt")
    
    console.print("Transcrevendo áudio...")
    transcription = transcribe_audio(audio_path, model_size)
    
    if transcription:
        # Combina o cabeçalho com a transcrição
        full_content = f"{header}\n{transcription}"
        
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(full_content)
        
        if not keep_audio:
            os.remove(audio_path)
            console.print("Arquivo de áudio removido.")
        
        console.print(f"Transcrição concluída! Arquivo salvo em: {transcription_path}")
        return True
    else:
        return False