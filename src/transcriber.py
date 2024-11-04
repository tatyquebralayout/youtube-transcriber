# src/transcriber.py
import whisper
import yt_dlp
from pathlib import Path
import torch
import os

def download_youtube_video(url, output_dir="downloads"):
    """
    Baixa o áudio de um vídeo do YouTube
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(title)s.%(ext)s',
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
            audio_path = f"{output_dir}/{video_title}.mp3"
            return audio_path, video_title
    except Exception as e:
        print(f"Erro ao baixar vídeo: {str(e)}")
        return None, None

def transcribe_audio(audio_path, model_size="base"):
    """
    Transcreve um arquivo de áudio usando o modelo Whisper
    """
    try:
        model = whisper.load_model(model_size)
        print("Iniciando transcrição... Isso pode levar alguns minutos.")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Erro na transcrição: {str(e)}")
        return None

def process_youtube_video(url, output_dir="transcricoes", model_size="base", keep_audio=False):
    """
    Processa um vídeo do YouTube: baixa, extrai áudio e realiza a transcrição
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Iniciando processamento do vídeo do YouTube")
    
    print("Baixando áudio do YouTube...")
    audio_path, video_title = download_youtube_video(url)
    
    if not audio_path or not video_title:
        print("Falha ao baixar o vídeo.")
        return False
    
    transcription_path = f"{output_dir}/{video_title}_transcricao.txt"
    
    print("Transcrevendo áudio...")
    transcription = transcribe_audio(audio_path, model_size)
    
    if transcription:
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        if not keep_audio:
            os.remove(audio_path)
        
        print(f"Transcrição concluída! Arquivo salvo em: {transcription_path}")
        return True
    else:
        return False 
 
