 # main.py
from src.transcriber import process_youtube_video
import torch

def main():
    # Verifica se CUDA está disponível para GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    # URL do vídeo do YouTube
    url = "https://www.youtube.com/watch?v=iyOkMhEXKeU"  # Substitua pela URL desejada
    
    # Configurações
    model_size = "base"  # Pode escolher entre: "tiny", "base", "small", "medium", "large"
    keep_audio = False   # Se True, mantém o arquivo MP3 após a transcrição
    
    success = process_youtube_video(url, model_size=model_size, keep_audio=keep_audio)
    
    if success:
        print("Processamento concluído com sucesso!")
    else:
        print("Houve um erro no processamento.")

if __name__ == "__main__":
    main()
