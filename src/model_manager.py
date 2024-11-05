from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import logging
import whisper
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuração do modelo de transcrição"""
    model_size: str = "base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"
    batch_size: int = 16
    num_workers: int = 4
    beam_size: int = 5

class ModelManager:
    """Gerenciador do modelo PyTorch para transcrição"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Inicializa o gerenciador do modelo.
        
        Args:
            config: Configuração do modelo. Se None, usa configuração padrão.
        """
        self.config = config or ModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Configuração do dispositivo e ambiente
        self._setup_environment()
        
        # Carregamento do modelo
        self.model = self._load_model()
        
        # Estatísticas de uso
        self.stats = {
            "total_processed": 0,
            "avg_processing_time": 0,
            "gpu_memory_used": 0 if torch.cuda.is_available() else None
        }

    def _setup_environment(self) -> None:
        """Configura o ambiente PyTorch"""
        if self.config.device == "cpu":
            torch.set_num_threads(self.config.num_workers)
        
        torch.set_default_dtype(
            torch.float32 if self.config.compute_type == "float32" else torch.float16
        )
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
        self.logger.info(f"PyTorch configurado: device={self.config.device}, "
                      f"compute_type={self.config.compute_type}")

    def _load_model(self) -> whisper.Whisper:
        """Carrega e otimiza o modelo Whisper."""
        try:
            model = whisper.load_model(self.config.model_size)
            model = model.to(self.config.device)

            if self.config.device == "cuda":
                model = self._optimize_for_inference(model)

            self.logger.info(f"Modelo {self.config.model_size} carregado com sucesso")
            return model

        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise

    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Otimiza o modelo para inferência."""
        if self.config.device == "cuda" and self.config.compute_type == "float16":
            model = model.half()
        model.eval()
        return model

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Transcreve um arquivo de áudio."""
        try:
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            mel = whisper.log_mel_spectrogram(audio).to(self.config.device)
            
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            options = {
                "beam_size": self.config.beam_size,
                "language": detected_lang,
                "task": "transcribe"
            }
            
            result = self.model.transcribe(audio_path, **options)
            self._update_stats()
            
            return {
                "text": result["text"],
                "segments": result["segments"],
                "language": detected_lang,
                "confidence": probs[detected_lang]
            }
            
        except Exception as e:
            self.logger.error(f"Erro na transcrição: {str(e)}")
            raise

    def _update_stats(self) -> None:
        """Atualiza estatísticas de uso do modelo"""
        self.stats["total_processed"] += 1
        
        if torch.cuda.is_available():
            self.stats["gpu_memory_used"] = torch.cuda.max_memory_allocated() / 1024**3

    def get_stats(self) -> Dict[str, float]:
        """Retorna estatísticas de uso do modelo"""
        return self.stats.copy()

    def cleanup(self) -> None:
        """Limpa recursos do modelo"""
        if hasattr(self, 'model'):
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.logger.info("Recursos do modelo liberados")
