from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
import random
import threading

class TranscriptionState(Enum):
    """Estados possíveis do processo de transcrição"""
    STARTING = "starting"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    FINISHING = "finishing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class MessageTemplate:
    """Template para mensagens com emoji e texto"""
    text: str
    emoji: str

class Messages:
    """Gerenciador de mensagens do sistema"""

    def __init__(self, use_emojis: bool = True):
        self._message_counter: int = 0
        self._promo_counter: int = 0
        self._last_message: Optional[str] = None
        self._use_emojis = use_emojis
        self._lock = threading.Lock()
        self._initialize_messages()

    def _initialize_messages(self) -> None:
        """Inicializa todas as mensagens do sistema"""
        self._messages: Dict[TranscriptionState, List[MessageTemplate]] = {
            TranscriptionState.STARTING: [
                MessageTemplate("Preparando os sistemas", "🚀"),
                MessageTemplate("Iniciando o processo", "⚡"),
                MessageTemplate("Configurando o ambiente", "⚙️"),
            ],
            TranscriptionState.DOWNLOADING: [
                MessageTemplate("Baixando o vídeo", "📥"),
                MessageTemplate("Obtendo conteúdo", "🎥"),
                MessageTemplate("Preparando download", "💾"),
            ],
            TranscriptionState.PROCESSING: [
                MessageTemplate("Processando o áudio", "🎵"),
                MessageTemplate("Analisando conteúdo", "📊"),
                MessageTemplate("Preparando transcrição", "🔍"),
            ],
            TranscriptionState.TRANSCRIBING: [
                MessageTemplate("Convertendo áudio em texto", "📝"),
                MessageTemplate("Realizando transcrição", "✍️"),
                MessageTemplate("Processando o texto", "📋"),
            ],
            TranscriptionState.FINISHING: [
                MessageTemplate("Finalizando processo", "✨"),
                MessageTemplate("Aplicando últimos ajustes", "🔧"),
                MessageTemplate("Preparando resultado", "📦"),
            ],
            TranscriptionState.COMPLETED: [
                MessageTemplate("Processo concluído!", "✅"),
                MessageTemplate("Transcrição finalizada!", "🎉"),
                MessageTemplate("Tudo pronto!", "🌟"),
            ],
            TranscriptionState.ERROR: [
                MessageTemplate("Ops! Algo deu errado", "❌"),
                MessageTemplate("Encontramos um problema", "⚠️"),
                MessageTemplate("Erro no processamento", "🔴"),
            ]
        }

        self._promo_messages = [
            MessageTemplate("Gostou? Dê uma estrela no GitHub!", "⭐"),
            MessageTemplate("Contribua com o projeto no GitHub", "🌟"),
            MessageTemplate("Siga para mais atualizações", "👋"),
        ]

    def get_message(self, progress: int) -> str:
        """
        Retorna uma mensagem apropriada baseada no progresso atual.

        Args:
            progress: Número inteiro representando o progresso (0-100) ou -1 para erro

        Returns:
            Mensagem formatada com emoji (se habilitado)
        """
        with self._lock:
            self._message_counter += 1
            
            # Mapeia o progresso para um estado
            state = self._get_state_from_progress(progress)
            
            # Evita repetir a última mensagem
            available_messages = [
                msg for msg in self._messages[state]
                if msg.text != self._last_message
            ]

            if not available_messages:
                available_messages = self._messages[state]

            template = random.choice(available_messages)
            self._last_message = template.text

            return self._format_message(template)

    def _get_state_from_progress(self, progress: int) -> TranscriptionState:
        """
        Converte o progresso numérico em um estado de transcrição.

        Args:
            progress: Número inteiro representando o progresso (0-100) ou -1 para erro

        Returns:
            TranscriptionState correspondente
        """
        if progress < 0:
            return TranscriptionState.ERROR
        elif progress == 0:
            return TranscriptionState.STARTING
        elif progress < 30:
            return TranscriptionState.DOWNLOADING
        elif progress < 60:
            return TranscriptionState.PROCESSING
        elif progress < 80:
            return TranscriptionState.TRANSCRIBING
        elif progress < 100:
            return TranscriptionState.FINISHING
        else:
            return TranscriptionState.COMPLETED

    def _format_message(self, template: MessageTemplate) -> str:
        """Formata a mensagem com ou sem emojis."""
        if self._use_emojis:
            return f"{template.emoji} {template.text}"
        else:
            return template.text

    def reset(self) -> None:
        """Reseta o contador de mensagens e última mensagem."""
        with self._lock:
            self._message_counter = 0
            self._promo_counter = 0
            self._last_message = None