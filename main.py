from src.transcriber import Transcriber
import os
from src.messages import Messages, TranscriptionState
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.theme import Theme
from rich.table import Table
from pathvalidate import sanitize_filename
from pathlib import Path
from typing import Dict, Tuple
import logging
import sys
from datetime import datetime

# Configuração do tema
THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "highlight": "magenta"
})

# Configuração dos modelos disponíveis
MODELS: Dict[str, Tuple[str, str, str]] = {
    "1": ("tiny", "Rápido, menor precisão", "~1GB RAM"),
    "2": ("base", "Equilibrado", "~2GB RAM"),
    "3": ("small", "Boa precisão", "~4GB RAM"),
    "4": ("medium", "Alta precisão", "~8GB RAM"),
    "5": ("large", "Máxima precisão", "~16GB RAM")
}
class YouTubeTranscriberCLI:
    """Interface de linha de comando para o YouTube Transcriber"""

    def __init__(self, vault_path: str):
        self.console = Console(theme=THEME)
        self.base_dir = Path(__file__).parent
        self.messages = Messages()  # Instancia o gerenciador de mensagens
        self.vault_path = vault_path  # Armazena o vault_path
        self.setup_logging()


    def setup_logging(self) -> None:
        """Configura o sistema de logging"""
        self.base_dir.joinpath('logs').mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_dir / 'logs/app.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self) -> None:
        """Configura os diretórios necessários"""
        directories = {
            'downloads': self.base_dir / 'downloads',
            'transcricoes': self.base_dir / 'transcricoes',
            'logs': self.base_dir / 'logs'
        }

        for name, path in directories.items():
            path.mkdir(parents=True, exist_ok=True)
            self.console.print(f"[info]✓ Diretório {name} configurado[/info]")

    def show_header(self) -> None:
        """Exibe o cabeçalho da aplicação"""
        header = Panel.fit(
            "[bold cyan]YouTube Transcriber[/bold cyan]\n"
            "[dim]Transforme vídeos em texto com facilidade[/dim]",
            border_style="cyan",
            padding=(1, 2),
            title="🎥 v1.0.0",
            subtitle="Desenvolvido por Seu Nome"
        )
        self.console.print(header)
        self.console.print()

    def check_system(self) -> str:
        """Verifica o sistema e retorna o dispositivo disponível"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        icon = "🚀" if device == "cuda" else "💻"
        
        self.console.print(f"[info]{icon} Sistema usando: {device.upper()}[/info]")
        return device

    def validate_url(self, url: str) -> bool:
        """Valida se a URL é do YouTube"""
        return any(domain in url.lower() for domain in ['youtube.com', 'youtu.be'])

    def get_url(self) -> str:
        """Obtém e valida a URL do vídeo"""
        while True:
            url = Prompt.ask(
                "[bold cyan]Digite a URL do YouTube[/bold cyan]",
                console=self.console
            )
            
            if self.validate_url(url):
                return url
            
            self.console.print(
                "[error]❌ URL inválida. Por favor, insira uma URL válida do YouTube.[/error]"
            )

    def show_model_selection(self) -> str:
        """Exibe o menu de seleção do modelo e retorna o modelo escolhido"""
        table = Table(
            title="Modelos Disponíveis",
            show_header=True,
            header_style="bold cyan",
            title_style="bold cyan"
        )
        
        table.add_column("Opção", style="cyan", justify="center")
        table.add_column("Modelo", style="white")
        table.add_column("Descrição", style="dim")
        table.add_column("Requisitos", style="yellow")

        for key, (model, desc, reqs) in MODELS.items():
            table.add_row(key, model, desc, reqs)

        self.console.print(table)
        self.console.print()

        while True:
            choice = Prompt.ask(
                "[bold cyan]Escolha o modelo (1-5)[/bold cyan]",
                default="2",
                console=self.console
            )
            
            if choice in MODELS:
                return MODELS[choice][0]
            
            self.console.print("[error]Escolha inválida. Tente novamente.[/error]")

    def process_video(self, url: str, model_size: str, keep_audio: bool) -> None:
    # ... código omitido ...

    try:
        transcriber = Transcriber(
            model_size=model_size,
            output_dir=self.base_dir / 'transcricoes',
            downloads_dir=self.base_dir / 'downloads',
            keep_audio=keep_audio,
            obsidian_vault=self.vault_path  # Usa o vault_path armazenado
        )
        
        success = transcriber.process_video(url)
        
        if success:
            self.show_success_message()
        else:
            self.show_error_message("Falha no processamento do vídeo")
            
    except Exception as e:
        self.logger.error(f"Erro no processamento: {str(e)}")
        self.show_error_message(str(e))


    def show_success_message(self) -> None:
        """Exibe mensagem de sucesso"""
        message = self.messages.get_message(100)  # Mensagem de conclusão
        success_panel = Panel.fit(
            f"{message}\n"
            "✨ [success]Processamento concluído com sucesso![/success]",
            border_style="green",
            padding=(1, 2)
        )
        self.console.print(success_panel)

    def show_error_message(self, error: str) -> None:
        """Exibe mensagem de erro"""
        message = self.messages.get_message(-1)  # Mensagem de erro
        error_panel = Panel.fit(
            f"{message}\n"
            f"❌ [error]Erro: {error}[/error]",
            border_style="red",
            padding=(1, 2)
        )
        self.console.print(error_panel)

    def run(self) -> None:
        """Executa a aplicação"""
        try:
            self.show_header()
            self.setup_directories()
            self.check_system()
            
            # Obter dados do usuário
            url = self.get_url()
            model_size = self.show_model_selection()
            keep_audio = Confirm.ask(
                "\n[bold cyan]Manter arquivo de áudio?[/bold cyan]",
                default=False,
                console=self.console
            )
            
            # Processar vídeo
            self.process_video(url, model_size, keep_audio)
            
        except KeyboardInterrupt:
            self.console.print("\n[warning]⚠️ Operação cancelada pelo usuário[/warning]")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Erro inesperado: {str(e)}")
            self.show_error_message(f"Erro inesperado: {str(e)}")
            sys.exit(1)

def main():
    # Configuração do vault do Obsidian
    VAULT_PATH = r"F:\pasta estudos\Code Brain"
    
    try:
        # Verifica se o caminho do vault existe
        if not os.path.exists(VAULT_PATH):
            raise ValueError(f"O caminho do vault '{VAULT_PATH}' não existe.")
        
        # Cria a instância do CLI passando o VAULT_PATH
        cli = YouTubeTranscriberCLI(vault_path=VAULT_PATH)
        cli.run()
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Erro na inicialização: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()