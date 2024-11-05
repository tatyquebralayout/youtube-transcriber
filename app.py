from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging
from pathlib import Path
import torch
from typing import Optional, Dict
from datetime import datetime
import os

from src.transcriber import Transcriber
from src.messages import Messages

app = Flask(__name__)

# Configuração de diretórios
BASE_DIR = Path(__file__).resolve().parent
TRANSCRICOES_DIR = BASE_DIR / 'transcricoes'
DOWNLOADS_DIR = BASE_DIR / 'downloads'
LOGS_DIR = BASE_DIR / 'logs'

# Certifique-se de que os diretórios existem
for directory in [TRANSCRICOES_DIR, DOWNLOADS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('YouTubeTranscriber')

class YouTubeTranscriberApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'your-secret-key'
        
        # Diretórios
        self.base_dir = Path(__file__).parent
        self.dirs = {
            'transcricoes': self.base_dir / 'transcricoes',
            'downloads': self.base_dir / 'downloads',
            'logs': self.base_dir / 'logs'
        }
        
        # Inicializa o gerenciador de mensagens
        self.messages_manager = Messages()
        
        self._setup_directories()
        self._setup_cuda()
        self._setup_transcriber()
        self._setup_routes()
        
        # Status das transcrições
        self.transcription_status = {}

    def _setup_directories(self):
        for name, path in self.dirs.items():
            if not path.exists():
                path.mkdir(parents=True)
                logger.info(f"Created directory: {path}")

    def _setup_cuda(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def _setup_transcriber(self):
        self.transcriber = Transcriber(
            output_dir=self.dirs['transcricoes'],
            downloads_dir=self.dirs['downloads']
        )

    def _setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/transcribe', 'transcribe', 
                            self.transcribe, methods=['POST'])
        self.app.add_url_rule('/status/<video_id>', 'status', self.status)
        self.app.add_url_rule('/transcricoes', 'list_transcricoes', 
                            self.list_transcricoes)
        self.app.add_url_rule('/download/<filename>', 'download', self.download)

    def update_status(self, video_id: str, status: str, progress: int, error: str = None):
        """Atualiza o status de uma transcrição"""
        message = self.messages_manager.get_message(progress)
        self.transcription_status[video_id] = {
            'status': status,
            'progress': progress,
            'message': message,
            'error': error
        }
        logger.debug(f"Status updated: {video_id} - {status} - {progress}% - {message}")

    def process_video(self, url: str, video_id: str):
        """Processa o vídeo em background"""
        try:
            self.update_status(video_id, 'downloading', 20)
            success = self.transcriber.process_video(url)
            
            if success:
                self.update_status(video_id, 'completed', 100)
                logger.info(f"Video processed successfully: {video_id}")
            else:
                self.update_status(video_id, 'error', -1, error='Falha na transcrição')
                logger.error(f"Failed to process video: {video_id}")
                
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            self.update_status(video_id, 'error', -1, error=str(e))

    def index(self):
        """Página inicial"""
        return render_template('index.html')

    def transcribe(self):
        """Endpoint de transcrição"""
        try:
            data = request.get_json()
            url = data.get('url')

            if not url:
                return jsonify({'error': 'URL não fornecida'}), 400

            from urllib.parse import urlparse
            video_id = self.get_video_id(url)
            if not video_id:
                return jsonify({'error': 'URL do YouTube inválida'}), 400

            from threading import Thread
            Thread(target=self.process_video, args=(url, video_id), 
                  daemon=True).start()
            
            logger.info(f"Started transcription for video: {video_id}")
            return jsonify({'video_id': video_id}), 202

        except Exception as e:
            logger.error(f"Error in transcribe endpoint: {e}")
            return jsonify({'error': str(e)}), 500

    def get_video_id(self, url: str) -> Optional[str]:
        """Extrai o ID do vídeo do YouTube"""
        from urllib.parse import urlparse, parse_qs
        try:
            parsed_url = urlparse(url)
            
            if parsed_url.netloc == 'youtu.be':
                return parsed_url.path[1:]
            
            if 'youtube.com' in parsed_url.netloc:
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query)['v'][0]
                if parsed_url.path.startswith(('/embed/', '/v/')):
                    return parsed_url.path.split('/')[2]
            
            return None
        except Exception as e:
            logger.error(f"Error extracting video ID: {e}")
            return None

    def status(self, video_id: str):
        """Status da transcrição"""
        status_data = self.transcription_status.get(video_id, {
            'status': 'not_found',
            'message': self.messages_manager.get_message(-1),
            'progress': 0
        })
        return jsonify(status_data)

    def list_transcricoes(self):
        """Lista de transcrições"""
        try:
            files = []
            for file in self.dirs['transcricoes'].glob('*_transcricao.txt'):
                files.append({
                    'name': file.name,
                    'date': datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    'size': file.stat().st_size
                })
            return jsonify({'files': sorted(files, key=lambda x: x['date'], 
                                          reverse=True)})
        except Exception as e:
            logger.error(f"Error listing transcriptions: {e}")
            return jsonify({'error': str(e)}), 500

    def download(self, filename: str):
        """Download de arquivo"""
        try:
            file_path = self.dirs['transcricoes'] / secure_filename(filename)
            if not file_path.is_file():
                return jsonify({'error': 'Arquivo não encontrado'}), 404
            return send_file(file_path, as_attachment=True, download_name=filename)
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return jsonify({'error': str(e)}), 500

    def get_app(self):
        """Retorna a aplicação Flask"""
        return self.app

def create_app():
    """Cria e retorna uma nova instância da aplicação"""
    transcriber_app = YouTubeTranscriberApp()
    return transcriber_app.get_app()

if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(host='0.0.0.0', port=5000, debug=True)