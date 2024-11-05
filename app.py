from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import logging
from pathlib import Path
import torch
from typing import Dict
from datetime import datetime
import os
import secrets
from threading import Thread

from src.messages import Messages
from src.transcriber import Transcriber, TranscriberConfig


# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('YouTubeTranscriber')

class YouTubeTranscriberApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = secrets.token_hex(32)
        
        # Configuração do Obsidian
        self.obsidian_vault = r"F:\pasta estudos\Code Brain"
        
        # Diretórios
        self.base_dir = Path(__file__).parent
        self.dirs = {
            'transcricoes': self.base_dir / 'transcricoes',
            'downloads': self.base_dir / 'downloads',
            'logs': self.base_dir / 'logs'
        }
        
        # Verifica se o caminho do vault existe
        if not os.path.exists(self.obsidian_vault):
            raise ValueError(f"O caminho do vault '{self.obsidian_vault}' não existe.")
        
        # Configuração do Transcriber
        transcriber_config = TranscriberConfig(
            model_size="base",
            output_dir=str(self.dirs['transcricoes']),
            downloads_dir=str(self.dirs['downloads']),
            obsidian_vault=self.obsidian_vault,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32",
            batch_size=16,
            num_workers=4
        )
        
        # Inicializa o transcriber com a nova configuração
        self.transcriber = Transcriber(config=transcriber_config)
        
        # Inicializa o gerenciador de mensagens
        self.messages_manager = Messages()
        
        # Status das transcrições
        self.transcription_status: Dict[str, dict] = {}
        
        # Configuração de diretórios e rotas
        self._setup_directories()
        self._setup_routes()

    def _setup_directories(self):
        """Configura os diretórios necessários"""
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {path}")

    def _setup_routes(self):
        """Configura as rotas da aplicação"""
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
            self.update_status(video_id, 'starting', 0)
            self.update_status(video_id, 'downloading', 20)
            
            success = self.transcriber.process_video(url)
            
            if success:
                self.update_status(video_id, 'completed', 100)
                logger.info(f"Video processed successfully: {video_id}")
            else:
                self.update_status(video_id, 'error', -1, 
                                 error='Falha no processamento do vídeo')
                
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

            video_id = self.get_video_id(url)
            if not video_id:
                return jsonify({'error': 'URL do YouTube inválida'}), 400

            Thread(target=self.process_video, args=(url, video_id), 
                  daemon=True).start()
            
            logger.info(f"Started transcription for video: {video_id}")
            return jsonify({'video_id': video_id}), 202

        except Exception as e:
            logger.error(f"Error in transcribe endpoint: {e}")
            return jsonify({'error': str(e)}), 500

    def get_video_id(self, url: str) -> str:
        """Extrai o ID do vídeo da URL do YouTube"""
        from urllib.parse import urlparse, parse_qs
        
        try:
            parsed_url = urlparse(url)
            
            if 'youtu.be' in parsed_url.netloc:
                return parsed_url.path[1:]
            
            if 'youtube.com' in parsed_url.netloc:
                if 'v' in parse_qs(parsed_url.query):
                    return parse_qs(parsed_url.query)['v'][0]
                if '/v/' in parsed_url.path:
                    return parsed_url.path.split('/v/')[1]
                if '/embed/' in parsed_url.path:
                    return parsed_url.path.split('/embed/')[1]
            
            return None
        except Exception:
            return None

    def status(self, video_id):
        """Status da transcrição"""
        status_data = self.transcription_status.get(video_id, {
            'status': 'not_found',
            'progress': 0,
            'message': self.messages_manager.get_message(0),
            'error': None
        })
        return jsonify(status_data)

    def list_transcricoes(self):
        """Lista de transcrições disponíveis"""
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

    def download(self, filename):
        """Download de arquivo"""
        try:
            file_path = self.dirs['transcricoes'] / secure_filename(filename)
            if not file_path.is_file():
                return jsonify({'error': 'Arquivo não encontrado'}), 404
            return send_file(file_path, as_attachment=True, 
                           download_name=filename)
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return jsonify({'error': str(e)}), 500

def create_app():
    """Cria e retorna uma nova instância da aplicação"""
    return YouTubeTranscriberApp().app

if __name__ == '__main__':
    flask_app = create_app()
    flask_app.run(host='0.0.0.0', port=5000, debug=True)