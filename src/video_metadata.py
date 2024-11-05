from dataclasses import dataclass
from typing import List

@dataclass
class VideoMetadata:
    """Estrutura de dados para metadados do vídeo"""
    title: str
    url: str
    channel: str
    channel_url: str
    duration: str
    tags: List[str]
    view_count: int
    like_count: int
    upload_date: str
    transcript_date: str = None