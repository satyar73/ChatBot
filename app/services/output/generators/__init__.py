"""
Output generators module exports.
"""
from app.services.output.generators.docs_service import DocsService
from app.services.output.generators.slides_service import SlidesService

__all__ = ["DocsService", "SlidesService"]