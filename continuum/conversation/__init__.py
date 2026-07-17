# Continuum Conversational AI Module
# Chat templates, conversation management, and dataset pipelines

from .template import ChatTemplate, Conversation, Role
from .dataset import ConversationalDataset, get_openassistant_data, get_dolly_data
from .manager import ConversationManager

__all__ = [
    "ChatTemplate",
    "Conversation",
    "Role",
    "ConversationalDataset",
    "get_openassistant_data",
    "get_dolly_data",
    "ConversationManager",
]
