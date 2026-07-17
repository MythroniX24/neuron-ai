"""
Chat Template System for Continuum Conversational AI.

Supports multiple chat formats:
- Llama-style: <|system|>...<|user|>...<|assistant|>...
- OpenAI-style: <|im_start|>system...<|im_end|>
- Simple: [INST]...[/INST]

The model uses special tokens reserved in the vocabulary.
"""

from enum import Enum
from typing import List, Optional, Dict, Tuple
import re


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message:
    """A single message in a conversation."""
    
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"<Message {self.role.value}: {self.content[:50]}...>"


class ChatTemplate:
    """
    Chat template for formatting conversations.
    
    Uses Llama-style format by default:
    <|system|>
    You are a helpful AI assistant.
    <|user|>
    Hello!
    <|assistant|>
    Hi! How can I help you today?
    """
    
    # Special tokens (must match tokenizer vocabulary)
    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    END_TOKEN = "<|end|>"
    
    # All special tokens for the chat format
    SPECIAL_TOKENS = [SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN]
    
    def __init__(self, add_generation_prompt: bool = True):
        self.add_generation_prompt = add_generation_prompt
    
    def format_messages(self, messages: List[Message]) -> str:
        """
        Format a list of messages into a single string.
        
        Args:
            messages: List of Message objects
        
        Returns:
            Formatted string ready for tokenization
        """
        parts = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                parts.append(f"{self.SYSTEM_TOKEN}\n{msg.content}\n{self.END_TOKEN}")
            elif msg.role == Role.USER:
                parts.append(f"{self.USER_TOKEN}\n{msg.content}\n{self.END_TOKEN}")
            elif msg.role == Role.ASSISTANT:
                parts.append(f"{self.ASSISTANT_TOKEN}\n{msg.content}\n{self.END_TOKEN}")
            elif msg.role == Role.TOOL:
                parts.append(f"{self.SYSTEM_TOKEN}\nTool: {msg.content}\n{self.END_TOKEN}")
        
        # Add generation prompt (for assistant to continue)
        if self.add_generation_prompt:
            parts.append(f"{self.ASSISTANT_TOKEN}\n")
        
        return "\n".join(parts)
    
    def format_conversation(
        self,
        system_prompt: str,
        user_message: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """
        Format a conversation with history for inference.
        
        Args:
            system_prompt: System prompt
            user_message: Current user message
            history: List of (user, assistant) tuples from previous turns
        
        Returns:
            Formatted string ready for tokenization
        """
        messages = []
        
        # System prompt
        if system_prompt:
            messages.append(Message(Role.SYSTEM, system_prompt))
        
        # History
        if history:
            for user_turn, assistant_turn in history:
                messages.append(Message(Role.USER, user_turn))
                messages.append(Message(Role.ASSISTANT, assistant_turn))
        
        # Current user message
        messages.append(Message(Role.USER, user_message))
        
        return self.format_messages(messages)
    
    def parse_response(self, response: str) -> str:
        """
        Parse model response to extract just the assistant's message.
        
        Removes any special tokens that might have been generated.
        """
        for token in self.SPECIAL_TOKENS:
            response = response.replace(token, "")
        return response.strip()
    
    def count_tokens_estimate(self, text: str) -> int:
        """Rough token count estimate (chars / 4 for English text)."""
        return len(text) // 4 + 1


class Conversation:
    """
    A single conversation session with history.
    """
    
    def __init__(
        self,
        system_prompt: str = "You are Continuum, a helpful, harmless, and honest AI assistant. You respond concisely and accurately.",
        max_history_turns: int = 10,
        template: Optional[ChatTemplate] = None,
    ):
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.template = template or ChatTemplate()
        self.history: List[Tuple[str, str]] = []  # (user, assistant) pairs
        self.messages: List[Message] = []
        
        # Add system prompt
        self.messages.append(Message(Role.SYSTEM, system_prompt))
    
    def add_user_message(self, message: str):
        """Add a user message to the conversation."""
        self.messages.append(Message(Role.USER, message))
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation."""
        self.messages.append(Message(Role.ASSISTANT, message))
        self.history.append((self.messages[-2].content, message))
        
        # Trim history if too long
        if len(self.history) > self.max_history_turns:
            excess = len(self.history) - self.max_history_turns
            # Keep system prompt, remove old history
            self.messages = [self.messages[0]]  # Keep system
            for user_turn, asst_turn in self.history[-self.max_history_turns:]:
                self.messages.append(Message(Role.USER, user_turn))
                self.messages.append(Message(Role.ASSISTANT, asst_turn))
    
    def get_prompt(self, add_generation_prompt: bool = True) -> str:
        """Get the formatted prompt for the model."""
        # Create a local template so we don't mutate self.template
        local_template = ChatTemplate(add_generation_prompt=add_generation_prompt)
        return local_template.format_messages(self.messages)
    
    def to_dict(self) -> Dict:
        """Serialize conversation to dict."""
        return {
            "system_prompt": self.system_prompt,
            "history": self.history,
            "messages": [(m.role.value, m.content) for m in self.messages],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Conversation":
        """Load conversation from dict."""
        conv = cls(system_prompt=data.get("system_prompt", ""))
        conv.history = data.get("history", [])
        conv.messages = [
            Message(Role(role), content) for role, content in data.get("messages", [])
        ]
        return conv
    
    def clear(self):
        """Clear conversation but keep system prompt."""
        self.history = []
        self.messages = [Message(Role.SYSTEM, self.system_prompt)]
