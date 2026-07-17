"""
Conversation Manager — manages multi-turn conversation state.

Integrates with:
- ContinuumModel for generation
- PersistentMemoryBank for long-term topic retention
- ChatTemplate for proper formatting
- State serialization for app lifecycle
"""

import os
import json
import torch
from typing import Dict, List, Optional, Tuple, Generator

from continuum.conversation.template import ChatTemplate, Conversation, Message, Role
from continuum.model.model import ContinuumModel
from continuum.inference.engine import ContinuumInference


class ConversationManager:
    """
    End-to-end conversation manager for Continuum SLM.
    
    Handles multi-turn conversations with:
    - Chat template formatting
    - State persistence between turns
    - PMB-based long-term memory
    - System prompt management
    - Token streaming
    
    Usage:
        manager = ConversationManager(model, tokenizer)
        
        # Single turn
        response = manager.chat("Hello!")
        
        # Multi-turn (state carried automatically)
        response2 = manager.chat("What did I just say?")
        
        # Streaming
        for token in manager.chat_stream("Tell me a story"):
            print(token, end="")
        
        # Save state
        manager.save("conversation_state.pt")
    """
    
    def __init__(
        self,
        model: ContinuumModel,
        tokenizer,
        system_prompt: Optional[str] = None,
        device: str = "cpu",
        quantize: bool = False,
        max_history_turns: int = 10,
        max_context_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Default system prompt for Hindi/English mixed conversation
        self.system_prompt = system_prompt or (
            "You are Continuum, a helpful AI assistant. "
            "You can communicate in Hindi, English, or Hinglish as needed. "
            "You are helpful, harmless, and honest. "
            "Keep responses concise and accurate."
        )
        
        self.max_history_turns = max_history_turns
        self.max_context_length = max_context_length
        
        # Create inference engine
        self.engine = ContinuumInference(
            model=model,
            tokenizer=tokenizer,
            device=device,
            quantize=quantize,
        )
        
        # Current conversation
        self.conversation = Conversation(
            system_prompt=self.system_prompt,
            max_history_turns=max_history_turns,
        )
        
        # Generation parameters
        self.temperature = 0.8
        self.top_k = 40
        self.top_p = 0.9
        self.max_new_tokens = 512
        self.repetition_penalty = 1.05
    
    def chat(
        self,
        message: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Single chat turn (non-streaming).
        
        Args:
            message: User's message
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Assistant's response text
        """
        # Add user message
        self.conversation.add_user_message(message)
        
        # Get formatted prompt
        prompt = self.conversation.get_prompt(add_generation_prompt=True)
        
        # Generate response
        response = self.engine.generate(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature or self.temperature,
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            stream=False,
        )
        
        # Parse response (engine returns string when stream=False)
        response = self.conversation.template.parse_response(response)
        
        # Add to conversation history
        self.conversation.add_assistant_message(response)
        
        return response
    
    def chat_stream(
        self,
        message: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        Streaming chat turn.
        
        Yields tokens one at a time.
        """
        # Add user message
        self.conversation.add_user_message(message)
        
        # Get formatted prompt
        prompt = self.conversation.get_prompt(add_generation_prompt=True)
        
        # Generate streaming response
        collected_tokens = []
        for token in self.engine.generate(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature or self.temperature,
            top_k=kwargs.get("top_k", self.top_k),
            top_p=kwargs.get("top_p", self.top_p),
            repetition_penalty=kwargs.get("repetition_penalty", self.repetition_penalty),
            stream=True,
        ):
            collected_tokens.append(token)
            yield token
        
        # Parse and add to history
        full_response = "".join(collected_tokens)
        parsed = self.conversation.template.parse_response(full_response)
        self.conversation.add_assistant_message(parsed)
    
    def new_conversation(self, system_prompt: Optional[str] = None):
        """Start a new conversation, resetting state."""
        if system_prompt:
            self.system_prompt = system_prompt
        self.conversation = Conversation(
            system_prompt=self.system_prompt,
            max_history_turns=self.max_history_turns,
        )
        self.engine.start_conversation()
    
    def save_state(self, path: str):
        """Save full conversation state (messages + model states + PMB)."""
        state_dict = self.engine.get_state_dict()
        
        state = {
            "conversation": self.conversation.to_dict(),
            "model_state": state_dict if state_dict is not None else None,
            "config": {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
            },
        }
        
        torch.save(state, path)
        file_size_kb = os.path.getsize(path) / 1024
        return f"Saved to {path} ({file_size_kb:.0f} KB, {len(self.conversation.history)} turns)"
    
    def load_state(self, path: str):
        """Load conversation state from disk."""
        if path.endswith(".pt"):
            state = torch.load(path, map_location=self.device, weights_only=False)
            
            if "conversation" in state:
                self.conversation = Conversation.from_dict(state["conversation"])
            
            if "model_state" in state and state["model_state"]:
                # Load model states
                self.engine.resume_conversation(path)
            
            if "config" in state:
                self.temperature = state["config"].get("temperature", self.temperature)
                self.top_k = state["config"].get("top_k", self.top_k)
                self.top_p = state["config"].get("top_p", self.top_p)
            
            return f"Loaded conversation ({len(self.conversation.history)} turns)"
        
        elif path.endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
            self.conversation = Conversation.from_dict(data)
            return f"Loaded conversation ({len(self.conversation.history)} turns)"
        
        raise ValueError(f"Unknown format: {path}")
    
    def get_stats(self) -> Dict:
        """Get conversation and model statistics."""
        stats = {
            "num_turns": len(self.conversation.history),
            "system_prompt": self.system_prompt[:50] + "...",
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }
        
        # Add model stats
        engine_stats = self.engine.get_stats()
        stats.update(engine_stats)
        
        return stats
    
    def set_presets(self, preset: str = "balanced"):
        """
        Set generation parameters from preset.
        
        Presets:
        - 'creative': High temperature, diverse
        - 'balanced': Medium temperature
        - 'precise': Low temperature, focused
        """
        presets = {
            "creative": {"temperature": 0.95, "top_k": 60, "top_p": 0.95, "max_new_tokens": 512},
            "balanced": {"temperature": 0.8, "top_k": 40, "top_p": 0.90, "max_new_tokens": 512},
            "precise": {"temperature": 0.4, "top_k": 20, "top_p": 0.85, "max_new_tokens": 256},
        }
        params = presets.get(preset, presets["balanced"])
        self.temperature = params["temperature"]
        self.top_k = params["top_k"]
        self.top_p = params["top_p"]
        self.max_new_tokens = params["max_new_tokens"]
