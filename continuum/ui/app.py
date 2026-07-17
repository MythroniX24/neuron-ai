"""
Continuum Chat UI — Flask Backend.

Serves a ChatGPT-style chat interface optimized for mobile devices.
Supports streaming token generation via Server-Sent Events (SSE).
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flask import Flask, render_template, request, Response, jsonify, stream_with_context
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Lazy-loaded model
_inference_engine = None
_tokenizer = None
_model_loaded = False


def get_model():
    """Lazy-load the model on first request."""
    global _inference_engine, _tokenizer, _model_loaded

    if not _model_loaded:
        try:
            from continuum.model.model import create_continuum_nano
            from continuum.inference.engine import ContinuumInference
            from continuum.tokenizer.bpe import ContinuumTokenizer

            print("Loading Continuum-Nano model...")
            model = create_continuum_nano()
            tokenizer = ContinuumTokenizer(vocab_size=model.config.vocab_size)

            engine = ContinuumInference(
                model=model,
                tokenizer=tokenizer,
                device="cpu",
                quantize=True,
            )

            _inference_engine = engine
            _tokenizer = tokenizer
            _model_loaded = True

            stats = engine.get_stats()
            print(f"Model loaded: {stats['model_params']:,} params, "
                  f"{stats['model_size_mb']:.1f} MB")

        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Running in demo mode (random responses).")
            _model_loaded = True  # Prevent retry

    return _inference_engine, _tokenizer


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def index():
    """Serve the chat interface."""
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Non-streaming chat endpoint."""
    data = request.get_json()
    message = data.get("message", "").strip()
    temperature = data.get("temperature", 0.8)
    top_k = data.get("top_k", 40)
    top_p = data.get("top_p", 0.9)
    max_tokens = data.get("max_tokens", 256)

    if not message:
        return jsonify({"error": "Empty message"}), 400

    engine, tokenizer = get_model()

    if engine is None:
        # Demo mode
        response_text = _demo_response(message)
    else:
        response_text = engine.generate(
            message,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stream=False,
        )

    return jsonify({
        "response": response_text,
        "timestamp": time.time(),
    })


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    data = request.get_json()
    message = data.get("message", "").strip()
    temperature = data.get("temperature", 0.8)
    top_k = data.get("top_k", 40)
    top_p = data.get("top_p", 0.9)
    max_tokens = data.get("max_tokens", 256)

    if not message:
        return jsonify({"error": "Empty message"}), 400

    engine, tokenizer = get_model()

    def generate_stream():
        if engine is None:
            # Demo mode: stream word by word
            demo = _demo_response(message)
            for word in demo.split():
                yield f"data: {json.dumps({'token': word + ' ', 'done': False})}\n\n"
                time.sleep(0.05)
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        else:
            for token in engine.generate(
                message,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stream=True,
            ):
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/conversation/new", methods=["POST"])
def new_conversation():
    """Start a new conversation."""
    engine, _ = get_model()
    if engine:
        msg = engine.start_conversation()
    else:
        msg = "New conversation started (demo mode)."
    return jsonify({"message": msg})


@app.route("/api/conversation/save", methods=["POST"])
def save_conversation():
    """Save current conversation state."""
    data = request.get_json()
    path = data.get("path", "conversation_state.pt")

    engine, _ = get_model()
    if engine:
        msg = engine.save_conversation(path)
    else:
        msg = "Demo mode: no state to save."
    return jsonify({"message": msg})


@app.route("/api/conversation/resume", methods=["POST"])
def resume_conversation():
    """Resume conversation from saved state."""
    data = request.get_json()
    path = data.get("path", "conversation_state.pt")

    engine, _ = get_model()
    if engine:
        msg = engine.resume_conversation(path)
    else:
        msg = "Demo mode: no state to resume."
    return jsonify({"message": msg})


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return model and conversation statistics."""
    engine, _ = get_model()
    if engine:
        return jsonify(engine.get_stats())
    return jsonify({
        "mode": "demo",
        "model_params": "5,000,000 (approx)",
        "model_size_mb": 20.0,
    })


# ============================================================================
# Demo mode responses (when model isn't loaded)
# ============================================================================

def _demo_response(message: str) -> str:
    """Generate a simple demo response for testing the UI."""
    import random

    responses = [
        f"I understand you're saying: \"{message[:50]}{'...' if len(message) > 50 else ''}\"\n\nThis is a demo response from Continuum SLM. The model will generate much better answers once trained!",
        f"Thanks for your message! In demo mode, I can show you the UI works, but I can't generate thoughtful responses yet.\n\nYour message was {len(message)} characters long.",
        f"Hello! 👋 I'm Continuum Nano (~5M parameters). I'm running locally on your device with no server connection needed. This demo shows the chat interface.",
    ]
    return random.choice(responses)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Continuum Chat Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--demo", action="store_true", help="Force demo mode (no model)")
    args = parser.parse_args()

    if args.demo:
        print("Running in demo mode (UI only, no model).")
    else:
        print("Starting Continuum Chat Server...")
        print(f"Open http://<your-device-ip>:{args.port} on your phone to chat!")

    app.run(host=args.host, port=args.port, debug=args.debug)
