# 🤝 Contributing to Neuron AI (Continuum SLM)

Thank you for your interest in contributing! This is a custom small language model project built from scratch, and every contribution helps make it better.

## 🐛 Bug Reports & Feature Requests

- Open a [GitHub Issue](https://github.com/MythroniX24/neuron-ai/issues)
- Use a clear title and description
- For bugs: include error logs, Python version, and reproduction steps
- For features: explain the use case and expected behavior

## 💻 Code Contributions

### Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/your-username/neuron-ai.git
   cd neuron-ai
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Setup

```bash
pip install -r continuum/requirements.txt
pip install pytest pytest-cov
```

### Coding Guidelines

- **Python 3.10+** — use modern Python features (f-strings, type hints, dataclasses)
- **Type hints** — all function signatures must include type annotations
- **Docstrings** — Google-style docstrings for all public methods
- **No debug prints** — use `logging` or remove before committing
- **Tests** — add/update tests in `continuum/model/test_*.py`
- **Preserve architecture** — DO NOT change GLT, ADL, Anchor Attention, or PMB behavior without discussion

### Run Tests

```bash
cd continuum
python -m pytest model/test_model.py -v
python -m pytest model/test_layers.py -v
python -m pytest model/test_attention.py -v
```

### Commit Messages

```
[COMPONENT] Brief description of change

- Bullet point details
- Why the change was made
```

Examples:
```
[Trainer] Fix gradient accumulation for large batches

- Updated loss scaling for multi-GPU training
- Added scaler state checkpoint save/load
```

### Pull Request Process

1. Ensure all tests pass
2. Update the README.md if needed (API changes, new features)
3. Update `continuum-slm-architecture.md` if architecture is affected
4. Create a PR with a clear description of changes
5. Reference related issues

## 🧠 Architecture Contributions

If you want to propose architecture changes:

1. Open a **Discussion** first (not a PR)
2. Reference the relevant section in `continuum-slm-architecture.md`
3. Explain the tradeoffs — this project values honest limitations over marketing claims

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

**Questions?** Open a [Discussion](https://github.com/MythroniX24/neuron-ai/discussions) or an [Issue](https://github.com/MythroniX24/neuron-ai/issues).
