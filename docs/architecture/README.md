# THAI-MOD Architecture Documentation

## Overview

This directory contains C4 architecture documentation for the THAI-MOD toxicity detection system,
organized in two versions:

- **v1 (Current)**: The system as deployed today -- TF-IDF + Logistic Regression baseline
- **v2 (Future)**: The target architecture -- WangchanBERTa + authentication + monitoring

## File Index

### C4 Diagrams

| File | C4 Level | Description |
|---|---|---|
| [c4-1-context.md](c4-1-context.md) | Level 1: Context | System scope, users, external systems. Shared across both versions. |
| [c4-2-container-v1-current.md](c4-2-container-v1-current.md) | Level 2: Container | Current containers: FastAPI, Moderator UI, Model Service, Cache, Datasets |
| [c4-2-container-v2-future.md](c4-2-container-v2-future.md) | Level 2: Container | Future containers: adds Auth Service, Monitoring Service, Metrics Store, Login UI, CI/CD |
| [c4-3-component-ml-v1-current.md](c4-3-component-ml-v1-current.md) | Level 3: Component | Current ML pipeline internals: TF-IDF vectorizer, LR classifier, cache manager, tokenizer |
| [c4-3-component-ml-v2-future.md](c4-3-component-ml-v2-future.md) | Level 3: Component | Future ML pipeline: BERT tokenizer, WangchanBERTa model, device manager, LR fallback |
| [c4-4-code-model-service.md](c4-4-code-model-service.md) | Level 4: Code | Class diagram, method signatures, sequence diagrams for model_service.py |

### Supporting Documents

| File | Description |
|---|---|
| [design-decisions.md](design-decisions.md) | 10 key design decisions with alternatives, rationale, and trade-offs |

### Reports (separate directory)

| File | Description |
|---|---|
| [../reports/lr-vs-bert-comparison.tex](../reports/lr-vs-bert-comparison.tex) | LaTeX report comparing LR baseline vs WangchanBERTa (compile with pdflatex or Overleaf) |

## How to Read

**For presentation**: Start with Level 1 (context), then Level 2 (containers), then Level 3 (ML components), then Level 4 (code). Use design-decisions.md for the Q&A section.

**For development**: Jump directly to the level relevant to your task:
- Adding a new external system? -> Level 1
- Adding a new container (e.g., auth service)? -> Level 2
- Changing ML pipeline internals? -> Level 3
- Modifying model_service.py? -> Level 4

## How to Update

### When P2-P5 are completed

Each task will affect specific files:

| Task | Files to update |
|---|---|
| P2 (Auth) | Update `c4-2-container-v1-current.md` to add auth containers, or just reference v2 as the new current |
| P3 (Testing + CI) | Update `c4-2-container-v1-current.md` to add CI/CD external system |
| P4 (Monitoring) | Update `c4-2-container-v1-current.md` to add monitoring containers |
| P5 (Docs) | No architecture changes needed |
| Model swap (LR -> BERT) | `c4-2-container-v1-current.md` becomes v2; update `c4-3-component-ml-v1-current.md` similarly |

### When v1 catches up to v2

Once all P2-P5 tasks are done and BERT is deployed:
1. The v2 files become the "current" documentation
2. You can either rename v2 files to replace v1, or simply note that v1 is historical
3. design-decisions.md stays relevant regardless

### Diagram format

All diagrams use **Mermaid** syntax. To render:
- GitHub renders Mermaid in markdown automatically
- VS Code: install "Markdown Preview Mermaid Support" extension
- Export to PNG: use `mmdc` CLI (`npm install -g @mermaid-js/mermaid-cli`) then `mmdc -i file.md -o file.png`
- Overleaf/LaTeX: export as PNG first, then include with `\includegraphics`

## Quick Reference: Current System

```
Moderator -> [Web UI] -> [FastAPI API] -> [ToxicityModelService] -> [TF-IDF + LR Pipeline]
                                |                   |
                          serves static        loads/saves
                            files              model cache
                                              (models/*.joblib)
```

## Quick Reference: Future System

```
Moderator -> [Login] -> [Auth] -> [Web UI] -> [FastAPI API] -> [ToxicityModelService]
                                                    |                   |
                                              [Monitoring]        [WangchanBERTa]
                                                    |              (GPU preferred)
                                              [Metrics Store]         |
                                                              [LR Fallback]
```
