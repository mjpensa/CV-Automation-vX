# CV-to-PowerPoint Automation System (Gold Standard)

**DETERMINISTIC** production-grade system for automated CV-to-PowerPoint generation with complete traceability.

## 🎯 Key Features

### DETERMINISM GUARANTEE ✓
- **Temperature: 0.0** (zero-variance output)
- **top_k: 1, top_p: 0.1** (maximum consistency)
- **Gemini 2.5 Flash Preview** (latest deterministic model)
- **Identical outputs** on repeated runs with same input

### 6-Agent Pipeline
1. **Agent 1 - Input Validator**: Multi-format CV parsing (.pdf, .docx, .txt)
2. **Agent 2 - Template Loader**: PowerPoint template analysis
3. **Agent 3 - Semantic Extractor**: AI-powered CV extraction with semantic enrichment
4. **Agent 4 - Mapping Engine**: Intelligent CV-to-placeholder mapping
5. **Agent 5 - Placeholder Analyzer**: Context-aware rule generation
6. **Agent 6 - Content Transformer**: Hard-limit enforcement & formatting

### Advanced Capabilities
- **Temporal Anchoring** (REQ-SEM-01): Distinguish current vs past roles
- **Industry Disambiguation** (REQ-SEM-03): Separate employer vs client industries
- **Hard Character Limits** (REQ-A6-01): Multi-stage truncation with smart boundaries
- **Mixed Markdown Formatting** (REQ-CON-01): **Bold** companies, *italic* dates, • bullets
- **Complete Traceability** (REQ-EXP-01): Before/after logging, JSON/Excel export

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key

### Installation

```bash
# Clone repository
git clone <repo-url>
cd CV-Automation-vX

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Basic Usage

```bash
python -m src.main --cv examples/resume.pdf --template templates/template.pptx
```

### Advanced Usage

```bash
# Specify output path
python -m src.main --cv resume.docx --template template.pptx --output result.pptx

# Show traceability
python -m src.main --cv resume.txt --template template.pptx --trace
```

## 📁 Project Structure

```
CV-Automation-vX/
├── src/
│   ├── agents/           # 6 agent implementations
│   ├── core/             # Schemas, validators, exceptions
│   ├── utils/            # Logger, file handlers
│   ├── pipeline/         # Orchestrator, traceability
│   └── main.py           # Entry point
├── config/               # Configuration files
├── outputs/              # Generated presentations
│   └── traceability/     # Traceability matrices
├── tests/                # Test suite
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash-preview-09-2025
TEMPERATURE=0.0           # MUST be 0.0 for determinism
```

### System Config (config/system_config.yaml)
```yaml
agents:
  agent3_semantic_extractor:
    temperature: 0.0      # DETERMINISM MANDATE
    top_p: 0.1
    top_k: 1
  agent6_content_transformer:
    temperature: 0.0      # DETERMINISM MANDATE
    hard_limit_enforcement: true

pipeline:
  deterministic_mode: true
  generate_traceability: true
```

## 📊 Output Files

After execution, you'll get:

1. **Generated PowerPoint**: `outputs/generated_pipeline_YYYYMMDD_HHMMSS.pptx`
2. **Traceability JSON**: `outputs/traceability/traceability_*.json`
3. **Traceability Excel**: `outputs/traceability/traceability_*.xlsx`
4. **System Log**: `outputs/system.log`

## 🧪 Testing Determinism

To verify deterministic output:

```bash
# Run twice with same input
python -m src.main --cv test.pdf --template template.pptx --output run1.pptx
python -m src.main --cv test.pdf --template template.pptx --output run2.pptx

# Compare outputs (should be identical)
diff run1.pptx run2.pptx
```

## 📋 Requirements

### Critical Non-Negotiable
- ✅ **REQ-DET-01**: Temperature = 0.0 (enforced in code)
- ✅ **REQ-DET-02**: Progressive retry with fallback
- ✅ **REQ-DET-03**: Deterministic fallback structures
- ✅ **REQ-A6-01**: Hard character limit enforcement
- ✅ **REQ-CON-01**: Mixed markdown formatting
- ✅ **REQ-CON-02**: Active voice conversion
- ✅ **REQ-EXP-01**: Complete before/after logging
- ✅ **REQ-EXP-02**: JSON & Excel export

### Semantic Intelligence
- ✅ **REQ-SEM-01**: Temporal anchoring with confidence scores
- ✅ **REQ-SEM-03**: Industry disambiguation (employer vs clients)
- ✅ **REQ-SEM-04**: Enhanced skill extraction from context

## 🔒 Quality Guarantees

- **Deterministic Output**: Identical results on re-run (temperature=0.0)
- **Hard Limits**: Never exceeds max_chars (multi-stage truncation)
- **Schema Validation**: JSON schema enforcement on all data
- **Error Handling**: Never fails silently (fallback structures)
- **Traceability**: Complete audit trail for every transformation

## 📝 Example Output

```
Pipeline Execution Summary:
  - Total Placeholders: 15
  - Mapped Placeholders: 12
  - Transformations: 12
  - Full Compliance: 10
  - Partial Compliance: 2
  - Quality Score: 95%
```

## 🛠️ Development

### Adding New Agents
All agents must inherit from `BaseAgent` with `temperature=0.0`:

```python
from src.agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("my_agent", temperature=0.0)  # REQUIRED
```

### Running Tests
```bash
pytest tests/ -v --cov=src
```

## 📚 Documentation

- **Architecture**: See `Implementation Guide`
- **API Reference**: Docstrings in all modules
- **Traceability Matrix**: Auto-generated in outputs/traceability/

## ⚠️ Important Notes

1. **API Key Required**: Set GEMINI_API_KEY before running
2. **Temperature Fixed**: Hardcoded to 0.0 for determinism
3. **File Formats**: Supports .pdf, .docx, .txt for CVs
4. **Template**: Must be valid .pptx PowerPoint file

## 🤝 Contributing

This is a production-grade deterministic system. Any changes must:
- Maintain temperature=0.0
- Include comprehensive tests
- Preserve traceability
- Update documentation

## 📄 License

[Add your license here]

## 🙋 Support

For issues or questions:
1. Check the Implementation Guide
2. Review traceability logs
3. Check system.log for errors
4. Verify API key configuration

---

**Built with DETERMINISM in mind. Temperature=0.0 guaranteed.**
