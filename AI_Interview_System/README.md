# AI Interview Analyzer

A complete **FREE** system for analyzing interview responses using AI. This tool automatically transcribes audio answers, scores responses on multiple dimensions, extracts skills, calculates relevance, and generates analytics summaries.

## 🎯 What This System Does

This AI Interview Analyzer evaluates job interview answers by performing 5 key tasks:

1. **📝 Audio Transcription** - Converts speech to text using free speech recognition
2. **📊 Multi-dimensional Scoring** - Evaluates answers on 5 criteria (1-10 scale)
3. **🔍 Skill Extraction** - Finds technical skills in both resume and answer
4. **🎯 Relevance Scoring** - Calculates similarity (0-1) between resume and answer skills
5. **📋 Analytics Summary** - Generates 3-5 bullet points with insights

## 🏗️ How It Works

### Architecture Flow

```
Audio File (MP3/M4A/WAV)
    ↓
[Step 1] Audio Transcription → Text transcript
[Step 2] Answer Scoring → 5 scores (1-10)
[Step 3] Skill Extraction → Skills from resume & answer
[Step 4] Similarity Calculation → Relevance score (0-1)
[Step 5] Summary Generation → 3-5 bullet points
    ↓
JSON Output with all results
```

### Approach & Methodology

#### Step 1: Audio Transcription
- **Method**: Google Speech Recognition API (free tier)
- **Process**: 
  - Converts audio file (MP3/M4A) to WAV format automatically
  - Sends audio to Google Speech Recognition
  - Returns transcribed text
- **Output**: Full text transcript of the interview answer

#### Step 2: Answer Scoring
- **Method**: Ollama + Llama3.2 LLM (local, free)
- **Process**:
  - Sends interview question and transcribed answer to LLM
  - LLM evaluates on 5 dimensions using structured prompts
  - Returns JSON with scores (1-10) for each dimension
- **Dimensions Evaluated**:
  - **Communication**: How well the candidate expresses thoughts
  - **Technical Relevance**: How relevant the answer is to the question
  - **Confidence**: How confident the candidate appears
  - **Clarity**: How clear and understandable the answer is
  - **Overall Quality**: Overall assessment

#### Step 3: Skill Extraction
- **Method**: Ollama + Llama3.2 LLM (local, free)
- **Process**:
  - Extracts technical skills from resume text using LLM
  - Extracts technical skills from transcribed answer using LLM
  - Returns lists of skills (programming languages, tools, frameworks, etc.)
- **Output**: Two lists - skills from resume and skills from answer

#### Step 4: Similarity Calculation
- **Method**: Sentence-BERT embeddings + Cosine Similarity
- **Process**:
  - Creates embeddings for all skills using Sentence-BERT model
  - Calculates average embedding for resume skills
  - Calculates average embedding for answer skills
  - Computes cosine similarity between the two embeddings
- **Output**: Relevance score between 0.0 (no match) and 1.0 (perfect match)

#### Step 5: Summary Generation
- **Method**: Ollama + Llama3.2 LLM (local, free)
- **Process**:
  - Sends all analysis results (scores, skills, relevance) to LLM
  - LLM generates concise 3-5 bullet point summary
  - Highlights key insights and areas for improvement
- **Output**: List of summary bullet points

### Technology Stack

- **Speech-to-Text**: Google Speech Recognition (free API, no key needed)
- **LLM**: Ollama + Llama3.2 (2GB model, runs locally, 100% free)
- **Embeddings**: Sentence-BERT all-MiniLM-L6-v2 (384 dimensions, local)
- **Audio Processing**: pydub (format conversion)
- **Similarity**: scikit-learn cosine similarity

## 🚀 Installation

### Prerequisites

- Python 3.8+
- macOS or Linux
- Homebrew (for macOS) or package manager

### Step 1: Install Ollama

```bash
# Install Ollama
brew install ollama

# Start Ollama server (keep this running in a terminal)
ollama serve
```

**In a NEW terminal**, download the model:

```bash
# Download Llama3.2 model (~2GB, one-time download)
# Downloads from: https://ollama.com/library/llama3.2
ollama pull llama3.2
```

### Step 2: Install Python Dependencies

```bash
# Navigate to project directory
cd /path/to/ai

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Basic Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis with example files
python main.py --question "Tell me about your Python experience" --audio examples/Audio.m4a --resume examples/sample_resume.txt --output results.json

# Or with your own files
python main.py \
  --question "Tell me about your Python experience" \
  --audio your_answer.m4a \
  --resume examples/sample_resume.txt \
  --output results.json
```

### Required Files

1. **Interview Question** (text)
   - Example: `"Tell me about your Python experience"`

2. **Audio Answer** (MP3, M4A, or WAV file)
   - Must contain actual speech (not music)
   - Supported: MP3, M4A, WAV, OGG, FLAC

3. **Resume Text** (text file)
   - Your resume as plain text
   - Example: `examples/sample_resume.txt`

### Output

The system creates a JSON file with:
- Full transcript
- Scores (communication, technical relevance, confidence, clarity, overall)
- Skills from resume
- Skills from answer
- Relevance score (0-1)
- Summary bullet points

Example output:

```json
{
  "transcript": "I have been working with Python for 5 years...",
  "scores": {
    "communication": 8.5,
    "technical_relevance": 9.0,
    "confidence": 7.5,
    "clarity": 8.0,
    "overall_quality": 8.2
  },
  "skills_resume": ["Python", "JavaScript", "Django", "AWS"],
  "skills_answer": ["Python", "Django", "REST APIs"],
  "relevance_score": 0.756,
  "summary": [
    "Strong technical relevance with score of 9.0/10",
    "Good alignment between resume and answer (75.6% match)"
  ]
}
```

## 🧪 Testing

### Quick System Test

Run the test script to verify everything works:

```bash
source venv/bin/activate
python test.py
```

This will test:
- Ollama connection
- Model availability
- Audio transcription
- All core functions

### Full Analysis Test

Test the complete system with example files:

```bash
source venv/bin/activate
python main.py --question "Tell me about your Python experience" --audio examples/Audio.m4a --resume examples/sample_resume.txt --output results.json
```

**Note**: Make sure `ollama serve` is running in another terminal before testing!

## 🌐 API Server (Optional)

Start the API server:

```bash
source venv/bin/activate
python run_api.py
```

Send a POST request:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "question=Tell me about your Python experience" \
  -F "resume_text=$(cat examples/sample_resume.txt)" \
  -F "audio_file=@your_answer.m4a"
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Ollama model
OLLAMA_MODEL = "llama3.2"  # Options: llama3.2, llama3.1, mistral, phi3

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

## 📁 Project Structure

```
.
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── main.py               # CLI entry point
├── run_api.py            # API server entry point
├── test.py               # Test script
├── src/                  # Source code
│   ├── analyzer.py       # Core analysis engine
│   ├── api.py            # FastAPI endpoint
│   ├── config.py         # Configuration settings
│   └── main.py           # CLI implementation
└── examples/             # Example files
    ├── sample_resume.txt  # Example resume
    └── Audio.m4a         # Example audio
```

## ⚠️ Troubleshooting

### "Ollama not running"
- Make sure `ollama serve` is running
- Check: `curl http://localhost:11434/api/tags`

### "Model not found"
- Pull the model: `ollama pull llama3.2`
- Check available: `ollama list`

### "Transcription failed"
- Ensure audio contains actual speech (not music)
- Check audio format is supported
- Install whisper-cpp for better quality: `brew install whisper-cpp`

### Low scores
- Give a real interview answer (not just "hello")
- Speak clearly and mention technical skills
- Reference skills from your resume

## 💰 Cost

**$0.00 - Everything is FREE!**

- Ollama + Llama3.2: Runs locally
- Google Speech Recognition: Free tier
- Sentence-BERT: Local embeddings
- All processing: On your machine

## 📚 Technical Details

- **Transcription**: Google Speech Recognition (free API)
- **LLM**: Llama3.2 via Ollama (2GB, downloads from https://ollama.com/library/llama3.2)
- **Embeddings**: Sentence-BERT all-MiniLM-L6-v2 (384 dimensions)
- **Performance**: ~15-30 seconds for a 1-minute interview answer
- **System Requirements**: 4GB+ RAM, ~5GB storage

---

**Ready to analyze interviews? Start Ollama and run your first analysis!** 🚀
