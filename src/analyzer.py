"""AI Interview Analyzer - Core analysis engine using FREE models."""
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Any
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config

# Try to import speech recognition (free, no API key needed for Google)
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Try to import faster-whisper (optional, better quality)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Try to import pydub for audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


class InterviewAnalyzer:
    """Main analyzer class for processing interview responses using free local models."""
    
    def __init__(self):
        """Initialize the analyzer with required models."""
        print("Loading models (this may take a moment on first run)...")
        
        # Try to load faster-whisper (best quality, optional)
        self.whisper_model = None
        if FASTER_WHISPER_AVAILABLE:
            try:
                print("Loading Whisper model (faster-whisper)...")
                self.whisper_model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
                print(" Faster-Whisper loaded")
            except Exception as e:
                print(f" Faster-Whisper not available: {str(e)}")
        
        # Initialize speech recognition (fallback, free Google API)
        self.speech_recognizer = None
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.speech_recognizer = sr.Recognizer()
                print(" Speech Recognition available (Google free API)")
            except Exception as e:
                print(f" Speech Recognition not available: {str(e)}")
        
        # Load embedding model (free, local)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        print(" Embedding model loaded")
        
        # Test Ollama connection
        try:
            ollama.list()
            print(" Ollama connected")
        except Exception as e:
            print(f" Warning: Ollama not running. Please start Ollama first.")
            print(f"  Install: https://ollama.com")
            print(f"  Then run: ollama pull {config.OLLAMA_MODEL}")
        
        print("All models ready!\n")
    
    def _convert_to_wav(self, audio_path: str) -> str:
        """Convert audio file to WAV format for speech recognition."""
        if not PYDUB_AVAILABLE:
            return None
        
        try:
            # Get file extension
            ext = os.path.splitext(audio_path)[1].lower()
            if ext == '.wav':
                return audio_path  # Already WAV
            
            print(f"Converting {ext} to WAV format...")
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav.close()
            
            # Export as WAV
            audio.export(temp_wav.name, format="wav")
            print(f" Converted to WAV: {temp_wav.name}")
            return temp_wav.name
        except Exception as e:
            print(f" Audio conversion failed: {str(e)}")
            return None
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using free methods (Whisper or Google Speech Recognition).
        
        Args:
            audio_path: Path to audio file (MP3, WAV, etc.)
            
        Returns:
            Transcribed text
        """
        temp_wav_file = None
        
        try:
            # Method 1: Try faster-whisper (best quality)
            if self.whisper_model:
                try:
                    print(f"Transcribing with faster-whisper: {audio_path}...")
                    segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments]).strip()
                    if transcript:
                        print(f" Transcription complete ({len(transcript)} characters)")
                        return transcript
                except Exception as e:
                    print(f" Faster-Whisper failed: {str(e)}, trying fallback...")
            
            # Method 2: Try Google Speech Recognition (free, no API key needed)
            if self.speech_recognizer:
                try:
                    # Convert to WAV if needed
                    wav_path = audio_path
                    if not audio_path.lower().endswith('.wav'):
                        wav_path = self._convert_to_wav(audio_path)
                        if wav_path:
                            temp_wav_file = wav_path
                        else:
                            raise Exception("Could not convert to WAV")
                    
                    print(f"Transcribing with Google Speech Recognition: {wav_path}...")
                    with sr.AudioFile(wav_path) as source:
                        # Adjust for ambient noise
                        self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = self.speech_recognizer.record(source)
                    
                    transcript = self.speech_recognizer.recognize_google(audio)
                    if transcript:
                        print(f" Transcription complete ({len(transcript)} characters)")
                        return transcript
                except sr.UnknownValueError:
                    print(" Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(f" Google Speech Recognition error: {str(e)}")
                except Exception as e:
                    print(f" Speech Recognition failed: {str(e)}")
            
            # Method 3: Try whisper-cpp (if installed via brew)
            try:
                print(f"Trying whisper-cpp: {audio_path}...")
                # whisper-cpp uses different command format
                result = subprocess.run(
                    ["whisper-cpp", "-m", "/opt/homebrew/share/whisper-cpp/ggml-base.en.bin", "-f", audio_path, "-t", "4"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0 and result.stdout:
                    transcript = result.stdout.strip()
                    if transcript:
                        print(f" Transcription complete ({len(transcript)} characters)")
                        return transcript
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                # Try alternative whisper-cpp paths
                try:
                    result = subprocess.run(
                        ["whisper-cpp", "-f", audio_path],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0 and result.stdout:
                        transcript = result.stdout.strip()
                        if transcript:
                            print(f" Transcription complete ({len(transcript)} characters)")
                            return transcript
                except:
                    pass
            
            raise Exception(
                "Transcription failed. Please install one of:\n"
                "  1. faster-whisper: pip install faster-whisper (may need Python 3.13)\n"
                "  2. whisper CLI: brew install whisper.cpp\n"
                "  3. Or ensure pydub is installed for audio conversion: pip install pydub"
            )
        finally:
            # Clean up temporary WAV file
            if temp_wav_file and os.path.exists(temp_wav_file):
                try:
                    os.unlink(temp_wav_file)
                except:
                    pass
    
    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call Ollama LLM (free, local).
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLM response
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = ollama.chat(
                model=config.OLLAMA_MODEL,
                messages=messages
            )
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Ollama call failed: {str(e)}. Make sure Ollama is running and model is pulled.")
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from text response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Try to find JSON array
        json_match = re.search(r'\[[^\[\]]*\]', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # If no JSON found, try parsing the whole text
        try:
            return json.loads(text)
        except:
            return {}
    
    def score_answer(self, question: str, answer: str) -> Dict[str, float]:
        """
        Score the candidate's answer on multiple dimensions using Ollama (FREE).
        
        Args:
            question: The interview question
            answer: The transcribed answer
            
        Returns:
            Dictionary with scores for each dimension
        """
        prompt = f"""You are an expert interview evaluator. Analyze the following interview answer and provide scores (1-10) for each dimension.

Interview Question: {question}

Candidate Answer: {answer}

Evaluate and provide scores for:
1. Communication: How well the candidate communicates their thoughts
2. Technical Relevance: How relevant the answer is to the technical question
3. Confidence: How confident the candidate appears
4. Clarity: How clear and understandable the answer is
5. Overall Quality: Overall assessment of the answer

Respond ONLY with a valid JSON object in this exact format:
{{
    "communication": <score 1-10>,
    "technical_relevance": <score 1-10>,
    "confidence": <score 1-10>,
    "clarity": <score 1-10>,
    "overall_quality": <score 1-10>
}}

Do not include any other text, only the JSON object."""

        try:
            system_prompt = "You are a precise JSON response generator. Always return valid JSON only."
            response = self._call_ollama(prompt, system_prompt)
            scores = self._extract_json(response)
            
            # Ensure all required keys exist with defaults
            default_scores = {
                "communication": 5.0,
                "technical_relevance": 5.0,
                "confidence": 5.0,
                "clarity": 5.0,
                "overall_quality": 5.0
            }
            
            for key in default_scores:
                if key not in scores:
                    scores[key] = default_scores[key]
                # Ensure scores are floats and in range
                scores[key] = float(max(1, min(10, scores[key])))
            
            return scores
        except Exception as e:
            print(f"Warning: Scoring failed, using default scores. Error: {str(e)}")
            return {
                "communication": 5.0,
                "technical_relevance": 5.0,
                "confidence": 5.0,
                "clarity": 5.0,
                "overall_quality": 5.0
            }
    
    def extract_skills_from_text(self, text: str, source: str = "answer") -> List[str]:
        """
        Extract skills mentioned in text using Ollama (FREE).
        
        Args:
            text: Text to extract skills from
            source: "resume" or "answer" for context
            
        Returns:
            List of extracted skills
        """
        # Limit text length for efficiency
        text_snippet = text[:1000] if len(text) > 1000 else text
        
        prompt = f"""Extract all technical skills, programming languages, tools, frameworks, and technologies mentioned in the following {source} text.

{source.capitalize()} Text:
{text_snippet}

Return ONLY a JSON array of skill names, like: ["Python", "JavaScript", "React", "AWS", "Docker"]
Do not include any other text, only the JSON array."""

        try:
            system_prompt = "You are a skill extraction tool. Return only JSON arrays."
            response = self._call_ollama(prompt, system_prompt)
            result = self._extract_json(response)
            
            # Handle different possible response formats
            if isinstance(result, list):
                skills = result
            elif isinstance(result, dict):
                skills = result.get("skills", result.get("skill_list", list(result.values())[0] if result else []))
            else:
                skills = []
            
            if isinstance(skills, str):
                skills = [skills]
            
            # Clean and filter skills
            cleaned_skills = []
            for skill in skills:
                if isinstance(skill, str) and len(skill.strip()) > 0:
                    cleaned_skills.append(skill.strip())
            
            return cleaned_skills if isinstance(cleaned_skills, list) else []
        except Exception as e:
            print(f"Warning: Skill extraction failed. Error: {str(e)}")
            return []
    
    def calculate_similarity(self, resume_skills: List[str], answer_skills: List[str]) -> float:
        """
        Calculate similarity score between resume skills and answer skills using Sentence-BERT (FREE).
        
        Args:
            resume_skills: List of skills from resume
            answer_skills: List of skills from answer
            
        Returns:
            Similarity score between 0 and 1
        """
        if not resume_skills or not answer_skills:
            return 0.0
        
        # Create embeddings for all skills
        all_skills = list(set(resume_skills + answer_skills))
        if not all_skills:
            return 0.0
        
        embeddings = self.embedding_model.encode(all_skills)
        
        # Create skill sets with indices
        resume_set = set(resume_skills)
        answer_set = set(answer_skills)
        
        # Get embeddings for resume and answer skills
        resume_indices = [i for i, skill in enumerate(all_skills) if skill in resume_set]
        answer_indices = [i for i, skill in enumerate(all_skills) if skill in answer_set]
        
        if not resume_indices or not answer_indices:
            return 0.0
        
        resume_embeddings = embeddings[resume_indices]
        answer_embeddings = embeddings[answer_indices]
        
        # Calculate average embeddings
        resume_avg = np.mean(resume_embeddings, axis=0).reshape(1, -1)
        answer_avg = np.mean(answer_embeddings, axis=0).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_avg, answer_avg)[0][0]
        
        # Ensure it's between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
    
    def generate_summary(self, question: str, answer: str, scores: Dict[str, float], 
                        resume_skills: List[str], answer_skills: List[str], 
                        relevance_score: float) -> List[str]:
        """
        Generate a short analytics summary using Ollama (FREE).
        
        Args:
            question: Interview question
            answer: Transcribed answer
            scores: Scoring dictionary
            resume_skills: Skills from resume
            answer_skills: Skills from answer
            relevance_score: Similarity score
            
        Returns:
            List of summary bullet points
        """
        answer_snippet = answer[:500] if len(answer) > 500 else answer
        
        prompt = f"""Generate a concise analytics summary (3-5 bullet points) for this interview analysis.

Interview Question: {question}
Answer: {answer_snippet}...

Scores:
- Communication: {scores.get('communication', 0)}
- Technical Relevance: {scores.get('technical_relevance', 0)}
- Confidence: {scores.get('confidence', 0)}
- Clarity: {scores.get('clarity', 0)}
- Overall Quality: {scores.get('overall_quality', 0)}

Resume Skills: {', '.join(resume_skills[:10])}
Answer Skills: {', '.join(answer_skills[:10])}
Skill Relevance Score: {relevance_score:.2f}

Return ONLY a JSON object with a "summary" key containing an array of 3-5 bullet point strings.
Format: {{"summary": ["bullet point 1", "bullet point 2", ...]}}
Do not include any other text."""

        try:
            system_prompt = "You are an analytics summary generator. Return only JSON."
            response = self._call_ollama(prompt, system_prompt)
            result = self._extract_json(response)
            summary = result.get("summary", [])
            
            if isinstance(summary, list) and len(summary) > 0:
                return summary
            else:
                # Fallback summary
                return [
                    f"Overall quality score: {scores.get('overall_quality', 0)}/10",
                    f"Skill relevance: {relevance_score:.2%} match between resume and answer",
                    f"Communication score: {scores.get('communication', 0)}/10"
                ]
        except Exception as e:
            print(f"Warning: Summary generation failed. Error: {str(e)}")
            return [
                f"Overall quality score: {scores.get('overall_quality', 0)}/10",
                f"Skill relevance: {relevance_score:.2%} match between resume and answer",
                f"Communication score: {scores.get('communication', 0)}/10"
            ]
    
    def analyze(self, question: str, audio_path: str, resume_text: str) -> Dict[str, Any]:
        """
        Complete analysis pipeline using FREE models.
        
        Args:
            question: Interview question
            audio_path: Path to audio file
            resume_text: Resume text content
            
        Returns:
            Complete analysis result as dictionary
        """
        print("Starting analysis with FREE models...")
        print("=" * 60)
        
        # Step 1: Transcribe audio
        print("\nStep 1/5: Transcribing audio (using local Whisper)...")
        transcript = self.transcribe_audio(audio_path)
        
        # Step 2: Score the answer
        print("\nStep 2/5: Scoring answer (using Ollama)...")
        scores = self.score_answer(question, transcript)
        
        # Step 3: Extract skills
        print("\nStep 3/5: Extracting skills (using Ollama)...")
        skills_resume = self.extract_skills_from_text(resume_text, "resume")
        skills_answer = self.extract_skills_from_text(transcript, "answer")
        
        # Step 4: Calculate similarity
        print("\nStep 4/5: Calculating skill similarity (using Sentence-BERT)...")
        relevance_score = self.calculate_similarity(skills_resume, skills_answer)
        
        # Step 5: Generate summary
        print("\nStep 5/5: Generating summary (using Ollama)...")
        summary = self.generate_summary(question, transcript, scores, skills_resume, 
                                       skills_answer, relevance_score)
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete!")
        print("=" * 60)
        
        return {
            "transcript": transcript,
            "scores": scores,
            "skills_resume": skills_resume,
            "skills_answer": skills_answer,
            "relevance_score": round(relevance_score, 3),
            "summary": summary
        }
