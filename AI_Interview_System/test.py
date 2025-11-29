#!/usr/bin/env python3
"""Test script for AI Interview Analyzer."""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all imports work."""
    print("Testing imports...")
    try:
        from analyzer import InterviewAnalyzer
        from config import OLLAMA_MODEL, EMBEDDING_MODEL
        import ollama
        from sentence_transformers import SentenceTransformer
        print(" All imports successful")
        return True
    except Exception as e:
        print(f" Import failed: {str(e)}")
        return False

def test_ollama():
    """Test Ollama connection."""
    print("\nTesting Ollama connection...")
    try:
        import ollama
        models = ollama.list()
        print(f" Ollama connected")
        
        # Handle different response formats
        if hasattr(models, 'models'):
            model_list = models.models
        elif isinstance(models, dict) and 'models' in models:
            model_list = models['models']
        elif isinstance(models, list):
            model_list = models
        else:
            model_list = []
        
        # Extract model names
        model_names = []
        for m in model_list:
            if hasattr(m, 'model'):
                model_names.append(m.model)
            elif isinstance(m, dict):
                model_names.append(m.get('name', m.get('model', '')))
            elif isinstance(m, str):
                model_names.append(m)
        
        if model_names:
            print(f"  Available models: {', '.join(model_names)}")
            if any('llama3.2' in name.lower() or 'llama' in name.lower() for name in model_names):
                llama_models = [n for n in model_names if 'llama' in n.lower()]
                print(f" Llama model found: {', '.join(llama_models)}")
                return True
            else:
                print(" Llama3.2 model not found. Run: ollama pull llama3.2")
                return False
        else:
            print(" No models found. Run: ollama pull llama3.2")
            return False
    except Exception as e:
        print(f" Ollama connection failed: {str(e)}")
        print("  Make sure 'ollama serve' is running")
        return False

def test_embedding_model():
    """Test embedding model."""
    print("\nTesting embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL
        
        print(f"Loading {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        test_embedding = model.encode('test')
        print(f" Embedding model loaded (dimension: {len(test_embedding)})")
        return True
    except Exception as e:
        print(f" Embedding model failed: {str(e)}")
        return False

def test_analyzer_init():
    """Test analyzer initialization."""
    print("\nTesting analyzer initialization...")
    try:
        from analyzer import InterviewAnalyzer
        analyzer = InterviewAnalyzer()
        print(" Analyzer initialized successfully")
        return True
    except Exception as e:
        print(f" Analyzer initialization failed: {str(e)}")
        return False

def test_audio_file():
    """Test if example audio file exists."""
    print("\nTesting example files...")
    audio_path = "examples/Audio.m4a"
    resume_path = "examples/sample_resume.txt"
    
    audio_exists = os.path.exists(audio_path)
    resume_exists = os.path.exists(resume_path)
    
    if audio_exists:
        print(f" Example audio found: {audio_path}")
    else:
        print(f" Example audio not found: {audio_path}")
    
    if resume_exists:
        print(f" Example resume found: {resume_path}")
    else:
        print(f" Example resume not found: {resume_path}")
    
    return audio_exists and resume_exists

def main():
    """Run all tests."""
    print("=" * 60)
    print("AI Interview Analyzer - Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Ollama", test_ollama()))
    results.append(("Embedding Model", test_embedding_model()))
    results.append(("Analyzer Init", test_analyzer_init()))
    results.append(("Example Files", test_audio_file()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed! System is ready to use.")
        return 0
    else:
        print("\n Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

