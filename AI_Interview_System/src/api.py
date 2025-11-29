"""FastAPI endpoint for the AI Interview Analyzer."""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from analyzer import InterviewAnalyzer

app = FastAPI(title="AI Interview Analyzer API", version="1.0.0")

analyzer = InterviewAnalyzer()


class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    question: str
    resume_text: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Interview Analyzer API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Upload audio file, question, and resume text for analysis"
        }
    }


@app.post("/analyze")
async def analyze_interview(
    question: str = Form(...),
    resume_text: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Analyze an interview response.
    
    Args:
        question: The interview question
        resume_text: The candidate's resume text
        audio_file: Audio file containing the candidate's answer
        
    Returns:
        JSON response with analysis results
    """
    # Validate file type
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm'}
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await audio_file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Run analysis
        result = analyzer.analyze(question, tmp_path, resume_text)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

