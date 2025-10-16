#!/usr/bin/env python3
"""
FastAPI Embedding Service for Phenotype Analysis
Provides REST API endpoints for anthropometric phenotype matching
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from anthropometric_llm_matcher import AnthropometricLLMMatcher
import pickle
import os
from pathlib import Path
from typing import Optional
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phenotype Analysis API",
    description="Anthropometric facial analysis using MediaPipe + Claude LLM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global matcher instance
matcher: Optional[AnthropometricLLMMatcher] = None
reference_profiles: Optional[dict] = None

# Configuration
REFERENCE_PROFILES_PATH = os.environ.get(
    "REFERENCE_PROFILES_PATH",
    "reference_profiles.pkl"
)


@app.on_event("startup")
async def startup_event():
    """Load reference profiles on startup"""
    global matcher, reference_profiles

    logger.info("🚀 Starting Phenotype Analysis API...")

    # Initialize matcher
    logger.info("🔧 Initializing AnthropometricLLMMatcher...")
    matcher = AnthropometricLLMMatcher()

    # Load reference profiles
    logger.info(f"📁 Loading reference profiles from {REFERENCE_PROFILES_PATH}...")

    if not Path(REFERENCE_PROFILES_PATH).exists():
        logger.error(f"❌ Reference profiles not found: {REFERENCE_PROFILES_PATH}")
        logger.error("Please run: python precompute_reference_profiles.py")
        raise FileNotFoundError(f"Reference profiles not found: {REFERENCE_PROFILES_PATH}")

    with open(REFERENCE_PROFILES_PATH, 'rb') as f:
        reference_profiles = pickle.load(f)

    logger.info(f"✅ Loaded {len(reference_profiles)} reference profiles")
    logger.info("🎉 API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global matcher

    if matcher:
        logger.info("🧹 Cleaning up resources...")
        matcher.cleanup()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Phenotype Analysis API",
        "version": "1.0.0",
        "reference_profiles": len(reference_profiles) if reference_profiles else 0
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "matcher_loaded": matcher is not None,
        "reference_profiles_loaded": reference_profiles is not None,
        "num_profiles": len(reference_profiles) if reference_profiles else 0
    }


@app.post("/analyze-phenotype")
async def analyze_phenotype(
    file: UploadFile = File(...),
    top_k: int = Form(10)
):
    """
    Analyze uploaded photo and return top phenotype matches

    Args:
        file: Image file (JPEG, PNG)
        top_k: Number of top matches to return (default: 10)

    Returns:
        JSON with user profile and ranked matches
    """

    if not matcher or not reference_profiles:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Reference profiles not loaded."
        )

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        try:
            # Save uploaded file
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            temp_path = tmp.name

            logger.info(f"📸 Analyzing image: {file.filename}")

            # Analyze
            result = matcher.analyze_user(
                temp_path,
                reference_profiles,
                top_k=top_k
            )

            logger.info(f"✅ Analysis complete. Top match: {result['matches'][0]['phenotype']}")

            return {
                "success": True,
                "filename": file.filename,
                "profile": {
                    "cephalic_index": result['user_profile']['cephalic_index'],
                    "head_shape": result['user_profile']['head_shape'],
                    "facial_index": result['user_profile']['facial_index'],
                    "face_shape": result['user_profile']['face_shape'],
                    "nasal_index": result['user_profile']['nasal_index'],
                    "nose_shape": result['user_profile']['nose_shape'],
                    "upper_facial_index": result['user_profile']['upper_facial_index'],
                    "jaw_face_width_ratio": result['user_profile']['jaw_face_width_ratio'],
                },
                "matches": result['matches']
            }

        except ValueError as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@app.post("/extract-profile")
async def extract_profile(file: UploadFile = File(...)):
    """
    Extract anthropometric profile from image without LLM matching
    Useful for debugging and testing

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        Anthropometric profile measurements
    """

    if not matcher:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Matcher not loaded."
        )

    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        try:
            # Save uploaded file
            contents = await file.read()
            tmp.write(contents)
            tmp.flush()
            temp_path = tmp.name

            logger.info(f"📸 Extracting profile from: {file.filename}")

            # Extract profile
            profile = matcher.extract_comprehensive_profile(temp_path)

            logger.info(f"✅ Profile extracted successfully")

            return {
                "success": True,
                "filename": file.filename,
                "profile": {
                    "cephalic_index": profile['cephalic_index'],
                    "head_shape": profile['head_shape'],
                    "facial_index": profile['facial_index'],
                    "face_shape": profile['face_shape'],
                    "nasal_index": profile['nasal_index'],
                    "nose_shape": profile['nose_shape'],
                    "upper_facial_index": profile['upper_facial_index'],
                    "jaw_face_width_ratio": profile['jaw_face_width_ratio'],
                    "mouth_facial_width_ratio": profile['mouth_facial_width_ratio'],
                    "intercanthal_index": profile['intercanthal_index'],
                }
            }

        except ValueError as e:
            logger.error(f"❌ Profile extraction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Profile extraction failed: {str(e)}")

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@app.get("/phenotypes")
async def list_phenotypes():
    """
    List all available reference phenotypes

    Returns:
        List of phenotype names and their profiles
    """

    if not reference_profiles:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Reference profiles not loaded."
        )

    phenotypes = []
    for name, profile in reference_profiles.items():
        phenotypes.append({
            "name": name,
            "cephalic_index": profile['cephalic_index'],
            "head_shape": profile['head_shape'],
            "facial_index": profile['facial_index'],
            "face_shape": profile['face_shape'],
            "nasal_index": profile['nasal_index'],
            "nose_shape": profile['nose_shape'],
        })

    return {
        "success": True,
        "count": len(phenotypes),
        "phenotypes": sorted(phenotypes, key=lambda x: x['name'])
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "embedding_service:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
