#!/usr/bin/env python3
"""
Test script for AnthropometricLLMMatcher
Tests the entire pipeline with a single image
"""

import sys
import pickle
from pathlib import Path
from anthropometric_llm_matcher import AnthropometricLLMMatcher
import json


def test_single_image(image_path: str, reference_profiles_path: str = "reference_profiles.pkl"):
    """
    Test the matcher with a single image

    Args:
        image_path: Path to test image
        reference_profiles_path: Path to reference profiles pickle file
    """

    print("="*70)
    print("ANTHROPOMETRIC LLM MATCHER - TEST")
    print("="*70)
    print()

    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    # Check if reference profiles exist
    if not Path(reference_profiles_path).exists():
        print(f"❌ Error: Reference profiles not found: {reference_profiles_path}")
        print()
        print("Please run first:")
        print("  python precompute_reference_profiles.py")
        sys.exit(1)

    # Load reference profiles
    print(f"📁 Loading reference profiles from {reference_profiles_path}...")
    with open(reference_profiles_path, 'rb') as f:
        reference_profiles = pickle.load(f)
    print(f"✅ Loaded {len(reference_profiles)} reference profiles")
    print()

    # Initialize matcher
    print("🔧 Initializing matcher...")
    matcher = AnthropometricLLMMatcher()
    print("✅ Matcher ready")
    print()

    # Analyze image
    print(f"📸 Analyzing image: {image_path}")
    print("   This will:")
    print("   1. Extract facial landmarks with MediaPipe")
    print("   2. Calculate anthropometric measurements")
    print("   3. Use Claude LLM to find best matches")
    print()

    try:
        result = matcher.analyze_user(image_path, reference_profiles, top_k=10)

        # Display user's profile
        print("="*70)
        print("YOUR ANTHROPOMETRIC PROFILE")
        print("="*70)
        print()

        profile = result['user_profile']
        print(f"📏 HEAD SHAPE:")
        print(f"   Cephalic Index: {profile['cephalic_index']:.1f}")
        print(f"   Classification: {profile['head_shape']}")
        print()

        print(f"📏 FACE SHAPE:")
        print(f"   Facial Index: {profile['facial_index']:.1f}")
        print(f"   Classification: {profile['face_shape']}")
        print()

        print(f"📏 NOSE SHAPE:")
        print(f"   Nasal Index: {profile['nasal_index']:.1f}")
        print(f"   Classification: {profile['nose_shape']}")
        print()

        print(f"📏 OTHER MEASUREMENTS:")
        print(f"   Upper Facial Index: {profile['upper_facial_index']:.1f}")
        print(f"   Jaw-to-Face Width Ratio: {profile['jaw_face_width_ratio']:.1f}%")
        print(f"   Mouth-to-Face Width Ratio: {profile['mouth_facial_width_ratio']:.1f}%")
        print(f"   Intercanthal Index: {profile['intercanthal_index']:.1f}%")
        print()

        # Display matches
        print("="*70)
        print("TOP 10 PHENOTYPE MATCHES")
        print("="*70)
        print()

        for i, match in enumerate(result['matches'], 1):
            print(f"{i}. {match['phenotype']}")
            print(f"   Overall Similarity: {match['similarity']:.1f}%")
            print(f"   ├─ LLM Reasoning Score: {match['llm_similarity']:.1f}%")
            print(f"   └─ Measurement Score: {match['measurement_similarity']:.1f}%")
            print()
            print(f"   Reasoning: {match['reasoning']}")
            print()

        # Summary
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print()
        print(f"✅ Analysis complete!")
        print(f"   Best match: {result['matches'][0]['phenotype']} ({result['matches'][0]['similarity']:.1f}%)")
        print(f"   Top 3 matches: {', '.join([m['phenotype'] for m in result['matches'][:3]])}")
        print()

        # Save detailed results
        output_file = "test_results.json"
        print(f"💾 Saving detailed results to {output_file}...")

        # Convert to JSON-serializable format
        json_result = {
            'image': image_path,
            'profile': {
                'cephalic_index': float(profile['cephalic_index']),
                'head_shape': profile['head_shape'],
                'facial_index': float(profile['facial_index']),
                'face_shape': profile['face_shape'],
                'nasal_index': float(profile['nasal_index']),
                'nose_shape': profile['nose_shape'],
                'upper_facial_index': float(profile['upper_facial_index']),
                'jaw_face_width_ratio': float(profile['jaw_face_width_ratio']),
                'mouth_facial_width_ratio': float(profile['mouth_facial_width_ratio']),
                'intercanthal_index': float(profile['intercanthal_index']),
            },
            'matches': [
                {
                    'rank': i,
                    'phenotype': m['phenotype'],
                    'similarity': float(m['similarity']),
                    'llm_similarity': float(m['llm_similarity']),
                    'measurement_similarity': float(m['measurement_similarity']),
                    'reasoning': m['reasoning'],
                }
                for i, m in enumerate(result['matches'], 1)
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(json_result, f, indent=2)

        print(f"✅ Results saved to {output_file}")
        print()

    except Exception as e:
        print()
        print("="*70)
        print("❌ ERROR")
        print("="*70)
        print()
        print(f"Analysis failed: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        matcher.cleanup()


def test_profile_extraction_only(image_path: str):
    """
    Test just the profile extraction (no LLM call)
    Useful for quick testing and debugging

    Args:
        image_path: Path to test image
    """

    print("="*70)
    print("PROFILE EXTRACTION TEST (No LLM)")
    print("="*70)
    print()

    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"📸 Extracting profile from: {image_path}")
    print()

    try:
        matcher = AnthropometricLLMMatcher()
        profile = matcher.extract_comprehensive_profile(image_path)

        print("✅ Profile extracted successfully!")
        print()
        print(f"Cephalic Index: {profile['cephalic_index']:.1f} ({profile['head_shape']})")
        print(f"Facial Index: {profile['facial_index']:.1f} ({profile['face_shape']})")
        print(f"Nasal Index: {profile['nasal_index']:.1f} ({profile['nose_shape']})")
        print(f"Upper Facial Index: {profile['upper_facial_index']:.1f}")
        print(f"Jaw-to-Face Width Ratio: {profile['jaw_face_width_ratio']:.1f}%")
        print()

        matcher.cleanup()

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_matcher.py <image_path> [reference_profiles.pkl]")
        print()
        print("  Or for quick profile extraction test:")
        print("  python test_matcher.py --extract-only <image_path>")
        print()
        print("Examples:")
        print("  python test_matcher.py test_portrait.jpg")
        print("  python test_matcher.py test_portrait.jpg my_profiles.pkl")
        print("  python test_matcher.py --extract-only test_portrait.jpg")
        sys.exit(1)

    if sys.argv[1] == "--extract-only":
        if len(sys.argv) < 3:
            print("❌ Error: Image path required")
            print("Usage: python test_matcher.py --extract-only <image_path>")
            sys.exit(1)
        test_profile_extraction_only(sys.argv[2])
    else:
        image_path = sys.argv[1]
        reference_profiles_path = sys.argv[2] if len(sys.argv) > 2 else "reference_profiles.pkl"
        test_single_image(image_path, reference_profiles_path)
