#!/usr/bin/env python3
"""
Precompute Reference Profiles
Processes all phenotype reference images once to build measurement database
Run this once to generate reference_profiles.pkl
"""

import pickle
from pathlib import Path
from anthropometric_llm_matcher import AnthropometricLLMMatcher
import json
from typing import Dict
import sys


def precompute_all_profiles(
    phenotype_images_dir: str = "phenotype_images",
    output_file: str = "reference_profiles.pkl"
) -> Dict:
    """
    Process all phenotype reference images and save profiles

    Args:
        phenotype_images_dir: Directory containing phenotype reference images
        output_file: Where to save the pickle file

    Returns:
        Dictionary of phenotype profiles
    """

    # Initialize matcher
    print("🔧 Initializing AnthropometricLLMMatcher...")
    matcher = AnthropometricLLMMatcher()

    # Find all images
    images_path = Path(phenotype_images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Directory not found: {phenotype_images_dir}")

    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(f"*{ext}"))
        image_files.extend(images_path.glob(f"*{ext.upper()}"))

    if not image_files:
        raise ValueError(f"No images found in {phenotype_images_dir}")

    print(f"📁 Found {len(image_files)} images in {phenotype_images_dir}")
    print()

    # Process each image
    reference_profiles = {}
    success_count = 0
    failed = []

    for i, image_path in enumerate(sorted(image_files), 1):
        # Extract phenotype name from filename
        # e.g., "Arabid_map.gif" or "Arabid.jpg" -> "Arabid"
        phenotype_name = image_path.stem
        # Remove common suffixes
        for suffix in ['_map', '_face', '_portrait', '_ref', '_reference']:
            if phenotype_name.endswith(suffix):
                phenotype_name = phenotype_name[:-len(suffix)]

        try:
            print(f"[{i}/{len(image_files)}] Processing {phenotype_name}...", end=" ")

            # Extract profile
            profile = matcher.extract_comprehensive_profile(str(image_path))

            # Save
            reference_profiles[phenotype_name] = profile

            # Success
            print(f"✅ Cephalic: {profile['cephalic_index']:.1f}, "
                  f"Facial: {profile['facial_index']:.1f}, "
                  f"Nasal: {profile['nasal_index']:.1f}")

            success_count += 1

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            failed.append((phenotype_name, str(e)))

    print()
    print("="*70)
    print(f"✅ Successfully processed: {success_count}/{len(image_files)}")

    if failed:
        print(f"❌ Failed: {len(failed)}")
        print("\nFailed images:")
        for name, error in failed:
            print(f"  - {name}: {error}")

    # Save to pickle
    print()
    print(f"💾 Saving profiles to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(reference_profiles, f)

    print(f"✅ Saved {len(reference_profiles)} profiles")

    # Also save as JSON for human inspection
    json_output = output_file.replace('.pkl', '.json')
    print(f"💾 Saving human-readable version to {json_output}...")

    # Convert to JSON-serializable format
    json_profiles = {}
    for name, profile in reference_profiles.items():
        json_profiles[name] = {
            'cephalic_index': round(profile['cephalic_index'], 2),
            'head_shape': profile['head_shape'],
            'facial_index': round(profile['facial_index'], 2),
            'face_shape': profile['face_shape'],
            'nasal_index': round(profile['nasal_index'], 2),
            'nose_shape': profile['nose_shape'],
            'upper_facial_index': round(profile['upper_facial_index'], 2),
            'jaw_face_width_ratio': round(profile['jaw_face_width_ratio'], 2),
            'mouth_facial_width_ratio': round(profile['mouth_facial_width_ratio'], 2),
            'intercanthal_index': round(profile['intercanthal_index'], 2),
        }

    with open(json_output, 'w') as f:
        json.dump(json_profiles, f, indent=2)

    print(f"✅ Saved human-readable profiles")

    # Cleanup
    matcher.cleanup()

    return reference_profiles


def verify_profiles(profiles_file: str = "reference_profiles.pkl"):
    """Verify the generated profiles file"""

    print(f"\n🔍 Verifying {profiles_file}...")

    with open(profiles_file, 'rb') as f:
        profiles = pickle.load(f)

    print(f"✅ Loaded {len(profiles)} profiles")
    print("\nSample profiles:")

    for i, (name, profile) in enumerate(list(profiles.items())[:5], 1):
        print(f"\n{i}. {name}:")
        print(f"   Cephalic Index: {profile['cephalic_index']:.1f} ({profile['head_shape']})")
        print(f"   Facial Index: {profile['facial_index']:.1f} ({profile['face_shape']})")
        print(f"   Nasal Index: {profile['nasal_index']:.1f} ({profile['nose_shape']})")

    return profiles


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        phenotype_dir = sys.argv[1]
    else:
        phenotype_dir = "phenotype_images"

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = "reference_profiles.pkl"

    print("="*70)
    print("PRECOMPUTE REFERENCE PROFILES")
    print("="*70)
    print()

    try:
        # Precompute
        profiles = precompute_all_profiles(phenotype_dir, output_file)

        # Verify
        verify_profiles(output_file)

        print()
        print("="*70)
        print("🎉 SUCCESS!")
        print("="*70)
        print()
        print("Next steps:")
        print(f"1. Your reference database is ready: {output_file}")
        print(f"2. Human-readable version: {output_file.replace('.pkl', '.json')}")
        print("3. You can now use this with the FastAPI service")
        print()
        print("Test it:")
        print("  python test_matcher.py <your_test_image.jpg>")

    except Exception as e:
        print()
        print("="*70)
        print("❌ ERROR")
        print("="*70)
        print(f"\n{str(e)}\n")
        sys.exit(1)
