"""
Anthropometric LLM Matcher
Combines MediaPipe facial measurements with Claude LLM reasoning for phenotype classification
"""

import mediapipe as mp
import cv2
import numpy as np
import anthropic
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os


class AnthropometricLLMMatcher:
    """
    Hybrid phenotype matcher using:
    1. MediaPipe for precise facial landmark detection
    2. Anthropometric measurements (cephalic index, facial width, etc.)
    3. Claude LLM for intelligent reasoning about measurements
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize MediaPipe and Claude API"""

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Claude API
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

        # Key landmark indices for anthropometric measurements
        self.landmarks = {
            # Head outline
            'top_head': 10,
            'chin': 152,
            'left_head': 234,
            'right_head': 454,

            # Face width
            'left_cheek': 234,
            'right_cheek': 454,

            # Nose
            'nose_tip': 1,
            'nose_bridge': 6,
            'left_nose': 98,
            'right_nose': 327,

            # Eyes
            'left_eye_outer': 33,
            'left_eye_inner': 133,
            'right_eye_inner': 362,
            'right_eye_outer': 263,

            # Mouth
            'left_mouth': 61,
            'right_mouth': 291,
            'top_lip': 13,
            'bottom_lip': 14,

            # Jaw
            'left_jaw': 172,
            'right_jaw': 397,
        }

    def extract_comprehensive_profile(self, image_path: str) -> Dict:
        """
        Extract comprehensive anthropometric profile from a face image

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing all measurements and ratios
        """

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected in image")

        # Get landmarks
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape

        # Convert normalized landmarks to pixel coordinates
        landmarks_px = {}
        for name, idx in self.landmarks.items():
            landmark = face_landmarks.landmark[idx]
            landmarks_px[name] = (landmark.x * w, landmark.y * h, landmark.z * w)

        # Calculate all measurements
        measurements = self._calculate_measurements(landmarks_px)

        # Calculate derived ratios and indices
        profile = self._calculate_anthropometric_profile(measurements)

        return profile

    def _calculate_measurements(self, landmarks: Dict) -> Dict:
        """Calculate raw distances from landmarks"""

        def distance(p1, p2):
            """Euclidean distance between two 3D points"""
            return np.sqrt(
                (p1[0] - p2[0])**2 +
                (p1[1] - p2[1])**2 +
                (p1[2] - p2[2])**2
            )

        measurements = {
            # Head dimensions
            'head_height': distance(landmarks['top_head'], landmarks['chin']),
            'head_width': distance(landmarks['left_head'], landmarks['right_head']),

            # Face dimensions
            'face_width': distance(landmarks['left_cheek'], landmarks['right_cheek']),
            'face_height': distance(landmarks['nose_bridge'], landmarks['chin']),

            # Bizygomatic width (cheekbone to cheekbone)
            'bizygomatic_width': distance(landmarks['left_cheek'], landmarks['right_cheek']),

            # Nose dimensions
            'nose_width': distance(landmarks['left_nose'], landmarks['right_nose']),
            'nose_height': distance(landmarks['nose_bridge'], landmarks['nose_tip']),

            # Eye dimensions
            'eye_separation': distance(landmarks['left_eye_inner'], landmarks['right_eye_inner']),
            'left_eye_width': distance(landmarks['left_eye_outer'], landmarks['left_eye_inner']),
            'right_eye_width': distance(landmarks['right_eye_outer'], landmarks['right_eye_inner']),

            # Mouth dimensions
            'mouth_width': distance(landmarks['left_mouth'], landmarks['right_mouth']),
            'mouth_height': distance(landmarks['top_lip'], landmarks['bottom_lip']),

            # Jaw dimensions
            'jaw_width': distance(landmarks['left_jaw'], landmarks['right_jaw']),
        }

        return measurements

    def _calculate_anthropometric_profile(self, m: Dict) -> Dict:
        """
        Calculate anthropometric indices and ratios

        Args:
            m: Raw measurements dictionary

        Returns:
            Comprehensive anthropometric profile
        """

        # Prevent division by zero
        eps = 1e-6

        profile = {
            # Raw measurements
            'measurements': m,

            # Classical anthropometric indices
            'cephalic_index': (m['head_width'] / (m['head_height'] + eps)) * 100,

            # Facial proportions
            'facial_index': (m['face_height'] / (m['face_width'] + eps)) * 100,
            'upper_facial_index': (m['nose_height'] / (m['bizygomatic_width'] + eps)) * 100,

            # Nose indices
            'nasal_index': (m['nose_width'] / (m['nose_height'] + eps)) * 100,

            # Eye proportions
            'intercanthal_index': (m['eye_separation'] / (m['bizygomatic_width'] + eps)) * 100,
            'eye_width_ratio': ((m['left_eye_width'] + m['right_eye_width']) / 2) / (m['face_width'] + eps) * 100,

            # Mouth proportions
            'mouth_facial_width_ratio': (m['mouth_width'] / (m['face_width'] + eps)) * 100,
            'mouth_height_width_ratio': (m['mouth_height'] / (m['mouth_width'] + eps)) * 100,

            # Jaw proportions
            'jaw_face_width_ratio': (m['jaw_width'] / (m['face_width'] + eps)) * 100,

            # Overall proportions
            'face_head_width_ratio': (m['face_width'] / (m['head_width'] + eps)) * 100,
            'face_head_height_ratio': (m['face_height'] / (m['head_height'] + eps)) * 100,
        }

        # Add categorical classifications
        profile['head_shape'] = self._classify_head_shape(profile['cephalic_index'])
        profile['face_shape'] = self._classify_face_shape(profile['facial_index'])
        profile['nose_shape'] = self._classify_nose_shape(profile['nasal_index'])

        return profile

    def _classify_head_shape(self, cephalic_index: float) -> str:
        """Classify head shape based on cephalic index"""
        if cephalic_index < 75:
            return "dolichocephalic (long-headed)"
        elif cephalic_index < 80:
            return "mesocephalic (medium)"
        else:
            return "brachycephalic (round-headed)"

    def _classify_face_shape(self, facial_index: float) -> str:
        """Classify face shape based on facial index"""
        if facial_index < 85:
            return "euryprosopic (broad-faced)"
        elif facial_index < 90:
            return "mesoprosopic (medium)"
        else:
            return "leptoprosopic (narrow-faced)"

    def _classify_nose_shape(self, nasal_index: float) -> str:
        """Classify nose shape based on nasal index"""
        if nasal_index < 70:
            return "leptorrhine (narrow nose)"
        elif nasal_index < 85:
            return "mesorrhine (medium)"
        else:
            return "platyrrhine (broad nose)"

    def _format_profile_for_llm(self, profile: Dict) -> str:
        """Format anthropometric profile for LLM analysis"""

        return f"""
ANTHROPOMETRIC PROFILE:

HEAD MEASUREMENTS:
- Cephalic Index: {profile['cephalic_index']:.1f} ({profile['head_shape']})
- Face-to-Head Width Ratio: {profile['face_head_width_ratio']:.1f}%
- Face-to-Head Height Ratio: {profile['face_head_height_ratio']:.1f}%

FACIAL PROPORTIONS:
- Facial Index: {profile['facial_index']:.1f} ({profile['face_shape']})
- Upper Facial Index: {profile['upper_facial_index']:.1f}
- Jaw-to-Face Width Ratio: {profile['jaw_face_width_ratio']:.1f}%

NOSE:
- Nasal Index: {profile['nasal_index']:.1f} ({profile['nose_shape']})

EYES:
- Intercanthal Index: {profile['intercanthal_index']:.1f}%
- Eye Width Ratio: {profile['eye_width_ratio']:.1f}%

MOUTH:
- Mouth-to-Face Width Ratio: {profile['mouth_facial_width_ratio']:.1f}%
- Mouth Height-to-Width Ratio: {profile['mouth_height_width_ratio']:.1f}%
"""

    def compare_profiles_with_llm(
        self,
        user_profile: Dict,
        reference_profiles: Dict[str, Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Use Claude to analyze anthropometric profiles and find best matches

        Args:
            user_profile: User's anthropometric profile
            reference_profiles: Dictionary mapping phenotype names to their profiles
            top_k: Number of top matches to return

        Returns:
            List of matches with similarity scores and reasoning
        """

        # Format user profile
        user_profile_text = self._format_profile_for_llm(user_profile)

        # Format reference profiles (show key measurements only)
        reference_summaries = []
        for phenotype_name, ref_profile in reference_profiles.items():
            summary = f"""
{phenotype_name}:
- Cephalic: {ref_profile['cephalic_index']:.1f} ({ref_profile['head_shape']})
- Facial: {ref_profile['facial_index']:.1f} ({ref_profile['face_shape']})
- Nasal: {ref_profile['nasal_index']:.1f} ({ref_profile['nose_shape']})
- Upper Facial: {ref_profile['upper_facial_index']:.1f}
- Jaw Ratio: {ref_profile['jaw_face_width_ratio']:.1f}%
"""
            reference_summaries.append(summary)

        # Create prompt for Claude
        prompt = f"""You are an expert anthropologist analyzing facial anthropometric measurements.

USER'S MEASUREMENTS:
{user_profile_text}

REFERENCE PHENOTYPES:
{''.join(reference_summaries)}

Task: Based on these anthropometric measurements, determine the {top_k} most similar phenotypes.

Consider:
1. Cephalic index (head shape) - most important
2. Facial index (face proportions)
3. Nasal index (nose shape)
4. Upper facial index
5. Jaw and facial width ratios
6. Overall pattern of measurements

For each match, provide:
- Phenotype name
- Similarity score (0-100%)
- Brief reasoning (2-3 sentences max)

Return ONLY valid JSON in this exact format:
{{
  "matches": [
    {{
      "phenotype": "name",
      "similarity": 85.5,
      "reasoning": "explanation"
    }}
  ]
}}
"""

        # Call Claude API
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Parse response
        response_text = message.content[0].text

        # Extract JSON (handle markdown code blocks)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)

        return result['matches'][:top_k]

    def _calculate_measurement_similarity(
        self,
        profile1: Dict,
        profile2: Dict
    ) -> float:
        """
        Calculate numerical similarity between two anthropometric profiles

        Returns:
            Similarity score (0-100)
        """

        # Key indices to compare
        indices = [
            'cephalic_index',
            'facial_index',
            'nasal_index',
            'upper_facial_index',
            'jaw_face_width_ratio',
            'mouth_facial_width_ratio',
            'intercanthal_index',
        ]

        # Calculate weighted differences
        total_diff = 0
        weights = {
            'cephalic_index': 3.0,  # Most important
            'facial_index': 2.5,
            'nasal_index': 2.0,
            'upper_facial_index': 1.5,
            'jaw_face_width_ratio': 1.0,
            'mouth_facial_width_ratio': 0.5,
            'intercanthal_index': 0.5,
        }

        total_weight = sum(weights.values())

        for index in indices:
            v1 = profile1[index]
            v2 = profile2[index]

            # Normalized difference (0-1 scale)
            # Assume max reasonable difference is 30 points for indices
            diff = abs(v1 - v2) / 30.0
            diff = min(diff, 1.0)  # Cap at 1.0

            # Apply weight
            total_diff += diff * weights[index]

        # Convert to similarity (0-100)
        avg_diff = total_diff / total_weight
        similarity = (1.0 - avg_diff) * 100

        return max(0, min(100, similarity))  # Clamp to 0-100

    def analyze_user(
        self,
        user_image_path: str,
        reference_profiles: Dict[str, Dict],
        top_k: int = 10
    ) -> Dict:
        """
        Complete analysis pipeline: extract user profile and find matches

        Args:
            user_image_path: Path to user's photo
            reference_profiles: Pre-computed reference profiles
            top_k: Number of top matches to return

        Returns:
            Dictionary with user profile and ranked matches
        """

        # Extract user's profile
        user_profile = self.extract_comprehensive_profile(user_image_path)

        # Get LLM-based matches
        llm_matches = self.compare_profiles_with_llm(
            user_profile,
            reference_profiles,
            top_k=top_k
        )

        # Calculate measurement-based similarity for hybrid scoring
        for match in llm_matches:
            phenotype_name = match['phenotype']
            if phenotype_name in reference_profiles:
                measurement_sim = self._calculate_measurement_similarity(
                    user_profile,
                    reference_profiles[phenotype_name]
                )

                # Hybrid score: 60% LLM reasoning + 40% measurement similarity
                llm_score = match['similarity']
                hybrid_score = (0.6 * llm_score) + (0.4 * measurement_sim)

                match['measurement_similarity'] = measurement_sim
                match['llm_similarity'] = llm_score
                match['similarity'] = hybrid_score

        # Re-sort by hybrid score
        llm_matches.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'user_profile': user_profile,
            'matches': llm_matches[:top_k]
        }

    def cleanup(self):
        """Release MediaPipe resources"""
        self.face_mesh.close()
