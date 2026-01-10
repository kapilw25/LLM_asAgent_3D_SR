"""
VLM Agent for Navigation Path Generation

Uses GPT-4V to generate navigation paths from room images.
Run on M1 Mac (API-based, no GPU required).

Usage:
    python src/m04_vlm_agent.py --image data/images/ai2thor/FloorPlan1/top_down.png
    python src/m04_vlm_agent.py --image data/images/ai2thor/FloorPlan1/ --task "bed to lamp"
    python src/m04_vlm_agent.py --test  # Run with sample image
"""

import os
import json
import base64
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPEN_API_KEY")


@dataclass
class VLMResponse:
    """Response from VLM agent."""
    path: List[str]
    raw_response: str
    success: bool
    error: Optional[str] = None


class VLMAgent:
    """
    Vision-Language Model agent for navigation.
    Uses GPT-4V to generate navigation paths from images.
    """

    VALID_DIRECTIONS = {"North", "South", "East", "West"}

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.api_key = OPENAI_API_KEY

        if not self.api_key:
            raise ValueError("OPEN_API_KEY not found in .env file")

    def generate_path(
        self,
        image_path: Path,
        start_object: str,
        goal_object: str,
        view_type: str = "top_down"
    ) -> VLMResponse:
        """
        Generate navigation path from image.

        Args:
            image_path: Path to room image (PNG)
            start_object: Starting object name (e.g., "bed")
            goal_object: Goal object name (e.g., "lamp")
            view_type: "top_down" or "first_person"

        Returns:
            VLMResponse with path and metadata
        """
        # Encode image
        base64_image = self._encode_image(image_path)

        # Build prompt
        prompt = self._build_prompt(start_object, goal_object, view_type)

        # Call GPT-4V
        try:
            response = self._call_gpt4v(base64_image, prompt)
            path = self._parse_path(response)

            return VLMResponse(
                path=path,
                raw_response=response,
                success=True
            )

        except Exception as e:
            return VLMResponse(
                path=[],
                raw_response="",
                success=False,
                error=str(e)
            )

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _build_prompt(
        self,
        start_object: str,
        goal_object: str,
        view_type: str
    ) -> str:
        """Build VLM prompt for navigation."""
        view_context = {
            "top_down": "This is a top-down (bird's eye) view of a room. North is up, South is down, East is right, West is left.",
            "first_person": "This is a first-person view from an agent in the room. Forward is North, backward is South, right is East, left is West."
        }

        return f"""You are a navigation agent. Analyze this room image and generate a navigation path.

{view_context.get(view_type, view_context["top_down"])}

TASK: Navigate from {start_object} to {goal_object}

RULES:
1. Output ONLY a JSON array of directions
2. Valid directions: "North", "South", "East", "West"
3. Each step moves one grid cell in that direction
4. Avoid obstacles (furniture, walls)
5. Find the shortest path possible

OUTPUT FORMAT (JSON array only, no explanation):
["North", "East", "North", "East"]

Generate the navigation path:"""

    def _call_gpt4v(self, base64_image: str, prompt: str) -> str:
        """Call GPT-4V API with image."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_path(self, response: str) -> List[str]:
        """Parse path from VLM response."""
        # Try to extract JSON array from response
        response = response.strip()

        # Handle markdown code blocks
        if "```" in response:
            # Extract content between code blocks
            parts = response.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    response = part
                    break

        # Try to parse as JSON
        try:
            path = json.loads(response)
            if isinstance(path, list):
                # Validate directions
                valid_path = []
                for direction in path:
                    if direction in self.VALID_DIRECTIONS:
                        valid_path.append(direction)
                    else:
                        print(f"Warning: Invalid direction '{direction}' ignored")
                return valid_path
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract directions from text
        path = []
        for direction in self.VALID_DIRECTIONS:
            # Count occurrences in order
            idx = 0
            while True:
                found = response.find(direction, idx)
                if found == -1:
                    break
                path.append((found, direction))
                idx = found + len(direction)

        # Sort by position and extract directions
        path.sort(key=lambda x: x[0])
        return [d for _, d in path]


def load_scene_metadata(scene_dir: Path) -> Optional[Dict]:
    """Load metadata.json from scene directory."""
    metadata_path = scene_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def run_demo():
    """Run demo with a sample or placeholder image."""
    print("=" * 60)
    print("VLM AGENT DEMO")
    print("=" * 60)

    # Check for sample images
    sample_dir = PROJECT_ROOT / "data" / "images" / "ai2thor"

    if sample_dir.exists():
        scenes = list(sample_dir.iterdir())
        scenes = [s for s in scenes if s.is_dir() and (s / "top_down.png").exists()]

        if scenes:
            scene = scenes[0]
            print(f"\nUsing scene: {scene.name}")

            agent = VLMAgent()

            # Load metadata to get objects
            metadata = load_scene_metadata(scene)
            if metadata and metadata.get("objects"):
                objects = metadata["objects"][:2]
                start = objects[0]["type"] if objects else "start"
                goal = objects[1]["type"] if len(objects) > 1 else "goal"
            else:
                start, goal = "start", "goal"

            # Test with top-down view
            top_down = scene / "top_down.png"
            print(f"\nTask: Navigate from {start} to {goal}")
            print(f"Image: {top_down}")

            result = agent.generate_path(top_down, start, goal, "top_down")

            print(f"\nResult:")
            print(f"  Success: {result.success}")
            print(f"  Path: {result.path}")
            print(f"  Path Length: {len(result.path)}")

            if result.error:
                print(f"  Error: {result.error}")

            return result

    print("\nNo AI2-THOR images found.")
    print("Run on GPU server first: python src/m03_ai2thor_capture.py --scenes 5")
    print("\nOr provide an image with: python src/m03_vlm_agent.py --image <path>")


def main():
    parser = argparse.ArgumentParser(
        description="VLM Agent for navigation path generation"
    )
    parser.add_argument(
        "--image", type=str,
        help="Path to room image or scene directory"
    )
    parser.add_argument(
        "--start", type=str, default="start",
        help="Starting object (default: start)"
    )
    parser.add_argument(
        "--goal", type=str, default="goal",
        help="Goal object (default: goal)"
    )
    parser.add_argument(
        "--task", type=str,
        help="Task description (e.g., 'bed to lamp')"
    )
    parser.add_argument(
        "--view", type=str, default="top_down",
        choices=["top_down", "first_person"],
        help="View type (default: top_down)"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model (default: gpt-4o)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run demo with sample image"
    )

    args = parser.parse_args()

    if args.test:
        run_demo()
        return

    if not args.image:
        parser.print_help()
        print("\n--- Running Demo ---")
        run_demo()
        return

    # Parse task if provided
    if args.task:
        parts = args.task.lower().split(" to ")
        if len(parts) == 2:
            args.start = parts[0].strip()
            args.goal = parts[1].strip()

    # Handle image path
    image_path = Path(args.image)

    if image_path.is_dir():
        # Scene directory - use top_down or first_person based on view type
        image_file = f"{args.view}.png"
        image_path = image_path / image_file

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"\nVLM Agent")
    print(f"  Image: {image_path}")
    print(f"  Task: {args.start} -> {args.goal}")
    print(f"  View: {args.view}")
    print(f"  Model: {args.model}")

    agent = VLMAgent(model=args.model)
    result = agent.generate_path(image_path, args.start, args.goal, args.view)

    print(f"\nResult:")
    print(f"  Success: {result.success}")
    print(f"  Path: {result.path}")
    print(f"  Path Length: {len(result.path)}")

    if result.error:
        print(f"  Error: {result.error}")

    if result.raw_response and not result.success:
        print(f"\nRaw Response:\n{result.raw_response}")


if __name__ == "__main__":
    main()
