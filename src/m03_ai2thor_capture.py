"""
AI2-THOR Image Capture for VLM Navigation

Run on GPU Server to capture room images for VLM evaluation.
Captures 5 sample scenes (POC) with top-down and first-person views.

Usage (on GPU Server):
    python src/m03_ai2thor_capture.py --scenes 5     # Capture 5 scenes
    python src/m03_ai2thor_capture.py --list         # List available scenes
    python src/m03_ai2thor_capture.py --scene FloorPlan1  # Capture single scene
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# AI2-THOR import (only available on GPU server)
try:
    from ai2thor.controller import Controller
    AI2THOR_AVAILABLE = True
except ImportError:
    AI2THOR_AVAILABLE = False
    print("Warning: ai2thor not installed. Install with: pip install ai2thor")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "images" / "ai2thor"


def get_available_scenes() -> Dict[str, List[str]]:
    """Get available AI2-THOR scene categories."""
    return {
        "kitchens": [f"FloorPlan{i}" for i in range(1, 31)],
        "living_rooms": [f"FloorPlan{i}" for i in range(201, 231)],
        "bedrooms": [f"FloorPlan{i}" for i in range(301, 331)],
        "bathrooms": [f"FloorPlan{i}" for i in range(401, 431)],
    }


def get_poc_scenes(n: int = 5) -> List[str]:
    """Get n sample scenes for POC (mix of room types)."""
    poc_scenes = [
        "FloorPlan1",    # Kitchen
        "FloorPlan2",    # Kitchen
        "FloorPlan201",  # Living room
        "FloorPlan301",  # Bedroom
        "FloorPlan401",  # Bathroom
    ]
    return poc_scenes[:n]


class AI2THORCapture:
    """Capture images and metadata from AI2-THOR scenes."""

    def __init__(self, width: int = 600, height: int = 600):
        if not AI2THOR_AVAILABLE:
            raise RuntimeError("ai2thor not installed")

        self.width = width
        self.height = height
        self.controller = None

    def start(self):
        """Initialize AI2-THOR controller."""
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            scene="FloorPlan1",
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=self.width,
            height=self.height,
            fieldOfView=90,
        )
        print("AI2-THOR controller initialized")

    def stop(self):
        """Stop AI2-THOR controller."""
        if self.controller:
            self.controller.stop()
            print("AI2-THOR controller stopped")

    def capture_scene(self, scene_name: str, output_dir: Path) -> Dict:
        """
        Capture images and metadata for a single scene.

        Returns:
            Dict with scene metadata
        """
        # Reset to scene
        self.controller.reset(scene=scene_name)

        # Get scene metadata
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]

        # Get objects in scene
        objects = self._get_scene_objects()

        # Calculate grid bounds
        grid_info = self._calculate_grid_info(reachable_positions)

        # Capture first-person view (agent POV)
        first_person_img = self._capture_first_person()

        # Capture top-down view
        top_down_img = self._capture_top_down()

        # Create output directory
        scene_dir = output_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Save images
        first_person_path = scene_dir / "first_person.png"
        top_down_path = scene_dir / "top_down.png"

        self._save_image(first_person_img, first_person_path)
        self._save_image(top_down_img, top_down_path)

        # Create metadata
        metadata = {
            "scene_name": scene_name,
            "grid_size": grid_info["grid_size"],
            "grid_bounds": grid_info["bounds"],
            "reachable_count": len(reachable_positions),
            "objects": objects,
            "images": {
                "first_person": str(first_person_path.relative_to(PROJECT_ROOT)),
                "top_down": str(top_down_path.relative_to(PROJECT_ROOT)),
            }
        }

        # Save metadata
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Captured: {scene_name}")
        print(f"    - Grid: {grid_info['grid_size']}")
        print(f"    - Objects: {len(objects)}")
        print(f"    - Saved to: {scene_dir}")

        return metadata

    def _get_scene_objects(self) -> List[Dict]:
        """Get list of objects in current scene."""
        objects = []
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["pickupable"] or obj["receptacle"]:
                objects.append({
                    "name": obj["name"],
                    "type": obj["objectType"],
                    "position": {
                        "x": round(obj["position"]["x"], 2),
                        "y": round(obj["position"]["y"], 2),
                        "z": round(obj["position"]["z"], 2),
                    },
                    "pickupable": obj["pickupable"],
                    "receptacle": obj["receptacle"],
                })
        return objects

    def _calculate_grid_info(self, positions: List[Dict]) -> Dict:
        """Calculate grid info from reachable positions."""
        if not positions:
            return {"grid_size": (0, 0), "bounds": {}}

        x_coords = [p["x"] for p in positions]
        z_coords = [p["z"] for p in positions]

        min_x, max_x = min(x_coords), max(x_coords)
        min_z, max_z = min(z_coords), max(z_coords)

        # Estimate grid size (assuming 0.25 step size)
        step = 0.25
        rows = int((max_z - min_z) / step) + 1
        cols = int((max_x - min_x) / step) + 1

        return {
            "grid_size": (rows, cols),
            "bounds": {
                "min_x": round(min_x, 2),
                "max_x": round(max_x, 2),
                "min_z": round(min_z, 2),
                "max_z": round(max_z, 2),
            }
        }

    def _capture_first_person(self):
        """Capture first-person view from agent."""
        event = self.controller.step(action="Pass")
        return event.frame

    def _capture_top_down(self):
        """Capture top-down (bird's eye) view."""
        # Move camera to top-down position
        event = self.controller.step(
            action="ToggleMapView"
        )
        if event.metadata["lastActionSuccess"]:
            return event.frame

        # Fallback: use regular view if map view not available
        return self._capture_first_person()

    def _save_image(self, frame, path: Path):
        """Save numpy frame as PNG image."""
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(path)
        except ImportError:
            print("Warning: Pillow not installed. Cannot save images.")


def capture_scenes(scene_names: List[str], output_dir: Path):
    """Capture multiple scenes."""
    print(f"\nCapturing {len(scene_names)} scenes...")
    print(f"Output directory: {output_dir}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = AI2THORCapture()
    capture.start()

    results = []
    for scene in scene_names:
        try:
            metadata = capture.capture_scene(scene, output_dir)
            results.append(metadata)
        except Exception as e:
            print(f"  Error capturing {scene}: {e}")

    capture.stop()

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "scenes_captured": len(results),
            "scenes": [r["scene_name"] for r in results],
        }, f, indent=2)

    print(f"\nCapture complete: {len(results)}/{len(scene_names)} scenes")
    print(f"Summary saved to: {summary_path}")

    return results


def list_scenes():
    """List all available AI2-THOR scenes."""
    scenes = get_available_scenes()
    print("\nAvailable AI2-THOR Scenes:")
    print("-" * 40)
    for category, scene_list in scenes.items():
        print(f"\n{category.upper()} ({len(scene_list)} scenes):")
        print(f"  {scene_list[0]} - {scene_list[-1]}")

    print("\n" + "-" * 40)
    print("POC Scenes (recommended for testing):")
    for scene in get_poc_scenes(5):
        print(f"  - {scene}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture AI2-THOR images for VLM navigation"
    )
    parser.add_argument(
        "--scenes", type=int, default=5,
        help="Number of POC scenes to capture (default: 5)"
    )
    parser.add_argument(
        "--scene", type=str,
        help="Capture a single specific scene (e.g., FloorPlan1)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scenes"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR),
        help="Output directory for images"
    )

    args = parser.parse_args()

    if args.list:
        list_scenes()
        return

    if not AI2THOR_AVAILABLE:
        print("\nError: ai2thor not installed.")
        print("Install on GPU server with: pip install ai2thor")
        print("\nUse --list to see available scenes without ai2thor.")
        return

    output_dir = Path(args.output)

    if args.scene:
        # Capture single scene
        capture_scenes([args.scene], output_dir)
    else:
        # Capture POC scenes
        scenes = get_poc_scenes(args.scenes)
        capture_scenes(scenes, output_dir)


if __name__ == "__main__":
    main()
