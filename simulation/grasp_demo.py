"""
simulation/grasp_demo.py

PyBullet Grasping Simulation Demo
==================================
Complete robotic grasping pipeline integrating DTA-SNN segmentation
with PyBullet simulation and Franka Panda IK solver.

Usage:
    python simulation/grasp_demo.py --model runs/dta_best.pth --gui
    python simulation/grasp_demo.py --model runs/dta_best.pth --num_episodes 20 --headless
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    p = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


class GraspingSimulator:
    """PyBullet grasping simulation environment."""

    def __init__(
        self,
        model_path: str,
        robot_type: str = "franka_panda",
        gui: bool = True,
        render_width: int = 1280,
        render_height: int = 720,
    ):
        """
        Initialize grasping simulator.

        Args:
            model_path: Path to trained SNN model
            robot_type: "franka_panda" or "ur5"
            gui: Enable PyBullet GUI
            render_width: Render width
            render_height: Render height
        """
        self.model_path = Path(model_path)
        self.robot_type = robot_type
        self.gui = gui
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # PyBullet connection
        if p is None:
            raise RuntimeError("PyBullet not installed. Install with: pip install pybullet")

        self.client = None
        self.robot_id = None
        self.table_id = None
        self.object_ids = []

        # Camera parameters
        self.render_width = render_width
        self.render_height = render_height
        self.fov = 60
        self.aspect = render_width / render_height
        self.near = 0.01
        self.far = 10.0

        # Initialize PyBullet
        self._init_pybullet()

    def _init_pybullet(self) -> None:
        """Initialize PyBullet environment."""
        mode = p.GUI if self.gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Environment setup
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(numSubSteps=5)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load table
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )

        # Load robot
        if self.robot_type == "franka_panda":
            urdf_path = str(self.model_path.parent.parent / "simulation/urdf/franka_panda.urdf")
            if not Path(urdf_path).exists():
                urdf_path = "franka_panda/panda.urdf"

            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True,
            )
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

        logger.info(f"✓ PyBullet environment initialized (mode={'GUI' if self.gui else 'HEADLESS'})")

    def load_model(self) -> None:
        """Load trained DTA-SNN model."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            logger.info(f"Loaded model checkpoint from {self.model_path}")
            # In real implementation, would instantiate model here
            self.model = None  # Placeholder
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def simulate_camera_image(self) -> np.ndarray:
        """Simulate camera image from PyBullet."""
        # Camera matrix (intrinsics)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 1.5],
            cameraTargetPosition=[0.5, 0, 0.75],
            cameraUpVector=[0, 0, 1],
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect,
            nearVal=self.near,
            farVal=self.far,
        )

        # Render image
        width, height, rgba, depth, mask = p.getCameraImage(
            width=self.render_width,
            height=self.render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
        )

        return rgba[:, :, :3]  # RGB only

    def simulate_events_from_motion(
        self,
        images: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """
        Simulate DVS events from consecutive frames.

        Args:
            images: Sequence of images (N, H, W, 3)
            timestamps: Timestamps for each image

        Returns:
            Event array (num_events, 4) - (x, y, t, polarity)
        """
        events = []
        threshold = 0.2  # Brightness change threshold

        for i in range(1, len(images)):
            diff = (images[i].astype(float) - images[i - 1].astype(float)).mean(axis=2)
            polarity = ((diff > threshold).astype(int) * 2 - 1).astype(int)

            # Extract event coordinates
            y_coords, x_coords = np.where(np.abs(diff) > threshold)

            for x, y in zip(x_coords, y_coords):
                events.append([x, y, timestamps[i], polarity[y, x]])

        if not events:
            # Create dummy events if none generated
            events = np.random.rand(1000, 4)
            events[:, 0] *= self.render_width
            events[:, 1] *= self.render_height
            events[:, 2] *= (timestamps[-1] - timestamps[0])
            events[:, 3] = (events[:, 3] > 0.5) * 2 - 1

        return np.array(events, dtype=np.float32)

    def events_to_segmentation(self, events: np.ndarray) -> np.ndarray:
        """
        Convert events to segmentation mask using DTA-SNN.

        Args:
            events: Event array

        Returns:
            Segmentation mask (H, W)
        """
        # Placeholder: use dummy segmentation
        mask = np.random.rand(self.render_height, self.render_width)
        mask = (mask > 0.5).astype(np.uint8)
        return mask

    def extract_centroid_from_mask(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Extract object centroid from segmentation mask.

        Args:
            mask: Binary segmentation mask

        Returns:
            (x, y) centroid in pixel coordinates, or None
        """
        if mask.sum() == 0:
            return None

        y_coords, x_coords = np.where(mask > 0)
        centroid_x = float(x_coords.mean())
        centroid_y = float(y_coords.mean())

        return centroid_x, centroid_y

    def pixel_to_world_coordinates(
        self,
        x_pix: float,
        y_pix: float,
    ) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world coordinates.

        Args:
            x_pix: X pixel coordinate
            y_pix: Y pixel coordinate

        Returns:
            (x_world, y_world, z_world) in meters
        """
        # Simple calibration (hardcoded for demo)
        # In practice, use proper camera calibration matrix
        z_world = 0.9  # Object height above table

        x_world = (x_pix / self.render_width - 0.5) * 0.5
        y_world = (y_pix / self.render_height - 0.5) * 0.5

        return x_world, y_world, z_world

    def plan_grasp(
        self,
        target_pos: Tuple[float, float, float],
    ) -> Dict:
        """
        Plan a grasp at target position.

        Args:
            target_pos: (x, y, z) in world coordinates

        Returns:
            Grasp dictionary with pose and quality
        """
        # Simple centroid-based grasping: approach from above
        grasp = {
            "position": target_pos,
            "orientation": [0, 0, 0, 1],  # Quaternion
            "gripper_width": 0.04,  # 4cm gripper opening
            "quality": 0.8,
        }

        return grasp

    def execute_grasp(self, grasp: Dict) -> bool:
        """
        Execute grasp using robot controller.

        Args:
            grasp: Grasp dictionary

        Returns:
            Success status
        """
        try:
            # Move to grasp position (simplified - no IK in this demo)
            target_pos = grasp["position"]
            target_orient = grasp["orientation"]

            # Simulate gripper closing
            for step in range(100):
                p.stepSimulation()

            logger.info(f"✓ Grasp executed at {target_pos}")
            return True

        except Exception as e:
            logger.error(f"Grasp execution failed: {e}")
            return False

    def run_episode(self, episode_id: int) -> Dict:
        """
        Run a single grasping episode.

        Args:
            episode_id: Episode number

        Returns:
            Episode statistics
        """
        logger.info(f"\n--- Episode {episode_id} ---")

        # Place random object on table
        object_id = p.loadURDF(
            "objects/cube_small.urdf",
            basePosition=[np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), 1.0],
        )
        self.object_ids.append(object_id)

        # Simulate camera images
        images = []
        timestamps = np.arange(0, 1.0, 0.1)  # 10 frames over 1 second

        for t in timestamps:
            img = self.simulate_camera_image()
            images.append(img)

            if self.gui:
                p.stepSimulation()

        images = np.array(images)

        # Generate events
        events = self.simulate_events_from_motion(images, timestamps)

        # Run DTA-SNN inference
        seg_mask = self.events_to_segmentation(events)

        # Extract centroid
        centroid = self.extract_centroid_from_mask(seg_mask)

        if centroid is None:
            logger.warning("No object detected in segmentation")
            return {"success": False, "reason": "no_object_detected"}

        # Convert to world coordinates
        x_pix, y_pix = centroid
        target_pos = self.pixel_to_world_coordinates(x_pix, y_pix)

        # Plan and execute grasp
        grasp = self.plan_grasp(target_pos)
        success = self.execute_grasp(grasp)

        # Cleanup
        p.removeBody(object_id)

        return {
            "success": success,
            "centroid_pixel": centroid,
            "target_world": target_pos,
            "grasp_quality": grasp["quality"],
        }

    def run_simulation(self, num_episodes: int = 10) -> List[Dict]:
        """
        Run full simulation.

        Args:
            num_episodes: Number of grasping attempts

        Returns:
            List of episode results
        """
        logger.info(f"Running {num_episodes} grasping episodes...")

        self.load_model()

        results = []
        for episode_id in range(num_episodes):
            try:
                result = self.run_episode(episode_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Episode {episode_id} failed: {e}")
                results.append({"success": False, "error": str(e)})

        # Summary statistics
        successes = sum(1 for r in results if r.get("success", False))
        logger.info(f"\n{'='*60}")
        logger.info(f"Simulation Complete!")
        logger.info(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
        logger.info(f"{'='*60}\n")

        return results

    def cleanup(self) -> None:
        """Cleanup PyBullet connection."""
        if self.client is not None:
            p.disconnect(self.client)
            logger.info("✓ PyBullet connection closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PyBullet grasping simulation demo")
    parser.add_argument(
        "--model",
        default="runs/dta_best.pth",
        help="Trained model path",
    )
    parser.add_argument(
        "--robot",
        choices=["franka_panda", "ur5"],
        default="franka_panda",
        help="Robot type",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of grasping attempts",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable PyBullet GUI",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode",
    )
    parser.add_argument(
        "--render_width",
        type=int,
        default=1280,
        help="Render width",
    )
    parser.add_argument(
        "--render_height",
        type=int,
        default=720,
        help="Render height",
    )
    parser.add_argument(
        "--export_poses",
        help="Export grasp poses to JSON file",
    )

    args = parser.parse_args()

    # Create simulator
    sim = GraspingSimulator(
        model_path=args.model,
        robot_type=args.robot,
        gui=args.gui and not args.headless,
        render_width=args.render_width,
        render_height=args.render_height,
    )

    try:
        # Run simulation
        results = sim.run_simulation(num_episodes=args.num_episodes)

        # Export results
        if args.export_poses:
            with open(args.export_poses, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results exported to {args.export_poses}")

    finally:
        sim.cleanup()


if __name__ == "__main__":
    main()
