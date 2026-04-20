import logging
from itertools import product
from typing import List
from pathlib import Path
import json

import hydra
import numpy as np
from create_data import OccludedArticulate3DCreator
from omegaconf import DictConfig
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.1",
    config_path="./config",
    config_name="config_data_creation",
)
def create_data(cfg: DictConfig) -> None:
    """
    Main entry point for data generation pipeline using Hydra configuration.

    Generates OccludedViewResult for each combination of:
    - Scene ID
    - Camera intrinsics
    - Viewing range
    - Orientation
    - And samples multiple camera poses per configuration

    Stores results in the Hydra output directory.

    Args:
        cfg: Hydra configuration dictionary
    """
    logger.info("Starting data generation pipeline")
    logger.info(f"Configuration:\n{cfg}")

    scene_ids: List[str] = cfg.data_creation.scene_ids
    if not scene_ids:
        logger.warning("No scene IDs provided in configuration")
        return

    output_path: Path = Path(cfg.data_creation.output_dir).absolute()
    if not output_path.exists():
        logger.info(f"Output directory does not exist, creating: {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

    if not (output_path / "views").exists():
        (output_path / "views").mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(scene_ids)} scene(s)")

    # Get view generation parameters
    view_config = cfg.data_creation.view_generation

    camera_intrinsics_list: list[dict[str, float]] = cfg.data_creation.camera_intrinsics
    range_values: list[float] = OmegaConf.to_container(view_config.range_values)  # type: ignore

    orientations: list[tuple] = OmegaConf.to_container(view_config.orientations)  # type: ignore
    point_density: float = view_config.observation_point_density
    samples_per_scene: int = view_config.samples_per_scene
    do_visualize: bool = view_config.visualize

    logger.info(f"Camera intrinsics configs: {len(camera_intrinsics_list)}")
    logger.info(f"Range values: {range_values}")
    logger.info(f"Orientations: {len(orientations)}")
    logger.info(f"Point density: {point_density}")
    logger.info(f"Samples per scene: {samples_per_scene}")

    all_results = {}
    failed_scenes = []

    for scene_id in scene_ids:
        try:
            logger.info(f"\n{'=' * 60}Processing scene: {scene_id}{'=' * 60}")
            creator = OccludedArticulate3DCreator(scene_id)
            logger.info(f"Mesh loaded: {creator.mesh_path}")
        except FileNotFoundError as e:
            logger.error(f"Failed to load scene {scene_id}: {e}")
            failed_scenes.append(scene_id)
            break
        except Exception as e:
            logger.error(f"Unexpected error for scene {scene_id}: {e}")
            failed_scenes.append(scene_id)
            break

        observation_points = creator.sample_inside_points(spacing=point_density)

        num_available = len(observation_points)
        if num_available == 0:
            logger.warning(f"No observation points available for scene {scene_id}")
            continue
        elif num_available < samples_per_scene:
            camera_indices = np.arange(num_available)
        else:
            num_samples = min(samples_per_scene, num_available)
            camera_indices = np.random.choice(
                num_available, size=num_samples, replace=False
            )

        # Generate combinations of all parameters
        parameter_combinations = list(
            product(camera_intrinsics_list, range_values, orientations)
        )

        logger.info(f"Parameter combinations: {len(parameter_combinations)}")

        per_scene_counter = 0
        for intrinsics_config, range_val, orientation_obj in parameter_combinations:
            for cam_idx in camera_indices:
                per_scene_counter += 1

                try:
                    result = creator.generate_view(
                        camera_pos=observation_points[cam_idx],
                        orientation_quat=np.array(orientation_obj),
                        intrinsics=dict(intrinsics_config),
                        range=float(range_val),
                    )

                    if not result.is_valid:
                        continue

                    name = f"{scene_id}_{per_scene_counter}"
                    local_path = Path(f"{name}.json")
                    result_file = output_path / local_path
                    result_dict = result.to_dict()
                    with open(result_file, "w") as f:
                        json.dump(result_dict, f)

                    result_dict.pop("visible_mesh_indices", None)
                    result_dict.pop("visible_triangle_indices", None)
                    result_dict["path"] = local_path.as_posix()
                    all_results[name] = result_dict

                    logger.info(
                        f"Generated view: scene={scene_id}_{per_scene_counter}, cam_idx={cam_idx}, "
                        f"visible_points={len(result.visible_mesh_indices)}, "
                        f"valid={result.is_valid}"
                        f" (saved to {result_file})"
                    )

                    # Visualize if enabled
                    if do_visualize and result.is_valid:
                        logger.info("Visualizing view...")
                        creator.visualize(result)

                except Exception as e:
                    logger.error(f"Failed to generate view for camera {cam_idx}: {e}")

        logger.info(f"\nScene {scene_id} complete: {per_scene_counter} views generated")

    # Save results to file
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline complete: {len(all_results)} views generated")

    results_file = output_path / "dataset.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f)
    logger.info(f"All results saved to {results_file}")


if __name__ == "__main__":
    create_data()
