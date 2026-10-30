"""Use the released image's environment, without importing the CMA policy."""
from .perception import Camera
from scripts.export_cma_submission import ACTIONS


def read_current_observation(env, camera, instruction):
    """Re-render only the current view through Habitat-Lab's public sensor suite.

    Unlike native get_sensor_observations, this preserves configured depth
    normalization. No pose arguments, simulator state, measures or task stepping.
    """
    from .perception import adapt_observation
    # The released human runtime normally drains this queue in env.step().
    # Drain already-pending updates before rendering too, without stepping the
    # task, changing the camera or reading human/agent ground-truth positions.
    human_runtime = getattr(env, 'havlnce_tool', None)
    if human_runtime is not None:
        human_runtime._handle_signals()
    # With no pose arguments, True skips the wrapper's unnecessary pose restore.
    sensors = env.habitat_env.sim.get_observations_at(keep_agent_at_new_pose=True)
    if sensors is None:
        raise ValueError('Current public RGB-D unavailable')
    return adapt_observation(dict(rgb=sensors['rgb'], depth=sensors['depth'],
                                  instruction=dict(text=instruction)), camera)


def runtime_imports(source=None):
    import habitat
    import habitat_sim
    if habitat.__version__ != '0.1.7' or habitat_sim.__version__ != '0.1.7':
        raise ValueError('Use the released HA-VLN Habitat 0.1.7 image')
    import habitat_extensions  # noqa: F401 -- register image sensors/tasks
    import vlnce_baselines
    import vlnce_baselines.common
    if source is not None:
        # Match the CMA bootstrap's package search order, without loading a policy.
        baseline = source / 'agent/VLN-CE/vlnce_baselines'
        for package, path in ((vlnce_baselines, baseline),
                              (vlnce_baselines.common, baseline / 'common')):
            if str(path) not in package.__path__:
                package.__path__.append(str(path))
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.environments import HAVLNCEDaggerEnv
    return get_config, HAVLNCEDaggerEnv


def environment_config(get_config, runtime_config, data_root, split, ids, gpu, seed):
    config = get_config(str(runtime_config))
    config.defrost()
    config.EVAL.SPLIT = split
    cfg = config.TASK_CONFIG
    cfg.DATASET.SPLIT = split
    cfg.DATASET.DATA_PATH = str(data_root) + '/HA-R2R/{split}/{split}.json.gz'
    cfg.DATASET.SCENES_DIR = str(data_root / 'scene_datasets')
    cfg.DATASET.EPISODES_ALLOWED = ids
    cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu
    cfg.SIMULATOR.HUMAN_GLB_PATH = str(data_root / 'HAPS2_0')
    cfg.SIMULATOR.HUMAN_INFO_PATH = str(data_root / 'Multi-Human-Annotations/human_motion.json')
    cfg.SIMULATOR.RECOMPUTE_NAVMESH_PATH = str(data_root / 'recompute_navmesh')
    cfg.SIMULATOR.AGENT_0.SENSORS = ['RGB_SENSOR', 'DEPTH_SENSOR']
    cfg.TASK.SENSORS = ['INSTRUCTION_SENSOR']
    cfg.TASK.MEASUREMENTS = []
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 500
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    cfg.SEED = seed
    config.freeze()
    if tuple(cfg.TASK.POSSIBLE_ACTIONS) != ACTIONS:
        raise ValueError('Image action vocabulary/order differs from the six-action contract')
    if not cfg.SIMULATOR.ADD_HUMAN:
        raise ValueError('Human-enabled challenge simulator is required')
    return config


def camera_from_config(config):
    sim = config.TASK_CONFIG.SIMULATOR
    rgb, depth, robot = sim.RGB_SENSOR, sim.DEPTH_SENSOR, sim.AGENT_0
    # Action-relative pitch is tracked locally; initial camera rotation must be upright.
    if (any(float(v) != 0 for v in depth.ORIENTATION) or
            any(float(v) != 0 for v in rgb.ORIENTATION) or
            list(rgb.POSITION) != list(depth.POSITION)):
        raise ValueError('Requires co-located initially upright RGB/depth cameras')
    if (float(rgb.HFOV) != float(depth.HFOV) or
            rgb.WIDTH * depth.HEIGHT != depth.WIDTH * rgb.HEIGHT):
        raise ValueError('RGB/depth FOV and aspect ratio must match for box projection')
    return Camera(width=depth.WIDTH, height=depth.HEIGHT, hfov=depth.HFOV,
                  min_depth=depth.MIN_DEPTH, max_depth=depth.MAX_DEPTH,
                  normalized=depth.NORMALIZE_DEPTH, camera_height=depth.POSITION[1],
                  robot_height=robot.HEIGHT, robot_radius=robot.RADIUS,
                  forward_step=sim.FORWARD_STEP_SIZE, turn_angle=sim.TURN_ANGLE,
                  tilt_angle=float(getattr(sim, 'TILT_ANGLE', 15)),
                  camera_x=float(depth.POSITION[0]),camera_z=-float(depth.POSITION[2]))
