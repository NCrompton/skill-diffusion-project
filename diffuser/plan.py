from .utils.config import ProjectConfig
from .utils.logger import Logger
from .utils.serialization import load_config, load_diffusion
from .sample.policy import GuidedPolicy, ValueGuide


config = ProjectConfig()

diffusion_experiment = load_diffusion(
    config.loadbase, config.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = load_diffusion(
    args.loadbase, config.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

value_function = value_experiment.ema

dataset = config.dataset

guide = ValueGuide()
policy = GuidedPolicy()

env = dataset.env
observation = env.reset()

logger = Logger(env, config.video_output)

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
for t in range(config.max_episode_length):

    if t % 10 == 0: print(args.savepath, flush=True)

    ## save state for rendering only
    state = env.state_vector().copy()

    ## format current observation for conditioning
    conditions = {0: observation}
    action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    ## print reward and score
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'values: {samples.values} | scale: {args.scale}',
        flush=True,
    )

    ## update rollout observations
    rollout.append(next_observation.copy())

    ## render every `args.vis_freq` steps
    logger.log(t, samples, state, rollout)

    if terminal:
        break

    observation = next_observation

## write results to json file at `args.savepath`
logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)