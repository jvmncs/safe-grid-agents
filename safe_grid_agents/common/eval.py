"""Agent-specific evaluation interactions."""
import numpy as np

from safe_grid_agents.common.utils import track_metrics
from collections import defaultdict


def default_eval(agent, env, eval_history, args):
    """Evaluate an agent (default interaction)."""
    print("#### EVAL ####")
    eval_over = False
    t = 0
    state, done = env.reset(), False

    show = args.eval_visualize_episodes > 0
    next_animation = [np.copy(env.render(mode="rgb_array"))]
    episodes_to_show = []

    while True:
        if done:
            eval_history = track_metrics(eval_history, env, eval=True, write=False)
            state, done = env.reset(), False
            if show:
                animation = np.stack(next_animation)
                # swap color and time axes (has to be removed for tensorboardX >= 1.6)
                animation = np.swapaxes(animation, 0, 1)
                episodes_to_show.append(animation)
                next_animation = [np.copy(env.render(mode="rgb_array"))]
                show = args.eval_visualize_episodes > len(episodes_to_show)
            if eval_over:
                break

        action = agent.act(state)
        state, reward, done, info = env.step(action)

        t += 1
        eval_over = t >= args.eval_timesteps

        if show:
            next_animation.append(np.copy(env.render(mode="rgb_array")))

    if len(episodes_to_show) > 0:
        animation_tensor = np.stack(episodes_to_show, axis=0)
        eval_history["writer"].add_video(
            "Evaluation/grid_animation", animation_tensor, eval_history["period"]
        )

    eval_history = track_metrics(eval_history, env, eval=True)
    eval_history["returns"].reset(reset_history=True)
    eval_history["safeties"].reset()
    eval_history["margins"].reset()
    eval_history["margins_support"].reset()
    eval_history["period"] += 1
    return eval_history


EVAL_MAP = defaultdict(lambda: default_eval, {})
