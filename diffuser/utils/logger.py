import os
import json
from render import Renderer

class Logger:

    def __init__(self, env, logpath):
        self.renderer = Renderer(env)
        self.savepath = logpath

    def log(self, action):
        self.renderer.render(action)

    def finish(self, t, score, total_reward, terminal, diffusion_experiment, value_experiment):
        self.renderer.done()
        json_path = os.path.join(self.savepath, 'rollout.json')
        json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
            'epoch_diffusion': diffusion_experiment.epoch, 'epoch_value': value_experiment.epoch}
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
        print(f'[ utils/logger ] Saved log to {json_path}')
