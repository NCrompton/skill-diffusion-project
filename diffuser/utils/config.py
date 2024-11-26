from data.datasets import DatasetLoader
from model.diffuser import GaussianDiffusion
from model.temporal import TemporalUnet
from data.sequence import SequenceDataset
from utils.render import Renderer
from data.normalization import DatasetNormalizer

class ProjectConfig:
    def __init__(self):
        self.dataset_loader = DatasetLoader("D4RL/kitchen")
        self.ds = self.dataset_loader.get_dataset("mixed-v2")
        self.normalizer = "GaussianNormalizer"
        self.max_n_episodes = 10000
        self.max_path_length = 1000
        self.dataset = SequenceDataset(dataset=self.ds, skill="kettle", 
                                       normalizer=self.normalizer, 
                                       max_n_episodes=self.max_n_episodes, 
                                       max_path_length=self.max_path_length)
        self.env = self.ds.recover_environment(render_mode="rgb_array")
        self.renderer = Renderer(self.env, "./video")
        self.observation_dim = 59
        self.action_dim = 9
        self.total_epoch = 500
        self.model = TemporalUnet(horizon=64, 
                                  transition_dim=self.observation_dim+self.action_dim, 
                                  cond_dim=self.observation_dim,
                                  )
        self.diffusion = GaussianDiffusion(model= self.model, 
                                           horizon=64, 
                                           observation_dim=self.observation_dim, 
                                           action_dim=self.action_dim,
                                           n_timesteps=100).to("cuda")
        self.video_output = "../video/result.mp4"
        self.max_episode_length = 450
        self.loadbase = None
        self.skill = "kettle"