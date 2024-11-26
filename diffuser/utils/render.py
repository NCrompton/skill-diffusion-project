from mediapy import VideoWriter
# from .config import ProjectConfig

class Renderer:
    def __init__(self, env, save_path):
        self.env = env
        self.frames = []
        # self.config = ProjectConfig()
        self.save_path = save_path

    def reset_env(self):
        print("resetting the env")
        self.env.reset()
        
    def render(self, action):
        print("rendering action")
        pixel = self.env.render()
        self.frames.append(pixel)
        obs, reward, truncated, done, info = self.env.step(action)
        if done:
            self.env.reset()
            self.done()

    def done(self, path_name):
        file_output_path = f"./{self.save_path}/{path_name}.mp4"
        with VideoWriter(file_output_path, (480, 480)) as w:
            for img in self.frames:
                w.add_image(img)
        print(f" [ renderer ] finished ouputting video to {file_output_path}")

    # def composite(self, savepath, paths, dim=(1024, 256), **kwargs):

    #     render_kwargs = {
    #         'trackbodyid': 2,
    #         'distance': 10,
    #         'lookat': [5, 2, 0.5],
    #         'elevation': 0
    #     }
    #     images = []
    #     for path in paths:
    #         ## [ H x obs_dim ]
    #         path = atmost_2d(path)
    #         img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
    #         images.append(img)
    #     images = np.concatenate(images, axis=0)

    #     if savepath is not None:
    #         imageio.imsave(savepath, images)
    #         print(f'Saved {len(paths)} samples to: {savepath}')

    #     return images