
def preprocess_data(dataset):
    result = []
    for episode in dataset:
        tmp = dict()
        tmp['actions'] = episode.actions
        # https://github.com/Farama-Foundation/Minari/issues/74 last obs is next_obs of last action
        tmp['observations'] = episode.observations["observation"][0:-1]
        tmp['rewards'] = episode.rewards
        tmp['terminals'] = episode.terminations
        tmp['id'] = episode.id
        result.append(tmp)
    return result