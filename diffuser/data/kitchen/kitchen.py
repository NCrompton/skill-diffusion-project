from ..sequence import SequenceDataset

class KitchenDataset(SequenceDataset):
    def __init__(self):
        self = SequenceDataset("D4RL/kitchen", "mixed-v2")