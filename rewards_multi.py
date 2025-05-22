from firehose.rewards import Reward

class HelicopterReward(Reward):
    @classmethod
    def name(cls):
        return "HelicopterReward"

    def __call__(self, **kwargs):
        area_on_fire = (self.env.state > 0).sum()
        return -area_on_fire * 0.1

class DroneReward(Reward):
    @classmethod
    def name(cls):
        return "DroneReward"

    def __call__(self, **kwargs):
        center = (self.env.height // 2, self.env.width // 2)
        window = self.env.state[max(0, center[0]-3):center[0]+4, max(0, center[1]-3):center[1]+4]
        area_on_fire = (window > 0).sum()
        return -area_on_fire * 0.2

class GroundCrewReward(Reward):
    @classmethod
    def name(cls):
        return "GroundCrewReward"

    def __call__(self, **kwargs):
        crew_area = self.env.state[0:5, 0:5]
        fire = (crew_area > 0).sum()
        return -fire * 0.5
