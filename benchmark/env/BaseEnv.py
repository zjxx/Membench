class BaseEnv():
    def __init__(self, config) -> None:
        self.config = config

    def reset(self):
        """
        Reset the environment.
        """
        raise NotImplementedError()

    def step(self, action):
        """
        Transform (action -> observation, reward, terminated, info).
        """
        raise NotImplementedError()