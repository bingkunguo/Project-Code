from gym.envs.registration import register

register(
    id='kuka_handlit-v0',
    entry_point='kuka_handlit.envs:Kuka_HandlitGymEnv',
    kwargs={'renders' : True},
)
