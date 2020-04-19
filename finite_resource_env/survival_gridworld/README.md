**Description of Contents**

**See env_tutorial.ipynb for how to use the environment**

1. gridworld.py
 - Contains the raw implementation of the Survival Gridworld environment.
2. environment.py
 - The environment implementation for reinforcement learning.
 - Implemented state representation, reward function, agent interaction, and other miscellaneous functions.
3. env_tutorial.ipynb
 - Jupyter notebook code that contains the tutorials on how to use the environment.
4. gridworld_a3c.ipynb
 - Implementation of Asynchronous Advantage Actor Critic (A3C) experiments for Survival Gridworld environment.
 - Experimentation using the seven test environments from test_envs.ipynb.
 - Includes experimention for finite resource, infinte resource, and advantage estimate clipping.
5. gridworld_ppo.ipynb
 - Implementation of Proximal Policy Optimization (PPO) experiments for Survival Gridworld environment.
 - Experimentation using the seven test environments from test_envs.ipynb.
 - Includes experimention for finite resource, infinte resource, and advantage estimate clipping.
6. netfunctions.py
 - Contains the helper functions to conveniently build a network for TensorFlow 1.13 ~ 1.15.
7. test_envs.py
 - Contains the description, gridworld matrix, and parameters for the seven intuitive test environment.
 - Also contains the rendered image for each test environment.
8. a3c_models folder
 - Contains sample test results and trained models for a3c experiment using advantage estimate clipping for the test environments.
9. ppo_models folder
 - Contains sample test results and trained models for ppo experiment using advantage estimate clipping for the test environments.



