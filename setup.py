from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym-webots'))

setup(name='gym-webots',
      version='0.0.1',
      packages=find_packages(),
      install_requires=['gym>=0.12'],
      description='The OpenAI Gym for robotics: A toolkit for developing and comparing your reinforcement learning agents using webots and ROS.',
      url='https://github.com/talregev/gym-webots',
      author='Tal Regev',
      package_data={'gym-webots': ['envs/assets/*']},
)
