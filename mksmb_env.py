
from stable_baselines3.common.env_util import make_atari_env,make_vec_env , make_vec_env #, AtariWrapper
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv,NoopResetEnv,MaxAndSkipEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv,ClipRewardEnv # WarpFrame,
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecFrameStack,VecEnv
from typing import List,Optional,Tuple,Union,Any, Callable, Dict,Type, Union
import gym_super_mario_bros
from gym import spaces
from nes_py.wrappers import JoypadSpace
import numpy as np,gym
import os, cv2
import wandb
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT,RIGHT_ONLY

def Joy(env_id = 'SuperMarioBros-v0',movement=COMPLEX_MOVEMENT):
  def Joyenv():
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, movement)
    return env
  return Joyenv

class SMBMonitor(Monitor):
    
    def __init__(
        self,
        env,
        usewandb=False,
        **kwargs
    ):
        super().__init__(env=env,**kwargs)
        self.usewandb = usewandb
    
    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = super().step(action)
        info_keywords=('coins','score','time','x_pos')
        if done and self.usewandb:
          wandbinfo = info["episode"]
          for key in  info_keywords: 
            wandbinfo[key] = info[key]
          info["episode"] = wandbinfo
          wandb.log(wandbinfo)
        return observation, reward, done, info

class SMEpisodicLifeEnv(EpisodicLifeEnv):
    def step(self, action: int) -> GymStepReturn:
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        if 'ale' in dir(self.env.unwrapped): # atari
            lives = self.env.unwrapped.ale.lives()
        else: # super mario bors ときめうち
            lives = self.env.unwrapped._life
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return obs, reward, done, info
    def reset(self, **kwargs) -> np.ndarray:
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        if 'ale' in dir(self.env.unwrapped): # atari
            self.lives = self.env.unwrapped.ale.lives()
        else: # super mario bors ときめうち
            self.lives = self.env.unwrapped._life
        return obs

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype
        )
    def observation(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(frame[4:236,4:252,:], cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None] 
    
class RewardWraper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env
    ):
        super().__init__(env=env)
        self.myconins = 0

    def reset(self,**kargs):
        self.mycoins = 0 
        return self.env.reset(**kargs)

    def step(self, action: Union[np.ndarray, int]):
        observation, reward, done, info = self.env.step(action)
        coins = info['coins']
        if self.mycoins < coins:
            reward += 1
            self.mycoins = coins
        return observation, reward, done, info
  
class SMBWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_sizeH: int = 116,
        screen_sizeW: int = 124, 
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
    ):
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = SMEpisodicLifeEnv(env)
        env = WarpFrame(env, width=screen_sizeW, height=screen_sizeH)
        env = RewardWraper(env)
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)

def make_mario_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None, #-> Colab版にはない
) -> VecEnv:

    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def mario_wrapper(env: gym.Env) -> gym.Env:
        env = SMBWrapper(env, **wrapper_kwargs)
        return env

    return SMBmake_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=mario_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs #-> Colab版にはない
    )

def SMBmake_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:

    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs #-> Colab版にはない

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = SMBMonitor(env, filename=monitor_path, **monitor_kwargs)
            if wrapper_class is not None:
                env = wrapper_class(env)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

## 最適モデルの保存用のコールバック関数
# make_atari_env() は内部でベクトル化前の環境をMonit()でラップして csv 形式のログを記録しているので、そこから直近の報酬データを取り出し、
# 平均報酬最大尾も# 最適モデルの保存用のコールバック関数

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class recordModelCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1, usewandb=False):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = 0 # 直近2*check_freqの平均スコアのベスト
        self.usewandb=usewandb

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # スコアの検索
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          self.logger.record('timesteps', self.num_timesteps)
          self.logger.record('mean_reward', self.best_mean_reward)
          if self.usewandb:
            wandb.log({'mean_reward':self.best_mean_reward})
            wandb.log({'timesteps':self.num_timesteps})
          if len(x) > 0:
              # 直近100lifeのスコアの平均
              mean_reward = np.mean(y[-2*self.check_freq:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps} : ",end='')
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward/ep: {mean_reward:.2f}")

              # 直近の平均報酬が上昇した場合はモデルを保存
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)
        return True
