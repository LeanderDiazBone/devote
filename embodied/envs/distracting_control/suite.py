# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored from google-research/distracting_control.

Local modifications:
  - Relative imports for in-tree use.
  - ``add_pixel_wrapper`` flag so callers that render via ``physics.render``
    themselves (e.g. ``embodied.envs.dmc.DMC``) can skip the pixels.Wrapper
    and avoid double rendering / an unused ``pixels`` obs key.
  - Falsy ``background_dataset_path`` skips the background distractor so
    color/camera distractions can be used without the DAVIS dataset.
"""
try:
  from dm_control import suite  # pylint: disable=g-import-not-at-top
  from dm_control.suite.wrappers import pixels  # pylint: disable=g-import-not-at-top
except ImportError:
  suite = None

from . import background
from . import camera
from . import color
from . import suite_utils


def is_available():
  return suite is not None


def load(domain_name,
         task_name,
         difficulty=None,
         dynamic=False,
         background_dataset_path=None,
         background_dataset_videos="train",
         background_kwargs=None,
         camera_kwargs=None,
         color_kwargs=None,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False,
         render_kwargs=None,
         pixels_only=True,
         pixels_observation_key="pixels",
         env_state_wrappers=None,
         add_pixel_wrapper=True):
  if not is_available():
    raise ImportError("dm_control module is not available. Make sure you "
                      "follow the installation instructions from the "
                      "dm_control package.")

  if difficulty not in [None, "easy", "medium", "hard"]:
    raise ValueError("Difficulty should be one of: 'easy', 'medium', 'hard'.")

  render_kwargs = render_kwargs or {}
  if "camera_id" not in render_kwargs:
    render_kwargs["camera_id"] = 2 if domain_name == "quadruped" else 0

  assert suite is not None
  env = suite.load(
      domain_name,
      task_name,
      task_kwargs=task_kwargs,
      environment_kwargs=environment_kwargs,
      visualize_reward=visualize_reward)

  # Apply background distractions (skip if no dataset path was provided).
  apply_background = (
      (difficulty or background_kwargs) and background_dataset_path)
  if apply_background:
    final_background_kwargs = dict()
    if difficulty:
      num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
      final_background_kwargs.update(
          suite_utils.get_background_kwargs(domain_name, num_videos, dynamic,
                                            background_dataset_path,
                                            background_dataset_videos))
    else:
      final_background_kwargs.update(
          dict(
              dataset_path=background_dataset_path,
              dataset_videos=background_dataset_videos))
    if background_kwargs:
      final_background_kwargs.update(background_kwargs)
    env = background.DistractingBackgroundEnv(env, **final_background_kwargs)

  # Apply camera distractions.
  if difficulty or camera_kwargs:
    final_camera_kwargs = dict(camera_id=render_kwargs["camera_id"])
    if difficulty:
      scale = suite_utils.DIFFICULTY_SCALE[difficulty]
      final_camera_kwargs.update(
          suite_utils.get_camera_kwargs(domain_name, scale, dynamic))
    if camera_kwargs:
      final_camera_kwargs.update(camera_kwargs)
    env = camera.DistractingCameraEnv(env, **final_camera_kwargs)

  # Apply color distractions.
  if difficulty or color_kwargs:
    final_color_kwargs = dict()
    if difficulty:
      scale = suite_utils.DIFFICULTY_SCALE[difficulty]
      final_color_kwargs.update(suite_utils.get_color_kwargs(scale, dynamic))
    if color_kwargs:
      final_color_kwargs.update(color_kwargs)
    env = color.DistractingColorEnv(env, **final_color_kwargs)

  if env_state_wrappers is not None:
    for wrapper in env_state_wrappers:
      env = wrapper(env)
  if add_pixel_wrapper:
    env = pixels.Wrapper(
        env,
        pixels_only=pixels_only,
        render_kwargs=render_kwargs,
        observation_key=pixels_observation_key)

  return env
