"""Smoke tests: tasks register and configs build."""

import mjlab  # noqa: F401  (loads the mjlab.tasks entry point)
from mjlab.entity.entity import Entity
from mjlab.tasks.registry import list_tasks, load_env_cfg

from mj_lab_gecko_project.robots.gecko import get_gecko_robot_cfg

TASKS = ("Gecko-Locomotion-Flat",)


def test_tasks_registered():
  for task in TASKS:
    assert task in list_tasks()


def test_env_cfgs_load():
  for task in TASKS:
    load_env_cfg(task)
    load_env_cfg(task, play=True)


def test_robot_compiles():
  model = Entity(get_gecko_robot_cfg()).spec.compile()
  assert model.nu > 0
