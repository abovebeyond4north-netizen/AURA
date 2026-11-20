"""Evaluation tasks stay behind verifier process boundary; heldout seed namespace."""
from aura.environments.task_generator import task_seed

PROTECTED_TASKS=('key','switch','hidden','altered','noise')
def protected_seed(i): return task_seed('protected',i,secret='verifier-v1-fixed-blind')
