import dreamerv3.ninjax as nj
from optax import Schedule
import jax.numpy as jnp

i32 = jnp.int32
f32 = jnp.float32


class Scheduler(nj.Module):
    def __init__(self, scheduler: Schedule):
        self.scheduler = scheduler
        self.step = nj.Variable(jnp.array, 0, i32, name='step')

    def __call__(self, *args, **kwargs):
        step = self.step.read().astype(f32)
        scheduler_const = self.scheduler(step)
        return f32(scheduler_const)

    def update(self):
        self.step.write(self.step.read() + 1)
