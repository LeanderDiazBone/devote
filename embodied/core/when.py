import time


class Every:

    def __init__(self, every, initial=True):
        self._every = every
        self._initial = initial
        self._prev = None

    def __call__(self, step):
        step = int(step)
        if self._every < 0:
            return True
        if self._every == 0:
            return False
        if self._prev is None:
            self._prev = (step // self._every) * self._every
            return self._initial
        if step >= self._prev + self._every:
            self._prev += self._every
            return True
        return False


class Ratio:

    def __init__(self, ratio):
        assert ratio >= 0, ratio
        self._ratio = ratio
        self._prev = None

    def __call__(self, step):
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


class Once:

    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:

    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        step = int(step)
        if not self._until:
            return True
        return step < self._until


class Clock:

    def __init__(self, every, first=True):
        self._every = every
        self._prev = None
        self._first = first

    def __call__(self, step=None):
        if self._every < 0:
            return False
        if self._every == 0:
            return True
        now = time.time()
        if self._prev is None:
            self._prev = now
            return self._first
        if now >= self._prev + self._every:
            # self._prev += self._every
            self._prev = now
            return True
        return False


class ConditionalEvery:
    def __init__(self, expl_freqs: list, step_horizon: int = 1):
        expl_freqs.sort()
        # clock times --> which clock to apply at which time
        self.times = [freq[0] for freq in expl_freqs]
        self.step_horizon = step_horizon
        # frequency of each clock
        self.every = [Every(every=freq[-1], initial=False) for freq in expl_freqs]
        self.expl_steps = 0

    def __call__(self, step):
        step = int(step)
        ans = False
        for i, time in enumerate(self.times):
            # if step < current clock time
            if step <= time:
                ans = self.every[i]((step // self.step_horizon))
                self.expl_steps += ans
                return ans
        return ans


if __name__ == '__main__':
    horizon = 64
    clocks = [
        [10 * horizon, 2],
        [20 * horizon, 4],
        [60 * horizon, 16],
    ]
    cond_every = ConditionalEvery(clocks, step_horizon=horizon)
    for i in range(60):
        ans = cond_every(i * horizon)
        if ans:
            print(i, i * horizon, ans, cond_every.expl_steps)

