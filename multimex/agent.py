from dreamerv3.agent import DreamerAgent
import dreamerv3.jaxagent as jaxagent
from dreamerv3 import jaxutils
from multimex.nets import IntrinsicRewardModel
import embodied
import ruamel.yaml as yaml
import jax.numpy as jnp
import jax
import dreamerv3.ninjax as nj
from dreamerv3 import nets

f32 = jnp.float32
i32 = jnp.int32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}


class MultiMexAgent(DreamerAgent):
    configs = yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'configs.yaml').read())

    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space=obs_space, act_space=act_space, config=config)
        agent_name = config.run.agent_name
        model_kwargs = config.int_rew_model.model.flat

        self.stoch_size = config.dyn.rssm.stoch
        self.num_classes = config.dyn.rssm.classes
        dist = {}
        pooling_kwargs = {}
        shapes = {}
        if 'dist.obs' in model_kwargs:
            dist['obs'] = model_kwargs['dist.obs']
            embedding_size = config.int_rew_model.pooling_output
            pooling_kwargs['output_size'] = {'obs': embedding_size}
            shapes['obs'] = (embedding_size,)
            del model_kwargs['dist.obs']
        if 'dist.dyn' in model_kwargs:
            dist['dyn'] = model_kwargs['dist.dyn']
            shapes['dyn'] = (self.stoch_size, self.num_classes)
            del model_kwargs['dist.dyn']
        if 'dist.rew' in model_kwargs:
            dist['rew'] = model_kwargs['dist.rew']
            shapes['rew'] = ()
            del model_kwargs['dist.rew']
        if len(dist.keys()) == 0:
            assert len(dist.keys()) == 0, "Need to provide at least one target for the ensemble to learn."

        assert agent_name in ['Lexa', 'Plan2Explore', 'MultiMex']
        if agent_name == 'Lexa':
            assert 'dyn' in dist, "Lexa only learns a forward dynamics model but dynamics " \
                                    "kwargs are not provided in the config"
            dist_dyn = dist['dyn']
            dist = {}
            shapes = {}
            pooling_kwargs = {}
            dist['dyn'] = dist_dyn
            shapes['dyn'] = (self.stoch_size, self.num_classes)
        elif agent_name == 'Plan2Explore':
            assert 'obs' in dist, "Plan2Explore only learns a observation model but observation " \
                                    "kwargs are not provided in the config"
            dist_obs = dist['obs']
            dist = {}
            shapes = {}
            pooling_kwargs = {}
            dist['obs'] = dist_obs
            embedding_size = config.int_rew_model.pooling_output
            shapes['obs'] = (embedding_size,)
            pooling_kwargs['output_size'] = {'obs': embedding_size}

        if 'expl_rew_weights' in config.int_rew_model:
            expl_rew_weights = config.int_rew_model.expl_rew_weights.flat
        else:
            expl_rew_weights = None

        model_kwargs['shape'] = shapes
        # add actions to the model kwargs
        model_kwargs['inputs'] = model_kwargs['inputs'] + tuple([k for k in self.act_space.keys()])
        model_kwargs['dist'] = dist
        self.ens_keys = dist.keys()

        self.int_rew_model = IntrinsicRewardModel(
            num_heads=self.config.int_rew_model.num_heads,
            pooling=pooling_kwargs,
            model_kwargs=model_kwargs,
            #disg_norm=self.config.int_rew_model.disg_norm,
            disg_agg=self.config.int_rew_model.disg_agg,
            use_entropy=self.config.int_rew_model.use_entropy,
            exploration_reward_weights=expl_rew_weights,
            name='int_rew_model',
        )
        self.modules.append(self.int_rew_model)

        kwargs = {}
        kwargs['shape'] = {
            k: (*s.shape, s.classes) if s.discrete else s.shape
            for k, s in self.act_space.items()}
        kwargs['dist'] = {
            k: config.actor_dist_disc if v.discrete else config.actor_dist_cont
            for k, v in self.act_space.items()}
        self.expl_actor = nets.MLP(**kwargs, **config.actor, name='expl_actor')
        self.modules.append(self.expl_actor)
        self.expl_retnorm = jaxutils.Moments(**config.retnorm, name='expl_retnorm')
        self.expl_valnorm = jaxutils.Moments(**config.valnorm, name='expl_valnorm')
        self.expl_advnorm = jaxutils.Moments(**config.advnorm, name='expl_advnorm')

        # Critic
        self.expl_critic = nets.MLP((), name='expl_critic', **self.config.critic)
        self.modules.append(self.expl_critic)
        self.expl_slowcritic = nets.MLP(
            (), name='expl_slowcritic', **self.config.critic, dtype='float32')
        self.expl_updater = jaxutils.SlowUpdater(
            self.expl_critic, self.expl_slowcritic,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
            name='expl_updater')

    @property
    def policy_keys(self):
        return '/(enc|dyn|actor|expl_actor)/'

    def train(self, data, carry):
        outs, carry, metrics = super().train(data, carry)
        self.expl_updater()
        return outs, carry, metrics

    def policy(self, obs, carry, mode='train'):
        self.config.jax.jit and embodied.print(
            f'Tracing policy function for mode {mode}', color='yellow')
        prevlat, prevact = carry
        obs = self.preprocess(obs)
        embed = self.enc(obs, bdims=1)
        prevact = jaxutils.onehot_dict(prevact, self.act_space)
        lat, out = self.dyn.observe(
            prevlat, prevact, embed, obs['is_first'], bdims=1)
        if mode == 'explore':
            actor = self.expl_actor(out, bdims=1)
            act = sample(actor)
        else:
            actor = self.actor(out, bdims=1)
            act = sample(actor)

        outs = {}
        if self.config.replay_context:
            outs.update({k: out[k] for k in self.aux_spaces if k != 'stepid'})
            outs['stoch'] = jnp.argmax(outs['stoch'], -1).astype(jnp.int32)

        outs['finite'] = {
            '/'.join(x.key for x in k): (
                jnp.isfinite(v).all(range(1, v.ndim)),
                v.min(range(1, v.ndim)),
                v.max(range(1, v.ndim)))
            for k, v in jax.tree_util.tree_leaves_with_path(dict(
                obs=obs, prevlat=prevlat, prevact=prevact,
                embed=embed, act=act, out=out, lat=lat,
            ))}

        assert all(
            k in outs for k in self.aux_spaces
            if k not in ('stepid', 'finite', 'is_online')), (
            list(outs.keys()), self.aux_spaces)

        act = {
            k: jnp.nanargmax(act[k], -1).astype(jnp.int32)
            if s.discrete else act[k] for k, s in self.act_space.items()}
        return act, outs, (lat, act)

    def loss(self, data, carry, update=True):
        metrics = {}
        prevlat, prevact = carry

        # Replay rollout
        prevacts = {
            k: jnp.concatenate([prevact[k][:, None], data[k][:, :-1]], 1)
            for k in self.act_space}
        prevacts = jaxutils.onehot_dict(prevacts, self.act_space)
        # give o_{t: t + H} -> e_{t:t + H}. Converts observations into embeddings
        embed = self.enc(data)
        # p(s_t|s_t-1, a_t-1, e_t) --> s_{t: t + H} | s_{t-1}, a_{t:t+H-1}, e_{t: t+ H}
        newlat, outs = self.dyn.observe(prevlat, prevacts, embed, data['is_first'])
        rew_feat = outs if self.config.reward_grad else sg(outs)
        dists = dict(
            **self.dec(outs),
            reward=self.rew(rew_feat, training=True),
            cont=self.con(outs, training=True))
        losses = {k: -v.log_prob(f32(data[k])) for k, v in dists.items()}
        if self.config.contdisc:
            del losses['cont']
            softlabel = data['cont'] * (1 - 1 / self.config.horizon)
            losses['cont'] = -dists['cont'].log_prob(softlabel)
        dynlosses, mets = self.dyn.loss(outs, **self.config.rssm_loss)
        losses.update(dynlosses)
        metrics.update(mets)

        s_prev = {
            k: outs[k][:, :-1]  # s_{t:t+H-1}
            for k in prevlat.keys()}
        a_prev = {k: val[:, 1:] for k, val in prevacts.items()}  # a_{t:t + H -1}
        model_input = s_prev | a_prev  # (s_{k}, a_{k})_{k=t:t+H-1}

        potential_labels = {
            'obs': embed[:, :-1],  # e_{t:t+H-1}
            'dyn': outs['stoch'][:, 1:],  # s_{t+1: t+ H}
            'rew': data['reward'][:, :-1],  # r_{t:t+H-1}
        }
        labels = {key: sg(potential_labels[key]) for key in self.ens_keys}

        int_rew_loss, int_rew_loss_dict = self.int_rew_model.loss(
            labels=labels,
            model_input=sg(model_input),
        )
        # concatenate to add a dummy loss for the first element.
        int_rew_loss = jnp.concatenate([int_rew_loss[:, 0].reshape(-1, 1) * 0, int_rew_loss], axis=-1)
        losses['int_rew_model'] = int_rew_loss

        replay_outs = outs

        rew = data['reward']
        con = 1 - f32(data['is_terminal'])
        # use all the states in the buffer for imagination vs only using the last one
        if self.config.imag_start == 'all':
            B, T = data['is_first'].shape
            startlat = self.dyn.outs_to_carry(treemap(
                lambda x: x.reshape((B * T, 1, *x.shape[2:])), replay_outs))
            startout, startrew, startcon = treemap(
                lambda x: x.reshape((B * T, *x.shape[2:])),
                (replay_outs, rew, con))
        elif self.config.imag_start == 'last':
            startlat = newlat
            startout, startrew, startcon = treemap(
                lambda x: x[:, -1], (replay_outs, rew, con))
        else:
            raise NotImplementedError
        if self.config.imag_repeat > 1:
            N = self.config.imag_repeat
            startlat, startout, startrew, startcon = treemap(
                lambda x: x.repeat(N, 0), (startlat, startout, startrew, startcon))

        ac_loss, ac_metrics = self.actor_critic_loss(
            startout=startout, startrew=startrew, startcon=startcon, startlat=startlat, replay_outs=replay_outs,
            data=data, update=update)
        losses.update(ac_loss)
        metrics.update(ac_metrics)

        expl_ac_loss, expl_ac_metrics = self.expl_actor_critic_loss(
            startout=startout, startcon=startcon, startlat=startlat, update=update)

        # update counter for exploration policy
        # self.expl_reward_counter.update()
        losses.update(expl_ac_loss)
        metrics.update(expl_ac_metrics)
        # metrics.update({'exploration_phase': i32(self.expl_reward_counter() > 1e-8)})
        # metrics.update({'exploration_step': i32(self.expl_reward_counter.step.read())})
        metrics.update({f'{k}_loss': v.mean() for k, v in losses.items()})
        metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
        if 'reward' in dists:
            stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
            metrics.update({f'rewstats/{k}': v for k, v in stats.items()})
        if 'cont' in dists:
            stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
            metrics.update({f'constats/{k}': v for k, v in stats.items()})
        metrics['activation/embed'] = jnp.abs(embed).mean()
        for key, val in int_rew_loss_dict.items():
            metrics[f'int_rew_model/{key}_loss'] = val
        # metrics['activation/deter'] = jnp.abs(replay_outs['deter']).mean()

        # Combine
        losses = {k: v * self.scales[k] for k, v in losses.items()}
        loss = jnp.stack([v.mean() for k, v in losses.items()]).sum()
        newact = {k: data[k][:, -1] for k in self.act_space}
        outs = {'replay_outs': replay_outs, 'prevacts': prevacts, 'embed': embed}
        outs.update({f'{k}_loss': v for k, v in losses.items()})
        carry = (newlat, newact)
        return loss, (outs, carry, metrics)

    def expl_actor_critic_loss(self, startout, startcon, startlat, update: bool = True):

        losses = {}
        metrics = {}

        prefix = 'expl_'

        def imgstep(carry, _):
            lat, act = carry
            lat, out = self.dyn.imagine(lat, act, bdims=1)
            out['stoch'] = sg(out['stoch'])
            act = cast(sample(self.expl_actor(out, bdims=1)))
            return (lat, act), (out, act)

        startact = cast(sample(self.expl_actor(startout, bdims=1)))
        _, (outs, acts) = jaxutils.scan(
            imgstep, sg((startlat, startact)),
            jnp.arange(self.config.imag_length), self.config.imag_unroll)
        outs, acts = treemap(lambda x: x.swapaxes(0, 1), (outs, acts))
        outs, acts = treemap(
            lambda first, seq: jnp.concatenate([first, seq], 1),
            treemap(lambda x: x[:, None], (startout, startact)), (outs, acts))

        rew, int_rew_metrics = self.int_rew_model(outs, acts, update=update)
        # TODO: See if we need termination flag for exploration policy
        con = jnp.concatenate([startcon[:, None], self.con(outs).mean()[:, 1:]], 1)
        acts = sg(acts)
        inp = treemap({
                          'none': lambda x: sg(x),
                          'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1),
                          'all': lambda x: x,
                      }[self.config.ac_grads], outs)
        actor = self.expl_actor(inp)
        critic = self.expl_critic(inp)
        slowcritic = self.expl_slowcritic(inp)
        voffset, vscale = self.expl_valnorm.stats()
        val = critic.mean() * vscale + voffset
        slowval = slowcritic.mean() * vscale + voffset
        tarval = slowval if self.config.slowtar else val
        discount = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
        weight = jnp.cumprod(discount * con, 1) / discount

        # Return
        rets = [tarval[:, -1]]
        disc = con[:, 1:] * discount
        lam = self.config.return_lambda
        interm = rew[:, 1:] + (1 - lam) * disc * tarval[:, 1:]
        for t in reversed(range(disc.shape[1])):
            rets.append(interm[:, t] + disc[:, t] * lam * rets[-1])
        ret = jnp.stack(list(reversed(rets))[:-1], 1)

        # Actor
        roffset, rscale = self.expl_retnorm(ret, update)
        adv = (ret - tarval[:, :-1]) / rscale
        aoffset, ascale = self.expl_advnorm(adv, update)
        adv_normed = (adv - aoffset) / ascale
        logpi = sum([v.log_prob(sg(acts[k]))[:, :-1] for k, v in actor.items()])
        ents = {k: v.entropy()[:, :-1] for k, v in actor.items()}
        actor_loss = sg(weight[:, :-1]) * -(
                logpi * sg(adv_normed) + self.config.actent * sum(ents.values()))
        losses[prefix + 'actor'] = actor_loss

        # Critic
        voffset, vscale = self.expl_valnorm(ret, update)
        ret_normed = (ret - voffset) / vscale
        ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
        losses[prefix + 'critic'] = sg(weight)[:, :-1] * -(
                                                                  critic.log_prob(sg(ret_padded)) +
                                                                  self.config.slowreg * critic.log_prob(
                                                              sg(slowcritic.mean())))[:, :-1]

        # Metrics
        for k, v in int_rew_metrics.items():
            metrics[f'int_rew_model/{k}'] = v
        metrics.update(jaxutils.tensorstats(adv, prefix + 'adv'))
        metrics.update(jaxutils.tensorstats(rew, prefix + 'rew'))
        metrics.update(jaxutils.tensorstats(weight, prefix + 'weight'))
        metrics.update(jaxutils.tensorstats(val, prefix + 'val'))
        metrics.update(jaxutils.tensorstats(ret, prefix + 'ret'))
        metrics.update(jaxutils.tensorstats(
            (ret - roffset) / rscale, prefix + 'ret_normed'))

        metrics[prefix + 'td_error'] = jnp.abs(ret - val[:, :-1]).mean()
        metrics[prefix + 'ret_rate'] = (jnp.abs(ret) > 1.0).mean()
        for k, space in self.act_space.items():
            act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
            metrics.update(jaxutils.tensorstats(f32(act), f'{prefix}act/{k}'))
            if hasattr(actor[k], 'minent'):
                lo, hi = actor[k].minent, actor[k].maxent
                rand = ((ents[k] - lo) / (hi - lo)).mean(
                    range(2, len(ents[k].shape)))
                metrics.update(jaxutils.tensorstats(rand, f'{prefix}rand/{k}'))
            metrics.update(jaxutils.tensorstats(ents[k], f'{prefix}ent/{k}'))

        metrics[prefix + 'rew/max'] = jnp.abs(rew).max()
        metrics[prefix + 'rew/mean'] = rew.mean()
        metrics[prefix + 'rew/std'] = rew.std()
        return losses, metrics

    def actor_critic_loss(self, startout, startrew, startcon, startlat, replay_outs, data, update: bool = True):
        losses = {}
        metrics = {}

        def imgstep(carry, _):
            lat, act = carry
            lat, out = self.dyn.imagine(lat, act, bdims=1)
            out['stoch'] = sg(out['stoch'])
            act = cast(sample(self.actor(out, bdims=1)))
            return (lat, act), (out, act)

        startact = cast(sample(self.actor(startout, bdims=1)))
        _, (outs, acts) = jaxutils.scan(
            imgstep, sg((startlat, startact)),
            jnp.arange(self.config.imag_length), self.config.imag_unroll)
        outs, acts = treemap(lambda x: x.swapaxes(0, 1), (outs, acts))
        outs, acts = treemap(
            lambda first, seq: jnp.concatenate([first, seq], 1),
            treemap(lambda x: x[:, None], (startout, startact)), (outs, acts))

        rew = jnp.concatenate([startrew[:, None], self.rew(outs).mean()[:, 1:]], 1)

        con = jnp.concatenate([startcon[:, None], self.con(outs).mean()[:, 1:]], 1)
        acts = sg(acts)
        inp = treemap({
                          'none': lambda x: sg(x),
                          'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1),
                          'all': lambda x: x,
                      }[self.config.ac_grads], outs)
        actor = self.actor(inp)
        critic = self.critic(inp)
        slowcritic = self.slowcritic(inp)
        voffset, vscale = self.valnorm.stats()
        val = critic.mean() * vscale + voffset
        slowval = slowcritic.mean() * vscale + voffset
        tarval = slowval if self.config.slowtar else val
        discount = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
        weight = jnp.cumprod(discount * con, 1) / discount

        # Return
        rets = [tarval[:, -1]]
        disc = con[:, 1:] * discount
        lam = self.config.return_lambda
        interm = rew[:, 1:] + (1 - lam) * disc * tarval[:, 1:]
        for t in reversed(range(disc.shape[1])):
            rets.append(interm[:, t] + disc[:, t] * lam * rets[-1])
        ret = jnp.stack(list(reversed(rets))[:-1], 1)

        # Actor
        roffset, rscale = self.retnorm(ret, update)
        adv = (ret - tarval[:, :-1]) / rscale
        aoffset, ascale = self.advnorm(adv, update)
        adv_normed = (adv - aoffset) / ascale
        logpi = sum([v.log_prob(sg(acts[k]))[:, :-1] for k, v in actor.items()])
        ents = {k: v.entropy()[:, :-1] for k, v in actor.items()}
        actor_loss = sg(weight[:, :-1]) * -(
                logpi * sg(adv_normed) + self.config.actent * sum(ents.values()))
        losses['actor'] = actor_loss

        # Critic
        voffset, vscale = self.valnorm(ret, update)
        ret_normed = (ret - voffset) / vscale
        ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
        losses['critic'] = sg(weight)[:, :-1] * -(
                                                         critic.log_prob(sg(ret_padded)) +
                                                         self.config.slowreg * critic.log_prob(
                                                     sg(slowcritic.mean())))[
                                                 :, :-1]

        if self.config.replay_critic_loss:
            replay_critic = self.critic(
                replay_outs if self.config.replay_critic_grad else sg(replay_outs))
            replay_slowcritic = self.slowcritic(replay_outs)
            boot = dict(
                imag=ret[:, 0].reshape(data['reward'].shape),
                critic=replay_critic.mean(),
            )[self.config.replay_critic_bootstrap]
            rets = [boot[:, -1]]
            live = f32(~data['is_terminal'])[:, 1:] * (1 - 1 / self.config.horizon)
            cont = f32(~data['is_last'])[:, 1:] * self.config.return_lambda_replay
            interm = data['reward'][:, 1:] + (1 - cont) * live * boot[:, 1:]
            for t in reversed(range(live.shape[1])):
                rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
            replay_ret = jnp.stack(list(reversed(rets))[:-1], 1)
            voffset, vscale = self.valnorm(replay_ret, update)
            ret_normed = (replay_ret - voffset) / vscale
            ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
            losses['replay_critic'] = sg(f32(~data['is_last']))[:, :-1] * -(replay_critic.log_prob(sg(ret_padded)) +
                                                                            self.config.slowreg *
                                                                            replay_critic.log_prob(
                                                                                sg(replay_slowcritic.mean())))[:, :-1]
            metrics.update(jaxutils.tensorstats(replay_ret, 'replay_ret'))
        # Metrics
        metrics.update(jaxutils.tensorstats(adv, 'adv'))
        metrics.update(jaxutils.tensorstats(rew, 'rew'))
        metrics.update(jaxutils.tensorstats(weight, 'weight'))
        metrics.update(jaxutils.tensorstats(val, 'val'))
        metrics.update(jaxutils.tensorstats(ret, 'ret'))
        metrics.update(jaxutils.tensorstats(
            (ret - roffset) / rscale, 'ret_normed'))
        metrics['td_error'] = jnp.abs(ret - val[:, :-1]).mean()
        metrics['ret_rate'] = (jnp.abs(ret) > 1.0).mean()
        for k, space in self.act_space.items():
            act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
            metrics.update(jaxutils.tensorstats(f32(act), f'act/{k}'))
            if hasattr(actor[k], 'minent'):
                lo, hi = actor[k].minent, actor[k].maxent
                rand = ((ents[k] - lo) / (hi - lo)).mean(
                    range(2, len(ents[k].shape)))
                metrics.update(jaxutils.tensorstats(rand, f'rand/{k}'))
            metrics.update(jaxutils.tensorstats(ents[k], f'ent/{k}'))

        metrics['data_rew/max'] = jnp.abs(data['reward']).max()
        metrics['data_rew/mean'] = data['reward'].mean()
        metrics['data_rew/std'] = data['reward'].std()
        metrics['pred_rew/max'] = jnp.abs(rew).max()
        metrics['pred_rew/mean'] = rew.mean()
        metrics['pred_rew/std'] = rew.std()

        return losses, metrics


@jaxagent.Wrapper
class Agent(MultiMexAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
