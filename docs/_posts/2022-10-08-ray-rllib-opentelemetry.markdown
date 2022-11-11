---
layout: post
title: "Ray RLlib and OpenTelemetry"
date:   2022-10-08 17:00:48 +1300
--- 

*This blog post assumes the reader have basic understanding of reinforcement learning and PPO algorithm. Recommended to read: [https://spinningup.openai.com/en/latest/algorithms/ppo.html]()*

## Problem

I'm using [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html) to train an agent to play a card game. The time to train with RLlib's PPO was taking longer compared to the [self implemented PPO training](https://github.com/Wal8800/card-games/blob/main/algorithm/agent.py#L249) when using the same number of epoch.
My hunch was that there are some configuration and implementation differences between RLlib PPO training and my own implementation. **I need to be able to accurately measure the time taken as I change the configuration on RLlib PPO training runs to improve the training time.** 

By default, the logs from RLlib PPO training already contains metric. For example:

```console
  timers:
    learn_throughput: 182.101
    learn_time_ms: 22493.059
    synch_weights_time_ms: 6.081
    training_iteration_time_ms: 25297.398
```

For a single training epoch `training_iteration_time_ms`, it was taking ~25 seconds and `learn_time_ms` indicates it spend ~22 seconds to update the agent. The self implemented PPO training took around 1 to 2 seconds to update the agent. 

```console
2022-09-26 10:43:00,235 — train_ppo — INFO — epoch: 6, policy_loss: -0.014, value_loss: 596478.312, time to sample: 6.946 time to update network: 1.154
2022-09-26 10:43:00,235 — train_ppo — INFO — return: -46.531, ep_len: 16.327 win rate: 0.277 ep_hands_played: 28.331 ep_games_played: 235, 
Early stopping at step 10 due to reaching max kl: 0.015266045928001404
2022-09-26 10:43:08,619 — train_ppo — INFO — epoch: 7, policy_loss: -0.041, value_loss: 590175.938, time to sample: 6.976 time to update network: 1.405
2022-09-26 10:43:08,620 — train_ppo — INFO — return: -21.176, ep_len: 15.686 win rate: 0.265 ep_hands_played: 27.329 ep_games_played: 245, 
```

The default RLlib metric didn't indicates how long it spent sampling the environment. Before Ray v2.0, it provided `sample_time_ms` but the value doesn't make sense if we add it together wth `learn_time_ms`. This is because the sum didn't equal to the `training_iteration_time_ms`.

Fortunately, [Ray allows us to instrument the framework using OpenTelemetry](https://docs.ray.io/en/latest/ray-observability/ray-tracing.html) so we can generate traces and get a better understanding of the underlying operations. 


## Setting up OpenTelemetry

To enable tracing, we need to setup a tracing startup hook function for each local/remote Ray worker. This is done by passing the reference to the setup function in `ray.init` before we start training. For example:

_runner.py_

```python
    ray.init(_tracing_startup_hook="ray_custom_util:setup_tracing")
```

- `ray_custom_util` refers to the module where the function is located
- `setup_tracing` is the function name. 

In my setup, I have `ray_custom_util.py` sitting next to `runner.py` on the same directory level. The `setup_tracing` function looks like the following:

```python
def setup_tracing() -> None:
    resource = Resource(attributes={SERVICE_NAME: "rayrunner"})

    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
        udp_split_oversized_batches=True,
    )

    processor = BatchSpanProcessor(jaeger_exporter)
    provider = TracerProvider(resource=resource)

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

```

The setup function will be executed when initialising each local/remote Ray worker. It will configure the span processor and traces exporter. In the above example, it is configured to send the traces to a local Jaeger instances. The local Jaeger instance is created through a [docker-compose.yaml](https://github.com/Wal8800/card-games/blob/main/docker-compose.yaml) and we can view the traces at `localhost:16686` from Jaeger web interface.

Next, we can start a new span and run the PPO training after the `ray.init`. For example:


```python
    ppo_config = (
        PPOConfig()
        .environment(env=env_name)
        .framework(framework="tf2")
        .resources(num_gpus=1)
        .experimental(_disable_preprocessor_api=True)
        .multi_agent(
            policies=bigtwo_policies, 
            policy_mapping_fn=lambda agent_id: default_policy_id
        )
        .rollouts(
            num_rollout_workers=4, num_envs_per_worker=1, 
            batch_mode="truncate_episodes", compress_observations=True,
            rollout_fragment_length=512,
        )
        .training(
            lr=0.0001, use_gae=True, gamma=0.99, lambda_=0.9, 
            kl_coeff=0.2, kl_target=0.01, sgd_minibatch_size=512,
            num_sgd_iter=80, train_batch_size=4096, clip_param=0.3,
        )
    )

    ray.init(_tracing_startup_hook="ray_custom_util:setup_tracing")
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("ppo_training"):
        _ = tune.run(
            "PPO",
            stop={"timesteps_total": 41000},
            checkpoint_freq=100,
            local_dir="./temp_results/" + env_name,
            config=ppo_config.to_dict(),
            checkpoint_at_end=True,
        )
```

After the training run is completed, we can go to the Jaeger web interface and view the spans for the training.


<div style="text-align: center; padding-bottom: 15px">
    <img src="{{ '/assets/img/ray_rllib_opentelemetry/original_sampling_edited.png' | relative_url }}" />
    <figcaption style="text-align: center; font-style: italic;">Figure 1: Spans for sampling the environments</figcaption>
</div>

- The spans generated during training in local/remote workers are group under `ppo_training` which is the parent span. When each worker was created, the context from `ppo_training` span is extracted and injected into each worker. Any span created on the workers will be using the injected context, this means they are linked to the parent span.

- There are 4 `RolloutWorker.sample.ray.remote_worker` spans running at the same time. This indicates there are 4 workers sampling in parallel which matches with the training configuration. 
  
- The sampling occurred in two rounds (the green box and the red box) because the rollout fragment (the amount of step collected per worker) is set to 512 steps and the train batch size is set 4096. So in a single round, it collects 2048 steps (512 steps * 4 workers) therefore two rounds would satisfy the train batch size.
  
- The sampling span `RolloutWorker.sample.ray.remote` duration is shorter than it's child span `RolloutWorker.sample.ray.remote_worker`. Most likely due to asynchronous execution where the sampling span was the function that triggers the worker and exit after triggering the worker. The worker continues to sample until it meets the sampling target.  


<div style="text-align: center; padding-bottom: 15px">
    <img src="{{ '/assets/img/ray_rllib_opentelemetry/original_span.png' | relative_url }}" />
    <figcaption style="text-align: center; font-style: italic;">Figure 2: Child spans under a single training span</figcaption>
</div>


- The default spans under `PPO.train.ray.remote_worker` span doesn't tell us what happens during the update step for the agent. We can only see the sampling spans and weights update spans.

To create additional spans, I duplicated the PPO algorithm class from RLlib and override the `training_step` function to create individual spans for more granularity. For example: 

```python
class TracedPPO(PPO, ABC):
    def training_step(self) -> ResultDict:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("synchronous_parallel_sample"):
            if self._by_agent_steps:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config["train_batch_size"],
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_env_steps=self.config["train_batch_size"],
                )

    # other code in the function are excluded for demostration purposes.
```

_For the full example refer to this [file](https://github.com/Wal8800/card-games/blob/main/ray_runner/multi_agent_bigtwo_runner.py)._

Then I ran the training with the custom PPO algorithm class.

```python
    _ = tune.run(
        TracedPPO,
        name="TracedPPO",
        stop={"timesteps_total": 41000},
        checkpoint_freq=100,
        local_dir="./temp_results/" + env_name,
        config=ppo_config.to_dict(),
        checkpoint_at_end=True,
    )
```

Now it shows a span to indicate the previous missing gap.

<div style="text-align: center; padding-bottom: 15px">
    <img src="{{ '/assets/img/ray_rllib_opentelemetry/traced_ppo.png' | relative_url }}" />
    <figcaption style="text-align: center; font-style: italic;">Figure 3: Additional spans with TracedPPO</figcaption>
</div>


- Previous sampling spans are now grouped together under `synchronous_paralle_sample` and we can see the total time spent in sampling.
- The bottleneck is at the `train_one_step`  step like `learn_time_ms` indicated. Looking at the code, the `learn_time_ms` metric is actually measured within the train one step functions.
- Other steps such as `standardize_fields` and `sync_weights` are negilible in terms of time taken.

We can add more instrumentations in the training step by creating a custom `train_one_step` and `do_minibatch_sgd` function. Then we can see what it is doing.

<div style="text-align: center; padding-bottom: 15px">
    <img src="{{ '/assets/img/ray_rllib_opentelemetry/sgd_iter_span.png' | relative_url }}" />
    <figcaption style="text-align: center; font-style: italic;">Figure 4: sgd iteration spans</figcaption>
</div>


- Looks like `train_one_step` is comprises of multiple sgd update iterations. 
- Each sgd update is relatively fast but if there are many iterations then time taken adds up. There are always 80 iterations under the custom `do_minibatch_sgd` function.

With this view, we can see any impacts on time taken when changing the configurations or customising the implementation for RLlib PPO training. 


## Observations

After comparing implementation, changing configuration, and checking the training time on the Jaeger web interface, there are two main drivers of longer training time.

### Eager tracing not enabled

Eager tracing is about tracing the logic in a function upon the first call and then constructing a computation graph. In the subsequent function calls, it will execute the computation graph to get the result. [Using the computation graph can be much faster than running the python function code](https://www.tensorflow.org/guide/intro_to_graphs#the_benefits_of_graphs). In this case for PPO training, eager tracing affects the speed for calculating the loss value for the agent update and the action inference time based on the given observation.

By default, tensorflow 2 eager tracing is disabled in RLlib training configuration. This means I have been running RLlib PPO training without eager tracing. On the other hand, the self implemented PPO training have tracing enabled for updating the agent by [using tf.function decorators](https://www.tensorflow.org/guide/function). 

After enabling eager tracing , the time taken improvements are:

- update time went from ~22 seconds to ~18 seconds.
- sampling time went from ~2.8 seconds to ~2.0 seconds.


### Early stopping on large KL divergence

In self implemented PPO training, it contains early stopping logic for updating the policy network when the mean KL divergence is greater than a specified threshold. This is because my implementation is based on the [Spinning Up implementation](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and they used this additional mechanism to ensure the new policy doesn't stray too far from the old policy.

In the self implemented PPO training, the number of update iteration is set to 80. However, often times, the number of iteration that get executed is significiant less than 80 due to early stopping, usually around less than 10 update iterations. 

On the other hand, when I first tried RLlib PPO training, I kept the training hyperparameters the same. This means the the number of update iteration `num_sgd_iter` is set to 80. Since the RLlib PPO implementation doesn't have early stopping, it always ran 80 update iterations. As a result, this contributes to making the RLlib PPO training taking longer than the self impemented PPO training.

To enable early stopping in RLlib PPO training, I need to change the existing RLlib PPO training logic. Fortunately, I'm already customising the `do_minibatch_sgd` function to enable spans so I can implement the same early stopping logic in there as well. After training with early stopping, the number of iteration drops to ~30 iterations and reduces the time taken for updating the agent. However, the number of iterations are still more than the self implemented PPO training. 

After further investigation,  it looks like RLlib PPO implementation uses [both KL penalty and clipping simultaneously](https://github.com/ray-project/ray/issues/13837) where as my implementation only uses clipping. Potentially, for the RLlib PPO implementation, the mean KL divergence between old policy to new policy in a single update iteration is relatively less than my PPO implementation therfore RLlib PPO training performs more iteration until it reached the same early stopping threshold.

For now, I'm going to set the number of iteration to 30 to keep the time taken improvement. Then further experiment to understand the impact on the game play performance.

## Results

<div style="text-align: center; padding-bottom: 15px">
    <img src="{{ '/assets/img/ray_rllib_opentelemetry/final_improvements.png' | relative_url }}" />
    <figcaption style="text-align: center; font-style: italic;">Figure 5: Spans after adjustments</figcaption>
</div>

After enabling eager tracing and changing the number of update iteration to a lower value, the time to train a single epoch is much faster now. Especially for `train_one_step`, the time taken decrease from ~22 seconds to ~7 seconds.  

The RLlib PPO agent update is relatively slower than self implemented PPO training. However, due to the faster sampling in RLlib, the overall RLlib PPO training is at least the same or faster than self implemented PPO training. Furthermore, from initial experiments, the agent trained from RLlib PPO is peforming better (in terms of win rates) compared to self implemented PPO training when trained with the same number of epoch.
