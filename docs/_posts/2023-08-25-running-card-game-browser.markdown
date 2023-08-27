---
layout: post
title:  "Running Big Two in the browser"
date:   2023-08-25 17:47:48 +1300
---

For a while, I have been wanting to create a web app client for the Big Two bot I have trained. However, I didn’t want to rewrite the Big Two game logic from python to javascript. Then I heard about Pyodide, it's a python distribution that runs on the browser. This means we can run python code in the web browser! 

So I hacked together a small react app that uses the existing python logic and tensorflow model. With minimal experience with web frontend, I used react-create app to quickly get started. The app also uses react pixi.js library to render some simple graphics for the card game. Here are the interesting bits found while putting together the web app.

## Exporting the bot

The bot was trained using the Ray RLlib framework. There are [documentation](https://docs.ray.io/en/latest/serve/tutorials/rllib.html) on how to serve the bot using ray server but nothing specific for running it in the browser without any dependencies. 

There is an [export model](https://docs.ray.io/en/latest/rllib/package_ref/policy.html#saving-and-restoring) method available so I can export the bot to tensorflow saved model format. Then I can use Tensorflow.js to run the model in the browser. 

Exporting the model has an interesting implication, since my bot performs additional operations after the model’s output to ignore invalid actions, I need to replicate this operation in the browser as well.

To use the saved model format in the web browser, I need to convert the saved model files to Tensorflow.js graph model. I ran the following command to convert:

```shell
tensorflowjs_converter \
--input_format=tf_saved_model --output_node_names="test" \
--saved_model_tags=serve ./ray_runner/model ./ray_runner/webmodel
```

It takes in the saved model inside `./rayrunner/model` folder and generates a webmodel folder containing a json file describing the model and a binary file for the weights

## Running python code in the browser

### Loading pyodide

First, we need to install the pyodide package.

```console
npm i pyodide
```

Then in code, we can load the pyodide distribution by providing indexURL.

```javascript
const runTime = await loadPyodide({
  indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.4/full/",
});
```

If we don’t specify the `indexURL` like I initially did, you will encounter the following error.

```console
error loading dynamically imported module: http://localhost:3000/static/js/pyodide.asm.js
```

When we don’t provide the `indexURL` to point to the corresponding pyodide javascript module, it defaults to the index.html file. This will error out because the html file is not a javascript module.

### Python packages and custom code

Pyodide supports [a set of 3rd party libraries](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) to use in the browser. So in my case, the only 3rd party package I need is numpy. I first load the micropip module provided by pyodide, then run `micropip.install` to download the numpy package on the browser. 

```javascript
await runTime.loadPackage("micropip");
const micropip = await runTime.pyimport("micropip");
await micropip.install("numpy");
```

Next, how do we import custom python module or scripts to use in the web browser? [It’s recommended to package your python code into a wheel and install the package into pyodide.](https://pyodide.org/en/stable/usage/loading-custom-python-code.html)

To keep it simple, I manually packaged the Big Two python module into a zip file and unpack it at the current directory so the browser's python code can import the package directly
```javascript
const response = await fetch("card-games.zip"); 
const buffer = await response.arrayBuffer();
await runTime.unpackArchive(buffer, "zip");
```

### Using python variables and methods

To run python code, we can call `runPythonAsync` on the pyodide runtime. For the Big Two web app, I needed to run python code to create the Big Two game environment and import related python helper functions.

```javascript
await runTime.runPythonAsync(`
  from bigtwo.bigtwo import BigTwo
  from bigtwo.preprocessing import (
    create_action_cat_mapping,
    generate_action_mask,
    obs_to_ohe,
  )
  from playingcards.card import Card, Rank, Suit

  game = BigTwo()
  cat_to_raw_action, raw_action_to_cat = create_action_cat_mapping()
`);
```

These imported functions and variables live in the global python namespace. To use them on the javascript side, we can fetch them using `globals.get` method. This will return python proxy objects. 

```javascript
// variables
const game = runTime.globals.get("game");
const catToRawAction = runTime.globals.get("cat_to_raw_action");
const rawActionToCat = runTime.globals.get("raw_action_to_cat");

// functions
const generateActionMask = runTime.globals.get("generate_action_mask");
const obsToOhe = runTime.globals.get("obs_to_ohe");
```

We can use these python proxies by directly calling methods or fields on the proxies.

```javascript
obs = game.get_current_player_obs()
```

`obs` is a python proxy object since the value is a python class. If the returned value was an immutable type such as `int` or `str`, it will automatically convert to the corresponding javascript type.

For other types, we can call `to_js()` to explicitly convert the python proxy object to a native javascript type. This allows us to specify javascript type in downstream functions.

For example, I'm generating a one hot encoding of the in-game observation by calling `obsToOhe` which returns a numpy array. After that, we call `tolist` on the result (numpy array method) to convert it to python list type. Along with one hot encoding, we generate a valid action mask by calling `generateActionMask` which returns a python list of 1 and 0. Finally, the `predictAction` function expects the one hot encodings and mask to be a javascript array type so we call `to_js` on the proxy object.


```javascript
const ohe = obsToOhe(obs).tolist();
const mask = generateActionMask(rawActionToCat, obs);

const cat = predictAction(
  model,
  ohe.toJs(),
  mask.toJs()
);
```

We also need to clean up the proxy object after using them, otherwise, we run the risk of memory leak. For instance, as we iterate on `obs.last_cards_played`, we create `card` which is a proxy object. After using the proxy object to create a javascript representation, we clean up by calling `destroy()`.

```javascript
for (let card of obs.last_cards_played) {
  currLastPlayedCards.push(toPyCard(card.suit.value, card.rank.value));
  card.destroy();
}
```

Fortunately, pyodide does have logic to automatically clean up in more complex situations like [making nested attribute access or method call](https://github.com/pyodide/pyodide/issues/1617) and [passing the proxy object to a javascript function](https://github.com/pyodide/pyodide/issues/1607).

For reference on how the type conversion works, please refer to the [pyodide type translation documentation](https://pyodide.org/en/stable/usage/type-conversions.html), it has great examples and types mapping.


## Running the model

The webmodel folder is served from the public folder. To load the model, we call the `loadGraphModel` function. 

```javascript
// import tensorflowjs package
import * as tf from "@tensorflow/tfjs";

// during application load
const model = await tf.loadGraphModel("webmodel/model.json");
```

The `predictAction` function is created to replicate how we run inference with the model and apply invalid action mask.

```javascript
const predictAction = function (
  model: tf.GraphModel,
  inputs: number[],
  mask: number[]
): any {
  const inputTensor = tf.tensor(inputs).reshape([1, -1]);

  // The model returns two output tensor in an array.
  const output = model.predict({
    observations: inputTensor,
  }) as tf.Tensor<tf.Rank>[];

  const infMask = tf.tensor(1).sub(tf.tensor(mask)).mul(tf.scalar(-1e9));
  const finalOutput = output[0].squeeze().add(infMask);

  return finalOutput.argMax().arraySync();
};
```

It’s fairly similar to the [python logic](https://github.com/Wal8800/card-games/blob/698422bfebf2dc3034c027494c4b5f7520c5e62a/ray_runner/multi_agent_bigtwo_runner.py#L59) since tensorflow js provides similar APIs as the python tensorflow library. I did have to explicitly cast model output as `tf.Tensor<tf.Rank>[]` as the typescript compiler doesn’t recognise it’s an array and doesn’t let me retrieve the first element by index `output[0]`

To get the underlying number from the tensor, we can call `array()` or `arraySync()`. The former is asynchronous and returns a promise that we can resolve to get the value. The purpose is to avoid blocking the UI thread while waiting for computation to be completed as the `tf.Tensor` is actually a handler to the tensor value. 

For our Big Two game, we use `arraySync` to get the tensor value since we don’t need to render anything or handle user events while we run inferences for the bot's player turn. Furthermore, the computation is relatively quick so even if we do block the UI thread, it won’t be noticeable in game. 


## Putting it together

Now that we can run python code to create a Big Two game and run inference on the Big Two bot model, what’s next? We need to write two javascript functions to manage the python Big Two game states and call the model whenever it’s the bot turn to play cards. 

- [resetAndStart](https://github.com/Wal8800/bigtwo-web-client/blob/main/src/App.tsx#L99-L113): which reset the Big Two game and goes through each bot player until it’s the human player turn.
- [step](https://github.com/Wal8800/bigtwo-web-client/blob/main/src/App.tsx#L115-L147): apply the human player action and play through the bot player’s turn until it’s the human player turn.

After hooking up the observation output from these two functions to render the UI, adding the play and sort card button, with a bit of invalid play handling, we have a functional card game!

The game is deployed to [https://bigtwo-card-game.onrender.com](https://bigtwo-card-game.onrender.com) if you want to give it a go!


## Final thoughts

A question I had is around performance of running the tensorflow.js model and interacting with the python Big Two environment. After playing the game in the browser multiple times, it doesn't feel like it's significantly slower compared to running it in python. To compare more accurately, I added some code to measure time taken for 

- Running the bot model to get action.
- Running the python Big Two game step.
- Generating the valid action mask.

On the javascript side, `performance.now()` is used and on the python side, `time.monotonic_ns()` is used. Interestingly, for `performance.now()`, firefox has different accuracy (milliseconds) compared to the rest of the browsers (microseconds) so for this experiment, I'm using google chrome. 

_javascript_
```javascript
const startTime = performance.now()
const [updatedObs, done] = pyEnv.runStep(pyEnv.game, action);
const endTime = performance.now()
```

_python_
```python
start_time = time.monotonic_ns()
env.step(action)
end_time = time.monotonic_ns()
```

This is not exactly comparing apples to apples as there are some differences between the implementations. For example, on the browser side, there are type conversions needed when calling the functions under the hood. Whereas on the python side, I'm using the Ray RLLib library to use the bot model. it is still useful to get a sense of the change in performance after we ported the game to the browser.

Here are the results:

| Operations                            | Python (ms)  | Chrome Browser (ms) |
| ------------------------------------  | -----------  | ------------------- |
| Running the bot model to get action   | 3 to 5       | 5 to 30             |
| Running the python Big Two game step  | 0.01 to 0.02 | 0.1 to 0.2          |
| Generating the valid action mask      | 0.01 to 0.03 | 0.1 to 0.3          |


For running the bot model, the range of latency is wide. The browser performance ranges from as good as running the model in python to almost 10 times slower. After inspecting the input for each inference, I can't see a pattern that will lead to shorter/longer time taken. It's still great to see, after we convert the model and using tensorflow js, we can still achieve the same performance on the browser. On the other hand, for the python logic, we can see it is consistently 10 times slower in the browser compared to running the code in python. Fortunately, these operations aren't called frequently and the absolute time taken is very small so it doesn't have a perceivable impact on the game play.

Overall, Pyodide can be very useful if you want to reuse an existing python module or leverage existing python libraries. For a more performance critical application, the performance slow down might be unappealing however it is on [pyodide roadmap to improve the performance of python code](https://pyodide.org/en/stable/project/roadmap.html#improve-performance-of-python-code-in-pyodide).
