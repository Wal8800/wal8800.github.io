---
layout: post
title:  "Graph attention network"
date:   2024-03-10 17:47:48 +1300
---

_Paper_: [https://arxiv.org/pdf/1710.10903.pdf](https://arxiv.org/pdf/1710.10903.pdf)

Graph attention network (GAT) is a type of graph neural network with a self attention layer during the graph updates. The attention layer learns relative importance between the incoming source nodes for the target node. As a result, when updating the target node’s value during training, the source nodes with higher importance have a bigger impact on the target node’s updated value.

<br>
<figure>
<svg style="width: 100%; height: 100px;" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="241px" height="81px" viewBox="-0.5 -0.5 241 81"><defs/><g><g><path d="M 80 40 Q 80 40 152.13 40" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 158.88 40 L 149.88 44.5 L 152.13 40 L 149.88 35.5 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/></g><g><ellipse cx="40" cy="40" rx="40" ry="40" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="all"/></g><g><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 40px; margin-left: 1px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">Source</div></div></div></foreignObject><text x="40" y="44" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="12px" text-anchor="middle">Source</text></switch></g></g><g><ellipse cx="200" cy="40" rx="40" ry="40" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="all"/></g><g><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 78px; height: 1px; padding-top: 40px; margin-left: 161px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">Target</div></div></div></foreignObject><text x="200" y="44" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="12px" text-anchor="middle">Target</text></switch></g></g></g></svg>
<figcaption style="text-align: center; font-style: italic;">source node is the node where the edge is pointing from and target node is the edge is pointing to</figcaption>
</figure>
<br>

So how does it work? Let's go through the equations in the paper with code. 

Imagine, we have a 4 nodes graph and each node has an array of 3 elements representing their features

<figure>
<svg style="width: 100%; height: 300px;" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="201px" height="151px" viewBox="-0.5 -0.5 201 151"><defs/><g><g><path d="M 20 120 Q 20 150 100 150 Q 180 150 180 126.37" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 180 121.12 L 183.5 128.12 L 180 126.37 L 176.5 128.12 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/></g><g><ellipse cx="20" cy="100" rx="20" ry="20" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="all"/></g><g><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 100px; margin-left: 1px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">1</div></div></div></foreignObject><text x="20" y="104" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="12px" text-anchor="middle">1</text></switch></g></g><g><path d="M 160 100 L 126.37 100" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 121.12 100 L 128.12 96.5 L 126.37 100 L 128.12 103.5 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/></g><g><path d="M 168 84 L 123.82 25.09" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 120.67 20.89 L 127.67 24.39 L 123.82 25.09 L 122.07 28.59 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/></g><g><ellipse cx="180" cy="100" rx="20" ry="20" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" transform="rotate(90,180,100)" pointer-events="all"/></g><g><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 100px; margin-left: 161px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">4</div></div></div></foreignObject><text x="180" y="104" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="12px" text-anchor="middle">4</text></switch></g></g><g><path d="M 84 32 L 25.09 76.18" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 20.89 79.33 L 24.39 72.33 L 25.09 76.18 L 28.59 77.93 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/></g><g><path d="M 100 40 L 100 73.63" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 100 78.88 L 96.5 71.88 L 100 73.63 L 103.5 71.88 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/></g><g><ellipse cx="100" cy="20" rx="20" ry="20" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="all"/></g><g><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 20px; margin-left: 81px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">3</div></div></div></foreignObject><text x="100" y="24" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="12px" text-anchor="middle">3</text></switch></g></g><g><path d="M 80 100 L 46.37 100" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 41.12 100 L 48.12 96.5 L 46.37 100 L 48.12 103.5 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/></g><g><ellipse cx="100" cy="100" rx="20" ry="20" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="all"/></g><g><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 38px; height: 1px; padding-top: 100px; margin-left: 81px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">2</div></div></div></foreignObject><text x="100" y="104" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="12px" text-anchor="middle">2</text></switch></g></g></g></svg>

</figure>


To calculate the new value for node 1, we need to calculate the attention coefficients first.

The equation for calculating attention coefficient between the source node and target node is the following:

$$ e_{ij} = LeakyReLU(\vec{a}^T[\boldsymbol{W}\vec{h}_{i}||\boldsymbol{W}\vec{h}_{j}]) $$

- $$ \vec{h}_i $$ is the target node features of the edge with $$ F $$ number of features.
- $$ \vec{h}_j $$ is the source node features of the edge with $$ F $$ number of features.
- $$ \boldsymbol{W} $$ is the weight matrix applied to every node to transform the input features into higher level features. The shape is $$ F' \times F $$ where $$ F' $$ is the new feature shape.
- $$ \vec{a}^T $$ is the weight vector representing the single feed forward layer with LeakyReLU nonlinearity. This is the attention function layer for learning the importance. This vector shape is $$ 2F' $$.
- Apply the weight matrix to the nodes features and concatenate the result.
- Performs matrix multiplication with the concatenated result and the attention weight vector ( $$ .^T $$ for transposition on the feed forward weight vector).
- Apply $$ LeakyReLU $$ for nonlinearity.
- $$ e_{ij} $$ represents the importance of the source node features to the target node.


The paper injects graph structure by performing masked attention. We only calculate $$ e_{ij} $$ for first order neighbours of the target node (including the target node itself). This means all the nodes that are 1 or less edge away from the target node. **In our example, our target node is node 1 therefore our source nodes include node 1, 2 and 3.** 

- Node 1 is the target node so we include itself in the first order neighbours.
- Node 2 and 3 are one edge away, both have a directed edge pointing toward node 1.
- Node 4 is not part of the node 1 first order neighbours.
  - It takes 2 hops to get to node 1, either through node 2 and node 3. 
  - The edge connected between node 4 and node 1 is directed toward node 4 from node 1. 

To make the attention coefficients easily comparable across the different source nodes, softmax function is applied to normalise the attention coefficients.


$$ \alpha_{ij} = softmax_{j}(e_{ij}) = \frac{exp(e_{ij})}{\sum_{k \in N_{i}}exp(e_{ik})} $$
 

Translating these equations into python code using node 1 as the target node.


```python
import numpy as np

# The node features are column vectors, F is 3.
# Note: all the values are arbitrarily picked for demonstration purposes.
h1 = np.array([[1], [2], [3]])
h2 = np.array([[2], [2], [4]])
h3 = np.array([[-1], [2], [1]])


# The weight matrix, I kept the new shape F' to 3 for simplicity.
# The shape of the matrix is 3 x 3.
W = np.array([[0.1, 0.1, 0.1], [0.3, 0.4, 0.6], [-1, 1, -1]])


# A simple single layer of feed forward network for the attention function.
# The column vector is transposed into a row vector.
# shape = 2F' = 2 x 3 = 6
ff_w = np.array([[0.1, 0.2, 0.7, 0.3, 0.1, 0.6]])


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

# Unnormalised attention coefficients. 
# @ is the shorthand for matrix multiplication in numpy.
e11 = leaky_relu(ff_w @ (np.concatenate([W @ h1, W @ h1])), alpha=0.2)
e12 = leaky_relu(ff_w @ (np.concatenate([W @ h1, W @ h2])), alpha=0.2)
e13 = leaky_relu(ff_w @ (np.concatenate([W @ h1, W @ h3])), alpha=0.2)


# Applying softmax to normalise the value.
sum = np.exp(e11) + np.exp(e12) + np.exp(e13)
a11 = np.exp(e11) / sum # [[0.233]], rounded to 3dp for display purposes
a12 = np.exp(e12) / sum # [[0.189]]
a13 = np.exp(e13) / sum # [[0.578]]
```

Once we have the attention coefficients, we can use them to calculate new node 1 features:


$$ \vec{h'}_{i}  = σ (\sum_{j \in N_{i}} \alpha_{ij} W\vec{h}_{j})$$ 



```python
# Using leaky_relu as the nonlinearity.
# Implementation detail: called .item() to get the scalar value from the nested array.
new_h1 = leaky_relu(
    (a11.item() * (W @ h1)) + 
    (a12.item() * (W @ h2)) + 
    (a13.item() * (W @ h3))
)

# new_h1 value is [[0.407], [2.03], [-0.066]]
```

## Multi-head attention

The paper also uses multi-head attention to stabilise the learning process of self attention. This means there are multiple independent attention functions, a separate $$ \alpha_{ij} $$ and $$ W $$ for each $$ k $$ head. We concatenate the output from each head, so the overall output shape is $$ KF' $$ shapes

$$ \vec{h'}_{i}  =   \Big\Vert_{k=1}^K  σ (\sum_{j \in N_{i}} \alpha^k_{ij} W^k\vec{h}_{j})$$

```python
new_h1_k1 = leaky_relu(
    (a11_k1.item() * (W_k1 @ h1)) + 
    (a12_k1.item() * (W_k1 @ h2)) + 
    (a13_k1.item() * (W_k1 @ h3))
)

new_h1_k2 = leaky_relu(
    (a11_k2.item() * (W_k2 @ h1)) + 
    (a12_k2.item() * (W_k2 @ h2)) + 
    (a13_k2.item() * (W_k2 @ h3))
)
new_h1 = np.concatenate([new_h1_k1, new_h1_k2])
```

If the multi-head attention is on the prediction layer, then we need to average the output and apply the final nonlinearity afterward (either a softmax or logistic sigmoid for classification). This is to keep the output shape as $$ F' $$ to match the number of classes we are classifying or the predicted values.  


$$ \vec{h'}_{i}  = σ (\frac{1}{K} \sum^K_{k = 1} \sum_{j \in N_{i}} \alpha^k_{ij} W^k\vec{h}_{j}) $$

```python
new_h1 = leaky_relu((new_h1_k1 + new_h1_k2) / 2)
```


## GAT v2

_Paper_: [https://arxiv.org/pdf/2105.14491.pdf](https://arxiv.org/pdf/2105.14491.pdf)

### Static attention

In this GATv2 paper, it identifies the original GAT can only compute "static attention", that is when the attention function can only select the same key regardless of the input query. In the context of GAT, it means **the attention function is always favouring a particular source node (key) regardless of the target node (query input).**


The paper rearranges the original attention coefficient equation where $$ \alpha $$ is broken into two part
$$ \alpha = [\alpha_{1} || \alpha_{2} ] $$. This splits the attention function into two, applying the source node and target node separately


$$ e(h_{i}, h_{j}) = LeakyReLU(\vec{a}_{1}^T\boldsymbol{W}\vec{h}_{i} + \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{j}) $$

The attention coefficient is only influence by the source node part $$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{j} $$ as the target node part $$ \vec{a}_{1}^T\boldsymbol{W}\vec{h}_{i} $$ remains the same when calculating the coefficient for first order neighbours.

As a result, the attention function is learning a ranking between all the nodes in the graph since the output value is only dependent on the given node $$ \vec{h}_{j} $$. Source nodes that are higher in the ranking will always be more important regardless of the target node.

Let's look at a more concrete example, imagine after training, the attention function learnt the following ranking for all nodes.

$$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{3} > \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{4} > \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{1} > \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{2} $$

When calculating the attention coefficients for node 1, **node 3's coefficient is the largest** because the target node part $$ \vec{a}_{1}^T\boldsymbol{W}\vec{h}_{1} $$ is the same and $$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{3} $$ is greater than $$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{1} $$ and $$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{2} $$.

$$ e(h_{1}, h_{1}) = LeakyReLU(\vec{a}_{1}^T\boldsymbol{W}\vec{h}_{1} + \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{1}) $$

$$ e(h_{1}, h_{2}) = LeakyReLU(\vec{a}_{1}^T\boldsymbol{W}\vec{h}_{1} + \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{2}) $$

$$ e(h_{1}, h_{3}) = LeakyReLU(\vec{a}_{1}^T\boldsymbol{W}\vec{h}_{1} + \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{3}) $$

Then looking at calculating the attention coefficients for node 2, even though the target node has changed, it doesn't make a difference because the target node part will be the same value for each source and target node pair. Consequently, **node 3's coefficient is the largest again** because $$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{3} $$ is greater than $$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{3} $$ and $$ \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{4} $$.

$$ e(h_{2}, h_{2}) = LeakyReLU(\vec{a}_{1}^T\boldsymbol{W}\vec{h}_{2} + \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{2}) $$

$$ e(h_{2}, h_{3}) = LeakyReLU(\vec{a}_{1}^T\boldsymbol{W}\vec{h}_{2} + \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{3}) $$

$$ e(h_{2}, h_{4}) = LeakyReLU(\vec{a}_{1}^T\boldsymbol{W}\vec{h}_{2} + \vec{a}_{2}^T\boldsymbol{W}\vec{h}_{4}) $$


### Dynamic attention


Instead of static attention, the paper proposes "dynamic attention" where the attention function selects different keys based on different query input. This means **the attention function is able to favour different source node given the target node.**  

It's important to have dynamic attention because the model should be able to calculate the relative importance based on the local context (the target node), not only restricted to a ranking across all nodes like in static attention. For example, the model should be able to express that node 2 has highest importance to node 1 while node 3 has highest importance to node 2. 


To achieve dynamic attention, the paper applies the attention layer after the nonlinearity (LeakyReLU). Now the attention function is calculating the coefficient based on the target and source node pair

$$ e_{ij} = \vec{a}^T LeakyReLU([\boldsymbol{W}\vec{h}_{i} || \boldsymbol{W}\vec{h}_{j}]) $$

```python
e11 = ff_w @ leaky_relu(np.concatenate([W @ h1, W @ h1]), alpha=0.2)
```

No changes to other parts of the layer such as the softmax and the multi-head attention. **However, the paper noted that a single GATv2 head generalises better than a multi-head GAT layer.**

_Unfortunately, I don't fully understand the proof for the GATv2 layer to allow dynamic attention. Something about the layer being a universal approximator which is able to satisfy the property of dynamic attention. Please read the proof in the paper's appendix for more information._


## Example with TF-GNN 

Using the [intro_mutag_example](https://github.com/tensorflow/gnn/blob/main/examples/notebooks/intro_mutag_example.ipynb), we can replace the example SimpleConv layer with GATv2 and try it out immediately!

```python
# Add this import.
from tensorflow_gnn.models.gat_v2.layers import GATv2Conv


# Replace the layer in _build_model functions.
for i in range(num_message_passing):
    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={
            "atoms": tfgnn.keras.layers.NodeSetUpdate(
                {
                    # Use GATv2Conv layer instead of SimpleConv.
                    "bonds": GATv2Conv(
                        num_heads=1,
                        per_head_channels=1,
                        receiver_tag=tfgnn.TARGET,
                    )
                },
                tfgnn.keras.layers.NextStateFromConcat(dense(next_state_dim)),
            )
        }
    )(graph)
```
