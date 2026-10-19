# Hidden State Layer ID Selection

Most speculative decoding architectures use hidden states from multiple layers in the target model as an input (the primary exception is MTP, which always uses the final hidden state).
For such architectures (Eagle-3, DFlash2, DSpark, etc...), the list of hidden states can be specified with the `--target-layer-ids` training flag.
The default list of target layer IDs has three items: `[2, num_layers // 2, num_layers - 3]`.
This is an acceptable default and a good starting point.
However, different hidden states contain different information about upcoming tokens, thus hidden state layer selection can materially impact the final performance of the model.

## Principles to Keep In Mind

As indicated above, hidden states from different target layers contain different information.
Nearby layers (and especially adjacent layers, e.g. layer 30 and layer 31) tend to be highly correlated (e.g. see [Gromov, Tirumala et al., 2024](https://arxiv.org/pdf/2403.17887)).
Thus, if you already have layer 30 in your hidden state selection, adding layer 31 may not provide much **additional** information to the draft model.
In particular, here is an example of how **NOT** to do layer selection: 1. Do a sweep of all single-layer-id lists. 2. Select the N highest performing layer ids.
This will likely end up with many layers that are close together and highly correlated (e.g. if layer 30 is the top performer, it is likely layer 31 will be in the top N as well).

## Recommended Layer Selection Strategy

The following guidelines are based on the ablations reported in [RFC #1015](https://github.com/vllm-project/speculators/issues/1015), which indicate that performance differences due to layer selection are stable across a variety of training scales, when the selections being compared have the same quantity of hidden states. As an example, if `[3,7,12]` beats `[2,10,13]` on a short training run, then it would also tend to win on a long training run. However, if `[3,7,12]` beats `[2,10,13,15]` on a short training run, we are not as confident that it will win on a long training run (this is likely due to the fact that a longer list of hidden states will produce a draft model with more parameters in the hidden state projection layer and may take longer to converge, but might converge to a higher value).

The overall shape of this strategy is as follows.

1. Generate a collection of candidate layer ID lists
2. Do a small training run for each candidate list
3. Select the K top-performing candidate lists
4. Do a medium training run for each surviving candidate list
5. Select the top-performing candidate from (4.)

We provide a utility script to generate candidate layer-id lists:

```bash
# For a target model with 32 layers, this will generate
# 20 total candidate layer ID lists of 6 hidden state IDs.
# It will be 4 candidate lists per 'bias', for each of the
# 5 biases. The bias determines the probability of a layer
# ID being chosen based on its position. As an example,
# the 'deep' bias favors larger layer IDs over smaller layer
# IDs. The --minimum-gap parameter enforces a minimum spacing
# between layer IDs in each candidate list. In this example,
# [2,5,7,10,19,20] would be impossible, because the pair (19,20)
# does not satisfy the minimum-gap of 1.
python scripts/sample_layer_candidates.py \
    --num-layers 32 \
    --num-layer-ids 6 \
    --num-candidates-per-bias 4 \
    --bias uniform deep shallow tails center \
    --minimum-gap 1 \
    --seed 0
```

As in the RFC, using approximately one epoch 4000 samples for the small training runs and one epoch of 64000 samples for the medium training runs is a reasonable starting point.
This can, of course, be scaled up or down depending on your compute and data capacity.
Larger and longer small/medium training runs will provide more reliable layer selection at the cost of more compute.
