# LASTDANCE: Layerwise Activation Similarity to Training Data for Assessing Non-Conforming Events

This is the repo for LASTDANCE, a novel method for out-of-distribution detection which leverages feature trajectory through the network, rather than using a single "optimal" layer, or an "early exit" network/ensemble, which sacrifices performance on the task. Using this trajectory paradigm, we set a new state-of-the-art result on Competency Estimation.

TODO
project onto unit ball -> cosine similarity
johnson lindenstrauss bound maybe?

NOTES

a bug has led me to something funny about this I didn't realize before.

if you do not set the model with model.eval(), the batch norm layers will stay on and this overlap is far more likely to happen. If you set the model to model.eval(), it's more common for the ood points to land far away, but they can still land in distribution. However, methods like MC dropout, which rely on the model using stochastic layers like MC dropout, don't perform well out of distribution. I wonder why!!

I will leave the model in train mode so that this issue is more noticable and prominent