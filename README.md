 We’re preparing the code release for Regularized Score Centering (RSC). To experiment with RSC starting from slime, extend its score-
centering loss with a reverse-KL penalty toward the behavior policy that generated each sample. We estimate the SC center using 128
independent auxiliary token samples from the behavior policy at each history (MC128). Define `rho = exp(logp - logmu)`, using the learner’s
log-probability and the stored behavior log-probability. For uncapped KL, add `beta * stop_gradient(rho * log(rho)) * logp` to the loss being
minimized, keeping the behavior probabilities fixed. For RSC with TIS, also apply truncated importance weights consistently to the reward
contribution and its score center. We’ll release the implementation and example configurations soon.
