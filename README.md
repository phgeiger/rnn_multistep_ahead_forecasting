Introduction
===

This repo contains code for my [blog post on implementing time series multi-step ahead forecasts using recurrent neural networks in TensorFlow](http://geiger.onl/news/blog/2018/05/26/Implementing-time-series-multi-step-ahead-forecasts-using-recurrent-neural-networks-in-TensorFlow.html). A version of this blog can also be found below.


Content
===

- "rnnmultistepahead.py": contains the model functions, estimators, training, evlauation and plot for simple, preliminary examples of the "recursive" and the "joint" approach to multistep ahead time series forecasting. You should be able to run it just like that.
- "daywise_max_planck_cafeteria_data.csv": data that is used as example in "rnnmultistepahead.py". Each row contains contains 36 queue length measurements corresponding to one day in the Max Planck Tuebingen campus cafeteria, Germany, measured between 11:30 and 14:30, every 5 minutes. 



Blog: Implementing time series multi-step ahead forecasts using recurrent neural networks in TensorFlow
===


Recently I started to use recursive neural networks (RNNs) in TensorFlow (TF) for time series forecasting. I'd like to perform multistep ahead forecasts and was wondering how to do this (1) with RNNs in general and (2) in TF in particular. Here I summarize my insights and provide some code.

To give a specific example of the problem: for our Max Planck Tuebingen campus cafeteria I'd like to forecast congestion (more specifically: queue length). The cafeteria opens at 11:30. Say it's 12:30, and I'd like to predict congestion for the remaining open time till 14:30. In my current data, congestion is measured every 5 minutes. 

Before going into details, let me emphasize that these are just preliminary notes, I'm not claiming that I have fully understood the problem and available approaches. Please feel free to get in touch if you have any comments!

## High-level background

It's nice about generative (probabilistic/Bayesian) model-based inference approaches, such as the classical state space model and corresponding Kalman filter [1], that you learn parameters that encode all possible relationship/joint distributions between variables. This allows you to derive forecasts for any number of steps ahead in a principled way. 

From my understanding, this is not in general possible in classical neual net-based approaches (similar, I guess, as it would be with other "discriminative" methods). In a sense: whatever mapping you want, you have to train it explicitly.

To predict multiple steps ahead using RNNs, various approaches have been proposed nonetheless. Let us follow [2][3] to briefly introduce them (note that [2][3] do not explicitly restrict to RNNs - could be non-recursive NNs as well):
- "Joint" approach: Let h be the number of steps we'd like to predict ahead. The NN is build such that at each step, it always outputs predictions for all h steps, i.e., a vector of dimension h, instead of just predicting only the next observation. The NN is trained for exactly this task, which gives us some guarantees that it should work -- in principle. One downside is obviously that we have to fix h in advance.
- "Independent" approach: Similar as joint, but train a separate NN for each future observation i from 1 to h separately.
- "Recursive" (called "iterative" in [3]): This is a substantially different approach: for each step i from 1 to h, take the previous prediction and feed it into a single cell of the RNN (together with the current state) instead of feeding in actual observations (which are not available yet). The nice thing is that this allows arbitrary many steps ahead. The downside is that, except when limiting to say linear RNNs which may then converge to Kalman filter update rules, we don't have theoretical guarantees as to whether this is acutally the "optimal"/right thing to do (as indicated above) [3].
- As a side note, there seem to be efforts [4] to map parameters of the RNN onto a kalman model and this in turn onto parameters for multistep models, to make things more principled.

Generally, for me it would be interesting if there are other, more principled, approaches, so please get in touch if you know of any.

## Implementation and evaluation in TensorFlow

I implemented a first version of the recursive versus the joint approach in TF and applied it to data from our campus cafeteria which I mentioned above. Note that for now, I focused on intra-day forecasts, while the more generic use case for RNNs and the recursive approach would probably be between-day forecasts.

The joint approach was straight forward to code, and I built on [8].

The recursive approach was a bit more tricky and I haven't found a fully convinccing solution. I used the estimator API of TF. I coded one model function which has two "modes", distinguishable via parameters: (1) a classical 1-step ahead RNN (LSTM, to be specific) and (2) arbitrary steps ahead predictions based on the recursive approach. Then I train the model via (1), store the weights as a checkpoint, and define a new estimator based on the mode (2), but with the weights loaded from the training of (1).

The code is available on GitHub.

A first evaluation shows that the joint approach works better which does not come as a surprise, because what is evaluated is excatly what was trained. Nonetheless, I was surprised how well the recursive approach performed, even though I pretty much did no fine-tuning of hyper-parameters.

As two side notes regarding implementation in TF:
- Apparently natural language understanding (NLU) people deal with similar issues and there is some implementation for this in TensorFlow [5], but I haven't looked at the details.
- Maybe it's somehow possible to harness the fact that predictions and outputs are closely related in an RNN.


## References

- \[1\]: H. Luetkepohl: New Introduction to Multiple Time Series Analysis
- \[2\]: Kline et al.: Methods for Multi-Step Time Series Forecasting with Neural Networks
- \[3\]: Multi-Step-Ahead Chaotic Time Series Prediction using Coevolutionary Recurrent Neural Networks
- \[4\]: R. Wilson et al.: A Neural Implementation of the Kalman Filter
- \[5\]: https://www.quora.com/In-TensorFlow-given-an-RNN-is-it-possible-to-use-the-output-of-the-previous-time-step-as-input-for-the-current-time-step-during-the-training-phase-How
- \[6\]: Arun Venkatraman, Martial Hebert, and J. Andrew Bagnell: Improving Multi-step Prediction of Learned Time Series Models
- \[7\]: Girard et al.: Gaussian Process Priors With Uncertain Inputs Application to Multiple-Step Ahead Time Series Forecasting
- \[8\]: https://medium.com/google-cloud/how-to-do-time-series-prediction-using-rnns-and-tensorflow-and-cloud-ml-engine-2ad2eeb189e8
