import numpy as np
from decoder import *
from data import *


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, probs, beam_id):
        """Hypothesis constructor.
        Args:
          tokens: List of integers. The ids of the tokens that form the summary so far.
          log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
          state: Current state of the decoder, a LSTMStateTuple.
          attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
          p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
          coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.probs = probs
        self.beam_id = beam_id
    

    def extend(self, token, prob, beam_id):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.
        Args:
          token: Integer. Latest token produced by beam search.
          log_prob: Float. Log prob of the latest token.
          state: Current decoder state, a LSTMStateTuple.
          attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
          p_gen: Generation probability on latest step. Float.
          coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
          New Hypothesis for next step.
        """
        return Hypothesis(tokens = self.tokens + [token],
                      probs = self.probs + [prob],
                      beam_id = beam_id)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.probs)

    @property
    def avg_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.prob / len(self.tokens)


def run_beam_search(model, vocab, article_input, beam_size, max_len):
    """Performs beam search decoding on the given example.
    Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch
    Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """
    # Run the encoder to get the encoder hidden states and decoder initial state
    
    #symbols = [START_DECODING_ID]
    #beam_ids = [0]
    #probs = [1.]
    #result = np.array([[]*beam_width])
    
    model.eval_run_encoder(article_input)
    

    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[START_DECODING_ID],
                     probs=[0.0],
                    beam_id=0)]
    results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

    steps = 0
    while steps < max_len and len(results) < beam_size:
        latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
        beam_ids = [h.beam_id for h in hyps] 
        
        # Run one step of the decoder to get the new info
        hiddens = model.decode((LongTensor([latest_tokens]), LongTensor(beam_ids)))

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        num_orig_hyps = len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
        for i in range(num_orig_hyps):
            values, indices = hiddens[i].topk(beam_size * 2)
            mask = (indices < VOCAB_SIZE).long()
            indices = indices * mask
            values = torch.log(values)
            #print(indices)
            h = hyps[i]
            
            for j in range(beam_size * 2):  # for each of the top 2*beam_size hyps:
            # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(token=indices[j].item(),
                               prob=values[j].item(),
                               beam_id=beam_ids[i])
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = [] # will contain hypotheses for the next step
        for h in sort_hyps(all_hyps): # in order of most likely h
            if h.latest_token == STOP_DECODING_ID: # if stop token is reached...
            # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= 35:
                    results.append(h)
            else: # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)
            if len(hyps) == beam_size or len(results) == beam_size:
                # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted[0].tokens[1:]

def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_prob, reverse=True)