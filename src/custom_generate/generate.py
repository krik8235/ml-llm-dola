import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'critical'

import logging
logger = logging.getLogger(__name__)

from typing import cast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, RepetitionPenaltyLogitsProcessor
from transformers.generation.utils import GenerateNonBeamOutput, GenerateDecoderOnlyOutput
from transformers.generation.streamers import BaseStreamer


# ref - https://github.com/XiangLi1999/ContrastiveDecoding/blob/170e9142e92159c1237d731e240f5eb14aabf428/transformers/src/transformers/generation_logits_process.py#L235
def _relative_top_filter(
    scores: torch.FloatTensor,
    baseline_scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float('Inf'),
    base_filter_value: float = -1e-3,
    min_tokens_to_keep: int = 1,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """filter tokens with probabilities above the threshold - `relative_top` * max probability in the distribution"""

    scores_normalized = scores.log_softmax(dim=-1)

    # compute threshold
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)

    # filter tokens
    baseline_scores_normalized = baseline_scores.log_softmax(dim=-1)
    baseline_scores_normalized[scores_normalized < probs_thresh] = base_filter_value
    scores_normalized[scores_normalized < probs_thresh] = filter_value

    return scores_normalized, baseline_scores_normalized # type: ignore



def _dola_select_contrast(
    candidate_premature_layers: list[int],
    candidate_premature_logits: dict[int, torch.FloatTensor],
    final_logits: torch.FloatTensor,
) -> torch.FloatTensor:
    
    if len(candidate_premature_layers) == 1:
        base_logits = candidate_premature_logits[candidate_premature_layers[0]]
        final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
        logits = cast(torch.FloatTensor, final_logits - base_logits)
        return logits


    # apply softmax to mature_layer
    softmax_mature_layer = F.softmax(final_logits, dim=-1) # shape: (batch_size, vocab_size)

    # apply softmax to stacked premature_layers
    stacked_premature_layers = torch.stack([candidate_premature_logits[i] for i in candidate_premature_layers], dim=0)
    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1) # shape: (num_premature_layers, batch_size, vocab_size)

    # calculate the average distribution
    avg_dist = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers) # shape: (num_premature_layers, batch_size, vocab_size)

    # compute log-softmax
    log_softmax_mature_layer = F.log_softmax(final_logits, dim=-1) # shape: (batch_size, vocab_size)
    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1) # shape: (num_premature_layers, batch_size, vocab_size)

    # compute the kl divergence using log-softmax
    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], avg_dist, reduction="none").mean(-1) # shape: (num_premature_layers, batch_size)
    kl2 = F.kl_div(log_softmax_premature_layers, avg_dist, reduction="none").mean(-1) # shape: (num_premature_layers, batch_size)

    # compute the js divergence
    js_divs = 0.5 * (kl1 + kl2) # shape: (num_premature_layers, batch_size)

    # reduce the batchmean
    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
    premature_layer = candidate_premature_layers[int(js_divs.argmax().item())]

    # compute logits
    base_logits = candidate_premature_logits[premature_layer]
    final_logits, base_logits = _relative_top_filter(final_logits, base_logits)
    logits = cast(torch.FloatTensor, final_logits - base_logits)

    return logits



def _dola_decoding(
        model,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        streamer: BaseStreamer | None = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> GenerateDecoderOnlyOutput | torch.LongTensor:
        r"""
        Generates sequences of token ids for models with a language modeling head using **dola decoding** and can be
        used for decoder-only text models.
        The method is based on the paper "DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language
        Models" (https://huggingface.co/papers/2309.03883) in ICLR 2024.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            dola_layers (`Union[str, list[int]]`):
                The candidate layers used in contrasting layers of DoLa. It can be either 1) 'low' or 'high', which
                means the lower part or higher part of the model layers, respectively, or 2) a list of layer indices
                to be used for candidate layers. The 0-th layer is the word embedding layer of the model.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """

        dola_layers: str | list[int] = generation_config.dola_layers
        assert dola_layers is not None, 'dola_layers must be set to use DoLa decoding'

        # repetition penalty
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty)
            logits_processor.append(repetition_penalty_processor) # add the processor to the list to apply it every step
            if generation_config.repetition_penalty < 1.2:
                logger.warning(
                    f"`repetition_penalty` is set to a value of {generation_config.repetition_penalty}, which could induce unwanted repetition. "
                    "The recommended value for DoLa decoding is `repetition_penalty>=1.2`.",
                )
        
        # pre-check dola conditions: num_beams = 1, non stateful, decoder-only llms only
        if getattr(generation_config, 'num_beams', 1) != 1: raise ValueError('dola generation needs num_beams == 1')
        if model.config.is_encoder_decoder: raise ValueError('dola decoding is only available for decoder-only models')
        if getattr(model, '_is_stateful', False):
            raise ValueError(f'dola decoding is not supported with stateful models, such as {model.__class__.__name__}')


        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores, raw_logits, decoder_attentions, cross_attentions, decoder_hidden_states = (), (), (), (), ()

        # track sequences already finished
        batch_size, cur_length = input_ids.shape[:2]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = model._get_initial_cache_position(cur_length, input_ids.device, model_kwargs)

        this_peer_finished = False

        # prepare layers for dola decoding
        final_layer = model.config.get_text_config().num_hidden_layers
        # if the model has tied word embeddings, we skip the word embeddings (0-th) layer and start from the 2nd layer,
        # as the early exit from word embeddings will become identity function
        # if the model is really shallow (<=2 layers), we use the 1st layer if it's not the final layer and the 0-th
        # layer otherwise. Notice that DoLa does not help shallow models much.
        if not model.config.tie_word_embeddings: start_layer = 0
        elif final_layer > 2: start_layer = 2
        elif final_layer == 2: start_layer = 1
        else: start_layer = 0

        # For `N`-layer models with `N <= 40` layers, the layers of `range(0, N // 2, 2)` and `range(N // 2, N, 2)`
        # are used for `'low'` and `'high'` layers, respectively.
        # For models with `N > 40` layers, the layers of `range(0, 20, 2)` and `range(N - 20, N, 2)` are used for
        # `'low'` and `'high'` layers, respectively.
        if isinstance(dola_layers, str) and dola_layers == "low":
            if start_layer == final_layer // 2:
                candidate_premature_layers = [start_layer]
            else:
                candidate_premature_layers = list(range(start_layer, final_layer // 2, 2)) if final_layer <= 40 else list(range(start_layer, 20, 2))
                
        elif isinstance(dola_layers, str) and dola_layers == "high":
            candidate_premature_layers = (
                list(range(final_layer // 2, final_layer, 2))
                if final_layer <= 40
                else list(range(final_layer - 20, final_layer, 2))
            )
        # set the `dola_layers` to a list of integers for layer indices to contrast manually specified layers.
        elif isinstance(dola_layers, list):
            candidate_premature_layers = [i for i in dola_layers if i < final_layer]
        else:
            raise ValueError("dola_layers must be either 'low', 'high' or a list of integers.")

        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("DoLa is not supported for models that don't have output embeddings.")

        
        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

            # .float() is needed to retain precision for later logits manipulations
            final_layer_next_token_logits = outputs.logits[:, -1, :].detach().to(copy=True, dtype=torch.float32)
            final_logits = outputs.logits[:, -1, :].float()
            candidate_premature_logits = {}
            for candidate_premature_layer in candidate_premature_layers:
                candidate_premature_logits[candidate_premature_layer] = lm_head(
                    outputs.hidden_states[candidate_premature_layer][:, -1, :]
                ).to(final_logits.device)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = _dola_select_contrast(
                candidate_premature_layers, candidate_premature_logits, final_logits
            )
            next_token_logits = next_token_logits.to(input_ids.device)
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # record scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores: scores += (next_token_scores,)
                if output_logits: raw_logits += (final_layer_next_token_logits,)
                if output_attentions:
                    if model.config.is_encoder_decoder:
                        decoder_attentions += (outputs.decoder_attentions,)
                        cross_attentions += (outputs.cross_attentions,)
                    else:
                        decoder_attentions += (outputs.attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (outputs.decoder_hidden_states,) if model.config.is_encoder_decoder else (outputs.hidden_states,)

            if do_sample:  # sample
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:  # argmax
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = cast(torch.LongTensor, torch.cat([input_ids, next_tokens[:, None]], dim=-1))
            if streamer is not None: streamer.put(next_tokens.cpu())

            # stop when each sentence is finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None: streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=cast("tuple[torch.FloatTensor] | None", scores),
                logits=cast("tuple[torch.FloatTensor] | None", raw_logits),
                attentions=cast("tuple[tuple[torch.FloatTensor]]", decoder_attentions),
                hidden_states=cast("tuple[tuple[torch.FloatTensor]]", decoder_hidden_states),
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids
        


def generate(model, *args, **kwargs):
    """
    custom generate function for dola decoding.
    Args:
        model (`PreTrainedModel`):
            The model to generate from.

        dola_layers (str | list[int] | None): 
            The layers to use for DoLa decoding. If `None`, DoLa decoding is not used. If a string, it must be one of "low" or "high", which means using the lower part or higher part of the model layers, respectively.
            "low" means the first half of the layers up to the first 20 layers, and "high" means the last half of the
            layers up to the last 20 layers.
            If a list of integers, it must contain the indices of the layers to use for candidate premature layers in DoLa.
            The 0-th layer is the word embedding layer of the model. Set to `'low'` to improve long-answer reasoning tasks,
            `'high'` to improve short-answer tasks. Check the [documentation](https://huggingface.co/transformers-community/dola)
            or [the paper](https://huggingface.co/papers/2309.03883) for more details.
    """

    generation_outputs = model.generate(custom_generate=_dola_decoding, *args, **kwargs)
    return generation_outputs
