from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List, Iterable

import torch

from torch import nn
from torch.nn import functional as F

from transformers.file_utils import ModelOutput
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartClassificationHead,
    BartModel,
    BartPretrainedModel
)


@dataclass
class MultiTaskOutput(ModelOutput):
    '''
        Customized output dataclass for multitask model

        loss <Optional[torch.FloatTensor]> : language modeling loss
        cl_loss <Optional[torch.FloatTensor]> : classification loss
        logits <torch.FloatTensor>  : language modeling logits
        cl_logits <torch.FloatTensor> : classification logits
    '''
    loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cl_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class LabelSmoothingLoss(nn.Module):
    """ With label smoothing, KL-divergence between q_{smoothed ground truth prob.}(w) and p_{prob. computed by model}(w) is minimized. """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        
    def forward(self, output, target):
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        return F.kl_div(output, model_prob, reduction='sum')


class BartForMultiTask(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.config = config
        self.model = BartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        # Option
        self.classify_on_encoder = kwargs.pop("classify_on_encoder", True)
        self.num_classes = kwargs.pop("num_classes", config.num_labels)
        # Language modeling head
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )
        # Classification head
        self.cl_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            self.num_classes,
            config.classifier_dropout
        )
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        lm_labels=None,
        cl_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        is_generation=True,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if cl_labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Shift lm_labels
        if lm_labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    lm_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        # Forward BART
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Get last decoder hidden state
        decoder_hidden_state = outputs.last_hidden_state
        # Compute language modeling logits
        lm_logits = self.lm_head(decoder_hidden_state) + self.final_logits_bias
        # Compute classification logits
        # cl_logits = None
        encoder_hidden_state = outputs.encoder_last_hidden_state
        sentence_representation = encoder_hidden_state[:,0,:]    
        cl_logits = self.cl_head(sentence_representation)
        
        # If language modeling label is given, compute language modeling loss
        lm_loss = None
        if lm_labels is not None:
            # loss_fn = LabelSmoothingLoss(self.config.vocab_size, smoothing=0.1, ignore_index=self.config.pad_token_id)
            # lm_loss = loss_fn(
            #     lm_logits.contiguous().view(-1, self.config.vocab_size),
            #     lm_labels.contiguous().view(-1)
            # )
            lm_loss = F.cross_entropy(
                lm_logits.contiguous().view(-1, self.config.vocab_size),
                lm_labels.contiguous().view(-1),
                ignore_index=self.config.pad_token_id, reduction='none'
            )
        # If classification label is given, compute classification loss
        cl_loss = None
        if cl_labels is not None:
            # loss_fn = LabelSmoothingLoss(2, smoothing=0.1, ignore_index=-100)
            # cl_loss = loss_fn(cl_logits, cl_labels)
            cl_loss = F.cross_entropy(cl_logits, cl_labels, ignore_index=-100)
        
        # It is crucial to use the same name for GenerationMixin
        return MultiTaskOutput(
            loss=lm_loss,
            cl_loss=cl_loss,
            logits=lm_logits,
            cl_logits=cl_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids, 
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past