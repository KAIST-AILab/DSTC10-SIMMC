import torch
from torch import Tensor, nn
from typing import List, Dict, Optional
from transformer import MultiModalTransformer, UniModalTransformer
from torch.nn import CrossEntropyLoss
from transformers.generation_logits_process import (
    # EncoderNoRepeatNGramLogitsProcessor,
    # ForcedBOSTokenLogitsProcessor,
    # ForcedEOSTokenLogitsProcessor,
    # HammingDiversityLogitsProcessor,
    # InfNanRemoveLogitsProcessor,
    # LogitsProcessorList,
    # MinLengthLogitsProcessor,
    # NoBadWordsLogitsProcessor,
    # NoRepeatNGramLogitsProcessor,
    # PrefixConstrainedLogitsProcessor,
    # RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import ipdb
class Model(nn.Module):
    def __init__(
        self,
        args,
        vocab_size,
        backbone,
        d_model = 512,
    ):
        super().__init__()
        
        # index 0 is None object
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.with_image = args.with_image
    
        #  tokenizer
        if args.with_image:
            print("Building MultiModal Transformer ...")
            self.transformer = MultiModalTransformer(
                backbone=backbone,
                text_encoder_type="roberta-base"
            )
        # only use text encoder
        else:
            self.transformer = UniModalTransformer(
                text_encoder_type="roberta-base"
            )

        if args.add_special_tokens:
            self.transformer.text_encoder.resize_token_embeddings(self.vocab_size)
            self.transformer.text_encoder.vocab_size = self.vocab_size

        self.generation_head = nn.Linear(d_model, self.vocab_size, bias=False) 

       
    def with_query_pos_embed(self, tensor, pos : Optional[Tensor]):
        pass

    def forward(
        self,
        batch: Dict,
        task_type: str="response-generation",
        query_pos_embed: bool=False
    ):
        """
        samples["images"] List, (bs, n_c, h, w) NestedTensor.tensor .mask
        samples["input_text"]
        samples[""]
        samples["objects"] bs, max_num_item, including previous scene objects
        sampels["bbox"] bs, max_num_item, 4, including previous scene objects with negative value
        """
        # seq_len, bs, d_model
        # 1) tgt, src embedding space shared
        # 2) tgt build embedding space  
        if self.with_image:
            try :
                query_embedd = self.transformer.text_encoder.get_input_embeddings()(batch["belief"]) # bs, seq_len, dim
            except:
                ipdb.set_trace()
            
            hs = self.transformer(images=batch["image"], text=batch["predict"], query_embedd=query_embedd)
            logits = self.generation_head(hs) # seq_len, bs, vocab_size
            return logits.permute(1, 0, 2) # (bs, seq_len, vocab_size)
        
        else:
            try:
                query_embedd = self.transformer.text_encoder.get_input_embeddings()(batch["belief"]) # bs, seq_len, dim
            except:
                ipdb.set_trace()

            hs = self.transformer(text=batch["predict"], query_embedd=query_embedd)
            logits = self.generation_head(hs) # seq_len, bs, vocab_size

            return logits.permute(1, 0, 2) #(bs, seq_len, vocab_size)

    def encode(self, batch: Dict):
        if self.with_image:
            return self.transformer.encode(images=batch["image"], text=batch["predict"])
        else:
            return self.transformer.encode(text=batch["predict"])

    @torch.no_grad()
    def generation(
        self,
        top_k,
        top_p,
        batch, 
        decoder_start_token_id,  
        eos_token_id, 
        pad_token_id, 
        max_length=256,
        min_tokens_to_keep=1,
    ):

        memory, mask = self.encode(batch)
        bs = batch["predict"].input_ids.size(0)
        device = batch["predict"].input_ids.device

        def get_next_tokens_score(query):
            query_embedd = self.transformer.text_encoder.get_input_embeddings()(query) # tensor()
            hs = self.transformer.decode(query_embedd, memory, memory_key_padding_mask=mask)
            logits = self.generation_head(hs) # ?? # seq_len, bs, vocab_size
            return logits.permute(1, 0, 2)

        unfinished_sequences = torch.ones(bs).to(device) # (bs, )
        query = torch.tensor(decoder_start_token_id).view(1, -1).repeat(bs, 1).to(device) # bs, seq_len 
        cur_len = 0
        while True:
            logits = get_next_tokens_score(query) # bs, seq_len, vocab_size
            # top p sampling
            assert 0 <=top_p <=1.0
            next_token_scores = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1)(None, logits[:, -1, :]) # bs, vocab_size

            # top k sampling
            # assert top_k > 0 
            # next_token_scores = TopKLogitsWarper(top_k=top_k, filter_value=-float("Inf"), min_tokens_to_keep=min_tokens_to_keep)(None, logits[:, -1, :])

            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1) # (bs, )
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            query = torch.cat([query, next_tokens.view(-1, 1)], dim=-1).long().to(device)# bs, seq_len, 
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            # if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if unfinished_sequences.max() == 0 or cur_len >= max_length:
                break

        return query
            