from functools import partial

import torch
import transformers

# from bias_bench.debias.self_debias.modeling import GPT2Wrapper
# from bias_bench.debias.self_debias.modeling import MaskedLMWrapper


class BertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)


class AlbertModel:
    def __new__(self, model_name_or_path):
        return transformers.AlbertModel.from_pretrained(model_name_or_path)


class RobertaModel:
    def __new__(self, model_name_or_path):
        return transformers.RobertaModel.from_pretrained(model_name_or_path)


class GPT2Model:
    def __new__(self, model_name_or_path):
        return transformers.GPT2Model.from_pretrained(model_name_or_path)


class BertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)


class AlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)


class RobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)


class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)


class _SentenceDebiasModel:
    def __init__(self, model_name_or_path, bias_direction):
        def _hook(module, input_, output, bias_direction):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            bias_direction = bias_direction.to(x.device)

            # Debias the representation.
            for t in range(x.size(1)):
                x[:, t] = x[:, t] - torch.ger(
                    torch.matmul(x[:, t], bias_direction), bias_direction
                ) / bias_direction.dot(bias_direction)

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, bias_direction=bias_direction)


class _INLPModel:
    def __init__(self, model_name_or_path, projection_matrix):
        def _hook(module, input_, output, projection_matrix):
            # Debias the last hidden state.
            x = output["last_hidden_state"]

            # Ensure that everything is on the same device.
            projection_matrix = projection_matrix.to(x.device)

            for t in range(x.size(1)):
                x[:, t] = torch.matmul(projection_matrix, x[:, t].T).T

            # Update the output.
            output["last_hidden_state"] = x

            return output

        self.func = partial(_hook, projection_matrix=projection_matrix)


class SentenceDebiasBertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2Model(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model


class SentenceDebiasBertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2LMHeadModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


class INLPBertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2Model(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model.register_forward_hook(self.func)
        return model


class INLPBertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class INLPAlbertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class INLPRobertaForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model.transformer.register_forward_hook(self.func)
        return model


class CDABertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class CDAAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class CDARobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model


class CDAGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model


class CDABertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDAAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDARobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDAGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model


class DropoutBertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model


class DropoutGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model


class DropoutBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model


class BertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class AlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class RobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class GPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class SentenceDebiasBertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2ForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class INLPBertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2ForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class CDABertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDARobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutBertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutRobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


# class SelfDebiasBertForMaskedLM:
#     def __new__(self, model_name_or_path):
#         model = MaskedLMWrapper(model_name_or_path)
#         return model


# class SelfDebiasAlbertForMaskedLM:
#     def __new__(self, model_name_or_path):
#         model = MaskedLMWrapper(model_name_or_path)
#         return model


# class SelfDebiasRobertaForMaskedLM:
#     def __new__(self, model_name_or_path):
#         model = MaskedLMWrapper(model_name_or_path)
#         return model


# class SelfDebiasGPT2LMHeadModel:
#     def __new__(self, model_name_or_path):
#         model = GPT2Wrapper(model_name_or_path, use_cuda=False)
#         return model







class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

class BertPrefixForMaskedLM(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = transformers.BertLMHeadModel(config)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for _, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values
        )

        return outputs

class BertPrefixModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = transformers.BertModel(config)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for _, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        # past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values
        )

        return outputs

class BertPrefixModelForSequenceClassification(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = transformers.BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        # self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        for param in self.prefix_encoder.parameters():
            param.requires_grad = False

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105
    
    def reset_classifier(self, config): 
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertPrefixForSequenceClassification: 
    def __new__(self, model_name_or_path, config):
        model = BertPrefixModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
        )
        return model

class ADEPTBertForMaskedLM: 
    def __new__(self, model_name_or_path
    # , config
    ):
        model = BertPrefixForMaskedLM.from_pretrained(
            model_name_or_path,
            # config=config,
        )
        return model

class ADEPTBertModel: 
    def __new__(self, model_name_or_path
    # , config
    ):
        model = BertPrefixModel.from_pretrained(
            model_name_or_path,
            # config=config,
        )
        return model