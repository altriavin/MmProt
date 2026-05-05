import re
from collections import OrderedDict
from transformers import AutoModel, AutoTokenizer
from .configuration_bert import JinaBertConfig
import torch
from .modeling_bert import BertModel

def remap_state_dict(state_dict, config: JinaBertConfig):
    """
    Map the state_dict of a Huggingface BERT model to be flash_attn compatible.
    """

    # LayerNorm
    def key_mapping_ln_gamma_beta(key):
        key = re.sub(r"LayerNorm.gamma$", "LayerNorm.weight", key)
        key = re.sub(r"LayerNorm.beta$", "LayerNorm.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_ln_gamma_beta(k), v) for k, v in state_dict.items())

    # Layers
    def key_mapping_layers(key):
        return re.sub(r"^encoder.layer.", "encoder.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^embeddings.LayerNorm.", "emb_ln.", key)
        key = re.sub(
            r"^encoder.layers.(\d+).attention.output.LayerNorm.(weight|bias)",
            r"encoder.layers.\1.norm1.\2",
            key,
        )
        key = re.sub(
            r"^encoder.layers.(\d+).output.LayerNorm.(weight|bias)",
            r"encoder.layers.\1.norm2.\2",
            key,
        )
        key = re.sub(
            r"^cls.predictions.transform.LayerNorm.(weight|bias)",
            r"cls.predictions.transform.layer_norm.\1",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^encoder.layers.(\d+).intermediate.dense.(weight|bias)",
            r"encoder.layers.\1.mlp.fc1.\2",
            key,
        )
        key = re.sub(
            r"^encoder.layers.(\d+).output.dense.(weight|bias)",
            r"encoder.layers.\1.mlp.fc2.\2",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    last_layer_subset = getattr(config, "last_layer_subset", False)
    for d in range(config.num_hidden_layers):
        Wq = state_dict.pop(f"encoder.layers.{d}.attention.self.query.weight")
        Wk = state_dict.pop(f"encoder.layers.{d}.attention.self.key.weight")
        Wv = state_dict.pop(f"encoder.layers.{d}.attention.self.value.weight")
        bq = state_dict.pop(f"encoder.layers.{d}.attention.self.query.bias")
        bk = state_dict.pop(f"encoder.layers.{d}.attention.self.key.bias")
        bv = state_dict.pop(f"encoder.layers.{d}.attention.self.value.bias")
        if not (last_layer_subset and d == config.num_hidden_layers - 1):
            state_dict[f"encoder.layers.{d}.mixer.Wqkv.weight"] = torch.cat(
                [Wq, Wk, Wv], dim=0
            )
            state_dict[f"encoder.layers.{d}.mixer.Wqkv.bias"] = torch.cat([bq, bk, bv], dim=0)
        else:
            state_dict[f"encoder.layers.{d}.mixer.Wq.weight"] = Wq
            state_dict[f"encoder.layers.{d}.mixer.Wkv.weight"] = torch.cat([Wk, Wv], dim=0)
            state_dict[f"encoder.layers.{d}.mixer.Wq.bias"] = bq
            state_dict[f"encoder.layers.{d}.mixer.Wkv.bias"] = torch.cat([bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub(
            r"^encoder.layers.(\d+).attention.output.dense.(weight|bias)",
            r"encoder.layers.\1.mixer.out_proj.\2",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    def key_mapping_decoder_bias(key):
        return re.sub(r"^cls.predictions.bias", "cls.predictions.decoder.bias", key)

    state_dict = OrderedDict((key_mapping_decoder_bias(k), v) for k, v in state_dict.items())

    # Word embedding
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    if pad_vocab_size_multiple > 1:
        word_embeddings = state_dict["embeddings.word_embeddings.weight"]
        state_dict["embeddings.word_embeddings.weight"] = F.pad(
            word_embeddings, (0, 0, 0, config.vocab_size - word_embeddings.shape[0])
        )
        decoder_weight = state_dict["cls.predictions.decoder.weight"]
        state_dict["cls.predictions.decoder.weight"] = F.pad(
            decoder_weight, (0, 0, 0, config.vocab_size - decoder_weight.shape[0])
        )
        # If the vocab was padded, we want to set the decoder bias for those padded indices to be
        # strongly negative (i.e. the decoder shouldn't predict those indices).
        # TD [2022-05-09]: I don't think it affects the MLPerf training.
        decoder_bias = state_dict["cls.predictions.decoder.bias"]
        state_dict["cls.predictions.decoder.bias"] = F.pad(
            decoder_bias, (0, config.vocab_size - decoder_bias.shape[0]), value=-100.0
        )

    # LayerNorm
    def key_mapping_layernorm(key):
        return re.sub(r'^encoder.layers.(\d+).mlp.layernorm.(weight|bias)', r"encoder.layers.\1.norm2.\2", key)

    state_dict = OrderedDict((key_mapping_layernorm(k), v) for k, v in state_dict.items())

    return state_dict


v2_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
config = JinaBertConfig(vocab_size=30528, use_qk_norm=False, mlp_type='glu', hidden_act='gelu')
state_dict = v2_model.state_dict()
new_state_dict = remap_state_dict(state_dict, config)
flash_model = BertModel(config)
flash_model.load_state_dict(new_state_dict)


torch.save(new_state_dict, 'converted_weights.bin')
print(config.to_json_string())


"""
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
inp = tokenizer.batch_encode_plus(['Hello world', 'How is the weather today?', 'It is raining a lot in  Berlin'], return_tensors='pt', padding=True).to('cuda')
v2_model.eval()
flash_model.eval()
v2_model = v2_model.to('cuda', torch.float16)
flash_model = flash_model.to('cuda', torch.float16)
output_v2 = v2_model(**inp)
output_flash = flash_model(**inp)
x = output_v2.last_hidden_state
y = output_flash.last_hidden_state
print(torch.abs(x - y))
"""