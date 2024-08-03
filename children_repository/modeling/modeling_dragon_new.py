from transformers.models.deberta_v2 import modeling_deberta_v2
from transformers.models.roberta import modeling_roberta
from transformers.models.bert import modeling_bert
import torch
import torch.nn as nn
import types
import logging
import os
from typing import Optional, Union
from transformers.configuration_utils import PretrainedConfig
import copy
from threading import Thread
from transformers.utils.hub import get_checkpoint_shard_files
from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
from packaging import version
from zipfile import is_zipfile
from contextlib import contextmanager
import importlib.metadata
import inspect
from transformers.utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
from collections.abc import Sequence
import torch.nn.functional as F

from transformers.safetensors_conversion import auto_conversion
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13

from transformers.utils import (
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    cached_file,
    download_url,
    has_file,
    is_offline_mode,
    is_remote_url,
    is_safetensors_available,
    strtobool,
    is_accelerate_available,
    ContextManagers,
)

if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import (
        check_tied_parameters_on_same_device,
        extract_model_from_parallel,
        find_tied_parameters,
        get_balanced_memory,
        get_max_memory,
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
        set_module_tensor_to_device,
    )

    accelerate_version = version.parse(
        importlib.metadata.version("accelerate"))
    if accelerate_version >= version.parse("0.31"):
        from accelerate.utils.modeling import get_state_dict_from_offload


from utils import layers
from modeling import modeling_gnn


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file


def is_fsdp_enabled():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def is_local_dist_rank_0():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and int(os.environ.get("LOCAL_RANK", -1)) == 0
    )


def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    else:
        return next(state_dict.values()).dtype


TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}

_init_weights = True


@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights

    if _enable:
        _init_weights = False

        def _skip_init(*args, **kwargs):
            pass

        # # Save the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        _init_weights = old_init_weights
        if _enable:
            # # Restore the original initialization functions
            for name, init_func in TORCH_INIT_FUNCTIONS.items():
                setattr(torch.nn.init, name, init_func)


modeling_bert: types.ModuleType
modeling_roberta: types.ModuleType
modeling_deberta: types.ModuleType


logger = logging.getLogger(__name__)


INHERIT_BERT = os.environ.get('INHERIT_BERT', 0)
if INHERIT_BERT:
    PreTrainedModelClass = modeling_bert.BertPreTrainedModel
else:
    PreTrainedModelClass = modeling_deberta_v2.DebertaV2PreTrainedModel
    # PreTrainedModelClass = modeling_roberta.RobertaPreTrainedModel
print('PreTrainedModelClass', PreTrainedModelClass)


class DRAGON(nn.Module):
    def __init__(
        self,
        args,
        model_name='microsoft/deberta-v3-large',
        k=5,
        n_ntype=4,
        n_etype=38,
        n_concept=799273,
        concept_dim=200,
        concept_in_dim=1024,
        n_attention_head=2,
        fc_dim=200,
        n_fc_layer=0,
        p_emb=0.2,
        p_gnn=0.2,
        p_fc=0.2,
        pretrained_concept_emb=None,
        freeze_ent_emb=True,
        init_range=0.02,
        ie_dim=200,
        info_exchange=True,
        ie_layer_num=1,
        sep_ie_layers=False,
        layer_id=-1,
    ):
        super().__init__()

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.lmgnn, self.loading_info = LMGNN.from_pretrained(
            model_name,
            output_hidden_states=True,
            output_loading_info=True,
            args=args,
            model_name=model_name,
            k=k,
            n_ntype=n_ntype,
            n_etype=n_etype,
            n_concept=n_concept,
            concept_dim=concept_dim,
            concept_in_dim=concept_in_dim,
            n_attention_head=n_attention_head,
            fc_dim=fc_dim,
            n_fc_layer=n_fc_layer,
            p_emb=p_emb,
            p_gnn=p_gnn,
            p_fc=p_fc,
            pretrained_concept_emb=pretrained_concept_emb,
            freeze_ent_emb=freeze_ent_emb,
            init_range=init_range,
            ie_dim=ie_dim,
            info_exchange=info_exchange,
            ie_layer_num=ie_layer_num,
            sep_ie_layers=sep_ie_layers,
            layer_id=layer_id,
        )

    def batch_graph(
        self, edge_index_init, edge_type_init, pos_triples_init, neg_nodes_init, n_nodes
    ):
        """
        edge_index_init:  list of (n_examples, ). each entry is torch.tensor(2, E?)    ==> [2, total_E]
        edge_type_init:   list of (n_examples, ). each entry is torch.tensor(E?, )     ==> [total_E, ]
        pos_triples_init: list of (n_examples, ). each entry is [h,r,t] where h/r/t: torch.tensor(n_triple?, ) ==> [3, `total_n_triple`]
        neg_nodes_init:   list of (n_examples, ). each entry is torch.tensor(n_triple?, n_neg) ==> [`total_n_triple`, n_neg]
        """
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ *
                      n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]

        pos_triples = [[], [], []]
        for _i_ in range(n_examples):
            h = pos_triples_init[_i_][0] + _i_ * n_nodes  # tensor[n_triple?,]
            r = pos_triples_init[_i_][1]  # tensor[n_triple?,]
            t = pos_triples_init[_i_][2] + _i_ * n_nodes  # tensor[n_triple?,]
            pos_triples[0].append(h)
            pos_triples[1].append(r)
            pos_triples[2].append(t)
        pos_triples = torch.stack([
            torch.cat(item) for item in pos_triples
        ])  # [3, `total_n_triple`] where `total_n_triple` is sum of n_triple within batch
        assert pos_triples.size(0) == 3

        neg_nodes = [neg_nodes_init[_i_] + _i_ *
                     n_nodes for _i_ in range(n_examples)]
        neg_nodes = torch.cat(neg_nodes)  # [`total_n_triple`, n_neg]
        assert neg_nodes.dim() == 2
        assert pos_triples.size(1) == neg_nodes.size(0)
        return edge_index, edge_type, pos_triples, neg_nodes

    def forward(self, *inputs, cache_output=False, detail=False):
        """
        inputs_ids: (batch_size, num_choice, seq_len)    -> (batch_size * num_choice, seq_len)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        node_scores: [bs, nc, n_node, 1]
        adj_lengths: means the "actual" number of nodes (excluding padding)(batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )

        returns:
        logits: [bs, nc]
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        # Here, merge the batch dimension and the num_choice dimension
        assert (
            len(inputs) == 6 + 5 + 2 + 2
        )  # 6 lm_data, 5 gnn_data, (edge_index, edge_type), (pos_triples, neg_nodes)
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = (
            [x.reshape(x.size(0) * x.size(1), *x.size()[2:])
             for x in inputs[:6]]
            + [x.reshape(x.size(0) * x.size(1), *x.size()[2:])
               for x in inputs[6:11]]
            + [sum(x, []) for x in inputs[11:15]]
        )

        (
            *lm_inputs,
            concept_ids,
            node_type_ids,
            node_scores,
            adj_lengths,
            special_nodes_mask,
            edge_index,
            edge_type,
            pos_triples,
            neg_nodes,
        ) = _inputs
        # node_scores = torch.zeros_like(node_scores) #handled in LMGNN forward
        edge_index, edge_type, pos_triples, neg_nodes = self.batch_graph(
            edge_index, edge_type, pos_triples, neg_nodes, concept_ids.size(1)
        )
        device = node_type_ids.device
        adj = (edge_index.to(device), edge_type.to(device))
        lp_data = (pos_triples.to(device), neg_nodes.to(device))

        logits, lm_loss, link_losses = self.lmgnn(
            lm_inputs,
            concept_ids,
            node_type_ids,
            node_scores,
            adj_lengths,
            special_nodes_mask,
            adj,
            lp_data,
            emb_data=None,
            cache_output=cache_output,
        )
        # logits: [bs * nc], lm_loss: scalar, link_losses: (scalar, scalar, scalar)
        if logits is not None:
            logits = logits.view(bs, nc)
        lm_loss = lm_loss * bs
        link_losses = [item * bs for item in link_losses]
        if not detail:
            return logits, lm_loss, link_losses
        else:
            return (
                logits,
                lm_loss,
                link_losses,
                concept_ids.view(bs, nc, -1),
                node_type_ids.view(bs, nc, -1),
                edge_index_orig,
                edge_type_orig,
            )
            # edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            # edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )


def load_state_dict(checkpoint_file: Union[str, os.PathLike], is_quantized: bool = False):
    """
    Reads a PyTorch checkpoint file, returning properly formatted errors if they arise.
    """
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        return safe_load_file(checkpoint_file)
    try:
        if (
            (is_deepspeed_zero3_enabled() and torch.distributed.is_initialized()
             and torch.distributed.get_rank() > 0)
            or (is_fsdp_enabled() and not is_local_dist_rank_0())
        ) and not is_quantized:
            map_location = "meta"
        else:
            map_location = "cpu"
        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
            isinstance(checkpoint_file, str)
            and map_location != "meta"
            and version.parse(torch.__version__) >= version.parse("2.1.0")
            and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        weights_only_kwarg = {
            "weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            **weights_only_kwarg,
            **extra_args,
        )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


class LMGNN(PreTrainedModelClass):
    def __init__(
        self,
        config,
        model_name,
        args,
        k=5,
        n_ntype=4,
        n_etype=38,
        n_concept=799273,
        concept_dim=200,
        concept_in_dim=1024,
        n_attention_head=2,
        fc_dim=200,
        n_fc_layer=0,
        p_emb=0.2,
        p_gnn=0.2,
        p_fc=0.2,
        pretrained_concept_emb=None,
        freeze_ent_emb=True,
        init_range=0.02,
        ie_dim=200,
        info_exchange=True,
        ie_layer_num=1,
        sep_ie_layers=False,
        layer_id=-1,
    ):
        super().__init__(config)
        self.args = args
        self.config = config

        self.init_range = init_range

        self.k = k
        self.concept_dim = concept_dim
        self.n_attention_head = n_attention_head
        self.activation = layers.GELU()
        if k >= 0:
            self.concept_emb = layers.CustomizedEmbedding(
                concept_num=n_concept,
                concept_out_dim=concept_dim,
                use_contextualized=False,
                concept_in_dim=concept_in_dim,
                pretrained_concept_emb=pretrained_concept_emb,
                freeze_ent_emb=freeze_ent_emb,
            )
            self.pooler = layers.MultiheadAttPoolLayer(
                n_attention_head, config.hidden_size, concept_dim
            )

        concat_vec_dim = (
            concept_dim * 2 + config.hidden_size if k >= 0 else config.hidden_size
        )
        self.fc = layers.MLP(
            concat_vec_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True
        )

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

        if INHERIT_BERT:
            self.bert = TextKGMessagePassing(
                config,
                args=args,
                k=k,
                n_ntype=n_ntype,
                n_etype=n_etype,
                dropout=p_gnn,
                concept_dim=concept_dim,
                ie_dim=ie_dim,
                p_fc=p_fc,
                info_exchange=info_exchange,
                ie_layer_num=ie_layer_num,
                sep_ie_layers=sep_ie_layers,
            )  # this is equivalent to BertModel
            if args.mlm_task:
                self.cls = modeling_bert.BertPreTrainingHeads(config)
        else:
            self.deberta = TextKGMessagePassing(
                config,
                args=args,
                k=k,
                n_ntype=n_ntype,
                n_etype=n_etype,
                dropout=p_gnn,
                concept_dim=concept_dim,
                ie_dim=ie_dim,
                p_fc=p_fc,
                info_exchange=info_exchange,
                ie_layer_num=ie_layer_num,
                sep_ie_layers=sep_ie_layers,
            )
            self.pooler_deberta = modeling_deberta_v2.ContextPooler(self.deberta.config)

            if args.mlm_task:
                self.lm_head = modeling_deberta_v2.DebertaV2OnlyMLMHead(config)
            # self.roberta = TextKGMessagePassing(
            #     config,
            #     args=args,
            #     k=k,
            #     n_ntype=n_ntype,
            #     n_etype=n_etype,
            #     dropout=p_gnn,
            #     concept_dim=concept_dim,
            #     ie_dim=ie_dim,
            #     p_fc=p_fc,
            #     info_exchange=info_exchange,
            #     ie_layer_num=ie_layer_num,
            #     sep_ie_layers=sep_ie_layers,
            # )  # this is equivalent to RobertaModel
            # if args.mlm_task:
            #     self.lm_head = modeling_roberta.RobertaLMHead(config)

        self.layer_id = layer_id
        self.cpnet_vocab_size = n_concept

        if args.link_task:
            if args.link_decoder == 'DistMult':
                self.linkpred = modeling_gnn.DistMultDecoder(
                    args, num_rels=n_etype, h_dim=concept_dim
                )
            elif args.link_decoder == 'TransE':
                self.linkpred = modeling_gnn.TransEDecoder(
                    args, num_rels=n_etype, h_dim=concept_dim
                )
            elif args.link_decoder == 'RotatE':
                self.linkpred = modeling_gnn.RotatEDecoder(
                    args, num_rels=n_etype, h_dim=concept_dim
                )
            else:
                raise NotImplementedError
            if args.link_proj_headtail:
                self.linkpred_proj = nn.Linear(concept_dim, concept_dim)
            if args.link_normalize_headtail == 3:
                self.emb_LayerNorm = nn.LayerNorm(concept_dim)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        inputs,
        concept_ids,
        node_type_ids,
        node_scores,
        adj_lengths,
        special_nodes_mask,
        adj,
        lp_data,
        cache_output=False,
        emb_data=None,
    ):
        """
        concept_ids: (batch_size, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)
        adj: edge_index, edge_type
        lp_data: pos_triples, neg_nodes

        returns:
        logits: [bs]
        """
        # LM inputs
        (
            lm_input_ids,
            lm_labels,
            input_ids,
            attention_mask,
            token_type_ids,
            output_mask,
        ) = inputs
        if self.args.mlm_task:
            input_ids = lm_input_ids

        # GNN inputs
        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2
        if self.k >= 0:
            gnn_input = self.concept_emb(concept_ids - 1, emb_data).to(
                node_type_ids.device
            )
        else:
            gnn_input = (
                torch.zeros((
                    concept_ids.size(0),
                    concept_ids.size(1),
                    self.concept_dim,
                ))
                .float()
                .to(node_type_ids.device)
            )
        gnn_input[:, 0] = 0
        gnn_input = self.dropout_e(gnn_input)  # (batch_size, n_node, dim_node)

        # Normalize node sore (use norm from Z)
        if self.args.no_node_score:
            node_scores = node_scores.new_zeros(node_scores.size())
        else:
            _mask = (
                torch.arange(node_scores.size(1), device=node_scores.device)
                < adj_lengths.unsqueeze(1)
            ).float()  # 0 means masked out #[batch_size, n_node]
            node_scores = -node_scores
            node_scores = (
                node_scores - node_scores[:, 0:1, :]
            )  # [batch_size, n_node, 1]
            node_scores = node_scores.squeeze(2)  # [batch_size, n_node]
            node_scores = node_scores * _mask
            mean_norm = (torch.abs(node_scores)).sum(
                dim=1
            ) / adj_lengths  # [batch_size, ]
            node_scores = node_scores / (
                mean_norm.unsqueeze(1) + 1e-05
            )  # [batch_size, n_node]
            node_scores = node_scores.unsqueeze(2)  # [batch_size, n_node, 1]

        if INHERIT_BERT:
            bert_or_roberta = self.bert
        else:
            bert_or_roberta = self.deberta
            # bert_or_roberta = self.roberta
        # Merged core
        lm_outputs, gnn_output = bert_or_roberta(
            input_ids,
            token_type_ids,
            attention_mask,
            output_mask,
            gnn_input,
            adj,
            node_type_ids,
            node_scores,
            special_nodes_mask,
            output_hidden_states=True,
        )
        # lm_outputs: ([bs, seq_len, sent_dim], ([bs, seq_len, sent_dim] for _ in range(25)))

        #### lm_outputs: ([bs, seq_len, sent_dim], [bs, sent_dim], ([bs, seq_len, sent_dim] for _ in range(25)))
        # gnn_output: [bs, n_node, dim_node]

        # LM outputs
        all_hidden_states = lm_outputs[
            -1
        ]  # ([bs, seq_len, sent_dim] for _ in range(25))
        # [bs, seq_len, sent_dim]
        lm_hidden_states = all_hidden_states[self.layer_id]
        sent_vecs = self.pooler_deberta(lm_hidden_states)  # [bs, sent_dim]

        # sent_token_mask = output_mask.clone()
        # sent_token_mask[:, 0] = 0

        _bs, _seq_len, _ = lm_hidden_states.size()
        if self.args.mlm_task:
            loss_fct = nn.CrossEntropyLoss()
            if INHERIT_BERT:
                prediction_scores, _seq_relationship_score = self.cls(
                    lm_hidden_states, sent_vecs
                )
                masked_lm_loss = loss_fct(
                    prediction_scores.view(
                        _bs * _seq_len, -1), lm_labels.view(-1)
                )
                # loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                next_sentence_loss = 0.0
                lm_loss = masked_lm_loss + next_sentence_loss
            else:
                prediction_scores = self.lm_head(lm_hidden_states)
                lm_loss = loss_fct(
                    prediction_scores.view(
                        _bs * _seq_len, -1), lm_labels.view(-1)
                )
        else:
            lm_loss = 0.0

        # GNN outputs
        Z_vecs = gnn_output[:, 0]  # (batch_size, dim_node)

        node_mask = torch.arange(
            node_type_ids.size(1), device=node_type_ids.device
        ) >= adj_lengths.unsqueeze(1)  # [bs, nodes] 1 means masked out
        gnn_output = gnn_output * (~node_mask).float().unsqueeze(2)
        node_mask = node_mask | (
            node_type_ids == 3
        )  # pool over all KG nodes (excluding the context node)
        # a temporary solution to avoid zero node
        node_mask[node_mask.all(1), 0] = 0

        # sent_node_mask = special_nodes_mask.clone()
        # sent_node_mask[:, 0] = 0

        if self.args.link_task:
            pos_triples, neg_nodes = (
                # pos_triples: [3, `total_n_triple`],  neg_nodes: [`total_n_triple`, n_neg]
                lp_data
            )

            pos_samples = pos_triples  # [3, `total_n_triple`]

            _n_neg = neg_nodes.size(1)
            head_negative_sample = neg_nodes[
                :, : _n_neg // 2
            ]  # [`total_n_triple`, n_neg//2]
            tail_negative_sample = neg_nodes[
                :, _n_neg // 2: _n_neg // 2 * 2
            ]  # [`total_n_triple`, n_neg//2]

            _bs, _, gnn_dim = gnn_output.size()
            embs = gnn_output.view(-1, gnn_dim)  # [`total_n_nodes`, gnn_dim]

            if self.args.link_proj_headtail:
                embs = self.linkpred_proj(embs)
            if self.args.link_normalize_headtail == 1:
                embs = embs / torch.norm(embs, p=2,
                                         dim=1, keepdim=True).detach()
            elif self.args.link_normalize_headtail == 2:
                embs = torch.tanh(embs)
            elif self.args.link_normalize_headtail == 3:
                embs = self.emb_LayerNorm(embs)

            positive_score = self.linkpred(
                embs, pos_samples)  # [`total_n_triple`, 1]
            head_neg_scores = self.linkpred(
                embs, (pos_samples, head_negative_sample), mode='head-batch'
            )
            tail_neg_scores = self.linkpred(
                embs, (pos_samples, tail_negative_sample), mode='tail-batch'
            )
            negative_score = torch.cat(
                [head_neg_scores, tail_neg_scores], dim=-1
            )  # [`total_n_triple`, total_n_neg]
            scores = (positive_score, negative_score)

            link_loss, pos_link_loss, neg_link_loss = self.linkpred.loss(
                scores)
        else:
            link_loss = pos_link_loss = neg_link_loss = 0.0

        # Concatenated pool
        if self.args.end_task:
            sent_vecs_for_pooler = sent_vecs
            if self.k >= 0:
                graph_vecs, _ = self.pooler(
                    sent_vecs_for_pooler, gnn_output, node_mask
                )  # graph_vecs: [bs, node_dim]
                concat_pool = torch.cat((graph_vecs, sent_vecs, Z_vecs), 1)
            else:
                concat_pool = sent_vecs
            logits = self.fc(self.dropout_fc(concat_pool))  # [bs, 1]
        else:
            logits = None

        return logits, lm_loss, (link_loss, pos_link_loss, neg_link_loss)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:
              - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
              - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
              - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
              - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
              - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) one of:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                    - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                    - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                    - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            # For example purposes. Not runnable.
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        offload_buffers = kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        gguf_file = kwargs.pop("gguf_file", None)
        # Cache path to the GGUF file
        gguf_path = None

        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError(
                    "Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:

            model_kwargs = kwargs

        user_agent = {"file_type": "model", "framework": "pytorch",
                      "from_auto_class": from_auto_class}

        if pretrained_model_name_or_path is not None and gguf_file is None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, TF_WEIGHTS_NAME + ".index")
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, FLAX_WEIGHTS_NAME)
                ):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder,
                                 _add_variant(SAFE_WEIGHTS_NAME, variant))
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            SAFE_WEIGHTS_NAME, variant)
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            WEIGHTS_NAME, variant)
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder,
                                 _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif not use_safetensors and (
                    os.path.isfile(os.path.join(
                        pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"))
                    or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME))
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                        " `from_tf=True` to load this model from those weights."
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, FLAX_WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                        " to load this model from those weights."
                    )
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = os.path.join(
                    subfolder, pretrained_model_name_or_path + ".index")
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(
                    pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            if revision == "main":
                                resolved_archive_file, revision, is_sharded = auto_conversion(
                                    pretrained_model_name_or_path, **cached_file_kwargs
                                )
                            cached_file_kwargs["revision"] = revision
                            if resolved_archive_file is None:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                    "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                    "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                                )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, filename, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if not local_files_only and not is_offline_mode():
                        if resolved_archive_file is not None:
                            if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                                # If the PyTorch file was found, check if there is a safetensors file on the repository
                                # If there is no safetensors file on the repositories, start an auto conversion
                                safe_weights_name = SAFE_WEIGHTS_NAME
                                has_file_kwargs = {
                                    "revision": revision,
                                    "proxies": proxies,
                                    "token": token,
                                    "cache_dir": cache_dir,
                                    "local_files_only": local_files_only,
                                }
                                cached_file_kwargs = {
                                    "cache_dir": cache_dir,
                                    "force_download": force_download,
                                    "resume_download": resume_download,
                                    "local_files_only": local_files_only,
                                    "user_agent": user_agent,
                                    "subfolder": subfolder,
                                    "_raise_exceptions_for_gated_repo": False,
                                    "_raise_exceptions_for_missing_entries": False,
                                    "_commit_hash": commit_hash,
                                    **has_file_kwargs,
                                }
                                if not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs):
                                    Thread(
                                        target=auto_conversion,
                                        args=(pretrained_model_name_or_path,),
                                        kwargs={
                                            "ignore_errors_during_conversion": True, **cached_file_kwargs},
                                        name="Thread-autoconversion",
                                    ).start()
                        else:
                            # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
                            # We try those to give a helpful error message.
                            has_file_kwargs = {
                                "revision": revision,
                                "proxies": proxies,
                                "token": token,
                                "cache_dir": cache_dir,
                                "local_files_only": local_files_only,
                            }
                            if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                                    " Use `from_tf=True` to load this model from those weights."
                                )
                            elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                                    " `from_flax=True` to load this model from those weights."
                                )
                            elif variant is not None and has_file(
                                pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                            ):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                    f" {variant}. Use `variant=None` to load this model from those weights."
                                )
                            else:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                                )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(
                    f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.

        if (
            is_safetensors_available()
            and isinstance(resolved_archive_file, str)
            and resolved_archive_file.endswith(".safetensors")
        ):
            with safe_open(resolved_archive_file, framework="pt") as f:
                metadata = f.metadata()

            if metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info(
                    "A TensorFlow safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info(
                    "A Flax safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)
        hf_quantizer = None
        is_quantized = hf_quantizer is not None

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None

            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                            torch_dtype = config.torch_dtype
                            logger.info(
                                f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                        else:
                            if is_sharded and "dtype" in sharded_metadata:
                                torch_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                torch_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(
                                    resolved_archive_file[0])
                                torch_dtype = get_state_dict_dtype(
                                    one_state_dict)
                                del one_state_dict  # free CPU memory
                            logger.info(
                                "Since the `torch_dtype` attribute can't be found in model's config object, "
                                "will use torch_dtype={torch_dtype} as derived from model's weights"
                            )
                    elif hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                    else:
                        raise ValueError(
                            f'`torch_dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`, but received {torch_dtype}'
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
                (torch_dtype == torch.float16) or hasattr(
                    hf_quantizer, "use_keep_in_fp32_modules")
            )

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = list(state_dict.keys())

            if gguf_path is None and (low_cpu_mem_usage or (use_keep_in_fp32_modules and is_accelerate_available())):
                # In case some weights need to be kept in float32 and accelerate is not installed,
                # we later on want to take the path where state_dict is not None, that is the one
                # that do not require accelerate.
                state_dict = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            logger.info(
                "Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            init_contexts = [deepspeed.zero.Init(
                config_dict_or_path=deepspeed_config())] + init_contexts
        elif low_cpu_mem_usage:
            init_contexts.append(init_empty_weights())

        # We do not want to modify the config inplace in from_pretrained.
        config = copy.deepcopy(config)
        config = cls._autoset_attn_implementation(
            config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
        )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Check first if we are `from_pt`
        if use_keep_in_fp32_modules:
            if is_accelerate_available() and not is_deepspeed_zero3_enabled():
                low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
            )

            # We store the original dtype for quantized models as we cannot easily retrieve it
            # once the weights have been quantized
            # Note that once you have loaded a quantized model, you can't change its dtype so this will
            # remain a single source of truth
            config._pre_quantization_dtype = torch_dtype

        if isinstance(device_map, str):
            special_dtypes = {}

            if hf_quantizer is not None:
                special_dtypes.update(
                    hf_quantizer.get_special_dtypes_update(model, torch_dtype))

            special_dtypes.update(
                {
                    name: torch.float32
                    for name, _ in model.named_parameters()
                    if any(m in name for m in keep_in_fp32_modules)
                }
            )

            target_dtype = torch_dtype

            if hf_quantizer is not None:
                target_dtype = hf_quantizer.adjust_target_dtype(target_dtype)

            no_split_modules = model._get_no_split_modules(device_map)
            if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )

            device_map_kwargs = {"no_split_module_classes": no_split_modules}
            if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
                device_map_kwargs["special_dtypes"] = special_dtypes
            elif len(special_dtypes) > 0:
                logger.warning(
                    "This model has some weights that should be kept in higher precision, you need to upgrade "
                    "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
                )
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=max_memory,
                    **device_map_kwargs,
                )
            else:
                max_memory = get_max_memory(max_memory)
            if hf_quantizer is not None:
                max_memory = hf_quantizer.adjust_max_memory(max_memory)
            device_map_kwargs["max_memory"] = max_memory

            # Make sure tied weights are tied before creating the device map.
            model.tie_weights()
            device_map = infer_auto_device_map(
                model, dtype=target_dtype, **device_map_kwargs)

            if hf_quantizer is not None:
                hf_quantizer.validate_environment(device_map=device_map)

        elif device_map is not None:
            model.tie_weights()
            tied_params = find_tied_parameters(model)
            # check if we don't have tied param in different devices
            check_tied_parameters_on_same_device(tied_params, device_map)

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(
                    model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers.modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model, loading_info = load_tf2_checkpoint_in_pytorch_model(
                        model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True
                    )
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
                        " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
                        " instructions."
                    )
                    raise
        elif from_flax:
            try:
                from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(
                    model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
                    " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
                    " installation instructions."
                )
                raise
        elif from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                loaded_state_dict_keys,  # XXX: rename?
                resolved_archive_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                _fast_init=_fast_init,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=torch_dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_modules=keep_in_fp32_modules,
                gguf_path=gguf_path,
            )

        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if device_map is not None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
                "offload_buffers": offload_buffers,
            }
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
            # For HQQ method we force-set the hooks for single GPU envs
            if (
                "force_hooks" in inspect.signature(dispatch_model).parameters
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
            ):
                device_map_kwargs["force_hooks"] = True
            if (
                hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                device_map_kwargs["offload_buffers"] = True

            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if output_loading_info:
            if loading_info is None:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            return model, loading_info

        return model


if INHERIT_BERT:
    ModelClass = modeling_bert.BertModel
else:
    ModelClass = modeling_deberta_v2.DebertaV2Model
    # ModelClass = modeling_roberta.RobertaModel


class TextKGMessagePassing(ModelClass):
    def __init__(
        self,
        config,
        args,
        k=5,
        n_ntype=4,
        n_etype=38,
        dropout=0.2,
        concept_dim=200,
        ie_dim=200,
        p_fc=0.2,
        info_exchange=True,
        ie_layer_num=1,
        sep_ie_layers=False,
    ):
        super().__init__(config=config)

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.hidden_size = concept_dim
        self.emb_node_type = nn.Linear(self.n_ntype, concept_dim // 2)

        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, concept_dim // 2)
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

        self.k = k

        self.Vh = nn.Linear(concept_dim, concept_dim)
        self.Vx = nn.Linear(concept_dim, concept_dim)

        self.activation = layers.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.encoder = DeBERTav2GAT(  # RoBERTaGAT
            config,
            args,
            k=k,
            n_ntype=n_ntype,
            n_etype=n_etype,
            hidden_size=concept_dim,
            dropout=dropout,
            concept_dim=concept_dim,
            ie_dim=ie_dim,
            p_fc=p_fc,
            info_exchange=info_exchange,
            ie_layer_num=ie_layer_num,
            sep_ie_layers=sep_ie_layers,
        )

        self.sent_dim = config.hidden_size

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        special_tokens_mask,
        H,
        A,
        node_type,
        node_score,
        special_nodes_mask,
        cache_output=False,
        position_ids=None,
        head_mask=None,
        output_hidden_states=True,
    ):
        """
        input_ids: [bs, seq_len]
        token_type_ids: [bs, seq_len]
        attention_mask: [bs, seq_len]
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
            edge_index: [2, n_edges]
            edge_type: [n_edges]
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        # LM inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 1D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        if len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError('Attnetion mask should be either 1D or 2D.')

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extended_attention_mask [20,1,1,100]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(
                        0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        # embedding_output (20,100,768) 20:batch_size, 100:seq_len, 1024:embed_size
        # GNN inputs
        _batch_size, _n_nodes = node_type.size()

        # Embed type
        T = modeling_gnn.make_one_hot(
            node_type.view(-1).contiguous(), self.n_ntype
        ).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(
            self.emb_node_type(T)
        )  # [batch_size, n_node, dim/2]

        # Embed score
        if self.basis_f == 'sin':
            js = (
                torch.arange(self.hidden_size // 2)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(node_type.device)
            )  # [1,1,dim/2]
            js = torch.pow(1.1, js)  # [1,1,dim/2]
            B = torch.sin(js * node_score)  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(
                self.emb_score(B)
            )  # [batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(
                self.emb_score(B)
            )  # [batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            # [batch_size, n_node, dim/2]
            B = self.activation(self.B_lin(node_score))
            node_score_emb = self.activation(
                self.emb_score(B)
            )  # [batch_size, n_node, dim/2]

        X = H
        edge_index, edge_type = (
            # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
            A
        )
        _X = (
            X.view(-1, X.size(2)).contiguous()
        )  # [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous()  # [`total_n_nodes`, ]
        _node_feature_extra = (
            torch.cat([node_type_emb, node_score_emb], dim=2)
            .view(_node_type.size(0), -1)
            .contiguous()
        )  # [`total_n_nodes`, dim]
        # _X = torch.zeros(node_type.size(0), self.args.max_num_nodes, self.hidden_size,
        #                  device=node_type.device)
        # Merged core
        encoder_outputs, _X = self.encoder(
            embedding_output,
            extended_attention_mask,
            special_tokens_mask,
            head_mask,
            _X,
            edge_index,
            edge_type,
            _node_type,
            _node_feature_extra,
            special_nodes_mask,
            output_hidden_states=output_hidden_states,
        )

        encoded_layers = encoder_outputs[1]
        sequence_output = encoded_layers[-1]

        outputs = (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]
        # LM outputs
        # sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)

        # outputs = (
        #     sequence_output,
        #     pooled_output,
        # ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        # GNN outputs
        X = _X.view(
            node_type.size(0), node_type.size(1), -1
        )  # [batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return outputs, output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:
              - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
              - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
              - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
              - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
              - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) one of:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                    - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                    - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                    - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            # For example purposes. Not runnable.
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        offload_buffers = kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        gguf_file = kwargs.pop("gguf_file", None)
        # Cache path to the GGUF file
        gguf_path = None

        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError(
                    "Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:

            model_kwargs = kwargs

        user_agent = {"file_type": "model", "framework": "pytorch",
                      "from_auto_class": from_auto_class}

        if pretrained_model_name_or_path is not None and gguf_file is None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, TF_WEIGHTS_NAME + ".index")
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, FLAX_WEIGHTS_NAME)
                ):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder,
                                 _add_variant(SAFE_WEIGHTS_NAME, variant))
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            SAFE_WEIGHTS_NAME, variant)
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            WEIGHTS_NAME, variant)
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder,
                                 _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(
                            WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif not use_safetensors and (
                    os.path.isfile(os.path.join(
                        pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"))
                    or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME))
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                        " `from_tf=True` to load this model from those weights."
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path,
                                 subfolder, FLAX_WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                        " to load this model from those weights."
                    )
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = os.path.join(
                    subfolder, pretrained_model_name_or_path + ".index")
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(
                    pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            if revision == "main":
                                resolved_archive_file, revision, is_sharded = auto_conversion(
                                    pretrained_model_name_or_path, **cached_file_kwargs
                                )
                            cached_file_kwargs["revision"] = revision
                            if resolved_archive_file is None:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                    "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                    "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                                )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, filename, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if not local_files_only and not is_offline_mode():
                        if resolved_archive_file is not None:
                            if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                                # If the PyTorch file was found, check if there is a safetensors file on the repository
                                # If there is no safetensors file on the repositories, start an auto conversion
                                safe_weights_name = SAFE_WEIGHTS_NAME
                                has_file_kwargs = {
                                    "revision": revision,
                                    "proxies": proxies,
                                    "token": token,
                                    "cache_dir": cache_dir,
                                    "local_files_only": local_files_only,
                                }
                                cached_file_kwargs = {
                                    "cache_dir": cache_dir,
                                    "force_download": force_download,
                                    "resume_download": resume_download,
                                    "local_files_only": local_files_only,
                                    "user_agent": user_agent,
                                    "subfolder": subfolder,
                                    "_raise_exceptions_for_gated_repo": False,
                                    "_raise_exceptions_for_missing_entries": False,
                                    "_commit_hash": commit_hash,
                                    **has_file_kwargs,
                                }
                                if not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs):
                                    Thread(
                                        target=auto_conversion,
                                        args=(pretrained_model_name_or_path,),
                                        kwargs={
                                            "ignore_errors_during_conversion": True, **cached_file_kwargs},
                                        name="Thread-autoconversion",
                                    ).start()
                        else:
                            # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
                            # We try those to give a helpful error message.
                            has_file_kwargs = {
                                "revision": revision,
                                "proxies": proxies,
                                "token": token,
                                "cache_dir": cache_dir,
                                "local_files_only": local_files_only,
                            }
                            if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                                    " Use `from_tf=True` to load this model from those weights."
                                )
                            elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                                    " `from_flax=True` to load this model from those weights."
                                )
                            elif variant is not None and has_file(
                                pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                            ):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                    f" {variant}. Use `variant=None` to load this model from those weights."
                                )
                            else:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                                )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(
                    f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.

        if (
            is_safetensors_available()
            and isinstance(resolved_archive_file, str)
            and resolved_archive_file.endswith(".safetensors")
        ):
            with safe_open(resolved_archive_file, framework="pt") as f:
                metadata = f.metadata()

            if metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info(
                    "A TensorFlow safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info(
                    "A Flax safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)
        hf_quantizer = None
        is_quantized = hf_quantizer is not None

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None

            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                            torch_dtype = config.torch_dtype
                            logger.info(
                                f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                        else:
                            if is_sharded and "dtype" in sharded_metadata:
                                torch_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                torch_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(
                                    resolved_archive_file[0])
                                torch_dtype = get_state_dict_dtype(
                                    one_state_dict)
                                del one_state_dict  # free CPU memory
                            logger.info(
                                "Since the `torch_dtype` attribute can't be found in model's config object, "
                                "will use torch_dtype={torch_dtype} as derived from model's weights"
                            )
                    elif hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                    else:
                        raise ValueError(
                            f'`torch_dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`, but received {torch_dtype}'
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
                (torch_dtype == torch.float16) or hasattr(
                    hf_quantizer, "use_keep_in_fp32_modules")
            )

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = list(state_dict.keys())

            if gguf_path is None and (low_cpu_mem_usage or (use_keep_in_fp32_modules and is_accelerate_available())):
                # In case some weights need to be kept in float32 and accelerate is not installed,
                # we later on want to take the path where state_dict is not None, that is the one
                # that do not require accelerate.
                state_dict = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            logger.info(
                "Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            init_contexts = [deepspeed.zero.Init(
                config_dict_or_path=deepspeed_config())] + init_contexts
        elif low_cpu_mem_usage:
            init_contexts.append(init_empty_weights())

        # We do not want to modify the config inplace in from_pretrained.
        config = copy.deepcopy(config)
        config = cls._autoset_attn_implementation(
            config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
        )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Check first if we are `from_pt`
        if use_keep_in_fp32_modules:
            if is_accelerate_available() and not is_deepspeed_zero3_enabled():
                low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
            )

            # We store the original dtype for quantized models as we cannot easily retrieve it
            # once the weights have been quantized
            # Note that once you have loaded a quantized model, you can't change its dtype so this will
            # remain a single source of truth
            config._pre_quantization_dtype = torch_dtype

        if isinstance(device_map, str):
            special_dtypes = {}

            if hf_quantizer is not None:
                special_dtypes.update(
                    hf_quantizer.get_special_dtypes_update(model, torch_dtype))

            special_dtypes.update(
                {
                    name: torch.float32
                    for name, _ in model.named_parameters()
                    if any(m in name for m in keep_in_fp32_modules)
                }
            )

            target_dtype = torch_dtype

            if hf_quantizer is not None:
                target_dtype = hf_quantizer.adjust_target_dtype(target_dtype)

            no_split_modules = model._get_no_split_modules(device_map)
            if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )

            device_map_kwargs = {"no_split_module_classes": no_split_modules}
            if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
                device_map_kwargs["special_dtypes"] = special_dtypes
            elif len(special_dtypes) > 0:
                logger.warning(
                    "This model has some weights that should be kept in higher precision, you need to upgrade "
                    "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
                )
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=max_memory,
                    **device_map_kwargs,
                )
            else:
                max_memory = get_max_memory(max_memory)
            if hf_quantizer is not None:
                max_memory = hf_quantizer.adjust_max_memory(max_memory)
            device_map_kwargs["max_memory"] = max_memory

            # Make sure tied weights are tied before creating the device map.
            model.tie_weights()
            device_map = infer_auto_device_map(
                model, dtype=target_dtype, **device_map_kwargs)

            if hf_quantizer is not None:
                hf_quantizer.validate_environment(device_map=device_map)

        elif device_map is not None:
            model.tie_weights()
            tied_params = find_tied_parameters(model)
            # check if we don't have tied param in different devices
            check_tied_parameters_on_same_device(tied_params, device_map)

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(
                    model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers.modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model, loading_info = load_tf2_checkpoint_in_pytorch_model(
                        model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True
                    )
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
                        " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
                        " instructions."
                    )
                    raise
        elif from_flax:
            try:
                from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(
                    model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
                    " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
                    " installation instructions."
                )
                raise
        elif from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                loaded_state_dict_keys,  # XXX: rename?
                resolved_archive_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                _fast_init=_fast_init,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=torch_dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_modules=keep_in_fp32_modules,
                gguf_path=gguf_path,
            )

        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if device_map is not None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
                "offload_buffers": offload_buffers,
            }
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
            # For HQQ method we force-set the hooks for single GPU envs
            if (
                "force_hooks" in inspect.signature(dispatch_model).parameters
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
            ):
                device_map_kwargs["force_hooks"] = True
            if (
                hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                device_map_kwargs["offload_buffers"] = True

            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if output_loading_info:
            if loading_info is None:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            return model, loading_info

        return model


class DeBERTav2GAT(modeling_deberta_v2.DebertaV2Encoder):
    def __init__(
        self,
        config,
        args,
        k=5,
        n_ntype=4,
        n_etype=38,
        hidden_size=200,
        dropout=0.2,
        concept_dim=200,
        ie_dim=200,
        p_fc=0.2,
        info_exchange=True,
        ie_layer_num=1,
        sep_ie_layers=False,
    ):
        super().__init__(config)

        self.args = args
        self.k = k
        self.concept_dim = concept_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.info_exchange = info_exchange

        if k >= 1:
            self.edge_encoder = torch.nn.Sequential(
                torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
            )
            self.gnn_layers = nn.ModuleList([
                modeling_gnn.GATConvE(
                    args, hidden_size, n_ntype, n_etype, self.edge_encoder
                )
                for _ in range(k)
            ])
            self.activation = layers.GELU()
            self.dropout_rate = dropout

            self.sent_dim = config.hidden_size
            self.sep_ie_layers = sep_ie_layers
            if sep_ie_layers:
                self.ie_layers = nn.ModuleList([
                    layers.MLP(
                        self.sent_dim + concept_dim,
                        ie_dim,
                        self.sent_dim + concept_dim,
                        ie_layer_num,
                        p_fc,
                    )
                    for _ in range(k)
                ])
            else:
                self.ie_layer = layers.MLP(
                    self.sent_dim + concept_dim,
                    ie_dim,
                    self.sent_dim + concept_dim,
                    ie_layer_num,
                    p_fc,
                )
            if self.args.residual_ie == 2:
                self.ie_LayerNorm = nn.LayerNorm(self.sent_dim + concept_dim)

    def forward(
        self,
        hidden_states,
        attention_mask,
        special_tokens_mask,
        head_mask,
        _X,
        edge_index,
        edge_type,
        _node_type,
        _node_feature_extra,
        special_nodes_mask,
        output_attentions=False,
        output_hidden_states=True,
        query_states=None,
        relative_pos=None,

    ):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, 1, 1, seq_len]
        head_mask: list of shape [num_hidden_layers]

        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """

        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = attention_mask.sum(-2) > 0
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv

        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()

        # if _X is None:  # Add this check
        #     _X = torch.zeros(hidden_states.size(
        #         0), self.args.max_num_nodes, self.hidden_size, device=hidden_states.device)

        for i, layer_module in enumerate(self.layer):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                output_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                _X = self.gnn_layers[gnn_layer_index](
                    _X, edge_index, edge_type, _node_type, _node_feature_extra
                )
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training=self.training)

                # Exchange info between LM and GNN hidden states (Modality interaction)
                if self.info_exchange or (
                    self.info_exchange == 'every-other-layer'
                    and (i - self.num_hidden_layers + self.k) % 2 == 0
                ):
                    # [bs, max_num_nodes, node_dim]
                    X = _X.view(bs, -1, _X.size(1))
                    # [bs, sent_dim]
                    context_node_lm_feats = hidden_states[:, 0, :]
                    context_node_gnn_feats = X[:, 0, :]  # [bs, node_dim]
                    context_node_feats = torch.cat(
                        [context_node_lm_feats, context_node_gnn_feats], dim=1
                    )
                    if self.sep_ie_layers:
                        _context_node_feats = self.ie_layers[gnn_layer_index](
                            context_node_feats
                        )
                    else:
                        _context_node_feats = self.ie_layer(context_node_feats)
                    if self.args.residual_ie == 1:
                        context_node_feats = context_node_feats + _context_node_feats
                    elif self.args.residual_ie == 2:
                        context_node_feats = self.ie_LayerNorm(
                            context_node_feats + _context_node_feats
                        )
                    else:
                        context_node_feats = _context_node_feats
                    context_node_lm_feats, context_node_gnn_feats = torch.split(
                        context_node_feats,
                        [context_node_lm_feats.size(
                            1), context_node_gnn_feats.size(1)],
                        dim=1,
                    )
                    hidden_states[:, 0, :] = context_node_lm_feats
                    X[:, 0, :] = context_node_gnn_feats
                    _X = X.view_as(_X)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return (
            outputs,
            _X,
        )  # last-layer hidden state, (all hidden states), (all attentions)

# class RoBERTaGAT(BertEncoder):
#     def __init__(
#         self,
#         config,
#         args,
#         k=5,
#         n_ntype=4,
#         n_etype=38,
#         hidden_size=200,
#         dropout=0.2,
#         concept_dim=200,
#         ie_dim=200,
#         p_fc=0.2,
#         info_exchange=True,
#         ie_layer_num=1,
#         sep_ie_layers=False,
#     ):
#         super().__init__(config, args)

#         self.args = args
#         self.k = k
#         self.concept_dim = concept_dim
#         self.num_hidden_layers = config.num_hidden_layers
#         self.info_exchange = info_exchange
#         if k >= 1:
#             self.edge_encoder = torch.nn.Sequential(
#                 torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size),
#                 torch.nn.BatchNorm1d(hidden_size),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(hidden_size, hidden_size),
#             )
#             self.gnn_layers = nn.ModuleList([
#                 modeling_gnn.GATConvE(
#                     args, hidden_size, n_ntype, n_etype, self.edge_encoder
#                 )
#                 for _ in range(k)
#             ])
#             self.activation = layers.GELU()
#             self.dropout_rate = dropout

#             self.sent_dim = config.hidden_size
#             self.sep_ie_layers = sep_ie_layers
#             if sep_ie_layers:
#                 self.ie_layers = nn.ModuleList([
#                     layers.MLP(
#                         self.sent_dim + concept_dim,
#                         ie_dim,
#                         self.sent_dim + concept_dim,
#                         ie_layer_num,
#                         p_fc,
#                     )
#                     for _ in range(k)
#                 ])
#             else:
#                 self.ie_layer = layers.MLP(
#                     self.sent_dim + concept_dim,
#                     ie_dim,
#                     self.sent_dim + concept_dim,
#                     ie_layer_num,
#                     p_fc,
#                 )
#             if self.args.residual_ie == 2:
#                 self.ie_LayerNorm = nn.LayerNorm(self.sent_dim + concept_dim)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask,
#         special_tokens_mask,
#         head_mask,
#         _X,
#         edge_index,
#         edge_type,
#         _node_type,
#         _node_feature_extra,
#         special_nodes_mask,
#         output_attentions=False,
#         output_hidden_states=True,
#     ):
#         """
#         hidden_states: [bs, seq_len, sent_dim]
#         attention_mask: [bs, 1, 1, seq_len]
#         head_mask: list of shape [num_hidden_layers]

#         _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
#         edge_index: [2, n_edges]
#         edge_type: [n_edges]
#         _node_type: [bs * n_nodes]
#         _node_feature_extra: [bs * n_nodes, node_dim]
#         """
#         bs = hidden_states.size(0)
#         all_hidden_states = ()
#         all_attentions = ()
#         for i, layer_module in enumerate(self.layer):
#             # LM
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
#             hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#             if i >= self.num_hidden_layers - self.k:
#                 # GNN
#                 gnn_layer_index = i - self.num_hidden_layers + self.k
#                 _X = self.gnn_layers[gnn_layer_index](
#                     _X, edge_index, edge_type, _node_type, _node_feature_extra
#                 )
#                 _X = self.activation(_X)
#                 _X = F.dropout(_X, self.dropout_rate, training=self.training)

#                 # Exchange info between LM and GNN hidden states (Modality interaction)
#                 if self.info_exchange or (
#                     self.info_exchange == 'every-other-layer'
#                     and (i - self.num_hidden_layers + self.k) % 2 == 0
#                 ):
#                     X = _X.view(bs, -1, _X.size(1))  # [bs, max_num_nodes, node_dim]
#                     context_node_lm_feats = hidden_states[:, 0, :]  # [bs, sent_dim]
#                     context_node_gnn_feats = X[:, 0, :]  # [bs, node_dim]
#                     context_node_feats = torch.cat(
#                         [context_node_lm_feats, context_node_gnn_feats], dim=1
#                     )
#                     if self.sep_ie_layers:
#                         _context_node_feats = self.ie_layers[gnn_layer_index](
#                             context_node_feats
#                         )
#                     else:
#                         _context_node_feats = self.ie_layer(context_node_feats)
#                     if self.args.residual_ie == 1:
#                         context_node_feats = context_node_feats + _context_node_feats
#                     elif self.args.residual_ie == 2:
#                         context_node_feats = self.ie_LayerNorm(
#                             context_node_feats + _context_node_feats
#                         )
#                     else:
#                         context_node_feats = _context_node_feats
#                     context_node_lm_feats, context_node_gnn_feats = torch.split(
#                         context_node_feats,
#                         [context_node_lm_feats.size(1), context_node_gnn_feats.size(1)],
#                         dim=1,
#                     )
#                     hidden_states[:, 0, :] = context_node_lm_feats
#                     X[:, 0, :] = context_node_gnn_feats
#                     _X = X.view_as(_X)

#         # Add last layer
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         outputs = (hidden_states,)
#         if output_hidden_states:
#             outputs = outputs + (all_hidden_states,)
#         if output_attentions:
#             outputs = outputs + (all_attentions,)
#         return (
#             outputs,
#             _X,
#         )  # last-layer hidden state, (all hidden states), (all attentions)
