import logging
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    # cached_path,
    # hf_bucket_url,
    is_remote_url,
)

from transformers.utils.hub import (
    cached_file
)
# from transformers.file_utils import (
#     TF2_WEIGHTS_NAME,
#     TF_WEIGHTS_NAME,
#     WEIGHTS_NAME,
#     cached_path,
#     hf_bucket_url,
#     is_remote_url,
# )

from children_repository.modeling import modeling_gnn
from children_repository.modeling.modeling_bert_custom import BertEncoder
from children_repository.utils import layers

modeling_bert: types.ModuleType
modeling_roberta: types.ModuleType
try:
    from transformers import modeling_bert, modeling_roberta
except ImportError:
    from transformers.models.bert import modeling_bert
    from transformers.models.roberta import modeling_roberta


logger = logging.getLogger(__name__)


INHERIT_BERT = os.environ.get('INHERIT_BERT', 0)
if INHERIT_BERT:
    PreTrainedModelClass = modeling_bert.BertPreTrainedModel
else:
    PreTrainedModelClass = modeling_roberta.RobertaPreTrainedModel
print('PreTrainedModelClass', PreTrainedModelClass)


from typing import Any, BinaryIO, Dict, Optional, Tuple, Union

PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}
HUGGINGFACE_CO_PREFIX = "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"


def hf_bucket_url(model_id: str, filename: str, revision: Optional[str] = None, mirror=None) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    if mirror:
        endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)
        legacy_format = "/" not in model_id
        if legacy_format:
            return f"{endpoint}/{model_id}-{filename}"
        else:
            return f"{endpoint}/{model_id}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)



import os
from pathlib import Path
import requests
import sys
from hashlib import sha256
import fnmatch
from filelock import FileLock
from contextlib import contextmanager
from functools import partial, wraps
import tempfile
import json
from tqdm import tqdm

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )


default_cache_path = os.path.join(torch_cache_home, "transformers")


PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0, user_agent: Union[Dict, str, None] = None):
    """
    Donwload remote file. Do not gobble up errors.
    """
    headers = {"user-agent": http_user_agent(user_agent)}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()
    content_length = r.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    progress = tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc="Downloading",
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()

def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = "transformers/{}; python/{}".format('0.1', sys.version.split()[0])
    ua += "; torch/{}".format(torch.__version__)
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua

def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            headers = {"user-agent": http_user_agent(user_agent)}
            r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
            r.raise_for_status()
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise ValueError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:

            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, user_agent=user_agent)

        os.replace(temp_file.name, cache_path)

        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path

from urllib.parse import urlparse
from zipfile import ZipFile, is_zipfile
import tarfile
import shutil

def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

        return output_path_extracted

    return output_path

class DRAGON(nn.Module):
    def __init__(
        self,
        args,
        model_name='roberta-large',
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
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
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

        neg_nodes = [neg_nodes_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
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
            [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:6]]
            + [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[6:11]]
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
            self.roberta = TextKGMessagePassing(
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
            )  # this is equivalent to RobertaModel
            if args.mlm_task:
                self.lm_head = modeling_roberta.RobertaLMHead(config)

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
            bert_or_roberta = self.roberta
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
        # lm_outputs: ([bs, seq_len, sent_dim], [bs, sent_dim], ([bs, seq_len, sent_dim] for _ in range(25)))
        # gnn_output: [bs, n_node, dim_node]

        # LM outputs
        all_hidden_states = lm_outputs[
            -1
        ]  # ([bs, seq_len, sent_dim] for _ in range(25))
        lm_hidden_states = all_hidden_states[self.layer_id]  # [bs, seq_len, sent_dim]
        sent_vecs = bert_or_roberta.pooler(lm_hidden_states)  # [bs, sent_dim]

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
                    prediction_scores.view(_bs * _seq_len, -1), lm_labels.view(-1)
                )
                next_sentence_loss = 0.0  # loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                lm_loss = masked_lm_loss + next_sentence_loss
            else:
                prediction_scores = self.lm_head(lm_hidden_states)
                lm_loss = loss_fct(
                    prediction_scores.view(_bs * _seq_len, -1), lm_labels.view(-1)
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
        node_mask[node_mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        # sent_node_mask = special_nodes_mask.clone()
        # sent_node_mask[:, 0] = 0

        if self.args.link_task:
            pos_triples, neg_nodes = (
                lp_data  # pos_triples: [3, `total_n_triple`],  neg_nodes: [`total_n_triple`, n_neg]
            )

            pos_samples = pos_triples  # [3, `total_n_triple`]

            _n_neg = neg_nodes.size(1)
            head_negative_sample = neg_nodes[
                :, : _n_neg // 2
            ]  # [`total_n_triple`, n_neg//2]
            tail_negative_sample = neg_nodes[
                :, _n_neg // 2 : _n_neg // 2 * 2
            ]  # [`total_n_triple`, n_neg//2]

            _bs, _, gnn_dim = gnn_output.size()
            embs = gnn_output.view(-1, gnn_dim)  # [`total_n_nodes`, gnn_dim]

            if self.args.link_proj_headtail:
                embs = self.linkpred_proj(embs)
            if self.args.link_normalize_headtail == 1:
                embs = embs / torch.norm(embs, p=2, dim=1, keepdim=True).detach()
            elif self.args.link_normalize_headtail == 2:
                embs = torch.tanh(embs)
            elif self.args.link_normalize_headtail == 3:
                embs = self.emb_LayerNorm(embs)

            positive_score = self.linkpred(embs, pos_samples)  # [`total_n_triple`, 1]
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

            link_loss, pos_link_loss, neg_link_loss = self.linkpred.loss(scores)
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
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
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
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        local_files_only = kwargs.pop('local_files_only', False)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = (
                config if config is not None else pretrained_model_name_or_path
            )
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, TF_WEIGHTS_NAME + '.index'
                    )
                ):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, TF_WEIGHTS_NAME + '.index'
                    )
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, TF2_WEIGHTS_NAME
                    )
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, WEIGHTS_NAME
                    )
                else:
                    raise EnvironmentError(
                        'Error no file named {} found in directory {} or `from_tf` set to False'.format(
                            [
                                WEIGHTS_NAME,
                                TF2_WEIGHTS_NAME,
                                TF_WEIGHTS_NAME + '.index',
                            ],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
                pretrained_model_name_or_path
            ):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + '.index'):
                assert from_tf, 'We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint'.format(
                    pretrained_model_name_or_path + '.index'
                )
                archive_file = pretrained_model_name_or_path + '.index'
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    # use_cdn=use_cdn,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError as err:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg) from err

            if resolved_archive_file == archive_file:
                logger.info('loading weights file {}'.format(archive_file))
            else:
                logger.info(
                    'loading weights file {} from cache at {}'.format(
                        archive_file, resolved_archive_file
                    )
                )
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location='cpu')
            except Exception as err:
                raise OSError(
                    'Unable to load weights from pytorch checkpoint file. '
                    'If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. '
                ) from err

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(
                    model, config, resolved_archive_file[:-6]
                )  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(
                        model, resolved_archive_file, allow_missing_keys=True
                    )
                except ImportError:
                    logger.error(
                        'Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see '
                        'https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            all_keys = list(state_dict.keys())

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=''):
                local_metadata = (
                    {} if metadata is None else metadata.get(prefix[:-1], {})
                )
                module._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    True,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
            has_prefix_module = any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()
            )
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + '.')[-1]
                    for key in model.state_dict().keys()
                ]

                missing_keys.extend(
                    head_model_state_dict_without_base_prefix - base_model_state_dict
                )

            if len(unexpected_keys) > 0:
                logger.warning(
                    f'Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when '
                    f'initializing {model.__class__.__name__}: {unexpected_keys}\n'
                    f'- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task '
                    f'or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n'
                    f'- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect '
                    f'to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).'
                )
            else:
                print(
                    f'All model checkpoint weights were used when initializing {model.__class__.__name__}.\n'
                )
            if len(missing_keys) > 0:
                logger.warning(
                    f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} '
                    f'and are newly initialized: {missing_keys}\n'
                    f'You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.'
                )
            else:
                print(
                    f'All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n'
                    f'If your task is similar to the task the model of the ckeckpoint was trained on, '
                    f'you can already use {model.__class__.__name__} for predictions without further training.'
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    'Error(s) in loading state_dict for {}:\n\t{}'.format(
                        model.__class__.__name__, '\n\t'.join(error_msgs)
                    )
                )
        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'error_msgs': error_msgs,
                'all_keys': all_keys,
            }
            return model, loading_info

        if hasattr(config, 'xla_device') and config.xla_device:
            try:
                import torch_xla.core.xla_model as xm
            except ImportError as err:
                raise ImportError('No torch-xla module available') from err

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model


if INHERIT_BERT:
    ModelClass = modeling_bert.BertModel
else:
    ModelClass = modeling_roberta.RobertaModel


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

        self.encoder = RoBERTaGAT(
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
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
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
            B = self.activation(self.B_lin(node_score))  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(
                self.emb_score(B)
            )  # [batch_size, n_node, dim/2]

        X = H
        edge_index, edge_type = (
            A  # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
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

        # LM outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        # GNN outputs
        X = _X.view(
            node_type.size(0), node_type.size(1), -1
        )  # [batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return outputs, output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
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
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        force_download = kwargs.pop('force_download', False)
        resume_download = kwargs.pop('resume_download', False)
        proxies = kwargs.pop('proxies', None)
        output_loading_info = kwargs.pop('output_loading_info', False)
        local_files_only = kwargs.pop('local_files_only', False)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = (
                config if config is not None else pretrained_model_name_or_path
            )
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, TF_WEIGHTS_NAME + '.index'
                    )
                ):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, TF_WEIGHTS_NAME + '.index'
                    )
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, TF2_WEIGHTS_NAME
                    )
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, WEIGHTS_NAME
                    )
                else:
                    raise EnvironmentError(
                        'Error no file named {} found in directory {} or `from_tf` set to False'.format(
                            [
                                WEIGHTS_NAME,
                                TF2_WEIGHTS_NAME,
                                TF_WEIGHTS_NAME + '.index',
                            ],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(
                pretrained_model_name_or_path
            ):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + '.index'):
                assert from_tf, 'We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint'.format(
                    pretrained_model_name_or_path + '.index'
                )
                archive_file = pretrained_model_name_or_path + '.index'
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    # use_cdn=use_cdn,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError as err:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg) from err

            if resolved_archive_file == archive_file:
                logger.info('loading weights file {}'.format(archive_file))
            else:
                logger.info(
                    'loading weights file {} from cache at {}'.format(
                        archive_file, resolved_archive_file
                    )
                )
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location='cpu')
            except Exception as err:
                raise OSError(
                    'Unable to load weights from pytorch checkpoint file. '
                    'If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. '
                ) from err

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith('.index'):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(
                    model, config, resolved_archive_file[:-6]
                )  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(
                        model, resolved_archive_file, allow_missing_keys=True
                    )
                except ImportError:
                    logger.error(
                        'Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see '
                        'https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions.'
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            all_keys = list(state_dict.keys())

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=''):
                local_metadata = (
                    {} if metadata is None else metadata.get(prefix[:-1], {})
                )
                module._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    True,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ''
            model_to_load = model
            has_prefix_module = any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()
            )
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + '.')[-1]
                    for key in model.state_dict().keys()
                ]

                missing_keys.extend(
                    head_model_state_dict_without_base_prefix - base_model_state_dict
                )

            if len(unexpected_keys) > 0:
                logger.warning(
                    f'Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when '
                    f'initializing {model.__class__.__name__}: {unexpected_keys}\n'
                    f'- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task '
                    f'or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n'
                    f'- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect '
                    f'to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).'
                )
            else:
                logger.info(
                    f'All model checkpoint weights were used when initializing {model.__class__.__name__}.\n'
                )
            if len(missing_keys) > 0:
                logger.warning(
                    f'Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} '
                    f'and are newly initialized: {missing_keys}\n'
                    f'You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.'
                )
            else:
                logger.info(
                    f'All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n'
                    f'If your task is similar to the task the model of the ckeckpoint was trained on, '
                    f'you can already use {model.__class__.__name__} for predictions without further training.'
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    'Error(s) in loading state_dict for {}:\n\t{}'.format(
                        model.__class__.__name__, '\n\t'.join(error_msgs)
                    )
                )
        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys,
                'error_msgs': error_msgs,
                'all_keys': all_keys,
            }
            return model, loading_info

        if hasattr(config, 'xla_device') and config.xla_device:
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model


class RoBERTaGAT(BertEncoder):
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
        super().__init__(config, args)

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
        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

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
                    X = _X.view(bs, -1, _X.size(1))  # [bs, max_num_nodes, node_dim]
                    context_node_lm_feats = hidden_states[:, 0, :]  # [bs, sent_dim]
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
                        [context_node_lm_feats.size(1), context_node_gnn_feats.size(1)],
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
