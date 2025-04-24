import logging
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from lingua.tokenizer import build_tokenizer
from apps.mtp.transformer import (
    LMMTPArgs,
    LMTransformer,
)

from apps.mtp.train import (
    TrainArgs,
)

logger = logging.getLogger()


def get_num_params(model: nn.Module) -> int:
    """
    Get the total model params
    Args : only_trainable: whether to only count trainable params
    """
    numel = {n: p.numel() for n, p in model.named_parameters()}
    return sum(numel.values())

def pprint(n: int) -> str:
    return "_".join([str(n)[::-1][i : i + 3] for i in range(0, len(str(n)), 3)])[::-1]


def size(args, include_embedding=False):
    tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
    output_size = tokenizer.n_words
    if args.model.vocab_size < 0:
        logger.info(f"Setting {args.model.vocab_size} to model ouput")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), "Vocab size should be the same as output size"

    with torch.device("meta"):
        model = LMTransformer(args.model)
    n_params = get_num_params(model) 
    num_non_embed_params = n_params - args.model.vocab_size * args.model.dim
    dim = args.model.dim
    n_layers = args.model.n_layers
    n_heads = args.model.n_heads

    if include_embedding:
        print(f"Number of params w/ embedding (dim={dim}, n_layers={n_layers}, n_heads={n_heads}): {pprint(n_params)}")
    else:
        print(f"Number of params w/o embedding (dim={dim}, n_layers={n_layers}, n_heads={n_heads}): {pprint(num_non_embed_params)}")

def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        mode: LMMTPArgs

    @dataclass
    class LMMTPArgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMMTPArgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    size(cfg, True)


if __name__ == "__main__":
    main()

#python -m lingua.size config=apps/main/configs/debug.yaml model.dim=2400  model.n_layers=29  model.n_heads=16
