"""Fresh C1 training with the inherited best common tail schedule."""
import json
from pathlib import Path

from . import train
from .paper_schedule import make_schedule, effective_longconv_lr


def main():
    args = train.parse_args()
    config = json.loads(Path(args.config).read_text())
    spec = config['variant']['common_schedule']
    assert config['variant']['initialization'] == 'fresh_seed0'
    assert args.seed == args.data_seed == 0 and args.mtp == 0
    assert not (Path(args.run_dir)/'TAIL_FORK_rank0.json').exists()
    original = train.lr_schedule
    current = dict(step=-1)
    def factory(a, n):
        return make_schedule(original(a, n), a, n, spec, current)
    train.lr_schedule = factory
    # The step state is set by lr_at before both optimizer calls.
    if not args.dry_run:
        from .longconv_train import LongConvTrainer
        lc_step = LongConvTrainer.step
        def controlled_step(self, lr, wd, coefficient):
            return lc_step(self, effective_longconv_lr(lr, spec, current), wd, coefficient)
        LongConvTrainer.step = controlled_step
    return train.main()


if __name__ == '__main__':
    main()
