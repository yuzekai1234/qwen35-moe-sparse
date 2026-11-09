"""Keep DDP gradient bucket views across accumulation steps without extra forwards."""


def accumulate_without_sync(ddp, step, start_step, micro, accumulation):
    # The first backward after construction/resume must install bucket views.
    # Synchronizing that first optimizer step's micros preserves the same
    # globally averaged gradient; later steps can accumulate into those views.
    return ddp and step > start_step and micro < accumulation-1


def clear_gradients_to_none(ddp):
    # set_to_none discards the views, recreating a full gradient allocation
    # before the last synchronized backward of each accumulation step.
    return not ddp
