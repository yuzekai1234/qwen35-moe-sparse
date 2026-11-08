"""Common qualified-parent schedule, rescaled to each full C1 horizon."""


def make_schedule(original, args, steps, spec, current):
    start = int(spec['start_step'])
    assert 0 < start < steps - 1 and spec['mode'] == 'linear_zero'
    anchor = original(start)
    auxiliary = args.aux_coef
    def at(step):
        current['step'] = step
        args.aux_coef = auxiliary if step < start else 0.
        return original(step) if step < start else anchor * (1 - (step-start)/(steps-start-1))
    return at


def effective_longconv_lr(lr, spec, current):
    return lr * (spec['longconv_tail_lr_mult'] if current['step'] >= spec['start_step'] else 1.)
