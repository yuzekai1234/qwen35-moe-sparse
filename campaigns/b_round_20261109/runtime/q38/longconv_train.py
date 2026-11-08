"""Training state of the LongConv lag filters (modeling LongConv.taps), sharded across data-parallel ranks.

The taps [K, c, c] of every block are not trained by the hybrid optimizer or DDP. Rank r owns the lags
r*K/world ... (r+1)*K/world - 1 (FP32 master copy and Muon momentum) and, for the forward pass, the BF16 taps of
its output columns r*c/world ... (r+1)*c/world - 1 for all lags. The full taps (block.taps) stay on the CPU and are
gathered onto the GPU only for evaluation and checkpoints (gather_taps, release_taps). Per optimizer step:

1. begin_step: LongConv.begin_step caches the response of this rank's columns and a zero gradient accumulator;
2. after the micro batches, gradients(): LongConv.end_step forms the gradient of this rank's columns for all lags;
   an all-to-all regroups it into this rank's lags for all columns (mean over ranks, as DDP);
3. step(): Muon on the owned lag matrices (Nesterov momentum, Newton-Schulz orthogonalization, update RMS 0.2,
   decoupled weight decay, as q38.muon.Muon), with the global clipping coefficient applied to the gradient first;
   an all-to-all then regroups the updated lags (BF16) into each rank's columns.

GPU memory per rank and block, for P = K c^2 taps: 10 P / world bytes between steps (master, momentum, column
taps); during a step also the column response and its gradient accumulator, 16 (n/2+1) c^2 / world bytes. With
offload, the master and momentum are pinned CPU tensors that step() streams through the GPU in the Muon chunks of
lags (same arithmetic, same chunks), which leaves 2 P / world bytes between steps.
"""
import torch
import torch.distributed as dist

from .muon import MUON_MAX_CHUNK_ELEMENTS, _newton_schulz_cuda, newton_schulz


class LongConvTrainer:
    def __init__(self, model, world=1, rank=0, group=None, momentum=0.95, rms_scale=0.2, ns_steps=5, offload=False):
        self.blocks = list(model.model.longconv.values())
        self.world, self.rank, self.group, self.offload = world, rank, group, offload
        self.device = self.blocks[0].taps.device if self.blocks else torch.device("cpu")
        self.momentum, self.rms_scale, self.ns_steps = momentum, rms_scale, ns_steps
        self.master, self.buffers, self.columns, self.grads = [], [], [], []
        for block in self.blocks:
            K, c = block.taps.shape[0], block.taps.shape[2]
            if K % world or c % world:
                raise ValueError("LongConv taps and channels must be divisible by the data-parallel world size")
            taps = block.taps.detach()
            lags = taps[rank * K // world:(rank + 1) * K // world]
            if offload:
                self.master.append(lags.to("cpu", torch.float32).pin_memory())
                self.buffers.append(torch.zeros(self.master[-1].shape, pin_memory=True))
            else:
                self.master.append(lags.float().clone())
                self.buffers.append(torch.zeros_like(self.master[-1]))
            self.columns.append(taps[:, :, rank * c // world:(rank + 1) * c // world].to(torch.bfloat16).contiguous())
            block.taps.requires_grad_(False)
            block.taps.data = taps.to("cpu", torch.bfloat16)

    def begin_step(self):
        for block, columns in zip(self.blocks, self.columns):
            block.begin_step(self.world, self.rank, self.group, columns)

    def gradients(self):
        """Owned-lag gradients of every block; returns their squared norm summed over all ranks."""
        self.grads = []
        for block in self.blocks:
            column_grad = block.end_step()                                     # [K, c, c/world], mean over ranks
            K, c, width = column_grad.shape
            if self.world > 1:
                send = column_grad.view(self.world, K // self.world, c, width).contiguous()   # [q] = lags of rank q
                received = torch.empty_like(send)                                             # [p] = rank p's columns
                self._alltoall(received, send)
                grad = received.permute(1, 2, 0, 3).reshape(K // self.world, c, c)
            else:
                grad = column_grad
            self.grads.append(grad)
        total = torch.stack([g.float().pow(2).sum() for g in self.grads]).sum() if self.grads else torch.zeros(())
        if self.world > 1:
            dist.all_reduce(total, group=self.group)
        return total

    @staticmethod
    def _chunks(master):
        """Row slices of the Muon chunks of lags (q38.muon: at most MUON_MAX_CHUNK_ELEMENTS per Newton-Schulz call)."""
        lags, c = master.shape[0], master.shape[-1]
        chunk = max(1, MUON_MAX_CHUNK_ELEMENTS // (c * c))
        return [slice(start, min(lags, start + chunk)) for start in range(0, lags, chunk)]

    @torch.no_grad()
    def step(self, lr, weight_decay, clip_coef=1.0):
        for master, buffer, grad, columns in zip(self.master, self.buffers, self.grads, self.columns):
            c = master.shape[-1]
            ns = _newton_schulz_cuda if grad.is_cuda else newton_schulz
            lags = torch.empty(master.shape, device=grad.device, dtype=torch.bfloat16)
            for rows in self._chunks(master):
                m, b = master[rows].to(grad.device, non_blocking=True), buffer[rows].to(grad.device, non_blocking=True)
                g = grad[rows] * clip_coef
                b.mul_(self.momentum).add_(g)
                orthogonal = ns(g.add(b, alpha=self.momentum), self.ns_steps).float()
                if weight_decay:
                    m.mul_(1 - lr * weight_decay)
                m.add_(orthogonal, alpha=-lr * self.rms_scale * c ** .5)
                if self.offload:
                    master[rows].copy_(m, non_blocking=True)
                    buffer[rows].copy_(b, non_blocking=True)
                lags[rows] = m
            self._update_columns(lags, columns)
        if self.offload:
            torch.cuda.synchronize(self.device)     # the CPU copies are read by the next step and by state_dict
        self.grads = []

    def _lags_bf16(self, master):
        """This rank's lag matrices [K/world, c, c] in BF16 on the GPU."""
        if not self.offload:
            return master.to(torch.bfloat16)
        lags = torch.empty(master.shape, device=self.device, dtype=torch.bfloat16)
        for rows in self._chunks(master):
            lags[rows] = master[rows].to(self.device, non_blocking=True)
        return lags

    @torch.no_grad()
    def _update_columns(self, lags, columns):
        """This rank's columns for all lags [K, c, c/world] from every rank's BF16 lags [K/world, c, c]."""
        if self.world == 1:
            columns.copy_(lags)
            return
        count, c, width = lags.shape[0], lags.shape[1], lags.shape[2] // self.world
        send = lags.view(count, c, self.world, width).permute(2, 0, 1, 3).contiguous()   # [q] = columns of rank q
        received = torch.empty_like(send)                                                # [p] = lags of rank p
        self._alltoall(received, send)
        columns.copy_(received.view(self.world * count, c, width))

    @torch.no_grad()
    def gather_taps(self):
        """Full BF16 taps of every block on this rank's GPU, for evaluation and checkpoints (collective)."""
        for block, master in zip(self.blocks, self.master):
            lags = self._lags_bf16(master)
            full = torch.empty((self.world * lags.shape[0],) + tuple(lags.shape[1:]), device=lags.device,
                               dtype=torch.bfloat16)
            if self.world > 1:
                self._gather(full, lags)
            else:
                full.copy_(lags)
            block.taps.data = full

    @torch.no_grad()
    def release_taps(self):
        """Move the gathered taps back to the CPU."""
        for block in self.blocks:
            block.taps.data = block.taps.data.to("cpu")

    def _alltoall(self, out, inp):
        if dist.get_backend(self.group) == "nccl":
            dist.all_to_all_single(out, inp, group=self.group)
        else:   # Gloo (CPU tests) has no all-to-all
            everything = inp.new_empty((self.world,) + inp.shape)
            dist.all_gather(list(everything.unbind(0)), inp.contiguous(), group=self.group)
            out.copy_(everything[:, self.rank])

    def _gather(self, out, inp):
        if dist.get_backend(self.group) == "nccl":
            dist.all_gather_into_tensor(out, inp.contiguous(), group=self.group)
        else:
            dist.all_gather(list(out.view((self.world, -1) + tuple(out.shape[1:])).unbind(0)), inp.contiguous(), group=self.group)

    def state_dict(self):
        return dict(world=self.world, rank=self.rank, master=[m.cpu() for m in self.master],
                    buffers=[b.cpu() for b in self.buffers])

    def load_state_dict(self, state):
        """Restore this rank's shards and rebuild its column taps from them (collective)."""
        if (state["world"], state["rank"]) != (self.world, self.rank):
            raise ValueError("LongConv shards were saved with another world size or rank")
        for target, source in zip(self.master + self.buffers, state["master"] + state["buffers"]):
            target.copy_(source)
        for master, columns in zip(self.master, self.columns):
            self._update_columns(self._lags_bf16(master), columns)
