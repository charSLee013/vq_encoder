import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def avg_with_mask(x, mask):
    assert mask.dtype == torch.float, "Mask should be float"

    if mask.ndim == 2:
        mask = mask.unsqueeze(1)

    if mask.shape[1] == 1:
        mask = mask.expand_as(x)

    return (x * mask).sum() / mask.sum()


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)

    if titles is None:
        titles = [None for i in range(len(data))]

    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]

        if isinstance(mel, torch.Tensor):
            mel = mel.float().detach().cpu().numpy()

        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]

    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=0)
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str