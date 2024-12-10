import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torchaudio
import typing as T
import io
from scipy.io import wavfile


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/nets_utils.py#L64
    Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.long().tolist()

    bs = int(len(lengths))
    if maxlen is None:
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)
    else:
        assert xs is None
        assert maxlen >= int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/nets_utils.py#L184
    Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)


def mask_by_length(xs, lengths, fill=0):
    """Mask tensor according to length.

    Args:
        xs (Tensor): Batch of input tensor (B, `*`).
        lengths (LongTensor or List): Batch of lengths (B,).
        fill (int or float): Value to fill masked part.

    Returns:
        Tensor: Batch of masked input tensor (B, `*`).

    Examples:
        >>> x = torch.arange(5).repeat(3, 1) + 1
        >>> x
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]])
        >>> lengths = [5, 3, 2]
        >>> mask_by_length(x, lengths)
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 0, 0],
                [1, 2, 0, 0, 0]])

    """
    assert xs.size(0) == len(lengths)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l]
    return ret


def wav_bytes_from_spectrogram_image(image: Image.Image) -> T.Tuple[io.BytesIO, float]:
    """
    Reconstruct a WAV audio clip from a spectrogram image. Also returns the duration in seconds.
    """

    max_volume = 50
    power_for_image = 0.25
    Sxx = spectrogram_from_image(image, max_volume=max_volume, power_for_image=power_for_image)

    sample_rate = 22050  # [Hz]
    clip_duration_ms = 5000  # [ms]

    bins_per_image = 512
    n_mels = 512

    # FFT parameters
    window_duration_ms = 180  # [ms]
    padded_duration_ms = 400  # [ms]
    step_size_ms = 18  # [ms]

    # Derived parameters
    num_samples = int(image.width / float(bins_per_image) * clip_duration_ms) * sample_rate
    n_fft = int(padded_duration_ms / 1000.0 * sample_rate)
    hop_length = int(step_size_ms / 1000.0 * sample_rate)
    win_length = int(window_duration_ms / 1000.0 * sample_rate)

    samples = waveform_from_spectrogram(
        Sxx=Sxx,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        num_samples=num_samples,
        sample_rate=sample_rate,
        mel_scale=True,
        n_mels=n_mels,
        max_mel_iters=200,
        num_griffin_lim_iters=32,
    )

    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples.astype(np.int16))
    wav_bytes.seek(0)

    duration_s = float(len(samples)) / sample_rate

    return wav_bytes, duration_s


def waveform_from_spectrogram(
    Sxx: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    num_samples: int,
    sample_rate: int,
    mel_scale: bool = True,
    n_mels: int = 512,
    max_mel_iters: int = 200,
    num_griffin_lim_iters: int = 32,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Reconstruct a waveform from a spectrogram.

    This is an approximate inverse of spectrogram_from_waveform, using the Griffin-Lim algorithm
    to approximate the phase.
    """
    Sxx_torch = torch.from_numpy(Sxx).to(device)

    # TODO(hayk): Make this a class that caches the two things

    if mel_scale:
        mel_inv_scaler = torchaudio.transforms.InverseMelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0,
            f_max=7600,
            #f_max=10000,
            n_stft=n_fft // 2 + 1,
            norm=None,
            mel_scale="htk",
            max_iter=max_mel_iters,
        ).to(device)

        Sxx_torch = mel_inv_scaler(Sxx_torch)

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        power=1.0,
        n_iter=num_griffin_lim_iters,
    ).to(device)

    waveform = griffin_lim(Sxx_torch).cpu().numpy()

    return waveform


def spectrogram_from_image(
    image: Image.Image, max_volume: float = 50, power_for_image: float = 0.25
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    TODO(hayk): Add image_from_spectrogram and call this out as the reverse.
    """
    # Convert to a numpy array of floats
    data = np.array(image).astype(np.float32)

    # Flip Y take a single channel
    data = data[::-1, :, 0]

    # Invert
    data = 255 - data

    # Rescale to max volume
    data = data * max_volume / 255

    # Reverse the power curve
    data = np.power(data, 1 / power_for_image)

    return data

def _paraphrase(orig_instruction):
    
   
    synonyms_for_angry = ["angry", "Furious","Fuming"]
    synonyms_for_fear = ["fear", "Terror", "Fright"]
    synonyms_for_happy = ["happy", "joyful", "delighted"]
    synonyms_for_contempt = ["contempt", "sneering"]
    synonyms_for_disgusted = ["disgusted",  "sickened"]
    synonyms_for_sad = ["sad", "Sorrowful", "Unhappy"]
    synonyms_for_surprised = ["surprised", "Astonished", "Shocked",]
    synonyms_for_neutral_expression = ["neutral"]
    
    
   
    instruction_templates = ["Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
    "Employ [EXP] expression during our conversation",
    "Make use of [EXP] expression as we discuss.",
    "Please talk with the emotion of [EXP].",
    "I'd like you to talk as if you were feeling [EXP].",
    "Express yourself as if you were experiencing [EXP].",
   ]
    if orig_instruction == 'angry':
        new_instruction = np.random.choice(synonyms_for_angry).lower()
    elif orig_instruction == 'fear':
        new_instruction = np.random.choice(synonyms_for_fear).lower()
    elif orig_instruction == 'happy':
        new_instruction = np.random.choice(synonyms_for_happy).lower()
    elif orig_instruction == 'contempt':
        new_instruction = np.random.choice(synonyms_for_contempt).lower()
    elif orig_instruction == "disgusted":
        new_instruction = np.random.choice(synonyms_for_disgusted).lower()
    elif orig_instruction == "sad":
        new_instruction = np.random.choice(synonyms_for_sad).lower()
    elif orig_instruction == "surprised":
        new_instruction = np.random.choice(synonyms_for_surprised).lower()
    elif orig_instruction == "neutral":
        new_instruction = np.random.choice(synonyms_for_neutral_expression).lower()
    else:
        new_instruction = orig_instruction
        print('we haven\'t found the emotion in the instruction:', orig_instruction)
    if new_instruction:    
        random_template = np.random.choice(instruction_templates)
        final_instruction = random_template.replace('[EXP]', new_instruction)
    else:
        final_instruction = '[EMPTY]'
    return final_instruction



synonyms_for_angry = ["angry", "Furious", "Annoyed", "Irritated", "Mad", "Enraged", "Indignant", "Fuming", "Infuriated", "Raging" "In a rage", ]
synonyms_for_fear = ["fear", "Fright", "Panic", "Alarm", "Afraid",  "Scared", "Frightened", "Nervous", "Panicked", "Horrified"]
synonyms_for_happy = ["happy", "joyful", "content", "delighted", "pleased", "cheerful",  "satisfied", "Glad", "Thrilled", "Over the moon"]
synonyms_for_contempt = ["contempt", "disdain", "scorn", "disrespect", "mockery", "sneering"]
synonyms_for_disgusted = ["disgusted", "revolted",  "sickened" , "Displeased"]
synonyms_for_sad = ["sad", "Sorrowful", "Unhappy", "Depressed",  "Disheartened",  "Mournful", "Heartbroken",  "Down", "Despairing", "Desperate"]
synonyms_for_surprised = ["surprised", "Astonished", "Amazed", "Startled", "Shocked", "Astounded"]
synonyms_for_neutral_expression = ["neutral", "Expressionless", "Blank",  "Poker-faced"]

    
    
# instruction_templates = ["Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
# # "I'd appreciate it if you could utilize [EXP] expression while conversing.",
# # "Employ [EXP] expression during our conversation, if you don't mind.",
# "Converse with the help of [EXP] expression, if you wish.",
# "Make use of [EXP] expression as we discuss.", "Could you express [EXP]ly in your speech?",
# "speak with the expression of [EXP]?",
# "Please convey your thoughts with the emotion of [EXP].",
# "I'd like you to talk as if you were feeling [EXP].",
# "Incorporate [EXP] into your speech.",
# "Let your words reflect the sentiment of [EXP].",
# "Speak while embodying the spirit of [EXP].",
# "Imagine your words are a reflection of [EXP].",
# "Try to vocalize your thoughts with the emotion represented by [EXP].",
# # "Could you articulate your thoughts with the demeanor of [EXP]?",
# # "Share your perspective while embodying [EXP].",
# "Let [EXP] be your guide in speaking.",
# # "I'd like to hear your words tinged with [EXP].",
# "Express yourself as if you were experiencing [EXP].",
# "Speak as if your emotions were [EXP].",
# "Try conveying your message with [EXP] as your tone.",
# "Imagine your words bathed in the light of [EXP].",
# "Discuss the topic with the aura of [EXP].",
# "Could you infuse your speech with [EXP]?",
# "Let [EXP] color your words as you speak.",
# "Speak in a way that mirrors the essence of [EXP].",
# "Could you vocalize your thoughts with the spirit of [EXP]?",
# "Express your ideas as if they were painted with [EXP].",
# "Let your words be a reflection of the atmosphere of [EXP].",
# "Try to convey your message while embodying [EXP].",
# "Speak with the emotion of [EXP] guiding your words.",
# "Imagine [EXP] as the backdrop to your conversation.",
# "Could you channel [EXP] in your speech?",
# "Let your words resonate with the feeling of [EXP].",
# "Speak with the essence of [EXP] in mind.",
# "Express your thoughts as if they were infused with [EXP].",
# "I'd like you to communicate using [EXP] expression, please.",
# "Could you chat employing [EXP] expression?",
# "Convey your ideas with the emotional backdrop of [EXP].",
# "Try to vocalize your message with [EXP] as your inspiration.",
# "Imagine your words flowing in the current of [EXP].",
# "Speak as if your emotions were painted with the colors of [EXP].",
# "Discuss the topic while embracing the spirit of [EXP].",
# "Let's discuss things while incorporating [EXP] expression.",
# "Feel free to use [EXP] expression while we talk.","Could you articulate your thoughts with the aura of [EXP]?",
# "Speak as if you were surrounded by [EXP].",
# "Express yourself with the nuance of [EXP].",
# "Let the essence of [EXP] guide your words.",
# "Try conveying your message as if it were bathed in [EXP].",
# "Imagine your words unfolding in the presence of [EXP].",
# "Speak with the vibe of [EXP] in your voice.",
# "Share your perspective while immersed in [EXP].",
# "Let [EXP] be the melody of your speech.",
# "Could you converse with the essence of [EXP]?",
# "Speak with the backdrop of [EXP] coloring your words.",
# "Express your thoughts with the ambiance of [EXP].",
# "Let the spirit of [EXP] be your guide in conversation.",
# "Try to convey your message with the emotional hue of [EXP].",
# "Imagine your words flowing like a river of [EXP].",
# "Speak as if your emotions were reflected in the mirror of [EXP].",
# "Share your perspective while surrounded by the aura of [EXP].",
# "Let [EXP] set the tone for your dialogue.",
# "Could you communicate with the spirit of [EXP]?",
# "Speak as if your words were flavored with [EXP].",
# "Express your thoughts with the backdrop of [EXP] in mind.",
# "Let [EXP] be the guiding principle in your conversation.",
# "Try conveying your message while encapsulating [EXP].",
# "Imagine your words being carried by the winds of [EXP].",
# "Speak as if your emotions were a reflection of [EXP].",
# "Share your perspective while immersed in the world of [EXP].",
# "Let [EXP] be the driving force behind your dialogue.",
# "Speak as if [EXP] were the lens through which you view the topic.",
# "Express your ideas with the emotional atmosphere of [EXP].",
# "Let [EXP] set the mood for your conversation.",
# "Try conveying your message while encapsulating the essence of [EXP].",
# "Imagine your words flowing in the river of [EXP].",
# "Speak as if your emotions were sculpted by the hand of [EXP].",
# "Share your perspective while bathed in the light of [EXP].",
# "Let [EXP] be the guiding star in your dialogue.",
# "Could you converse with the aura of [EXP]?",
# "Speak as if [EXP] were the canvas upon which you paint your words.",
# "Express your thoughts with the backdrop of [EXP] enveloping you.",
# "Let [EXP] shape the texture of your conversation.",
# "Try conveying your message while radiating [EXP].",
# "Imagine your words swirling in the vortex of [EXP].",
# "Speak as if your emotions were melodies inspired by [EXP].",
# "Share your perspective while bathed in the essence of [EXP].",
# "Let [EXP] be the compass that guides your dialogue.",
# "Could you vocalize your thoughts with the ambiance of [EXP]?",
# "Speak as if [EXP] were the tapestry upon which you weave your words.",
# "Express your ideas with the emotional tapestry of [EXP].",
# "Let [EXP] be the heartbeat of your conversation.",
# "Try conveying your message while surrounded by the presence of [EXP].",
# "Imagine your words dancing to the rhythm of [EXP].",
# "Speak as if your emotions were the stars in the sky of [EXP].",
# "Share your perspective while bathed in the essence of [EXP]."]
instruction_templates = ["talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "talk with [EXP] emotion", "Please talk with a [EXP] expression.", 
"Please talk with a [EXP] face.", 
"Please talk with a [EXP] look.", 
"Please engage in a conversation using an [EXP] expression.",
"Converse with the help of [EXP] expression.",
"Make use of [EXP] expression as we discuss.", "Could you express [EXP]ly in your speech?",
"speak with the expression of [EXP]",
"Please convey your thoughts with the emotion of [EXP].",
"I'd like you to talk as if you were feeling [EXP].",
"Incorporate [EXP] into your speech.",
"Let your words reflect the sentiment of [EXP].",
"Let [EXP] be your guide in speaking.",
"Express yourself as if you were experiencing [EXP].",
"Speak as if your emotions were [EXP].",
"Try conveying your message with [EXP] as your tone.",
"Let [EXP] be your emotion as you speak.",
"Speak with the emotion of [EXP] guiding your words.",
"Let your words with the feeling of [EXP].",
"I'd like you to communicate using [EXP] expression, please.",
"Could you chat employing [EXP] expression?",
"Discuss the topic while embracing the spirit of [EXP].",
"Let's discuss things while incorporating [EXP] expression.",
"Feel free to use [EXP] expression while we talk.",
"Speak as if you were surrounded by [EXP].",
"Let the essence of [EXP] guide your words.",
"Let [EXP] set the tone for your dialogue.",
"Could you communicate with the spirit of [EXP]?",
"Express your thoughts with the [EXP] in mind.",
"Let [EXP] be the guiding principle in your conversation.",
"Try conveying your message while being [EXP].",
"Let [EXP] set the mood for your conversation.",
"Speak as if your emotions were [EXP].",
"Let [EXP] shape the texture of your conversation.",
"Try conveying your message while being [EXP].",
"Let [EXP] guides your dialogue.",
"Speak as if [EXP].",
"Express your ideas with the emotional of [EXP].",
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
"Employ [EXP] expression during our conversation",
"Make use of [EXP] expression as we discuss.",
"Please talk with the emotion of [EXP].",
"I'd like you to talk as if you were feeling [EXP].",
"Express yourself as if you were experiencing [EXP].",
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
"Employ [EXP] expression during our conversation",
"Make use of [EXP] expression as we discuss.",
"Please talk with the emotion of [EXP].",
"I'd like you to talk as if you were feeling [EXP].",
"Express yourself as if you were experiencing [EXP].",
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression."
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression.",
"Please talk with a [EXP] expression.", "Please talk with a [EXP] face.", "Please talk with a [EXP] look.", "Please engage in a conversation using an [EXP] expression."]
    
def _paraphrase_extend(orig_instruction):
    
    if orig_instruction == 'angry':
        new_instruction = np.random.choice(synonyms_for_angry).lower()
    elif orig_instruction == 'fear':
        new_instruction = np.random.choice(synonyms_for_fear).lower()
    elif orig_instruction == 'happy':
        new_instruction = np.random.choice(synonyms_for_happy).lower()
    elif orig_instruction == 'contempt':
        new_instruction = np.random.choice(synonyms_for_contempt).lower()
    elif orig_instruction == "disgusted":
        new_instruction = np.random.choice(synonyms_for_disgusted).lower()
    elif orig_instruction == "sad":
        new_instruction = np.random.choice(synonyms_for_sad).lower()
    elif orig_instruction == "surprised":
        new_instruction = np.random.choice(synonyms_for_surprised).lower()
    elif orig_instruction == "neutral":
        new_instruction = np.random.choice(synonyms_for_neutral_expression).lower()
    else:
        new_instruction = orig_instruction
        print('we haven\'t found the emotion in the instruction:', orig_instruction)
    if new_instruction:    
        random_template = np.random.choice(instruction_templates)
        final_instruction = random_template.replace('[EXP]', new_instruction)
    else:
        final_instruction = '[EMPTY]'
    return final_instruction




sentences_for_hdtf = ['talk with free expression.', 'talk with neutral expression', 'talk naturally', "being poker face when talking", "talk with poker face", "talk with neutral expression", "talk with poker face", "please be neutral when you are talking", "talk with emotionless face"]    

adv_dict = {'1':['intensity 1', 'intensity 1', 'intensity 1', 'intensity 1', 'intensity 1', 'intensity 1', 'intensity 1', 'intensity 1', 'intensity 1', 'slightly', 'slightly', 'slightly', 'slightly', 'slightly','slightly','lightly', 'somewhat', 'a bit', 'a little'], '2':['intensity 2','intensity 2','intensity 2','intensity 2','intensity 2','intensity 2','intensity 2','intensity 2','intensity 2','intensity 2','moderately', 'moderately', 'moderately', 'moderately', 'moderately', 'fairly', 'quite', 'pretty', 'reasonably'], '3':['intensity 3', 'intensity 3', 'intensity 3', 'intensity 3', 'intensity 3', 'intensity 3', 'intensity 3', 'intensity 3', 'intensity 3', 'intensity 3','extremely', 'extremely', 'extremely', 'extremely', 'extremely', 'very', 'highly', 'strongly', 'remarkably', 'extraordinarily'], '[EMPTY]':['']}


def _paraphrase_extend_intensity(orig_instruction, intensity):
    

    if orig_instruction == 'angry':
        new_instruction = np.random.choice(synonyms_for_angry).lower()
    elif orig_instruction == 'fear':
        new_instruction = np.random.choice(synonyms_for_fear).lower()
    elif orig_instruction == 'happy':
        new_instruction = np.random.choice(synonyms_for_happy).lower()
    elif orig_instruction == 'contempt':
        new_instruction = np.random.choice(synonyms_for_contempt).lower()
    elif orig_instruction == "disgusted":
        new_instruction = np.random.choice(synonyms_for_disgusted).lower()
    elif orig_instruction == "sad":
        new_instruction = np.random.choice(synonyms_for_sad).lower()
    elif orig_instruction == "surprised":
        new_instruction = np.random.choice(synonyms_for_surprised).lower()
    elif orig_instruction == "neutral":
        new_instruction = np.random.choice(synonyms_for_neutral_expression).lower()
    else:
        new_instruction = orig_instruction
        print('we haven\'t found the emotion in the instruction:', orig_instruction)
    if new_instruction:
        try:
            adv = random.choice(adv_dict[intensity])
        except:
            adv = ''
        if random.random() < 0.5:
            new_instruction = adv + ' ' + new_instruction
        else:
            new_instruction = new_instruction + ' ' + adv 
        random_template = np.random.choice(instruction_templates)
        final_instruction = random_template.replace('[EXP]', new_instruction)
    else:
        final_instruction = '[EMPTY]'
    return final_instruction




def all_paraphrase(instruction, inst_flag, audio_flag):
    if audio_flag and inst_flag:
        new_instruction = _paraphrase(instruction)
    elif audio_flag and (not inst_flag):
        new_instruction = random.choice(sentences_for_hdtf)
    elif (not audio_flag) and inst_flag:
        new_instruction = instruction
    else:
        print('we haven\'t found the emotion in the instruction:', instruction)
    return new_instruction if new_instruction else '[EMPTY]'

def all_paraphrase_extend(instruction, inst_flag, audio_flag):
    if audio_flag and inst_flag: # mead
        new_instruction = _paraphrase_extend(instruction)
    elif audio_flag and (not inst_flag):  # hdtf
        new_instruction = random.choice(sentences_for_hdtf)
    elif (not audio_flag) and inst_flag:  # text2motion
        new_instruction = instruction
    else:
        print('we haven\'t found the emotion in the instruction:', instruction)
    return new_instruction if new_instruction else '[EMPTY]'


def all_paraphrase_intensity(instruction, inst_flag, audio_flag, file_name):
    
    if audio_flag and inst_flag: # mead
        try:
            intensity = file_name.split('_')[3]
        except:
            intensity = '[EMPTY]'
            print('we haven\'t found the intensity in the file name:', file_name)
        new_instruction = _paraphrase_extend_intensity(instruction, intensity)
    elif audio_flag and (not inst_flag):  # hdtf
        new_instruction = random.choice(sentences_for_hdtf)
    elif (not audio_flag) and inst_flag:  # text2motion
        new_instruction = instruction
    else:
        print('we haven\'t found the emotion in the instruction:', instruction)
    return new_instruction if new_instruction else '[EMPTY]'



import random
def add_au_func(orig_instruction,  inst_flag, audio_flag, action_units):
    if audio_flag and inst_flag and action_units != '[EMPTY]':
        action_units_sentence = "; ".join([" ".join(each.split('_')) for each in action_units.split(',')])
        random_num = random.random()
        if random_num < 0.2:
            new_instruction = orig_instruction + ' with these action units: ' + action_units_sentence
        elif random_num < 0.7:
            new_instruction = 'talk with these action units: ' + action_units_sentence + '.'
        else:
            new_instruction = orig_instruction
        # new_instruction = 'talk with these action units: ' + action_units_sentence + '.'
        return new_instruction
    else:
        return orig_instruction
    
import json
import re  
au_gpt4v_para = json.load(open('/mnt/blob/xxxx/TETF/datasets/MEAD_all/au_detect/au_paraphrase_v.json', 'r'))    

def add_au_func_para(orig_instruction,  inst_flag, audio_flag, action_units, file_name):
    if audio_flag and inst_flag and action_units != '[EMPTY]':
        action_units_sentence = "; ".join([" ".join(each.split('_')) for each in action_units.split(',')])
        rand = random.random()
        if rand < 0.5:
            action_units_sentence = action_units_sentence + '.'
        else:
            if file_name in au_gpt4v_para.keys():
                action_units_sentences = au_gpt4v_para[file_name]
                single_sentence = random.choice(action_units_sentences.split('\n'))
                match = re.search(r'\d+\.\s*(.*)', single_sentence)
                if match:
                    action_units_sentence = match.group(1).strip().strip('"')
                
            
        random_num = random.random()
        if random_num < 0.2:
            new_instruction = orig_instruction + ' with these action units: ' + action_units_sentence
        elif random_num < 0.7:
            new_instruction = 'talk with these action units: ' + action_units_sentence + '.'
        else:
            new_instruction = orig_instruction
        # new_instruction = 'talk with these action units: ' + action_units_sentence + '.'
        return new_instruction
    else:
        return orig_instruction
    
def add_au_func_intensity(orig_instruction,  inst_flag, audio_flag, action_units, file_name):
    if audio_flag and inst_flag and action_units != '[EMPTY]':
        action_units_sentence = "; ".join([" ".join(each.split('_')) for each in action_units.split(',')])
        random_num = random.random()
        if random_num < 0.2:
            new_instruction = orig_instruction + ' with these action units: ' + action_units_sentence
        elif random_num < 0.7:
            new_instruction = 'talk with these action units: ' + action_units_sentence + '.'
        else:
            new_instruction = orig_instruction
        # new_instruction = 'talk with these action units: ' + action_units_sentence + '.'
        return new_instruction
    else:
        return orig_instruction
    
        
action_units_dict = {
    "inner_brow_raiser": 0,
    "outer_brow_raiser": 1,
    "brow_lowerer": 2,
    "upper_lid_raiser": 3,
    "cheek_raiser": 4,
    "lid_tightener": 5,
    "nose_wrinkler": 6,
    "upper_lip_raiser": 7,
    "nasolabial_deepener": 8,
    "lip_corner_puller": 9,
    "sharp_lip_puller": 10,
    "dimpler": 11,
    "lip_corner_depressor": 12,
    "lower_lip_depressor": 13,
    "chin_raiser": 14,
    "lip_pucker": 15,
    "tongue_show": 16,
    "lip_stretcher": 17,
    "lip_funneler": 18,
    "lip_tightener": 19,
    "lip_pressor": 20,
    "lips_part": 21,
    "jaw_drop": 22,
    "mouth_stretch": 23,
    "lip_bite": 24,
    "nostril_dilator": 25,
    "nostril_compressor": 26,
    "left_inner_brow_raiser": 27,
    "right_inner_brow_raiser": 28,
    "left_outer_brow_raiser": 29,
    "right_outer_brow_raiser": 30,
    "left_brow_lowerer": 31,
    "right_brow_lowerer": 32,
    "left_cheek_raiser": 33,
    "right_cheek_raiser": 34,
    "left_upper_lip_raiser": 35,
    "right_upper_lip_raiser": 36,
    "left_nasolabial_deepener": 37,
    "right_nasolabial_deepener": 38,
    "left_dimpler": 39,
    "right_dimpler": 40
}

def get_au_label(action_units):
    au_label = np.zeros(len(action_units_dict.keys()) + 1, dtype=np.float32)
    if action_units != '[EMPTY]':
        for each in action_units.split(','):
            if each.strip() in action_units_dict.keys():
                au_label[action_units_dict[each.strip()]] = 1
            else:
                au_label[-1] = 1
    return torch.tensor(au_label)
        
typical_emotion_dict = {'angry':  ['brow_lowerer', 'jaw_drop', 'nose_wrinkler', 'lid_tightener'],
                        'fear': ['inner_brow_raiser', 'jaw_drop', 'upper_lid_raiser', 'outer_brow_raiser'],
                        'happy': ['cheek_raiser', 'lip_corner_puller', 'jaw_drop', 'lid_tightener'],
                        'contempt': ['inner_brow_raiser', 'chin_raiser','lip_corner_puller', 'lid_tightener'],
                        'disgusted': ['brow_lowerer', 'cheek_raiser', 'lip_corner_depressor', 'nose_wrinkler'],
                        'sad': ['brow_lowerer', 'chin_raiser', 'inner_brow_raiser', 'lip_corner_depressor'],
                        'surprised': ['inner_brow_raiser', 'jaw_drop', 'outer_brow_raiser', 'lid_tightener'],
                        }
    