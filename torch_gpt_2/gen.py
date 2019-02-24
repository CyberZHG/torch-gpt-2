import torch


__all__ = ['generate']


def generate(net,
             bpe,
             texts,
             length=100,
             top_k=1):
    """Generate text after the given contexts.

    :param net: The trained net.
    :param bpe: Byte pair encoding object.
    :param texts: A list of texts.
    :param length: The length of following texts to be generated.
    :param top_k: Choose the next token from top K.
    :return: A list of generated texts.
    """
    batch_size = len(texts)
    encodes = [bpe.encode(text) for text in texts]
    text_lens = [len(encode) for encode in encodes]
    max_len = max(text_lens)
    input_data = [encode + [0] * (max_len - len(encode)) for encode in encodes]
    for shift in range(length):
        output_data = net(torch.LongTensor(input_data)).detach().numpy()
        for index in range(batch_size):
            probs = [(prob, i) for i, prob in enumerate(output_data[index, text_lens[index] + shift - 1])]
            probs.sort(reverse=True)
            probs = probs[:top_k]
            indices, probs = list(map(lambda x: x[1], probs)), list(map(lambda x: x[0], probs))
            scale = 1.0 / sum(probs)
            probs = [prob * scale for prob in probs]
            next_token = indices[torch.Tensor(probs).multinomial(1).numpy()[0]]
            input_data[index].append(0)
            input_data[index][text_lens[index] + shift] = next_token
    outputs = [bpe.decode(input_data[index][:text_lens[index] + length]) for index in range(batch_size)]
    return outputs
