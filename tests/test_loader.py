import os
from unittest import TestCase
import torch
import numpy as np
from torch_gpt_2 import load_trained_model_from_checkpoint
from keras_gpt_2 import load_trained_model_from_checkpoint as load_keras_model


class TestLoader(TestCase):

    def test_load_from_checkpoint(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        toy_checkpoint_path = os.path.join(current_path, 'toy_checkpoint')
        config_path = os.path.join(toy_checkpoint_path, 'hparams.json')
        checkpoint_path = os.path.join(toy_checkpoint_path, 'model.ckpt')
        net = load_trained_model_from_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
        print(net)

    def test_load_from_checkpoint_shorter(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        toy_checkpoint_path = os.path.join(current_path, 'toy_checkpoint')
        config_path = os.path.join(toy_checkpoint_path, 'hparams.json')
        checkpoint_path = os.path.join(toy_checkpoint_path, 'model.ckpt')
        net = load_trained_model_from_checkpoint(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            seq_len=10,
        )
        print(net)

    def test_same(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        toy_checkpoint_path = os.path.join(current_path, 'toy_checkpoint')
        config_path = os.path.join(toy_checkpoint_path, 'hparams.json')
        checkpoint_path = os.path.join(toy_checkpoint_path, 'model.ckpt')
        torch_net = load_trained_model_from_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
        keras_net = load_keras_model(config_path=config_path, checkpoint_path=checkpoint_path)
        x = torch.randint(1, 10, (2, 13)).type(torch.LongTensor)
        torch_y = torch_net(x).detach().numpy()
        keras_y = keras_net.predict(x.numpy())
        self.assertTrue(np.allclose(keras_y, torch_y, rtol=0.0, atol=1e-4), (keras_y[1, 2, :20], torch_y[1, 2, :20]))
