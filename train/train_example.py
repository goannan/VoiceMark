from speechtokenizer import SpeechTokenizer
from speechtokenizer.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
import argparse

from train import WMTrainer



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, help='Config file path')
    parser.add_argument('--continue_train', action='store_true', help='Continue to train from checkpoints')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    generator = SpeechTokenizer(cfg)
    discriminators = {'mpd': MultiPeriodDiscriminator(), 'msd': MultiScaleDiscriminator(),
                      'mstftd': MultiScaleSTFTDiscriminator(32)}

    trainer = WMTrainer(embedder=generator,
                        discriminators=discriminators,
                        cfg=cfg)

    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()