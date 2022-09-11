import torch

from fractal_zero.fractal_zero import FractalZero


if __name__ == "__main__":
    # TODO: arg parser
    trainer = torch.load("checkpoints/astral-meadow-111.checkpoint")
    fractal_zero: FractalZero = trainer.fractal_zero
    
    fractal_zero.eval()
    fractal_zero.play_game(render=True)