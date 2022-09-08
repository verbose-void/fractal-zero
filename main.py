from this import d
from time import sleep
import gym

import torch
from data.data_handler import DataHandler
from fmc import FMC
from fractal_zero import FractalZero

from models.dynamics import FullyConnectedDynamicsModel
from models.joint_model import JointModel
from models.prediction import FullyConnectedPredictionModel
from models.representation import FullyConnectedRepresentationModel
from data.replay_buffer import ReplayBuffer
from trainer import Trainer

import wandb
from tqdm import tqdm


if __name__ == "__main__":
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")

    env = gym.make("CartPole-v0")

    max_replay_buffer_size = 128
    embedding_size = 16
    out_features = 1

    num_games = 512
    batch_size = 128
    train_every = 1
    train_batches = 1
    evaluate_every = 1

    max_steps = 512
    num_walkers = 128
    lookahead_steps = 8

    use_wandb = True

    representation_model = FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(
        env, embedding_size, out_features=out_features
    )
    prediction_model = FullyConnectedPredictionModel(env, embedding_size)
    joint_model = JointModel(representation_model, dynamics_model, prediction_model).to(device)

    replay_buffer = ReplayBuffer(max_replay_buffer_size)
    data_handler = DataHandler(env, replay_buffer, device=device, batch_size=batch_size)
    trainer = Trainer(data_handler, joint_model, use_wandb=use_wandb)
    fractal_zero = FractalZero(env, joint_model)  # TODO: move into Trainer?

    for i in tqdm(range(num_games), desc="Playing games and training", total=num_games):
        game_history = fractal_zero.play_game(max_steps, num_walkers, lookahead_steps)
        replay_buffer.append(game_history)

        if i % train_every == 0:
            for _ in range(train_batches):
                trainer.train_step()

        if i % evaluate_every == 0:
            # TODO: move into trainer?
            game_history = fractal_zero.play_game(max_steps, num_walkers, lookahead_steps, render=False)
            if use_wandb:
                wandb.log({
                    "evaluation_episode_length": len(game_history),
                    "evaluation_cumulative_reward": sum(game_history.environment_reward_signals),
                })