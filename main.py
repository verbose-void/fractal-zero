from dataclasses import asdict
from this import d
from time import sleep
import gym

import torch
from fractal_zero.config import FractalZeroConfig
from fractal_zero.data.data_handler import DataHandler
from fractal_zero.fmc import FMC
from fractal_zero.fractal_zero import FractalZero

from fractal_zero.models.dynamics import FullyConnectedDynamicsModel
from fractal_zero.models.joint_model import JointModel
from fractal_zero.models.prediction import FullyConnectedPredictionModel
from fractal_zero.models.representation import FullyConnectedRepresentationModel
from fractal_zero.data.replay_buffer import ReplayBuffer
from fractal_zero.trainer import FractalZeroTrainer

import wandb
from tqdm import tqdm


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    embedding_size = 16
    out_features = 1

    train_every = 1
    train_batches = 2
    evaluate_every = 16

    representation_model = FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(
        env, embedding_size, out_features=out_features
    )
    prediction_model = FullyConnectedPredictionModel(env, embedding_size)
    joint_model = JointModel(representation_model, dynamics_model, prediction_model)

    config = FractalZeroConfig(
        env,
        joint_model,
        max_replay_buffer_size=512,
        num_games=5_000,
        max_game_steps=200,
        max_batch_size=128,
        unroll_steps=16,
        learning_rate=0.001,
        optimizer="SGD",
        num_walkers=64,
        balance=1.0,
        lookahead_steps=64,
        evaluation_lookahead_steps=64,
        wandb_config={"project": "fractal_zero_cartpole"},
    )

    # TODO: make this logic automatic in config somehow?
    config.joint_model = config.joint_model.to(config.device)

    data_handler = DataHandler(config)

    fractal_zero = FractalZero(config)

    trainer = FractalZeroTrainer(
        fractal_zero,
        data_handler,
    )

    for i in tqdm(range(config.num_games), desc="Playing games and training", total=config.num_games):
        fractal_zero.train()
        game_history = fractal_zero.play_game()
        data_handler.replay_buffer.append(game_history)

        if i % train_every == 0:
            for _ in range(train_batches):
                trainer.train_step()

        if i % evaluate_every == 0:
            # TODO: move into trainer?
            fractal_zero.eval()
            game_history = fractal_zero.play_game()

            if config.use_wandb:
                wandb.log(
                    {
                        "evaluation/episode_length": len(game_history),
                        "evaluation/cumulative_reward": sum(
                            game_history.environment_reward_signals
                        ),
                    },
                    commit=False,
                )
