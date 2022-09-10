from this import d
from time import sleep
import gym

import torch
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")

    env = gym.make("CartPole-v0")

    max_replay_buffer_size = 512
    embedding_size = 16
    out_features = 1

    num_games = 5_000
    train_every = 1
    train_batches = 2
    evaluate_every = 16
    max_batch_size = 128
    learning_rate = 0.001

    max_steps = 200
    num_walkers = 64
    balance = 1

    lookahead_steps = 64
    evaluation_lookahead_steps = 64
    unroll_steps = 16

    use_wandb = False

    representation_model = FullyConnectedRepresentationModel(env, embedding_size)
    dynamics_model = FullyConnectedDynamicsModel(
        env, embedding_size, out_features=out_features
    )
    prediction_model = FullyConnectedPredictionModel(env, embedding_size)
    joint_model = JointModel(representation_model, dynamics_model, prediction_model).to(
        device
    )

    replay_buffer = ReplayBuffer(max_replay_buffer_size)
    data_handler = DataHandler(env, replay_buffer, device=device, max_batch_size=max_batch_size)

    fractal_zero = FractalZero(env, joint_model, balance=balance)
    trainer = FractalZeroTrainer(
        fractal_zero,
        data_handler,
        unroll_steps=unroll_steps,
        learning_rate=learning_rate,
        use_wandb=use_wandb,
    )

    for i in tqdm(range(num_games), desc="Playing games and training", total=num_games):
        fractal_zero.train()
        game_history = fractal_zero.play_game(
            max_steps, num_walkers, lookahead_steps, use_wandb_for_fmc=use_wandb
        )
        replay_buffer.append(game_history)

        if i % train_every == 0:
            for _ in range(train_batches):
                trainer.train_step()

        if i % evaluate_every == 0:
            # TODO: move into trainer?
            fractal_zero.eval()
            game_history = fractal_zero.play_game(
                max_steps, num_walkers, evaluation_lookahead_steps, render=False
            )
            if use_wandb:
                wandb.log(
                    {
                        "evaluation/episode_length": len(game_history),
                        "evaluation/cumulative_reward": sum(
                            game_history.environment_reward_signals
                        ),
                    },
                    commit=False,
                )
