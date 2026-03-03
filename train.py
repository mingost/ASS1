import numpy as np
import matplotlib.pyplot as plt
from game_2048 import Game2048, Direction
from dqn_model import DQNAgent


def train_agent(episodes=50, batch_size=64):
    """Train the DQN agent to play 2048"""
    game = Game2048()
    state_size = 16  # 4x4 board
    action_size = 4  # UP, DOWN, LEFT, RIGHT

    agent = DQNAgent(state_size, action_size, epsilon_decay=0.99)

    scores = []
    max_tiles = []
    avg_scores = []

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0
        moves = 0

        # Show progress for first episode
        if episode == 0:
            print("Episode 1 started...")

        while not game.game_over:
            action = agent.act(state)
            direction = Direction(action)
            next_state, reward, done = game.move(direction)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            moves += 1

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)

        scores.append(game.score)
        max_tiles.append(game.get_max_tile())

        # Update target network every 5 episodes
        if episode % 5 == 0:
            agent.update_target_model()

        # Calculate average score
        avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
        avg_scores.append(avg_score)

        # Print progress every 5 episodes for better feedback
        if (episode + 1) % 5 == 0:
            print(f"Episode: {episode + 1}/{episodes}, Score: {game.score}, "
                  f"Max Tile: {game.get_max_tile()}, Moves: {moves}, "
                  f"Epsilon: {agent.epsilon:.3f}, Avg Score: {avg_score:.1f}")
        elif episode < 3:
            # Print first 3 episodes to show it's working
            print(f"Episode: {episode + 1}/{episodes}, Score: {game.score}, Max Tile: {game.get_max_tile()}")

        # Save model every 50 episodes
        if (episode + 1) % 50 == 0:
            agent.save(f'model_2048_episode_{episode + 1}.pth')
            print(f"Model saved at episode {episode + 1}")

    # Save final model
    agent.save('model_2048_final.pth')

    # Plot training results
    plot_training_results(scores, max_tiles, avg_scores)

    return agent, scores, max_tiles


def plot_training_results(scores, max_tiles, avg_scores):
    """Plot training statistics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot scores
    axes[0].plot(scores)
    axes[0].plot(avg_scores, 'r-', linewidth=2)
    axes[0].set_title('Training Scores')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].legend(['Score', 'Average Score'])
    axes[0].grid(True)

    # Plot max tiles
    axes[1].plot(max_tiles)
    axes[1].set_title('Maximum Tile Achieved')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Max Tile Value')
    axes[1].grid(True)

    # Plot max tile distribution
    unique, counts = np.unique(max_tiles, return_counts=True)
    axes[2].bar(range(len(unique)), counts)
    axes[2].set_xticks(range(len(unique)))
    axes[2].set_xticklabels(unique)
    axes[2].set_title('Max Tile Distribution')
    axes[2].set_xlabel('Tile Value')
    axes[2].set_ylabel('Frequency')
    axes[2].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining completed!")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Max Tile Ever: {np.max(max_tiles)}")


if __name__ == "__main__":
    print("Starting 2048 Q-Learning Training...")
    print("Training with 50 episodes")
    agent = None
    try:
        agent, scores, max_tiles = train_agent(episodes=50)
        print("Training finished!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        if agent is not None:
            print("Saving current model...")
            agent.save('model_2048_interrupted.pth')
            print("Model saved as 'model_2048_interrupted.pth'")
        else:
            print("Training stopped too early, no model to save.")
