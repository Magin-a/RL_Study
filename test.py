from __future__ import unicode_literals

import gym
import gym_gridworld
import numpy as np
import matplotlib.pyplot as plt

# 환경 생성
env = gym.make('gridworld-v0')

# 최대 에피소드 수 설정
max_episodes = 500
total_rewards = []
success_count = 0
success_target = env.target_state  # 목표 상태 (초기 설정)

# 학습 파라미터 설정
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

# Q-테이블 초기화
q_table = np.zeros((env.grid_shape[0], env.grid_shape[1], env.action_space.n))

# 학습 루프
for episode in range(max_episodes):
    observation = env.reset()  # 초기 상태를 관찰
    state = env.agent_state  # 초기 상태 좌표 (x, y)
    done = False
    total_reward = 0

    while not done:
        env.render(episode=episode + 1, max_episodes=max_episodes)

        # 탐험 또는 활용 결정
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # 탐험: 무작위 행동 선택
        else:
            action = np.argmax(q_table[state[0], state[1]])  # 활용: Q-값이 가장 높은 행동 선택

        # 행동 수행 및 새로운 상태, 보상 획득
        next_observation, reward, done, info = env.step(action)
        next_state = env.agent_state  # 다음 상태 좌표 (x, y)

        # Q-테이블 업데이트
        best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] += learning_rate * (
            reward + discount_factor * q_table[next_state[0], next_state[1], best_next_action] - q_table[state[0], state[1], action])

        state = next_state  # 현재 상태 업데이트
        total_reward += reward  # 총 보상 업데이트

    total_rewards.append(total_reward)  # 에피소드 보상 기록

    # 목표 도달 여부 확인
    if state == success_target:
        success_count += 1

np.save('q_table_v0.npy', q_table)

# 환경 닫기
env.close()

# 결과 출력
print(f"Total episodes: {max_episodes}")
print(f"Success count: {success_count} / {max_episodes}")
print(f"Average reward per episode: {sum(total_rewards) / max_episodes:.2f}")

# 학습된 경로 시각화 함수
def visualize_path(env, q_table):
    observation = env.reset()  # 초기 상태 관찰
    done = False
    path = []  # 경로 저장용 리스트
    env.render()  # 초기 상태 렌더링
    env.render(episode=max_episodes, max_episodes=max_episodes)

    # 목표 도달을 위해 루프 수행
    while not done:
        state = env.agent_state

        # Q-table에서 현재 상태에 대한 Q값들 출력
        q_values = q_table[state[0], state[1]]

        # 최적의 행동 선택
        action = np.argmax(q_values)

        # 선택된 행동을 기반으로 환경 업데이트
        next_observation, _, done, _ = env.step(action)
        new_state = env.agent_state

        # 상태 업데이트 후 경로에 추가
        path.append(new_state)

        # 목표 상태에 도달했는지 확인
        if done:
            print("Goal reached!")
            break

        # 상태가 더 이상 변경되지 않으면 중단 (무한 루프 방지)
        if new_state == state:
            print("Agent is stuck, stopping visualization.")
            break
        
        env.render(episode=max_episodes, max_episodes=max_episodes) 

    # 최종 경로 시각화
    env.result_render(path) 


# 학습된 최적 경로 시각화
visualize_path(env, q_table)

