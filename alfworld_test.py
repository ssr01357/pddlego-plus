#!/usr/bin/env python

import os
import json
import glob
import random
import argparse
from os.path import join as pjoin

import textworld
import textworld.gym

from alfworld.info import ALFWORLD_DATA
from alfworld.agents.utils.misc import add_task_to_grammar
from alfworld.agents.environment.alfred_tw_env import AlfredExpert, AlfredDemangler, AlfredExpertType


def main(args):
    print(123)
    outputs = []

    outputs.append(f"Playing '{args.problem}'.")
    GAME_LOGIC = {
        "pddl_domain": open(args.domain).read(),
        "grammar": open(args.grammar).read(),
    }

    # load state and trajectory files
    pddl_file = os.path.join(args.problem, 'initial_state.pddl')
    json_file = os.path.join(args.problem, 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)
    GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)

    # dump game file
    gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
    json.dump(gamedata, open(gamefile, "w"))

    # expert = AlfredExpert(expert_type=AlfredExpertType.PLANNER)
    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        score=True,
        max_score=True,
        intermediate_reward=True,
        extras=["expert_plan"]
    )
    env_id = textworld.gym.register_game(
        gamefile,
        request_infos,
        max_episode_steps=1000000,
        wrappers=[AlfredDemangler(), expert]
    )
    env = textworld.gym.make(env_id)

    print(234)
    # reset环境
    obs, infos = env.reset()
    # print(obs.split('\n'))
    init_obs = obs.split('\n')[2]
    goal = obs.split('\n')[-1]
    valid_actions = infos["admissible_commands"]
    outputs.append(f"Initial observation:\n{init_obs}")
    outputs.append(f"\nGoal: {goal}")
    outputs.append(f"\nValid actions: {valid_actions}")
    outputs.append(f"\ninfos keys: {infos.keys()}")

    # 这里直接使用wrapper提供的expert_plan，不再人工输入
    # expert_plan = infos.extras.get("expert_plan", [])
    expert_plan = ['hello', 'go to diningtable 1']

    # 如果expert_plan非空，就一步步执行
    for i, action in enumerate(expert_plan):
        outputs.append(f"--- Step {i+1} ---")
        outputs.append(f"Action: {action}")
        obs, score, done, infos = env.step(action)
        outputs.append(f"Observation:\n{obs}")
        outputs.append(f"Score: {score}")

        if done:
            outputs.append("You won!")
            break

    # 如果expert_plan为空或者没能在plan里完成，也可能会在这里结束
    # 可以根据需要再做一些判断或提示

    # 最后统一打印所有输出
    print("\n".join(str(line) for line in outputs))


if __name__ == "__main__":
    description = "Play the abstract text version of an ALFRED environment automatically (no manual input)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("problem", nargs="?", default=None,
                        help="Path to a folder containing PDDL and traj_data files."
                             f"Default: pick one at random found in {ALFWORLD_DATA}")
    parser.add_argument("--domain",
                        default=pjoin(ALFWORLD_DATA, "logic", "alfred.pddl"),
                        help="Path to a PDDL file describing the domain."
                             " Default: `%(default)s`.")
    parser.add_argument("--grammar",
                        default=pjoin(ALFWORLD_DATA, "logic", "alfred.twl2"),
                        help="Path to a TWL2 file defining the grammar used to generate text feedbacks."
                             " Default: `%(default)s`.")
    args = parser.parse_args()
    print('after args')

    if args.problem is None:
        problems = glob.glob(pjoin(ALFWORLD_DATA, "**", "initial_state.pddl"), recursive=True)

        # Remove problem which contains movable receptacles.
        problems = [p for p in problems if "movable_recep" not in p]

        if len(problems) == 0:
            raise ValueError(f"Can't find problem files in {ALFWORLD_DATA}. Did you run alfworld-data?")

        args.problem = os.path.dirname(random.choice(problems))

    if "movable_recep" in args.problem:
        raise ValueError("This problem contains movable receptacles, which is not supported by ALFWorld.")

    main(args)
