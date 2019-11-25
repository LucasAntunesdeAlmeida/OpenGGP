# -*- coding: utf-8 -*-


import argparse
from argparse import RawTextHelpFormatter
from agent import *


def parseArguments():
    parser = argparse.ArgumentParser(description='Artificial intelligence program design to play more than one game successfully', formatter_class=RawTextHelpFormatter)
    parser.add_argument("-v", "--version", action='version', version='OpenGGP v2.0\nRepository Link => https://github.com/LucasAntunesdeAlmeida/OpenGGP')
    parser.add_argument("--train", action="store_const", const=True, help="Set train mode to on", default=False)
    parser.add_argument("--test", action="store_const", const=True, help="Set test mode to on", default=False)
    parser.add_argument("--render", action="store_const", const=True, help="Set render mode to on", default=False)
    parser.add_argument("--resume", action="store_const", const=True, help="Set resume mode to on", default=False)
    parser.add_argument("-g", "--gamename", type=str, required=True, help="Define game from OpenAI Gym")
    parser.add_argument("-f", "--filename", type=str, help="Set the restore file path")
    parser.add_argument("-m", "--memory", type=int, help="Set the size of the replay memory")

    return parser.parse_args()


if __name__ == '__main__':
    args = parseArguments()

    if args.filename:
        path = "{0}/{1}.h5".format('saves', args.filename)
    else:
        path = "{0}/{1}.h5".format('saves', args.gamename)

    env = gym.make(args.gamename)

    action_size = env.action_space.n
    ggp_agent = Agent()
    ggp_agent.set_game(args.gamename)
    ggp_agent.set_restore_file_path(path)
    ggp_agent.set_action_size(action_size)
    ggp_agent.set_render(args.render)

    if args.memory:
        ggp_agent.set_replay_memory(args.memory)

    if args.train:
        if args.resume:
            ggp_agent.set_resume(True)
        ggp_agent.train(env)
    elif args.test:
        ggp_agent.test(env)

    env.close()
