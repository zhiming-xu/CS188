import json
import argparse
import glob

import pacman


# each evaluation case is a 3-tuple of
# 1. layout (-l in pacman.py)
# 2. agent number (-s in pacman.py)
# 3. weight (used for weight averaging the scores)
# evaluation_cases = [
#     # TODO specify the evaluation cases and their score weights
#     ('mediumMaze', 2, 1., 0.1),
#     ('mediumMaze', 2, 1., 0.2),
#     ('mediumMaze', 2, 1., 0.3),
#     ('mediumMaze', 4, 1., 0.1),
#     ('mediumMaze', 4, 1., 0.2),
#     ('mediumMaze', 4, 1., 0.3),
#     ('bigMaze', 4, 1., 0.2),
#     ('bigMaze', 4, 1., 0.3),
#     ('mediumCorners', 3, 1., 0.2),
#     ('tinyMaze', 4, 1., 0),
#     ('mediumCorners', 4, 1., 0),
#     ('mediumMaze', 1, 1., 0)
# ]


def get_test_cases():
    tests = glob.glob('layouts/*.lay')
    return [(t, 1.) for i, t in enumerate(tests)] # TODO add weighted?

evaluation_cases = get_test_cases()

def main(pacman_agent):
    total_scores = 0.
    total_weight = 0.
    for n_case, case in enumerate(evaluation_cases):
        print('running game {} / {}, {}'.format(n_case+1, len(evaluation_cases), case[0]))
        layout, score_weight = case

        # Run the pacman game and get its score
        pacman_cmd = 'python pacman.py --pacman {} -l {} -q'
        pacman_cmd_args = pacman_cmd.format(pacman_agent, layout)
        # skip 'python pacman.py' in the command line arguments above
        args = pacman.readCommand(pacman_cmd_args.split()[2:])
        games = pacman.runGames(**args)
        # Take the average of the game scores. Note that there should be only
        # one game in games, unless `-n` is used in pacman.py
        scores = [game.state.getScore() for game in games]
        game_score = sum(scores) / len(scores)

        total_scores += game_score * score_weight
        total_weight += score_weight

    final_score = total_scores / total_weight
    print("Final score: ", final_score)

    # Generate results.json
#    score_fields = {}
#    score_fields['score'] = final_score
#    score_fields['visibility'] = 'visible'
#    score_fields['leaderboard'] = [{"name": "Score", "value": final_score}]
#    score_fields['output'] = 'successfully run {} games!'.format(
#        len(evaluation_cases))
#    write_output(score_fields)


def write_output(score_fields):
    with open('/autograder/results/results.json', 'w') as output_file:
        json.dump(score_fields, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pacman', default='myAgents.py')
    args = parser.parse_args()
    main(args.pacman)
