import utils
import train_planner


def validate_entry_point(args):
    agent, train_env, val_envs = train_planner.train_setup(args)[:3]
    #agent.load(args.model_prefix)

    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        agent.env = val_env
        # teacher_results = agent.test(
        #     use_dropout=False, feedback='teacher', allow_cheat=True,
        #     beam_size=1)
        # teacher_score_summary, _ = evaluator.score_results(teacher_results)
        # for metric,val in teacher_score_summary.items():
        #     print("{} {}\t{}".format(env_name, metric, val))

        # results = agent.test(
        #     use_dropout=False, feedback='argmax', beam_size=args.beam_size)

        results = agent.test(args=args,
            use_dropout=False, feedback='argmax',
            loss_type=args.loss_type,
            debug_interval=args.debug_interval,
            )
        score_summary, _ = evaluator.score_results(results)

        if args.eval_file:
            eval_file = "{}_{}.json".format(args.eval_file, env_name)
            eval_results = []
            for instr_id, result in results.items():
                eval_results.append(
                    {'instr_id': instr_id, 'trajectory': result['trajectory']})
            with open(eval_file, 'w') as f:
                utils.pretty_json_dump(eval_results, f)

        # TODO: testing code, remove
        # score_summary_direct, _ = evaluator.score_results(agent.results)
        # assert score_summary == score_summary_direct

        for metric, val in sorted(score_summary.items()):
            print("{} {}\t{}".format(env_name, metric, val))


def make_arg_parser():
    parser = train_planner.make_arg_parser()
    #parser.add_argument("model_prefix")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--eval_file", required=True)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
