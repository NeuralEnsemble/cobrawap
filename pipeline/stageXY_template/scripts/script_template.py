import argparse


if __name__ == '__main__':
    CLI = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    CLI.add_argument("--PROFILE", nargs='?', type=str, required=True,
                     help="path to input data in neo format")
    args, unknown = CLI.parse_known_args()

    print(args.PROFILE)
    print(unknown)
