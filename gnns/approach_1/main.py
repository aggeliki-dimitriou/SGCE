import config
from trainer import Trainer


def main():
    print("----- STARTING -----")
    args = config
    trainer = Trainer(args)
    if args.LOAD_PATH:
        trainer.load()
    else:
        trainer.fit()

    if args.SIM:
        trainer.find_similarity()
    elif args.TEST_GRAPH_PATH:
        trainer.test()
    else:
        if args.SAVE_PATH:
            trainer.save()

        trainer.predict()

    print("-----  THE END  -----")


if __name__ == "__main__":
    main()
