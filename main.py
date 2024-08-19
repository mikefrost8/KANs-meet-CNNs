import argparse
import yaml
import training
import training_KAN
import test

def main():
    parser = argparse.ArgumentParser(description='Run training or testing?.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='Mode to run the script in: "train" or "test".')
    args = parser.parse_args()

    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if args.mode == 'train':
        training_KAN.train(config)
        #training.train(config)
    elif args.mode == 'test':
        test.test()  # Assuming the test module has a function called 'test'

if __name__ == '__main__':
    main()
