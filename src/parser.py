import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the Cora dataset.
    The default hyperparameters give a good quality model without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run GWNN.")

    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/cora_edges.csv",
	                help = "Edge list csv.")

    parser.add_argument("--features-path",
                        nargs = "?",
                        default = "./input/cora_features.json",
	                help = "Feature json.")

    parser.add_argument("--target-path",
                        nargs = "?",
                        default = "./input/cora_target.csv",
	                help = "Target classes csv.")

    parser.add_argument("--log-path",
                        nargs = "?",
                        default = "./logs/cora_logs.json",
	                help = "Log json.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 200,
	                help = "Number of training epochs. Default is 200.")

    parser.add_argument("--filters",
                        type = int,
                        default = 16,
	                help = "Filters (neurons) in convolution. Default is 16.")

    parser.add_argument("--approximation-order",
                        type = int,
                        default = 3,
	                help = "Order of Chebyshev polynomial. Default is 3.")

    parser.add_argument("--test-size",
                        type = float,
                        default = 0.2,
	                help = "Ratio of training samples. Default is 0.2.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.5,
	                help = "Dropout probability. Default is 0.5.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--tolerance",
                        type = float,
                        default = 10**-4,
	                help = "Sparsification parameter. Default is 10^-4.")

    parser.add_argument("--scale",
                        type = float,
                        default = 1,
	                help = "Heat kernel scale length. Default is 1.0.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 5*10**-4,
	                help = "Adam weight decay. Default is 5*10^-4.")

    return parser.parse_args()
