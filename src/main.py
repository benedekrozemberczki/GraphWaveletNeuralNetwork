from gwnn import GWNNTrainer
from utils import WaveletSparsifier
from parser import parameter_parser
from utils import tab_printer, graph_reader, feature_reader, target_reader, save_logs

def main():
    """
    Parsing command line parameters, reading data, doing sparsification, fitting a GWNN and saving the logs.
    """
    args = parameter_parser()
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)
    sparsifier = WaveletSparsifier(graph, args.scale, args.approximation_order, args.tolerance)
    sparsifier.calculate_all_wavelets()
    trainer = GWNNTrainer(args, sparsifier, features, target)
    trainer.fit()
    trainer.score()
    save_logs(args, trainer.logs)
    

if __name__ == "__main__":
    main()
