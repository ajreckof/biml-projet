from Projet import ShallowNetwork, grid_search_shallow, write_results_to_csv

if __name__ == '__main__':
    model, accuracy, results = grid_search_shallow(ShallowNetwork, [10, 50, 100, 200, 500, 1000], [0.01, 0.001, 0.0001])
    print(accuracy)
    write_results_to_csv(
        results,
        "results_shallow.csv",
        ["learning_rate", "number_of_neurones", "result", "time"],
    )