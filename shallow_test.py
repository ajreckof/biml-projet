from Projet import ShallowNetwork, grid_search_shallow
import csv

if __name__ == '__main__':
    results = grid_search_shallow(ShallowNetwork, [10, 50, 100, 200, 500, 1000], [0.01, 0.001, 0.0001])
    f = open('./results_shallow.csv', 'w')

    # create the csv writer
    writer = csv.writer(f)

    # write column names
    writer.writerow(["learning_rate", "number_of_neurones", "result"])

    # write data to the csv file
    writer.writerows(results)

    # close the file
    f.close()