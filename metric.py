import numpy as np


class ConfusionMatrix:
    """Streaming interface to allow for any source of predictions. Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""

    def __init__(self, number_of_labels=2):
        self.number_of_labels = number_of_labels
        self.confusion_matrix = np.zeros(shape=(self.number_of_labels, self.number_of_labels))

    def count_predicted(self, ground_truth, predicted, number_of_added_elements=1):
        self.confusion_matrix[ground_truth, predicted] += number_of_added_elements

    """labels are integers from 0 to number_of_labels-1"""

    def get_count(self, ground_truth_index, predicted_index):
        return self.confusion_matrix[ground_truth_index][predicted_index]

    """returns list of lists of integers; use it as result[ground_truth][predicted]
       to know how many samples of class ground_truth were reported as class predicted"""

    def get_confusion_matrix(self):
        return self.confusion_matrix

    """returns list of 64-bit floats"""

    def get_intersection_union_per_class(self):
        matrix_diagonal = [self.confusion_matrix[i][i] for i in range(self.number_of_labels)]
        errors_summed_by_row = [0] * self.number_of_labels

        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_row[row] += self.confusion_matrix[row][column]
        errors_summed_by_column = [0] * self.number_of_labels
        for column in range(self.number_of_labels):
            for row in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_column[column] += self.confusion_matrix[row][column]

        divisor = [0] * self.number_of_labels
        for i in range(self.number_of_labels):
            divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
            if matrix_diagonal[i] == 0:
                divisor[i] = 1

        return [float(matrix_diagonal[i]) / divisor[i] for i in range(self.number_of_labels)]

    """returns 64-bit float"""

    def get_overall_accuracy(self):
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                all_values += self.confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += self.confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        return float(matrix_diagonal) / all_values

    def get_average_intersection_union(self):
        values = self.get_intersection_union_per_class()
        return sum(values) / len(values)