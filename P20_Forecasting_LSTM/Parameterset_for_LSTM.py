from functools import total_ordering


@total_ordering
class ParameterSetLSTM:
    # unidades, dropout, lrate, epocas, batch,
    min_max_values = [
        (100, 400),  # Unidades LSTM 1            --- 0
        (100, 400),  # Unidades LSTM 2            --- 1
        (100, 400),  # Unidades LSTM 3            --- 2
        (10, 50),  # Epocas                       --- 3
        [1, 20],  # Batch Size                    --- 4
        (0.0001, 0.01),  # Learning rate          --- 5
        (0.2, 0.5),  # Dropout Capa 1             --- 6
        (0.2, 0.5),  # Dropout Capa 2             --- 7
        (0.2, 0.5),  # Dropout Capa 3             --- 8
    ]

    def __init__(self, init_parameters, init_rmse=0, name=''):
        self.parameters = init_parameters
        self.rmse = init_rmse
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, ParameterSetLSTM):
            return NotImplemented
        return self.rmse == other.rmse

    def __lt__(self, other):
        if not isinstance(other, ParameterSetLSTM):
            return NotImplemented

        return self.rmse < other.rmse



    def __str__(self):
        return f'Parameters: {str(self.parameters)} - RMSE: {str(self.rmse)}'

    def __repr__(self):
        return f'(Parameters: {str(self.parameters)} - RMSE: {str(self.rmse)})'

if __name__ == '__main__':
    print('Test: ')

