import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math_functions


def plot_confusion_matrix(df_confusion, title='Confusion matrix'):
    plt.matshow(df_confusion)
    plt.title(title)
    plt.colorbar()
    tick_marks_x = np.arange(len(df_confusion.columns))
    tick_marks_y = np.arange(len(df_confusion.index))
    plt.xticks(tick_marks_x, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks_y, df_confusion.index)
    plt.xlabel(df_confusion.columns.name)
    plt.ylabel(df_confusion.index.name)

    for (i, j), z in np.ndenumerate(df_confusion):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    plt.show()


def discrete_normalization(x):
    return max(min(round(x * 10) * 1.0 / 10.0, 1), 0.1)


def show_confusion_matrix(real_y, pred_y):
    real_res_y = pd.Series(
        np.array(list(map(discrete_normalization, np.reshape(real_y, (real_y.shape[0])))))
        , name="real"
    )
    pred_res_y = pd.Series(
        np.array(list(map(discrete_normalization, np.reshape(pred_y, (pred_y.shape[0])))))
        , name="pred"
    )

    print("outliners: ")
    print(np.array(list(
        filter(lambda x: not (0 <= x < 1.01),
               map(discrete_normalization, np.reshape(pred_y, (pred_y.shape[0])))
               )
    )))

    confusion = pd.crosstab(real_res_y, pred_res_y, normalize='index')

    plot_confusion_matrix(confusion)

