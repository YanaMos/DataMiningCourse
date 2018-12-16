import xgboost as xgb
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

label = {'ALL': 0,
         'AML': 1,
         999: 999}

BM_PB = {'BM': 1,
         'PB': 2,
         999: 999}

TB_if_ALL = {'B-cell': 1,
             'T-cell': 2,
             999: 999}

FAB_if_AML = {'M1': 1,
              'M2': 2,
              'M4': 3,
              'M5': 4,
              999: 999}

Gender = {'F': 1,
          'M': 2,
          999: 999}

Treatment_Response = {'Failure': 1,
                      'Success': 2,
                      999: 999}

Source = {'DFCI': 1,
          'CALGB': 2,
          'St-Jude': 3,
          'CCG': 4,
          999: 999}


def model(data, label, feature_name):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)

    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_name, missing=-999)
    dtest = xgb.DMatrix(X_test, y_test, feature_names=feature_name, missing=-999)

    print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))
    print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))

    # 'scale_pos_weight' - for unbalanced dataset (calculated as negative / positive)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'max_delta_step': 1,
        'scale_pos_weight': 3
    }

    num_rounds = 10
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_rounds, evals=evallist)

    return model, dtest


def predict(model, dtest):
    predict = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    return predict


def predict_info(model, predict, dtest):
    predicted_labels = predict > 0.5

    print('Accuracy: {0:.2f}'.format(accuracy_score(dtest.get_label(), predicted_labels)))
    print('Precision: {0:.2f}'.format(precision_score(dtest.get_label(), predicted_labels)))
    print('Recall: {0:.2f}'.format(recall_score(dtest.get_label(), predicted_labels)))
    print('F1: {0:.2f}'.format(f1_score(dtest.get_label(), predicted_labels)))

    importances = model.get_fscore()
    print(importances)

    xgb.plot_importance(model)
    plt.savefig('../DataMiningCourse/feature_importances')


def classification():
    with open('genes-leukemia.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)

    dataset = []
    labels = []
    feature_name = []
    for id, row in enumerate(data):
        if id == 0:
            for id_row, value in enumerate(row):
                if id_row == 2 or id_row > 4:
                    feature_name.append(value)

        else:
            line = []
            for id_row, value in enumerate(row):
                if not value:
                    value = 999

                if id_row == 0:
                    continue
                if id_row == 1:
                    labels.append(int(label[value]))
                    continue
                if id_row == 2:
                    line.append(int(BM_PB[value]))
                    continue
                if id_row == 3:
                    # line.append(int(TB_if_ALL[value]))
                    continue
                if id_row == 4:
                    # line.append(int(FAB_if_AML[value]))
                    continue
                if id_row == 6:
                    line.append(int(Gender[value]))
                    continue
                if id_row == 8:
                    line.append(int(Treatment_Response[value]))
                    continue
                if id_row == 10:
                    line.append(int(Source[value]))
                    continue

                else:
                    line.append(int(float(value)))

            dataset.append(line)

    modelXb, dtest = model(dataset, labels, feature_name)
    predictXb = predict(modelXb, dtest)
    predict_info(modelXb, predictXb, dtest)


classification()
