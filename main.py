
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import tensorflow as tf

training_data = pd.read_csv('./train.csv')

Z_training = training_data.drop(columns=["Survived"])

a_training = training_data.drop(columns=[
                                 "PassengerId",
                                 "Pclass",
                                 "Name",
                                 "Sex",
                                 "Age",
                                 "SibSp",
                                 "Parch",
                                 "Fare",
                                 "Embarked",
                                 "Ticket",
                                 "Cabin"
                                ])

test_data = pd.read_csv('./test.csv')
X_test = test_data
training_data[training_data["Name"] == "Allison, Miss. Helen Loraine"]
print("The average age of the training set is: {:.1f}".format(Z_training['Age'].mean()))
print("The median age of the training set is: {:.1f}".format(Z_training['Age'].median()))


sex_pivot = training_data.pivot_table(index="Sex", values="Survived")
sex_pivot.plot.bar()
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()

class_pivot = training_data.pivot_table(index="Pclass", values="Survived")
class_pivot.plot.bar()
plt.title("Survival Rate by Ticket Class")
plt.ylabel("Survival Rate")
plt.xlabel("Ticket Class")
plt.show()


training_data["UpperClass"] = np.where(training_data["Pclass"] <= 2, 1, 0)

upperclass_pivot = training_data.pivot_table(index="UpperClass", values="Survived")
upperclass_pivot.plot.bar()
plt.show()

Z_training.count()

X_test.count()
embarked_mode = Z_training['Embarked'].mode()[0]
age_median = Z_training['Age'].median()

Z_training['Embarked'] = Z_training['Embarked'].fillna(embarked_mode)
Z_training['Age'] = Z_training['Age'].fillna(age_median)

test_age_median = X_test['Age'].median()
test_fare_median = X_test['Fare'].median()

X_test['Age'] = X_test['Age'].fillna(test_age_median)
X_test['Fare'] = X_test['Fare'].fillna(test_fare_median)

Z_training['Embarked'] = pd.factorize(Z_training['Embarked'])[0]
Z_training['Sex'] = pd.factorize(Z_training['Sex'])[0]

X_test['Embarked'] = pd.factorize(X_test['Embarked'])[0]
X_test['Sex'] = pd.factorize(X_test['Sex'])[0]


def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df


Z_training = create_dummies(Z_training, "Pclass")
Z_training = create_dummies(Z_training, "Sex")
Z_training = create_dummies(Z_training, "Embarked")
Z_training.head()

X_test = create_dummies(X_test, "Pclass")
X_test = create_dummies(X_test, "Sex")
X_test = create_dummies(X_test, "Embarked")
X_test.head()


a_training = create_dummies(a_training, "Survived")
a_training.head()

Z_training = Z_training.drop(columns=["Name", "Ticket", "Cabin", "PassengerId", "Pclass", "Sex", "Embarked"])
Z_training.head()

X_test = X_test.drop(columns=["Name", "Ticket", "Cabin", "PassengerId","Pclass","Sex","Embarked"])
X_test.head()

a_training = a_training.drop(columns="Survived")


X_train_normalized = normalize(Z_training, 'l2', 0).T
X_test_normalized = normalize(X_test, 'l2', 0).T


X = tf.placeholder(tf.float64, [12, None], "X")
Y = tf.placeholder(tf.float64, [2, None], "Y")

initializer = tf.contrib.layers.xavier_initializer(seed=1, dtype=tf.float64)
num_layer1_hidden_units = 9
num_layer2_hidden_units = 5
num_layer_3_hidden_units = 9

W1 = tf.Variable(initializer([num_layer1_hidden_units, 12]), dtype=tf.float64, name="W1")
b1 = tf.Variable(tf.zeros([num_layer1_hidden_units, 1], dtype=tf.float64), name="b1")

W2 = tf.Variable(initializer([num_layer2_hidden_units, num_layer1_hidden_units]), dtype=tf.float64, name="W2")
b2 = tf.Variable(tf.zeros([num_layer2_hidden_units, 1], dtype=tf.float64), name="b2")

W3 = tf.Variable(initializer([num_layer_3_hidden_units, num_layer2_hidden_units]), dtype=tf.float64, name="W3")
b3 = tf.Variable(tf.zeros([num_layer_3_hidden_units, 1], dtype=tf.float64), name="b3")

W4 = tf.Variable(initializer([2, num_layer_3_hidden_units]), dtype=tf.float64, name="W4")
b4 = tf.Variable(tf.zeros([2, 1], dtype=tf.float64), name="b4")

print("W1 = " + str(W1))
print("W2 = " + str(W2))
print("W3 = " + str(W3))
print("W4 = " + str(W4))

Z1 = tf.add(tf.matmul(W1, X), b1)
A1 = tf.nn.relu(Z1)

Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.nn.relu(Z2)

Z3 = tf.add(tf.matmul(W3, A2), b3)
A3 = tf.nn.relu(Z3)

Z4 = tf.add(tf.matmul(W4, A3), b4)
A4 = tf.nn.sigmoid(Z4)

logits = tf.transpose(A4)
labels = tf.transpose(Y)

learning_rate = 0.01
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

tf.summary.histogram('cost_function', cost)

correct_prediction = tf.equal(tf.argmax(A4), tf.argmax(Y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

tf.summary.histogram('accuracy', accuracy)



merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

num_epochs = 20001
with tf.Session() as sess:

    sess.run(init)

    train_writer = tf.summary.FileWriter('./train-logs', sess.graph)

    for epoch in range(num_epochs):
        summary, _, e_cost = sess.run([merged, optimizer, cost], {X: X_train_normalized, Y: a_training.T})
        train_writer.add_summary(summary, epoch)
        if epoch % 10000 == 0:
            print("Cost after epoch %i: %f" % (epoch, e_cost))

    print("Train Accuracy:", accuracy.eval({X: X_train_normalized, Y: a_training.T}))

    # Zapisanie predictions
    predictions = np.argmax(A4.eval({X: X_test_normalized}), axis=0)

    # Zapisanie do pliku
    csv_data = pd.DataFrame()
    csv_data["PassengerId"] = test_data["PassengerId"]
    csv_data["Survived"] = predictions
    np.savetxt('predictions.csv', csv_data, fmt="%1.0f", header="PassengerId,Survived", delimiter=",", comments='')

    # Autor_1: lukasz_jamrozik.62455
    # Autor_2: tomasz_matyla.62502