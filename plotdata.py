import numpy as np
import matplotlib.pyplot as plt

# Accuracy data
classifiers = ['Linear', 'RBF', 'Logistic', 'SGD']
comprehension_acc = [0.48, 0.36, 0.45, 0.43]
production_acc     = [0.68, 0.50, 0.50, 0.50]

# Positions of the bars on the x-axis
x = np.arange(len(classifiers))
width = 0.35  # width of each bar



# Plot bars
plt.bar(x - width/2, comprehension_acc, width, label='Comprehension')
plt.bar(x + width/2, production_acc,     width, label='Production')

# Labeling
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.xticks(x, classifiers)

# Set y-axis from 0 to 1, with ticks at every 0.1
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1.1, 0.1))

plt.legend()
plt.show()
