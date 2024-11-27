# Copyright 2024 CTIC (Technological Center for Information and Communication).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import seaborn


def confusion_matrix_plot(confusion_matrix, labels, title='Confusion Matrix', save_path='', filename='confusion_matrix.png'):
    plt.figure(figsize=(8, 6))
    
    # Plotting the confusion matrix with heatmap
    seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=labels, yticklabels=labels, cbar=False)

    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path + filename, bbox_inches='tight')
    plt.show()