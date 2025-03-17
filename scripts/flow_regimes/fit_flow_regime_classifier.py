"""
Created 13 December 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


"""
Classify flow regime based on the following hierarchy:

If C1 and C2:
    then annular flow
else if C3 and C4:
    then slug/churn flow
else:
    bubbly flow

where the conditions are
C1: v_gs > 3.1 * (g * sigma * (rho_l - rho_g) / rho_g ** 2) ** (1/4)
C2: alpha >= 0.7
C3: v_gs > 1.08 * v_ls
C4: alpha >= 0.25

This is a simplified version of the hierarchy in the paper "Simplified two-phase flow modeling in wellbores" by 
Hasan et al. (2010). One simplification is that slug and churn flow has been combined into one class.
"""


def create_feature_dict(v_gs, v_ls, alpha):
    rho_g = 1
    rho_l = 900
    g = 9.81
    sigma = 0.033  # Really a function of the densities and temperature
    c1_const = 3.1 * (g * sigma * (rho_l - rho_g) / rho_g ** 2) ** (1 / 4)

    feature_dict = {
        'c1': np.tanh(v_gs - c1_const),
        'c2': np.tanh((alpha - 0.7) * 2),  # Multiplied to increase sensitivity
        'c3': np.tanh(v_gs - 1.08 * v_ls),
        'c4': np.tanh((alpha - 0.25) * 2),  # Multiplied to increase sensitivity
    }

    return feature_dict


def create_class_dict(feature_dict):
    c1 = feature_dict['c1']
    c2 = feature_dict['c2']
    c3 = feature_dict['c3']
    c4 = feature_dict['c4']

    annular = (c1 > 0) & (c2 > 0)
    slug = ~annular & (c3 > 0) & (c4 > 0)
    bubbly = ~(annular | slug)

    # Do some checks here
    print(sum(annular | slug | bubbly))
    print(sum(annular & slug))
    print(sum(annular & bubbly))
    print(sum(slug & bubbly))

    class_array = np.zeros(c1.size, dtype=int)
    class_array[annular] = 0  # Not needed, but added for completeness
    class_array[slug] = 1
    class_array[bubbly] = 2
    print(class_array)

    # One-hot encoded
    class_dict = {
        'annular': annular,
        'slug': slug,
        'bubbly': bubbly,
    }

    # Encoded as integer
    class_dict_v2 = {
        'regime': class_array
    }

    return class_dict, class_dict_v2


np.random.seed(1234)

# Create dataset
n = int(2e5)  # Number of datapoints

alpha = np.random.uniform(0, 1, n)
v_gs = np.random.uniform(0, 40, n)
v_ls = np.random.uniform(0, 40, n)

feature_dict = create_feature_dict(v_gs, v_ls, alpha)
print(feature_dict['c1'].shape)

class_dict, class_dict_v2 = create_class_dict(feature_dict)
class_array = class_dict_v2['regime']

all_dict = {**feature_dict, **class_dict, **class_dict_v2}
df = pd.DataFrame(all_dict)
x_cols = list(feature_dict.keys())
y_cols = list(class_dict.keys())
class_col = list(class_dict_v2.keys())

# lr = LogisticRegression(multi_class='multinomial', penalty='l1', C=1., solver='saga', fit_intercept=False)
lr = LogisticRegression(multi_class='multinomial', penalty='l2', C=0.01, solver='lbfgs', fit_intercept=True, max_iter=200)
lr.fit(df[x_cols], df[class_col].to_numpy().ravel())
pred = lr.predict(df[x_cols])
print(pred)

print('Errors:')
print(np.sum(class_array - pred != 0))

print('Coefficients')
print(lr.coef_)
print('Intercept')
print(lr.intercept_)

print('Class probabilities')
pred_proba = lr.predict_proba(df[x_cols])
print(pred_proba)


# Try to mimic model
def mimic(x):
    A = np.array([[3.17715258, 6.81938489, 0.30182974, 3.58362465],
                  [-1.47973427, -4.34033317, 2.58200006, 3.49656911],
                  [-1.6974183, -2.47905172, -2.8838298, -7.08019376]])
    b = np.array([-3.92904391, -1.46509477, 5.39413869])

    y = x.dot(A.T) + b
    exp_y = np.exp(y)
    proba = exp_y / exp_y.sum(axis=1, keepdims=True)

    return y, proba


y, proba = mimic(df[x_cols].to_numpy())

print('Compare')
print(pred_proba[12])
print(proba[12])

# # Brief test
# alpha = np.array([0.5, 0.8])
# v_gs = np.array([10., 20])
# v_ls = np.array([5., 10.])
# fd2 = create_feature_dict(v_gs, v_ls, alpha)
# df2 = pd.DataFrame(fd2)
# y2, proba2 = mimic(df2[x_cols].to_numpy())
# print(proba2)
# quit()

# Create new data for testing
n_test = 200
v_gs = np.linspace(0, 20, n_test)
v_ls = np.linspace(0, 20, n_test)

v_gs_mesh, v_ls_mesh = np.meshgrid(v_gs, v_ls)

alpha = 0.8 * np.ones(v_gs_mesh.size)  # Test with different alpha values to see effect of flow regime map
feature_dict = create_feature_dict(v_gs_mesh.ravel(), v_ls_mesh.ravel(), alpha)
print(feature_dict['c1'].shape)

class_dict, class_dict_v2 = create_class_dict(feature_dict)
class_array = class_dict_v2['regime']

test_dict = {**feature_dict, **class_dict, **class_dict_v2}
df_test = pd.DataFrame(test_dict)
test_proba = lr.predict_proba(df_test[x_cols])
test_proba = test_proba.reshape((n_test, n_test, 3))

fig, ax = plt.subplots()
ax.imshow(test_proba, origin='lower')

# Fix axes labels since imshow uses image size to set axes
ax.set_xticks(np.linspace(0, n_test, 5), np.linspace(0, v_gs.max(), 5))
ax.set_yticks(np.linspace(0, n_test, 5), np.linspace(0, v_ls.max(), 5))

ax.set_xlabel('Superficial gas velocity (m/s)')
ax.set_ylabel('Superficial liquid velocity (m/s)')

annular_patch = mpatches.Patch(color='red', label='Annular flow')
slug_patch = mpatches.Patch(color='blue', label='Slug/churn flow')
bubbly_patch = mpatches.Patch(color='green', label='Bubbly flow')
plt.legend(handles=[slug_patch, bubbly_patch, annular_patch])

plt.show()


