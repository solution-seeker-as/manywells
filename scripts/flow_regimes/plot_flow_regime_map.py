"""
Created 10 December 2024
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Generate a flow map based on a multiclass classification model:
    p = softmax(f(v_gs, v_ls, alpha)),
where p is a vector of probabilities of the different flow regimes.
f maps from the superficial velocities (v_gs, v_ls) and void fraction (alpha) to a vector (logits).
The logits are normalized by the softmax so that its elements add to one, resulting in p.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scripts.plt_config import set_mpl_config


def softmax(x: np.array, axis=None):
    if axis is None:
        axis = 0
    exp_logits = np.exp(x)
    return exp_logits / exp_logits.sum(axis=axis, keepdims=True)


def tanh(x: np.array):
    exp_x = np.exp(x)
    exp_nx = np.exp(-x)
    return (exp_x - exp_nx) / (exp_x + exp_nx)


def create_feature_dict(v_gs, v_ls, alpha):
    rho_g = 1
    rho_l = 900
    g = 9.81
    sigma = 0.033  # Really a function of the densities and temperature
    c1_const = 3.1 * (g * sigma * (rho_l - rho_g) / rho_g ** 2) ** (1 / 4)
    # c1_const = 12.821059412975487  # TODO: remove later
    print('c1_const:', c1_const)

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
    bubble = ~(annular | slug)

    # Do some checks here
    print(sum(annular | slug | bubble))
    print(sum(annular & slug))
    print(sum(annular & bubble))
    print(sum(slug & bubble))

    class_array = np.zeros(c1.size, dtype=int)
    class_array[annular] = 0  # Not needed, but added for completeness
    class_array[slug] = 1
    class_array[bubble] = 2
    print(class_array)

    # One-hot encoded
    class_dict = {
        'annular': annular,
        'slug': slug,
        'bubble': bubble,
    }

    # Encoded as integer
    class_dict_v2 = {
        'regime': class_array
    }

    return class_dict, class_dict_v2



# Classify flow regime
def classify(x):
    """
    :param x: Features computed from v_gs, v_ls, alpha
    :return: prediction (0: annular, 1: slug, 2: bubbly), probability vector
    """
    A = np.array([[ 3.17715258,  6.81938489,  0.30182974,  3.58362465],
                  [-1.47973427, -4.34033317,  2.58200006,  3.49656911],
                  [-1.6974183 , -2.47905172, -2.8838298 , -7.08019376]])
    b = np.array([-3.92904391, -1.46509477,  5.39413869])

    y = x.dot(A.T) + b

    # Softmax
    exp_y = np.exp(y)
    proba = exp_y / exp_y.sum(axis=1, keepdims=True)

    return y, proba



# Brief test
# alpha = np.array([0.8, 0.8])
# v_gs = np.array([15., 15])
# v_ls = np.array([5., 20])
#
# fd2 = create_feature_dict(v_gs, v_ls, alpha)
# df2 = pd.DataFrame(fd2)
# y2, proba2 = classify(df2.to_numpy())
# print(proba2)
# quit()

# # Brief test 2
# alpha = np.array([0.6, 0.8])
# v_gs = np.array([15., 15])
# v_ls = np.array([5., 5.])
#
# fd2 = create_feature_dict(v_gs, v_ls, alpha)
# df2 = pd.DataFrame(fd2)
# y2, proba2 = classify(df2.to_numpy())
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
x_cols = list(feature_dict.keys())

class_dict, class_dict_v2 = create_class_dict(feature_dict)
class_array = class_dict_v2['regime']

test_dict = {**feature_dict, **class_dict, **class_dict_v2}
df_test = pd.DataFrame(test_dict)
_, test_proba = classify(df_test[x_cols].to_numpy())
# test_proba = lr.predict_proba(df_test[x_cols])
test_proba = test_proba.reshape((n_test, n_test, 3))

# Use probabilities as alpha channel
alpha_annular = test_proba[:, :, 0]
alpha_slug = test_proba[:, :, 1]
alpha_bubbly = test_proba[:, :, 2]

# Set colors
color_annular = np.zeros((200, 200, 3))
rgb_annular = (0.992, 0.941, 0.463)
color_annular[:, :, 0] = rgb_annular[0]
color_annular[:, :, 1] = rgb_annular[1]
color_annular[:, :, 2] = rgb_annular[2]

color_slug = np.zeros((200, 200, 3))
rgb_slug = (0.537, 0.796, 0.651)
color_slug[:, :, 0] = rgb_slug[0]
color_slug[:, :, 1] = rgb_slug[1]
color_slug[:, :, 2] = rgb_slug[2]

color_bubbly = np.zeros((200, 200, 3))
rgb_bubbly = (0.471, 0.592, 0.694)
color_bubbly[:, :, 0] = rgb_bubbly[0]
color_bubbly[:, :, 1] = rgb_bubbly[1]
color_bubbly[:, :, 2] = rgb_bubbly[2]

# Append alpha channel
img_annular = np.append(color_annular, alpha_annular.reshape(200, 200, 1), axis=2)  # Add alpha channel
img_slug = np.append(color_slug, alpha_slug.reshape(200, 200, 1), axis=2)  # Add alpha channel
img_bubbly = np.append(color_bubbly, alpha_bubbly.reshape(200, 200, 1), axis=2)  # Add alpha channel

# Plot
set_mpl_config()

fig, ax = plt.subplots(figsize=(9, 9))
imgplot = ax.imshow(img_annular, origin='lower')
imgplot = ax.imshow(img_slug, origin='lower')
imgplot = ax.imshow(img_bubbly, origin='lower')

# Fix axes labels since imshow uses image size to set axes
ax.set_xticks(np.linspace(0, n_test, 5), np.linspace(0, v_gs.max(), 5))
ax.set_yticks(np.linspace(0, n_test, 5), np.linspace(0, v_ls.max(), 5))

ax.set_xlabel('Superficial gas velocity (m/s)')
ax.set_ylabel('Superficial liquid velocity (m/s)')

annular_patch = mpatches.Patch(color=rgb_annular, alpha=1, label='Annular flow')
slug_patch = mpatches.Patch(color=rgb_slug, alpha=1, label='Slug/churn flow')
bubbly_patch = mpatches.Patch(color=rgb_bubbly, alpha=1, label='Bubbly flow')
plt.legend(handles=[annular_patch, slug_patch, bubbly_patch], loc='upper left')

plt.tight_layout()

plt.show()


