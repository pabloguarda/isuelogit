from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isuelogit.mytypes import Union, Dict, List, Matrix, Proportion, ColumnVector

import matplotlib
import pylab

import matplotlib.pyplot as plt

from matplotlib.transforms import BlendedGenericTransform
import matplotlib.transforms as transforms
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

import seaborn as sns
# sns.set(rc={"figure.dpi":30, 'savefig.dpi':30})

from isuelogit.estimation import error_by_link

import networkx as nx
# from networkx.utils import is_string_like

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np



# To write with scienfic notation in y axis
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here


# https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


class Artist:
    def __init__(self,
                 folder_plots=None,
                 dim_subplots=None,
                 tex=False):

        self._folder = folder_plots
        self._dim_subplots = dim_subplots
        self.fontsize = 10

        if tex:
            from matplotlib import rc
            matplotlib.rcParams['text.usetex'] = True
            # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            # rc('text', usetex=True)
            # plt.rcParams['figure.dpi'] = 30
            # plt.rcParams['savefig.dpi'] = 30
        else:
            matplotlib.rcParams['text.usetex'] = False

        plt.rcParams['mathtext.default'] = 'regular'

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        self._folder = value

    @property
    def dim_subplots(self):
        return self._dim_subplots

    @dim_subplots.setter
    def dim_subplots(self, value):
        self._dim_subplots = value

    def save_fig(self, fig, folder, filename):
        if folder is None:
            folder = self.folder

        fig.savefig(folder + '/' + filename + '.pdf', pad_inches=0.1, bbox_inches="tight")  #

    def show_network(self, G):
        '''Visualization of network.
        :arg G: graph object

        '''
        fig = plt.subplots()
        # fig.set_tight_layout(False) #Avoid warning using matplot

        pos = nx.get_node_attributes(G, 'pos')

        if len(pos) == 0:
            pos = nx.spring_layout(G)

        nx.draw(G, pos)
        labels = nx.get_edge_attributes(G, 'weight')
        # nx.draw(G, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1')

        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nx.draw_networkx_labels(G, pos)

        plt.show()

    def draw_MultiDiNetwork(self, G: nx.graph, font_size=12, node_size=300, edge_curvature=0.3, nodes_pos=None,
                            show_edge_labels=True):
        ''' Visualization of Digraph or Multigraphs'''

        # https://stackoverflow.com/questions/60067022/multidigraph-edges-from-networkx-draw-with-connectionstyle

        def new_add_edge(G, a, b, c=0):
            if (a, b) in G.edges:
                max_rad = max(x[2]['rad'] for x in G.edges(data=True) if sorted(x[:2]) == sorted([a, b]))
            else:
                max_rad = 0  # np.pi* #0

            G.add_edge(a, b, c, rad=max_rad + edge_curvature)

        # Generate a copy of the graph
        G_copy = nx.MultiDiGraph()
        edges = list(G.edges)
        for edge in edges:
            if isinstance(G, nx.MultiDiGraph):
                if any(x in edges for x in [(edge[0], edge[1], 1), (edge[1], edge[0], 0)]):
                    new_add_edge(G=G_copy, a=edge[0], b=edge[1], c=edge[2])
                else:
                    G_copy.add_edge(edge[0], edge[1], edge[2], rad=0)

            elif isinstance(G, nx.DiGraph):
                if any(x in edges for x in [(edge[1], edge[0])]):
                    new_add_edge(G_copy, edge[0], edge[1], 0)
                else:
                    G_copy.add_edge(edge[0], edge[1], rad=0)

        # Add nodes information
        pos = nodes_pos
        if pos is None:
            pos = nx.spring_layout(G_copy)

        labels = {}
        for node in G.nodes():
            # set the node name as the key and the label as its value
            labels[node] = node + 1  # Avoid that nodes are shown starting from 0

        nx.draw_networkx_nodes(G_copy, pos=pos, node_size=node_size)
        # nx.draw(G_copy, with_labels=False)
        nx.draw_networkx_labels(G_copy, pos, labels, font_size=font_size)

        # G = G_copy

        # Add edges information
        for edge in G_copy.edges(data=True):
            nx.draw_networkx_edges(G_copy, pos, edgelist=[(edge[0], edge[1])],
                                   connectionstyle=f'arc3, rad = {edge[2]["rad"]}')

            # show_edge_labels = True
            if show_edge_labels == True:
                # nx.draw_networkx_edge_labels(G_copy, pos, edge_labels=labels, label_pos= 0.5)
                draw_networkx_digraph_edge_labels(G=G_copy, pos=pos)

        # plt.show()

    def plot_custom_networks(self, N, show_edge_labels, subfolder, filename):

        fig = plt.figure()

        plt.subplot(2, 2, 1)
        self.draw_MultiDiNetwork(G=N['N1'].G, nodes_pos={0: (0, 0), 1: (0.3, 0)}, show_edge_labels=show_edge_labels)
        plt.title('N1')

        plt.subplot(2, 2, 2)
        self.draw_MultiDiNetwork(G=N['N2'].G, nodes_pos={0: (0, 1), 1: (0.1, 1), 2: (0.2, 1)},
                                 show_edge_labels=show_edge_labels)
        plt.title('N2')

        plt.subplot(2, 2, 3)
        self.draw_MultiDiNetwork(G=N['N3'].G, nodes_pos={0: (0, 2), 1: (0, 0), 2: (1, 1), 3: (5, 1)},
                                 show_edge_labels=show_edge_labels)
        plt.title('N3')

        plt.subplot(2, 2, 4)
        self.draw_MultiDiNetwork(G=N['N4'].G, nodes_pos={0: (0, -1), 1: (2, -1), 2: (1, -3), 3: (3, 1)},
                                 show_edge_labels=show_edge_labels)
        plt.title('N4')

        plt.show()

        self.save_fig(fig=fig, folder=subfolder, filename=filename)

    def plot_all_networks(self, N, show_edge_labels, subfolder, filename):

        fig = plt.figure()

        for i, j in zip(N.keys(), range(1, len(N.keys()) + 1)):
            plt.subplot(3, 3, j)

            self.draw_MultiDiNetwork(G=N[i].G, show_edge_labels=show_edge_labels, node_size=300 / 2,
                                     font_size=11)
            plt.title(i)

        plt.show()

        self.save_fig(fig=fig, folder=subfolder, filename=filename)

    def regularization_error(self, filename, subfolder, errors: {}, lambdas: {}, N_labels: {}, color):

        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        # for i, j in zip(N_labels.values(), range(0, len(N_labels))):
        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            # Vertical line with optimal lambda
            test_error = np.array(list(errors[i].values()))
            idx = np.nanargmin(test_error)
            ax[pos_plot].axvline(x=lambdas[i][idx], color=color, linestyle='dashed', linewidth=0.5
                                 , label=r'Optimal $\lambda$')

            ax[pos_plot].plot(np.array(list(lambdas[i])), np.array(list(errors[i].values())), color=color
                              )
            ax[pos_plot].set_title(j)
            ax[pos_plot].set_xscale("log")
            ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # set labels
        plt.setp(ax[-1, :], xlabel='$\lambda$')
        plt.setp(ax[:, 0], ylabel='error')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=2
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        plt.tight_layout()
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def regularization_joint_error(self, filename, subfolder, lambdas: {}, errors: {}, N_labels, colors):

        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        # for i, j in zip(N_labels.values(), range(0, len(N_labels))):
        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            # Training: Error and Vertical lines with optimal lambda
            train_error = np.array(list(errors['train'][i].values()))
            validation_error = np.array(list(errors['validation'][i].values()))
            ax[pos_plot].axvline(x=lambdas['train'][i][np.nanargmin(train_error)], color=colors[0], linestyle='dashed',
                                 linewidth=0.5,
                                 label=r'Optimal $\lambda$ training')
            ax[pos_plot].plot(np.array(list(lambdas['train'][i])), np.array(list(errors['train'][i].values())),
                              label='Error training', color=colors[0])

            # Validation
            ax[pos_plot].axvline(x=lambdas['validation'][i][np.nanargmin(validation_error)], color=colors[1],
                                 linestyle='dashed', linewidth=0.5,
                                 label=r'Optimal $\lambda$ Validation')
            ax[pos_plot].plot(np.array(list(lambdas['validation'][i])),
                              np.array(list(errors['validation'][i].values())), label='Error validation',
                              color=colors[1])
            # plt.plot(np.array(list(errors_logit['train'][i].values())))
            ax[pos_plot].set_title(i)
            ax[pos_plot].set_xscale("log")
            # plt.title('train' + str(i))

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # plt.xscale("log")

        # set labels
        plt.setp(ax[-1, :], xlabel='$\lambda$')
        plt.setp(ax[:, 0], ylabel='error')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=2
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        plt.tight_layout()
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def regularization_path(self, filename, subfolder, lambdas, theta_estimate, errors, N_labels, key_attrs, color):
        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        key_attr_plotted = False

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            # Vertical line
            validation_error = np.array(list(errors[i].values()))
            idx = np.nanargmin(validation_error)
            ax[pos_plot].axvline(x=lambdas[i][idx], color=color, linestyle='dashed', linewidth=0.5
                                 , label=r'Optimal $\lambda$')

            theta_keys = list(theta_estimate[i][0].keys())

            for l in theta_keys:
                estimates_attr = []
                for m in theta_estimate[i].keys():
                    estimates_attr.append(theta_estimate[i][m][l])

                if l in key_attrs:
                    if key_attr_plotted:
                        ax[pos_plot].plot(lambdas[i], estimates_attr, color=color, linestyle='solid', linewidth=1)
                        key_attr_plotted = True
                    else:
                        ax[pos_plot].plot(lambdas[i], estimates_attr, color=color, linestyle='solid', linewidth=1
                                          , label=r"$\hat{\theta}$" + '(' + l + ')')
                else:
                    ax[pos_plot].plot(lambdas[i], estimates_attr, linestyle='dashed', linewidth=0.2)
                ax[pos_plot].set_title(j)
                ax[pos_plot].set_xscale("log")

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # set labels
        plt.setp(ax[-1, :], xlabel=r"$\lambda$")
        plt.setp(ax[:, 0], ylabel=r'$\hat{\theta}$')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=4
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        # plt.xscale("log")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, bottom=0.2)
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def regularization_consistency(self, filename, subfolder, errors: np.array, theta_true: {}, theta_estimate: {},
                                   N_labels: {}, color):

        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])
        counter = 0

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            validation_error = np.array(list(errors[i].values()))
            idx = np.nanargmin(validation_error)

            ax[pos_plot].plot(np.array(list(theta_true.values())), label=r"$\theta$", color=color, linestyle='solid',
                              linewidth=1)
            ax[pos_plot].plot(list(theta_estimate[i][idx].values()), label=r"$\hat{\theta}$", color=color,
                              linestyle='dashed', linewidth=1)

            # set labels
            plt.setp(ax[-1, :], xlabel=r"$i$")
            plt.setp(ax[:, 0], ylabel=r"$\theta_i$")

            # plt.plot(np.array(list(errors_logit['train'][i].values())))
            ax[pos_plot].set_title(j)

            counter += 1

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=2
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def regularization_joint_consistency(self, filename, subfolder, errors: np.array, theta_true: {},
                                         theta_estimate: {}, N_labels: {}, colors):

        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])
        counter = 0

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            key_train = list(errors['train'][i].keys())[np.nanargmin(np.array(list(errors['train'][i].values())))]
            key_validation = list(errors['validation'][i].keys())[
                np.nanargmin(np.array(list(errors['validation'][i].values())))]

            ax[pos_plot].plot(np.array(list(theta_true.values())), label=r"$\theta$", linestyle='solid', linewidth=1,
                              color='k')
            ax[pos_plot].plot(list(theta_estimate['train'][i][key_train].values()), label=r"$\hat{\theta}_{train}$",
                              color=colors[0], linestyle='dashed', linewidth=1)
            ax[pos_plot].plot(list(theta_estimate['validation'][i][key_validation].values()),
                              label=r"$\hat{\theta}_{validation}$", color=colors[1],
                              linestyle='dashed', linewidth=1)

            # set labels
            plt.setp(ax[-1, :], xlabel=r"$i$")
            plt.setp(ax[:, 0], ylabel=r"$\theta_i$")

            # plt.plot(np.array(list(errors_logit['train'][i].values())))
            ax[pos_plot].set_title(i)

            counter += 1

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=3
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def estimated_vs_true_theta(self, filename, subfolder, theta_est_t: {}, theta_c: {}, theta_true_t: {},
                                constraints_theta, N_labels: {}, color):

        # self = plot
        fig, ax = plt.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            learned_parameter_t = np.array(theta_est_t[i])
            true_parameter_t = np.array(theta_true_t[i])

            ax[pos_plot].scatter(learned_parameter_t, true_parameter_t, s=1, color='k', label='estimate')
            ax[pos_plot].plot([true_parameter_t.min(), true_parameter_t.max()],
                              [true_parameter_t.min(), true_parameter_t.max()],
                              'k--', lw=1, color=color, label='truth')
            ax[pos_plot].set_title(j)
            # ax[pos_plot].set_xscale("log")

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # set labels
        plt.setp(ax[-1, :], xlabel=r'$\theta_t$')
        plt.setp(ax[:, 0], ylabel=r'$\hat{\theta_t}$')

        plt.tight_layout()

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=2
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

        # fig, ax = plt.subplots()
        # ax.scatter(learned_parameter_t, true_parameter_t)
        # ax.plot([true_parameter_t.min(), true_parameter_t.max()], [true_parameter_t.min(), true_parameter_t.max()], 'k--', lw=4)
        # ax.set_xlabel('True theta')
        # ax.set_ylabel('Learned theta')
        # ax.set_title(filename + '\n coefficient')
        # plt.show()

        if constraints_theta['Z']['c'] != 0:

            fig, ax = plt.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

            for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
                pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

                learned_parameter_t = np.array(theta_est_t[i])
                true_parameter_t = np.array(theta_true_t[i])
                learned_parameter_c = np.array(theta_c[i])
                # Value of time
                learned_parameter_vot = 60 * learned_parameter_t / learned_parameter_c
                true_parameter_vot = 60 * true_parameter_t / constraints_theta['Z'][
                    'c']  # Multiply by 60 to convert to USD per hour

                ax[pos_plot].plot([true_parameter_vot.min(), true_parameter_vot.max()],
                                  [true_parameter_vot.min(), true_parameter_vot.max()], 'k--',
                                  lw=1, color=color, label='truth')
                ax[pos_plot].scatter(learned_parameter_vot, true_parameter_vot, s=1, color='k', label='estimate')
                ax[pos_plot].set_title(j)

            # set labels
            plt.setp(ax[-1, :], xlabel=r'$\theta_t/\theta_c$')
            plt.setp(ax[:, 0], ylabel=r'$\hat{\theta_t}/\hat{\theta_c}$')

            # Legend
            lines, labels = fig.axes[0].get_legend_handles_labels()
            fig.legend(lines, labels, loc='upper center', ncol=2
                       , bbox_to_anchor=[0.52, -0.45]
                       , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

            plt.tight_layout()
            plt.show()
            self.save_fig(fig=fig, filename=filename, folder=subfolder + '_vot')

            # fig, ax = plt.subplots()
            # ax.scatter(learned_parameter_vot, true_parameter_vot)
            # ax.plot([true_parameter_t.min(), true_parameter_t.max()], [true_parameter_t.min(), true_parameter_t.max()], 'k--',
            #         lw=4)
            # ax.set_xlabel('True value of time')
            # ax.set_ylabel('Learned value of time')
            # ax.set_title(filename + '\n Value of Time')
            # plt.show()

    def flows_vs_true_theta(self, filename, subfolder, x, f, N_labels):

        fig, ax = plt.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            x_N = np.array(x[i])
            f_N = np.array(f[i])

            # Standard deviation of flow assignment
            sdX_vs_theta_N = [np.sqrt(np.var(row)) for row in x_N]
            sdF_vs_theta_N = [np.sqrt(np.var(row)) for row in f_N]

            ax[pos_plot].plot(sdF_vs_theta_N, label='route predicted_counts')
            ax[pos_plot].set_title(j)

        plt.setp(ax[-1, :], xlabel=r'$\theta_t$')
        plt.setp(ax[:, 0], ylabel='std')

        # plt.tight_layout()
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

        # Plot
        # plt.plot(sdX_vs_theta_N, label='link predicted_counts')
        # plt.plot(sdF_vs_theta_N, label='route predicted_counts')
        # plt.title(filename + '\n Standard deviation of route predicted_counts')
        # plt.set_xlabel('Theta')
        # plt.set_ylabel('Standard deviation of route predicted_counts')
        # plt.show()

    def consistency_nonlinear_link_logit_estimation(self, filename, subfolder, theta_est: {}, display_parameters: {},
                                                    vot_est: {}, theta_true: {}, N_labels: {}, n_bootstraps: int):

        fig, ax = plt.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        # This suptitle rais errors
        # # Title does not show up on Pycharm bu it does in file saved
        # fig.suptitle("Estimates of logit parameters versus sample size (i.e. observed links predicted_counts)\n",
        #              y=1.05)
        # # ax = ax.flatten()

        y_label = ''

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            # x-axis: number of links
            x = np.array(list(theta_est[i].keys()))

            if display_parameters['tt']:
                # Travel time
                y = np.array([theta_est[i][l]['tt']['mean'] for l in theta_est[i].keys()])
                e = np.array(
                    [theta_est[i][l]['tt']['sd'] for l in theta_est[i].keys()]) / np.sqrt(
                    n_bootstraps)
                ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax[pos_plot].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='blue',
                                      label=r'$\hat{\theta_v}$', color='blue')
                ax[pos_plot].plot([min(x), max(x)], [theta_true[i]['tt'], theta_true[i]['tt']], label=r'$\theta_v$',
                                  color='blue', linestyle='dashed')
                # plt.setp(ax[(0, 0)], ylabel=r'$\hat{\theta_v}$')
                y_label = r'$\hat{\theta_v}$'

            if display_parameters['c']:
                # Cost
                y = np.array([theta_est[i][l]['c']['mean'] for l in theta_est[i].keys()])
                e = np.array(
                    [theta_est[i][l]['c']['sd'] for l in theta_est[i].keys()]) / np.sqrt(
                    n_bootstraps)
                ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax[pos_plot].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='red',
                                      label=r'$\hat{\theta_c}$', color='red')
                ax[pos_plot].plot([min(x), max(x)], [theta_true[i]['c'], theta_true[i]['c']], label=r'$\theta_c$',
                                  color='red', linestyle='dashed')
                y_label = r"$\hat{\theta_c}$"
                # plt.setp(ax[(0, 0)], ylabel=r"$\hat{\theta_c}$")

            if display_parameters['vot']:
                # VOT
                x = np.array(list(vot_est[i].keys()))
                y = np.array([vot_est[i][l]['mean'] for l in theta_est[i].keys()])
                e = np.array([vot_est[i][l]['sd'] for l in theta_est[i].keys()]) / np.sqrt(
                    n_bootstraps)
                ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax[pos_plot].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='black',
                                      label=r'$\theta_v/\theta_c$', color='black')
                ax[pos_plot].plot([min(x), max(x)],
                                  [theta_true[i]['tt'] / theta_true[i]['c'], theta_true[i]['tt'] / theta_true[i]['c']],
                                  label=r'$\hat{\theta_v}/\hat{\theta_c}$', color='black', linestyle='dashed')
                y_label = r'$\hat{\theta_v}/\hat{\theta_c}$'
                # plt.setp(ax[(1, 0)], ylabel=r'$\hat{\theta_v}/\hat{\theta_c}$')

            # Set title network
            ax[pos_plot].set_title(j)

        # Axis labels
        plt.setp(ax[-1, :], xlabel="Number of links")
        plt.setp(ax[:, 0], ylabel=y_label)

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=3
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        # Axis labels
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def regularization_error_nonlinear_link_logit_estimation(self, filename, subfolder, errors: {}, N_labels: {},
                                                             n_samples: int):

        fig, ax = plt.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        # This suptitle rais errors
        # # Title does not show up on Pycharm bu it does in file saved
        # fig.suptitle("Estimates of logit parameters versus sample size (i.e. observed links predicted_counts)\n",
        #              y=1.05)
        # # ax = ax.flatten()

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            # x-axis: number of links
            x = np.array([r"$\lambda = 0$", r"$\lambda = \lambda^{\star}$"])
            y = np.array([errors[i]['noreg']['mean'], errors[i]['reg']['mean']])
            e = np.array(
                [errors[i]['reg']['mean'], errors[i]['noreg']['sd']]) / np.sqrt(n_samples)
            ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[pos_plot].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='red',
                                  label='No regularization', color='red')

            # # Regularization
            # y = errors[i]['reg']['mean']
            # e = np.array(errors[i]['reg']['sd']) / np.sqrt(n_samples)
            # ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax[pos_plot].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='red',
            #                       label= 'Regularization', color='red')

            # Set title network
            ax[pos_plot].set_title(j)

        # Axis labels
        # plt.setp(ax[-1, :], xlabel="Number of links")
        plt.setp(ax[:, 0], ylabel=r'RMSE')

        # # Legend
        # lines, labels = fig.axes[0].get_legend_handles_labels()
        # fig.legend(lines, labels, loc='upper center', ncol=3
        #            , bbox_to_anchor=[0.52, -0.45]
        #            , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        # Axis labels
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    # def consistency_nonlinear_link_logit_estimation(self, filename, network_name, theta_est:{}, vot_est:{}, theta_true:{}, N_label, n_bootstraps: int):
    #
    #     fig, ax = plt.subplots(ncols=2, nrows=2)
    #
    #     # Title does not show up on Pycharm bu it does in file saved
    #     fig.suptitle("Estimates of logit parameters versus sample size (i.e. observed links predicted_counts)\n" + N_label,
    #                  y=1.05)
    #     # ax = ax.flatten()
    #
    #     # Travel time
    #     x = np.array(list(theta_est.keys()))
    #     y = np.array([theta_est[i]['tt']['mean'] for i in theta_est.keys()])
    #     e = np.array(
    #         [theta_est[i]['tt']['sd'] for i in theta_est.keys()]) / np.sqrt(
    #         n_bootstraps)
    #     ax[(0, 0)].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #     ax[(0, 0)].errorbar(x, y, e, linestyle='None', fmt='o',  markersize=3, mfc='blue', label=r'$\hat{\theta}$', color = 'blue')
    #     ax[(0, 0)].plot([min(x), max(x)], [float(theta_true['tt']), float(theta_true['tt'])], label=r'$\theta$', color = 'black')
    #     plt.setp(ax[(0, 0)], ylabel=r'$\hat{\theta_t}$')
    #
    #     # Cost
    #     x = np.array(list(theta_est.keys()))
    #     y = np.array([theta_est[i]['c']['mean'] for i in theta_est.keys()])
    #     e = np.array(
    #         [theta_est[i]['c']['sd'] for i in theta_est.keys()]) / np.sqrt(
    #         n_bootstraps)
    #     ax[(0, 1)].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #     ax[(0, 1)].errorbar(x, y, e, linestyle='None', fmt='o',  markersize=3, mfc='blue', label=r'$\hat{\theta}$', color = 'blue')
    #     ax[(0, 1)].plot([min(x), max(x)], [float(theta_true['c']), float(theta_true['c'])], label=r'$\theta$', color = 'black')
    #     plt.setp(ax[(0, 1)], ylabel=r"$\hat{\theta_c}$")
    #
    #     # VOT
    #     x = np.array(list(vot_est.keys()))
    #     y = np.array([vot_est[i]['mean'] for i in vot_est.keys()])
    #     e = np.array([vot_est[i]['sd'] for i in vot_est.keys()]) / np.sqrt(
    #         n_bootstraps)
    #     ax[(1, 0)].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #     ax[(1, 0)].errorbar(x, y, e, linestyle='None', fmt='o',  markersize=3, mfc='blue', label=r'$\hat{\theta}$', color = 'blue')
    #     ax[(1, 0)].plot([min(x), max(x)], [float(theta_true['tt']) / float(theta_true['c']), float(theta_true['tt']) / float(theta_true['c'])],
    #                     label=r'$\theta$', color = 'black')
    #     plt.setp(ax[(1, 0)], ylabel=r'$\hat{\theta_t}/\hat{\theta_c}$')
    #
    #     ax[(1, 1)].axis('off')
    #
    #     # x-axis labels
    #     plt.setp(ax[-1, :], xlabel="Number of links")
    #
    #     lines, labels = ax[(0, 0)].get_legend_handles_labels()
    #
    #     fig.legend(lines, labels
    #                , loc='upper center', ncol=2
    #                , bbox_to_anchor=[0.52, -0.3]
    #                , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes)
    #                )
    #
    #     # Axis labels
    #     plt.show()
    #     self.save_fig(fig=fig, filename=filename, network_name = network_name)

    def error_nonlinear_link_logit_estimation(self, filename, subfolder, theta_est: {}, vot_est: {}, theta_true: {},
                                              N_label, n_samples: int):

        fig, ax = plt.subplots(ncols=2, nrows=2)

        # # Title does not show up on Pycharm bu it does in file saved
        # fig.suptitle("Estimates of logit parameters versus sample size (i.e. observed links predicted_counts)\n" + N_label,
        #              y=1.05)
        # ax = ax.flatten()

        # Error no regularization
        x = np.array(list(theta_est.keys()))
        y = np.array([theta_est[i]['tt']['mean'] for i in theta_est.keys()])
        e = np.array(
            [theta_est[i]['tt']['sd'] for i in theta_est.keys()]) / np.sqrt(
            n_samples)
        ax[(0, 0)].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[(0, 0)].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='blue', label=r'$\hat{\theta}$',
                            color='blue')
        ax[(0, 0)].plot([min(x), max(x)], [float(theta_true['tt']), float(theta_true['tt'])], label=r'$\theta$',
                        color='black')
        plt.setp(ax[(0, 0)], ylabel=r'$\hat{\theta_t}$')

        # Cost
        x = np.array(list(theta_est.keys()))
        y = np.array([theta_est[i]['c']['mean'] for i in theta_est.keys()])
        e = np.array(
            [theta_est[i]['c']['sd'] for i in theta_est.keys()]) / np.sqrt(
            n_samples)
        ax[(0, 1)].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[(0, 1)].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='blue', label=r'$\hat{\theta}$',
                            color='blue')
        ax[(0, 1)].plot([min(x), max(x)], [float(theta_true['c']), float(theta_true['c'])], label=r'$\theta$',
                        color='black')
        plt.setp(ax[(0, 1)], ylabel=r"$\hat{\theta_c}$")

        # VOT
        x = np.array(list(vot_est.keys()))
        y = np.array([vot_est[i]['mean'] for i in vot_est.keys()])
        e = np.array([vot_est[i]['sd'] for i in vot_est.keys()]) / np.sqrt(
            n_samples)
        ax[(1, 0)].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[(1, 0)].errorbar(x, y, e, linestyle='None', fmt='o', markersize=3, mfc='blue', label=r'$\hat{\theta}$',
                            color='blue')
        ax[(1, 0)].plot([min(x), max(x)], [float(theta_true['tt']) / float(theta_true['c']),
                                           float(theta_true['tt']) / float(theta_true['c'])],
                        label=r'$\theta$', color='black')
        plt.setp(ax[(1, 0)], ylabel=r'$\hat{\theta_t}/\hat{\theta_c}$')

        ax[(1, 1)].axis('off')

        # x-axis labels
        plt.setp(ax[-1, :], xlabel="Number of links")

        lines, labels = ax[(0, 0)].get_legend_handles_labels()

        fig.legend(lines, labels
                   , loc='upper center', ncol=2
                   , bbox_to_anchor=[0.52, -0.3]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes)
                   )

        # Axis labels
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def vot_regularization_path(self, filename, subfolder, lambdas, vot_estimate, theta_true, errors, N_labels,
                                color) -> None:
        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            ax[pos_plot].set_title(i)
            ax[pos_plot].set_xscale("log")

            ax[pos_plot].plot(np.array(list(vot_estimate[i].keys())), np.array(list(vot_estimate[i].values())),
                              label=r"$\hat{\theta_t}^{\star}/\hat{\theta_c}^{\star}$", color=color, linestyle='dashed',
                              linewidth=1)

            # Vertical line (optimal lambda for validation)
            validation_error = np.array(list(errors[i].values()))
            idx = np.nanargmin(validation_error)
            ax[pos_plot].axvline(x=lambdas[i][idx], color=color, linestyle='dashed', linewidth=0.5
                                 , label=r'$\lambda^{\star}$')

            ax[pos_plot].plot([min(np.array(list(vot_estimate[i].keys()))), max(np.array(list(vot_estimate[i].keys())))]
                              , [float(theta_true['tt']) / float(theta_true['c']),
                                 float(theta_true['tt']) / float(theta_true['c'])],
                              label=r'$\theta_t/\theta_c$', color='black')

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # set labels
        plt.setp(ax[-1, :], xlabel=r"$\lambda$")
        plt.setp(ax[:, 0], ylabel=r'$\hat{\theta_t}/\hat{\theta_c}$')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=4
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        # plt.xscale("log")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, bottom=0.2)
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def vot_multidays_consistency(self, filename, subfolder, labels, vot_estimates, theta_true, N_labels,
                                  colors) -> None:
        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        curves_labels = list(labels.keys())
        legend_labels = list(labels.values())

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            ax[pos_plot].set_title(i)
            ax[pos_plot].set_xticks(np.array(list(vot_estimates[curves_labels[0]][i].keys())))
            ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax[pos_plot].set_ylim(bottom=-0.005)
            # ax[pos_plot].set_xscale("log")

            ax[pos_plot].plot(np.array(list(vot_estimates[curves_labels[0]][i].keys())),
                              np.array(list(vot_estimates[curves_labels[0]][i].values())), label=legend_labels[0],
                              color=colors[0], linestyle='dashed', linewidth=1)

            ax[pos_plot].plot(np.array(list(vot_estimates[curves_labels[1]][i].keys())),
                              np.array(list(vot_estimates[curves_labels[1]][i].values())), label=legend_labels[1],
                              color=colors[1], linestyle='dashed',
                              linewidth=1)

            # ax[pos_plot].plot(np.array(list(vot_estimates[i].keys())), np.array(list(vot_estimates[i].values())),
            #                   label=r"$\hat{\theta_t}^{\star}/\hat{\theta_c}^{\star}$", color=color, linestyle='dashed',
            #                   linewidth=1)

            # Vertical line (optimal lambda for validation)
            # validation_error = np.array(list(errors[i].values()))
            # idx = np.nanargmin(validation_error)
            # ax[pos_plot].axvline(x=lambdas[i][idx], color=color, linestyle='dashed', linewidth=0.5
            #                      , label=r'$\lambda^{\star}$')

            ax[pos_plot].plot([min(np.array(list(vot_estimates[curves_labels[0]][i].keys()))),
                               max(np.array(list(vot_estimates[curves_labels[0]][i].keys())))]
                              , [theta_true[i]['tt'] / theta_true[i]['c'], theta_true[i]['tt'] / theta_true[i]['c']],
                              label=r'$\theta_t/\theta_c$', color='black')

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # set labels
        plt.setp(ax[-1, :], xlabel="Days")
        plt.setp(ax[:, 0], ylabel=r'$\hat{\theta_t}/\hat{\theta_c}$')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=4
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        # plt.xscale("log")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, bottom=0.2)
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def q_multidays_consistency(self, filename, subfolder, labels, q_estimates, q_true, N_labels, colors) -> None:
        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        curves_labels = list(labels.keys())
        legend_labels = list(labels.values())

        q_errors = {}

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            ax[pos_plot].set_title(i)
            # ax[pos_plot].set_xscale("log")

            # Error in q estimation
            q_errors[curves_labels[0]] = [np.round(np.linalg.norm(q_estimate - q_true[i], 2), 1) for days, q_estimate in
                                          q_estimates[curves_labels[0]][i].items()]
            q_errors[curves_labels[1]] = [np.round(np.linalg.norm(q_estimate - q_true[i], 2), 1) for days, q_estimate in
                                          q_estimates[curves_labels[1]][i].items()]

            ax[pos_plot].plot(np.array(list(q_estimates[curves_labels[0]][i].keys())),
                              np.array(q_errors[curves_labels[0]]) / len(q_errors[curves_labels[0]]),
                              label=legend_labels[0], color=colors[0], linestyle='dashed', linewidth=1)

            ax[pos_plot].plot(np.array(list(q_estimates[curves_labels[1]][i].keys())),
                              np.array(q_errors[curves_labels[1]]) / len(q_errors[curves_labels[1]]),
                              label=legend_labels[1], color=colors[1], linestyle='dashed', linewidth=1)

            ax[pos_plot].set_ylim(bottom=-0.005)
            ax[pos_plot].set_xticks(np.array(list(q_estimates[curves_labels[0]][i].keys())))
            ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Vertical line (optimal lambda for validation)
            # validation_error = np.array(list(errors[i].values()))
            # idx = np.nanargmin(validation_error)
            # ax[pos_plot].axvline(x=lambdas[i][idx], color=color, linestyle='dashed', linewidth=0.5
            #                      , label=r'$\lambda^{\star}$')

            # ax[pos_plot].plot([min(np.array(list(vot_estimates[i].keys()))), max(np.array(list(vot_estimates[i].keys())))]
            #                   , [float(theta_true['tt']) / float(theta_true['c']), float(theta_true['tt']) / float(theta_true['c'])],
            #                   label=r'$\theta_t/\theta_c$', color='black')

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # set labels
        plt.setp(ax[-1, :], xlabel="Days")
        plt.setp(ax[:, 0], ylabel=r'$||\hat{Q}-\bar{Q}||_2$')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=4
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        # plt.xscale("log")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, bottom=0.2)
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def computational_time_multidays_consistency(self, filename: str, subfolder: str, computational_times: np.array,
                                                 N_labels: {}, colors: []) -> None:
        # self = plot
        fig = plt.figure()
        ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        for i, j, k in zip(N_labels.keys(), N_labels.values(), range(0, len(N_labels))):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])

            ax[pos_plot].set_title(i)
            ax[pos_plot].set_xticks(np.array(list(computational_times['end_theta_q'][i].keys())))
            ax[pos_plot].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # ax[pos_plot].set_xscale("log")

            ax[pos_plot].plot(np.array(list(computational_times['end_theta_q'][i].keys())),
                              np.array(list(computational_times['end_theta_q'][i].values())),
                              label=r"Endogenous $Q$ and $\theta$", color=colors[0], linestyle='dashed', linewidth=1)

            ax[pos_plot].plot(np.array(list(computational_times['end_q'][i].keys())),
                              np.array(list(computational_times['end_q'][i].values())), label=r"Endogenous $Q$",
                              color=colors[1],
                              linestyle='dashed', linewidth=1)

            ax[pos_plot].plot(np.array(list(computational_times['end_theta'][i].keys())),
                              np.array(list(computational_times['end_theta'][i].values())),
                              label=r"Endogenous $\theta$", color=colors[2],
                              linestyle='dashed', linewidth=1)

            # Vertical line (optimal lambda for validation)
            # validation_error = np.array(list(errors[i].values()))
            # idx = np.nanargmin(validation_error)
            # ax[pos_plot].axvline(x=lambdas[i][idx], color=color, linestyle='dashed', linewidth=0.5
            #                      , label=r'$\lambda^{\star}$')

            # ax[pos_plot].plot([min(np.array(list(computational_times[i].keys()))), max(np.array(list(computational_times[i].keys())))]
            #                   , [float(theta_true['tt']) / float(theta_true['c']), float(theta_true['tt']) / float(theta_true['c'])],
            #                   label=r'$\theta_t/\theta_c$', color='black')

        # Remove empty subplots
        for k in range(len(N_labels), self.dim_subplots[0] * self.dim_subplots[1]):
            pos_plot = int(np.floor(k / self.dim_subplots[0])), int(k % self.dim_subplots[1])
            ax[pos_plot].axis('off')

        # set labels
        plt.setp(ax[-1, :], xlabel="Days")
        plt.setp(ax[:, 0], ylabel=r'Computational time [s]')

        # Legend
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center', ncol=4
                   , bbox_to_anchor=[0.52, -0.45]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.flatten()[-2].transAxes))

        # plt.xscale("log")
        fig.tight_layout()
        # fig.subplots_adjust(top=0.9, bottom=0.2)
        plt.show()
        self.save_fig(fig=fig, filename=filename, folder=subfolder)

    def q_estimation_convergence(self, filename, subfolder, results_norefined_df, results_refined_df, methods):

        fig = plt.figure()
        # ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])
        ax = {}
        ax[(0, 0)] = plt.subplot(self.dim_subplots[0], self.dim_subplots[1], 1)
        ax[(0, 1)] = plt.subplot(self.dim_subplots[0], self.dim_subplots[1], 2, sharey=ax[(0, 0)])

        # matplotlib.rcParams['text.usetex'] = True

        # ax[(1, 0)] = plt.subplot(self.dim_subplots[0], self.dim_subplots[1], 3)
        # ax[(1, 1)] = plt.subplot(self.dim_subplots[0], self.dim_subplots[1], 4, sharey=ax[(1, 0)])

        # ii) Objective function values

        # - No refined

        if len(results_norefined_df['objective']) == 1:
            ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['objective'], marker='o', markersize=2)
        else:
            ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['objective'])

        ax[(0, 0)].set_xticks(np.arange(0, results_norefined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))

        # if results_norefined_df['objective'].min() != results_norefined_df['objective'].max():
        #     ax[(1,0)].set_ylim(results_norefined_df['objective'].min(), results_norefined_df['objective'].max())

        ax[(0, 0)].set_ylabel(r"$ ||(\hat{q}-q||_2^2 $")
        ax[(0, 0)].set_xlabel("iterations (" + methods[0] + ")")
        # ax[(1, 0)].yaxis.set_major_formatter(yfmt3)

        # - Refined
        if len(results_refined_df['objective']) == 1:
            ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['objective'], marker='o', markersize=2)
        else:
            ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['objective'])

        ax[(0, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))

        # if results_refined_df['objective'].min() != results_refined_df['objective'].max():
        #     ax[(1,1)].set_ylim(results_refined_df['objective'].min(), results_refined_df['objective'].max())

        ax[(0, 1)].set_xlabel("iterations (" + methods[1] + ")")
        # ax[(1, 1)].yaxis.set_major_formatter(yfmt4)

        fig.tight_layout()

        # fig.savefig(self.folder + '/' + network_name + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        plt.show()

        return fig

    def benchmark_optimization_methods(self, methods, results_experiment, colors):

        fig_loss, ax_loss = plt.subplots(figsize=(4, 4))
        fig_legend = pylab.figure(figsize=(3, 2))

        # colors = ['b','g', 'r','c', 'm', 'y', 'b', 'w']

        counter = 0

        # Loss plot

        for method in methods:
            ax_loss.plot('iter', 'loss', data=results_experiment[results_experiment['method'] == method], label=method,
                         color=colors[counter])
            ax_loss.set_ylabel(r"$ ||x(\hat{\theta})-\bar{x}||_2^2$")
            counter += 1

        ax_loss.set_xlabel("iterations")

        # Set font sizes
        for axi in reversed(fig_loss.get_axes()):

            axi.xaxis.get_label().set_fontsize(18)
            axi.yaxis.get_label().set_fontsize(18)
            # axi.legend(prop=dict(size=18))

            for label in (axi.get_xticklabels() + axi.get_yticklabels()):
                # label.set_fontname('Arial')
                label.set_fontsize(16)

        # Vot plot

        fig_vot, ax_vot = plt.subplots(figsize=(4, 4))

        counter = 0

        for method in methods:
            ax_vot.plot('iter', 'vot', data=results_experiment[results_experiment['method'] == method], label=method,
                        color=colors[counter])
            ax_vot.set_ylabel(r'$\hat{\theta_t}/\hat{\theta_c}$')
            counter += 1

        ax_vot.axhline(0.1667, linestyle='dashed', color='black')
        ax_vot.set_xlabel("iterations")

        lines, labels = [], []
        for axi in reversed(fig_vot.get_axes()):
            # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            linei, labeli = axi.get_legend_handles_labels()
            lines = linei + lines
            labels = labeli + labels

            # Set font sizes
            axi.xaxis.get_label().set_fontsize(18)
            axi.yaxis.get_label().set_fontsize(18)
            # axi.legend(prop=dict(size=18))

            for label in (axi.get_xticklabels() + axi.get_yticklabels()):
                # label.set_fontname('Arial')
                label.set_fontsize(16)

        # labels = ax.get_legend_handles_labels()
        fig_legend.legend(lines, labels, 'center', ncol=3)
        # ax.legend()
        plt.show()

        return fig_loss, fig_vot, fig_legend

    def histogram_errors(self,
                         best_results_norefined: pd.DataFrame,
                         best_results_refined: pd.DataFrame,
                         observed_counts: ColumnVector,
                         filename: str,
                         folder: str
                         ):

        # Distribution of errors across link counts

        best_x_norefined = np.array(list(best_results_norefined['x'].values()))[:, np.newaxis]
        best_x_refined = np.array(list(best_results_refined['x'].values()))[:, np.newaxis]

        fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(5, 5))

        x = error_by_link(observed_counts=observed_counts, predicted_counts=best_x_norefined)
        y = error_by_link(observed_counts=observed_counts, predicted_counts=best_x_refined)
        # We can set the number of bins with the `bins` kwarg
        # plt.hist([x,y], label = ['non-refined stage','refined stage'])

        bound = np.ceil(max(np.maximum(abs(x[~np.isnan(x)]),abs(y[~np.isnan(y)])))/1000)*1000

        bins = np.arange(-bound, bound + 500, 500)

        axs.hist(x, bins, alpha=0.5, label='non-refined')
        axs.hist(y, bins, alpha=0.5, label='refined')
        plt.xlabel("difference between predicted and observed counts")
        plt.ylabel("frequency")
        plt.legend(loc='upper center', bbox_to_anchor=[0.52, -0.15], ncol = 2,
                   prop={'size': self.fontsize}, title = 'optimization stage',
                   bbox_transform=BlendedGenericTransform(fig.transFigure, axs.transAxes))

        axs.xaxis.label.set_size(self.fontsize)
        axs.yaxis.label.set_size(self.fontsize)
        axs.tick_params(axis='y', labelsize=self.fontsize)
        axs.tick_params(axis='x', labelsize=self.fontsize)

        fig.savefig(folder + '/' + filename, pad_inches=0.1, bbox_inches="tight")

        plt.show()

        plt.close(fig)


    def convergence(self,
                    results_refined_df: pd.DataFrame,
                    results_norefined_df: pd.DataFrame,
                    filename: str,
                    methods: [str],
                    simulated_data: bool = False,
                    folder: str = None,
                    theta_true=None):

        '''
     Plot convergence to the true ratio of theta and the reduction of the objective function over iterations

     :argument results

     '''

        if folder is None:
            folder = self.folder

        fig = plt.figure(figsize=(7, 6))
        # ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])

        dim_subplots = (2, 2)

        ax = {}
        ax[(0, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 1)
        ax[(0, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 2, sharey=ax[(0, 0)])

        ax[(1, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 3)
        ax[(1, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 4, sharey=ax[(1, 0)])

        # To write with scienfic notation in y axis
        class ScalarFormatterForceFormat(ScalarFormatter):
            def _set_format(self):  # Override function that finds format to use.
                self.format = "%0.1f"  # Give format here

        # import matplotlib.ticker as mtick

        # Y axis with scientific notation
        yfmt1 = ScalarFormatterForceFormat()
        yfmt1.set_powerlimits((0, 0))

        yfmt2 = ScalarFormatterForceFormat()
        yfmt2.set_powerlimits((0, 0))

        class ScalarFormatterForceFormat(ScalarFormatter):
            def _set_format(self):  # Override function that finds format to use.
                self.format = "%1.1f"  # Give format here

        # Y axis with scientific notation
        yfmt3 = ScalarFormatterForceFormat()
        yfmt3.set_powerlimits((0, 0))

        yfmt4 = ScalarFormatterForceFormat()
        yfmt4.set_powerlimits((0, 0))

        # matplotlib.rcParams['text.usetex'] = True

        # i) Theta estimates in the two plots in the top

        if simulated_data and 'theta_c' in list(results_refined_df.keys()):
            # - No refined

            if len(results_norefined_df['vot']) == 1:
                ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['vot'], marker='o', markersize=2)
            else:
                ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['vot'])

            # ax[(0,0)].set_yticks(np.arange(results_norefined_df['vot'].min(), results_norefined_df['vot'].max(), 0.2))
            ax[(0, 0)].axhline(0.1667, linestyle='dashed')
            # ax[(0, 0)].set_ylabel(r'$\hat{\theta_t}/\hat{\theta_c}$')
            ax[(0, 0)].set_ylabel("value of time")
            # ax[(0, 0)].yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
            # ax[(0, 0)].yaxis.set_major_formatter(yfmt1)
            ax[(0, 0)].tick_params(labelbottom=False)
            # ax[(0, 1)].yaxis.major.formatter._useMathText = True

            # - Refined

            if len(results_refined_df['vot']) == 1:
                ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['vot'], marker='o', markersize=2)
            else:
                ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['vot'])

            # ax[(0,1)].set_yticks(np.arange(results_refined_df['vot'].min(), results_refined_df['vot'].max(), 0.2))
            ax[(0, 1)].axhline(float(theta_true['tt']) / float(theta_true['c']), linestyle='dashed')
            # ax[(0, 1)].yaxis.set_major_formatter(yfmt2)
            ax[(0, 1)].tick_params(labelbottom=False)
            # ax[(0,1)].yaxis.major.formatter._useMathText = True

        # Fresno network where vot may represent trade off between travel time and standard deviation of travel time
        elif 'vot' in list(results_norefined_df.keys()) and np.count_nonzero(
                ~np.isnan(results_norefined_df['vot'])) > 0:

            if len(results_norefined_df['vot']) == 1:
                ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['vot'], marker='o', markersize=2)
            else:
                ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['vot'])

            # ax[(0,0)].set_yticks(np.arange(results_norefined_df['vot'].min(), results_norefined_df['vot'].max(), 0.2))
            ax[(0, 0)].axhline(1, linestyle='dashed')
            # ax[(0, 0)].set_ylabel(r'$\hat{\theta_t}/\theta_{\hat{\sigma}}$')
            ax[(0, 0)].set_ylabel("value of travel time reliability")
            # ax[(0, 0)].yaxis.set_major_formatter(OOMFormatter(-4, "%1.1f"))
            # ax[(0, 0)].yaxis.set_major_formatter(yfmt1)
            ax[(0, 0)].tick_params(labelbottom=False)
            # ax[(0, 1)].yaxis.major.formatter._useMathText = True

            # - Refined

            if len(results_refined_df['vot']) == 1:
                ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['vot'], marker='o', markersize=2)
            else:
                ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['vot'])

            # ax[(0,1)].set_yticks(np.arange(results_refined_df['vot'].min(), results_refined_df['vot'].max(), 0.2))
            ax[(0, 1)].axhline(1, linestyle='dashed')
            # ax[(0, 1)].yaxis.set_major_formatter(yfmt2)
            ax[(0, 1)].tick_params(labelbottom=False)


        else:  # (no estimation of cost parameter but of travel time coefficient)
            # - No refined

            if len(results_norefined_df['theta_tt']) == 1:
                ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['theta_tt'], marker='o',
                                markersize=2)

            else:
                ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['theta_tt'])

            # ax[(0,0)].set_yticks(np.arange(results_norefined_df['vot'].min(), results_norefined_df['vot'].max(), 0.2))
            # ax[(0, 0)].set_ylabel(r'$\hat{\theta_t}$')
            ax[(0, 0)].set_ylabel("coefficient")
            # ax[(0, 0)].set_ticklabels([])

            ax[(0, 0)].tick_params(labelbottom=False)

            # - Refined
            if len(results_refined_df['theta_tt']) == 1:
                ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['theta_tt'], marker='o', markersize=2)
            else:
                ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['theta_tt'])

            # ax[(0,1)].set_yticks(np.arange(results_refined_df['vot'].min(), results_refined_df['vot'].max(), 0.2))
            ax[(0, 1)].tick_params(labelbottom=False)
            # ax[(0,1)].yaxis.set_major_formatter(yfmt2)

            if theta_true is None or 'tt' not in theta_true.keys():
                # Fresno case
                ax[(0, 0)].axhline(0, linestyle='dashed')
                ax[(0, 1)].axhline(0, linestyle='dashed')

            else:
                ax[(0, 0)].axhline(float(theta_true['tt']), linestyle='dashed')
                ax[(0, 1)].axhline(float(theta_true['tt']), linestyle='dashed')

        ax[(0, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1,
                                        int(np.ceil((results_norefined_df['iter'].max() - results_norefined_df[
                                            'iter'].min()) / 10))))

        ax[(0, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1,
                                        int(np.ceil((results_refined_df['iter'].max() - results_refined_df[
                                            'iter'].min()) / 10))))
        plt.setp(ax[(0, 1)].get_yticklabels(), visible=False)

        # ii) Objective function values

        # - No refined

        if len(results_norefined_df['objective']) == 1:
            ax[(1, 0)].plot(results_norefined_df['iter'], results_norefined_df['objective'], marker='o', markersize=2)
        else:
            ax[(1, 0)].plot(results_norefined_df['iter'], results_norefined_df['objective'])

        # ax[(1,0)].set_xticks(np.arange(0*results_norefined_df['iter'].min(), results_norefined_df['iter'].max()+1, results_norefined_df['iter'].max()/10))

        # if results_norefined_df['objective'].min() != results_norefined_df['objective'].max():
        #     ax[(1,0)].set_ylim(results_norefined_df['objective'].min(), results_norefined_df['objective'].max())

        # ax[(1, 0)].set_ylabel(r"$ ||(x(\hat{\theta})-\bar{x}||_2^2 $")
        ax[(1, 0)].set_ylabel("objective function")
        ax[(1, 0)].set_xlabel("iterations (" + methods[0] + ")")
        # ax[(1, 0)].yaxis.set_major_formatter(yfmt3)

        # - Refined
        if len(results_refined_df['objective']) == 1:
            ax[(1, 1)].plot(results_refined_df['iter'], results_refined_df['objective'], marker='o', markersize=2)
        else:
            ax[(1, 1)].plot(results_refined_df['iter'], results_refined_df['objective'])

        ax[(1, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1, int(
            np.ceil((results_norefined_df['iter'].max() - results_norefined_df['iter'].min()) / 10))))

        ax[(1, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))
        plt.setp(ax[(1, 1)].get_yticklabels(), visible=False)

        # if results_refined_df['objective'].min() != results_refined_df['objective'].max():
        #     ax[(1,1)].set_ylim(results_refined_df['objective'].min(), results_refined_df['objective'].max())

        ax[(1, 1)].set_xlabel("iterations (" + methods[1] + ")")
        # ax[(1, 1)].yaxis.set_major_formatter(yfmt4)

        for axi in fig.get_axes():
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        for axi in [ax[(1, 0)], ax[(1, 1)]]:
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axi.yaxis.set_major_formatter(yfmt)

        fig.tight_layout()

        if folder is not None:
            fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        return fig

    class OOMFormatter(matplotlib.ticker.ScalarFormatter):
        def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
            self.oom = order
            self.fformat = fformat
            matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

        def _set_orderOfMagnitude(self, nothing=None):
            self.orderOfMagnitude = self.oom

        def _set_format(self, vmin=None, vmax=None):
            self.format = self.fformat
            if self._useMathText:
                self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

    def convergence_models(self,
                    results_dfs: Dict[str, pd.DataFrame],
                    features: Dict[str, str],
                    filename: str,
                    folder: str = None):

        '''
     Plot convergence to the true ratio of theta and the reduction of the objective function over iterations

     :argument results

     '''

        if folder is None:
            folder = self.folder

        colors = ['gray', 'red', 'green', 'c', 'm', 'y', 'b', 'w']
        # colors = [u'black', u'g', u'r', u'c', u'm', u'y', u'k']

        n_models = len(list(results_dfs.keys()))

        # results_dfs['model_2']['theta_tt'] = 2*results_dfs['model_2']['theta_tt']

        fig,ax = plt.subplots(figsize=(int(3.5*n_models), 12), nrows = 4, ncols = n_models)
        # ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])
        axs_paths = ax[0, :]
        axs_loss = ax[1, :]
        axs_traveltime = ax[2,:]#plt.subplot(2, n_models, 1)
        axs_exogenous_parameters = ax[3,:]
        # axs_loss = plt.subplot(2, n_models, 2, sharex=axs_parameters)

        axs_traveltime[0].set_ylabel("coefficient")
        axs_exogenous_parameters[0].set_ylabel("exogenous attributes coefficients")
        axs_loss[0].set_ylabel("objective function")


        # Paths plots
        axs_paths[0].set_ylabel("Paths")

        #Y axes are shared and are removed from all plots starting from second column
        for row in range(0,ax.shape[0]):
            ax[row][0].get_shared_y_axes().join(ax[row][0], *ax[row][1:])
            for axi in ax[row][1:]:
                plt.setp(axi.get_yticklabels(), visible=False)



        # X ticks are shared within plots of the same column
        for n_model in range(n_models):
            ax[3][n_model].get_shared_x_axes().join(*[ax[2][n_model], ax[1][n_model],ax[0][n_model]])
            ax[-1][n_model].set_xlabel("iteration")
            for axi in ax[:][n_model]:
                plt.setp(axi.get_xticklabels(), visible=False)

            plt.setp(axs_traveltime[n_model].get_xticklabels(), visible=False)


        # axs_traveltime[0].get_shared_y_axes().join(axs_traveltime[0], *axs_traveltime[1:])
        # axs_exogenous_parameters[0].get_shared_y_axes().join(axs_traveltime[0], *axs_traveltime[1:])
        # axs_loss[0].get_shared_y_axes().join(axs_traveltime[0], *axs_traveltime[1:])
        # axs_paths[0].get_shared_y_axes().join(axs_paths[0], *axs_paths[1:])
        # axs_traveltime[0].get_shared_x_axes().join(*axs_traveltime)
        # axs_traveltime[0].get_shared_x_axes().join(*axs_traveltime)

        acc_iters = 0
        n_model = 0
        max_paths = 0
        for model, results_df in results_dfs.items():

            if results_df['n_paths'].max() > max_paths:
                max_paths = results_df['n_paths'].max()

            counter = 0
            for feature in features.keys():
                parameter = 'theta_' + feature
                if feature == 'tt':
                    axs_traveltime[n_model].plot('iter', parameter, data=results_df[['iter', parameter]],
                                                           label=feature, color = 'blue')
                if feature != 'tt':
                    if parameter in results_df.keys():
                        axs_exogenous_parameters[n_model].plot('iter', parameter, data=results_df[['iter',parameter]],
                                                               label=feature, color=colors[counter])
                    counter += 1

            # Set xticks

            for i in [0,1,2,3]:
                plt.sca(ax[i,n_model])
                plt.xticks(list(np.arange(results_df.iter.min()+1,results_df.iter.max()-1,2)) + [results_df.iter.max()],
                           list(np.arange(acc_iters+2,acc_iters+results_df.iter.max()-1,2))+ [acc_iters+results_df.iter.max()])

            acc_iters += results_df.shape[0]

            # Add horizontal line at y = 0 in parameter estimates plots
            axs_traveltime[n_model].axhline(0, linestyle='dashed', color = 'white')
            axs_exogenous_parameters[n_model].axhline(0, linestyle='dashed', color = 'white')
            # axs_traveltime[n_model].set_ylim(0, int(max_paths * 1.1))
            # axs_exogenous_parameters[n_model].set_ylim(0, int(max_paths * 1.1))

            axs_loss[n_model].plot('iter', 'objective', data=results_df[['iter', 'objective']])

            #Paths plots
            axs_paths[n_model].plot('iter', 'n_paths',
                                    data=results_df[['iter', 'n_paths']],
                                    label = 'total', color = 'black', linestyle='solid')
            axs_paths[n_model].plot('iter', 'n_paths_added',
                                    data=results_df[['iter', 'n_paths_added']],
                                    label = 'generated', color = 'black', linestyle='dashed')
            axs_paths[n_model].plot('iter', 'n_paths_effectively_added',
                                    data=results_df[['iter', 'n_paths_effectively_added']],
                                    label = 'added', color = 'black', linestyle='dotted')

            axs_paths[n_model].set_ylim(0,int(max_paths*1.1))

            # axs_paths[n_model].axhline(0, linestyle='dashed', color = 'gray')

            #Add proper legend
            # Feature Engineering model should have separate exogenous feature legend but using the same colors from their raw counterparts

            n_model += 1

        for axi in fig.get_axes():
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        # To write with scienfic notation in y axis
        class ScalarFormatterForceFormat(ScalarFormatter):
            def _set_format(self):  # Override function that finds format to use.
                self.format = "%0.1f"  # Give format here
        for axi in axs_loss:
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axi.yaxis.set_major_formatter(yfmt)

        for axi in axs_paths:
            yfmt = OOMFormatter(3, "%1.1f")
            yfmt.set_powerlimits((0, 0))
            axi.yaxis.set_major_formatter(yfmt)


        # Legend path plots
        lines1, labels1 = axs_paths[0].get_legend_handles_labels()

        fig.legend(lines1, labels1, title="paths", loc='upper center', ncol=3, prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.52, -0.25]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax[-1, 0].transAxes))

        # Legend attributes
        lines1, labels1 = axs_traveltime[1].get_legend_handles_labels()
        lines2, labels2 = axs_exogenous_parameters[1].get_legend_handles_labels()
        # lines2, labels2 = axs_exogenous_parameters[2].get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2

        #Replace labels
        for idx, label in zip(range(len(labels)),labels):
            labels[idx] = features.get(label, labels[idx])


        # g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)
        fig.legend(lines, labels, title="attributes", loc='upper center', ncol=len(features), prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.52, -0.5]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax[-1,0].transAxes))


        fig.tight_layout()

        plt.show()

        if folder is not None:
            fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        return fig

    def convergence_networks_experiment(self,
                                        results: Union[Dict, pd.DataFrame],
                                        filename: str,
                                        methods: [str],
                                        colors,
                                        labels,
                                        folder: str = None,
                                        theta_true=None):

        '''
     Plot convergence to the true ratio of theta and the reduction of the objective function over iterations

     :argument results

     '''

        if folder is None:
            folder = self.folder

        dim_subplots = (2, 2)

        fig = plt.figure(figsize=(7, 6))
        # ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])
        ax = {}
        ax[(0, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 1)
        ax[(0, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 2) #, sharey=ax[(0, 0)])
        # ax[(0, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 2)

        ax[(1, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 3)
        ax[(1, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 4, sharey=ax[(1, 0)])
        # ax[(1, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 4)

        # matplotlib.rcParams['text.usetex'] = True

        # i) Theta estimates in the two plots in the top

        # - No refined

        for network, color, label in zip(results.keys(), colors, labels):
            results_norefined_df = results[network][results[network]['stage'] == 'norefined']

            ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['theta_tt'], color=color, label=label)

        ax[(0, 0)].axhline(float(theta_true['tt']), linestyle='dashed', color = 'black')
        # ax[(0, 0)].set_ylabel(r'$\hat{\theta_t}$')
        ax[(0, 0)].set_ylabel("coefficient")
        # ax[(0, 0)].set_ticklabels([])

        ax[(0, 0)].tick_params(labelbottom=False)

        if results_norefined_df['theta_tt'].max() < 2 and results_norefined_df['theta_tt'].min() > -15:
            ax[(0, 0)].set_yticks(np.arange(-14, 2 + 2, 2))



        # - Refined
        for network, color, label in zip(results.keys(), colors, labels):
            results_refined_df = results[network][results[network]['stage'] == 'refined']
            ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['theta_tt'], color=color)

        ax[(0, 1)].axhline(float(theta_true['tt']), linestyle='dashed', color = 'black')
        ax[(0, 1)].tick_params(labelbottom=False)

        if results_refined_df['theta_tt'].max() < 2 and results_refined_df['theta_tt'].min() > -15:
            ax[(0, 1)].set_yticks(np.arange(-14, 2 + 2, 2))
        # ax[(0, 1)].yaxis.set_ticklabels([])
        # plt.setp(ax[(0, 1)].get_yticklabels(), visible=False)
        # ax[(0,1)].yaxis.set_major_formatter(yfmt2)

        ax[(0, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1,
                                        int(np.ceil((results_norefined_df['iter'].max() - results_norefined_df[
                                            'iter'].min()) / 10))))

        ax[(0, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1,
                                        int(np.ceil((results_refined_df['iter'].max() - results_refined_df[
                                            'iter'].min()) / 10))))

        # ii) Objective function values

        # - No refined
        for network, color, label in zip(results.keys(), colors, labels):
            results_norefined_df = results[network][results[network]['stage'] == 'norefined']

            ax[(1, 0)].plot(results_norefined_df['iter'], results_norefined_df['objective'], color=color, label=label)

        # if results_norefined_df['objective'].min() != results_norefined_df['objective'].max():
        #     ax[(1,0)].set_ylim(results_norefined_df['objective'].min(), results_norefined_df['objective'].max())

        ax[(1, 0)].axhline(0, linestyle='dashed')

        # ax[(1, 0)].set_ylabel(r"$ ||(x(\hat{\theta})-\bar{x}||_2^2 $")
        ax[(1, 0)].set_ylabel("objective function")
        ax[(1, 0)].set_xlabel("iterations (" + methods[0] + ")")
        # ax[(1, 0)].yaxis.set_major_formatter(yfmt3)

        # - Refined
        for network, color, label in zip(results.keys(), colors, labels):
            results_refined_df = results[network][results[network]['stage'] == 'refined']

            ax[(1, 1)].plot(results_refined_df['iter'], results_refined_df['objective'], color=color, label=label)

        ax[(1, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1, int(
            np.ceil((results_norefined_df['iter'].max() - results_norefined_df['iter'].min()) / 10))))

        ax[(1, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))

        #     ax[(1,1)].set_ylim(results_refined_df['objective'].min(), results_refined_df['objective'].max())
        # if results_refined_df['objective'].min() != results_refined_df['objective'].max():

        ax[(1, 1)].axhline(0, linestyle='dashed')

        ax[(1, 1)].set_xlabel("iterations (" + methods[1] + ")")
        # plt.setp(ax[(1, 1)].get_yticklabels(), visible=False)
        # ax[(1, 1)].yaxis.set_ticklabels([])
        # ax[(1, 1)].yaxis.set_major_formatter(yfmt4)

        # Change color style to white and black in box plots
        for axi in [ax[(0, 0)], ax[(0, 1)], ax[(1, 0)], ax[(1, 1)]]:
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        # Legend
        lines, labels = ax[(1, 1)].get_legend_handles_labels()
        # g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)
        fig.legend(lines, labels, loc='upper center', ncol=4, prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.52, -0.25]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax[(1, 1)].transAxes))

        fig.tight_layout()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        plt.show()

        return fig

    def convergence_network_experiment(self,
                                       results_df: Dict,
                                       colors,
                                       labels,
                                       filename: str,
                                       methods: [str],
                                       folder: str = None,
                                       theta_true=None):

        '''
        Plot convergence to the true ratio of theta and the reduction of the objective function over iterations

        :argument results

        '''

        if folder is None:
            folder = self.folder

        fig = plt.figure(figsize=(7, 6))

        dim_subplots = (2, 2)

        ax = {}
        ax[(0, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 1)
        ax[(0, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 2, sharey=ax[(0, 0)])

        ax[(1, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 3)
        ax[(1, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 4, sharey=ax[(1, 0)])

        # matplotlib.rcParams['text.usetex'] = True

        # i) Theta estimates in the two plots in the top

        # - No refined

        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_norefined_df = results_df[scenario][results_df[scenario]['stage'] == 'norefined']

            ax[(0, 0)].plot(results_norefined_df['iter'],
                            results_norefined_df['theta_tt'] / results_norefined_df['theta_c'], color=color,
                            label=label)

            ax[(0, 0)].axvline(results_norefined_df['objective'].argmin() + 1, linestyle='dotted', color=color)
            ax[(1, 0)].axvline(results_norefined_df['objective'].argmin() + 1, linestyle='dotted', color=color)

        ax[(0, 0)].axhline(float(theta_true['tt']) / float(theta_true['c']), linestyle='dashed', color='black')

        ax[(0, 0)].set_ylabel("value of time")
        # ax[(0, 0)].set_ylabel(r'$\hat{\theta_t}/\hat{\theta_c}$')
        # ax[(0, 0)].set_ticklabels([])

        ax[(0, 0)].tick_params(labelbottom=False)

        # - Refined
        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_refined_df = results_df[scenario][results_df[scenario]['stage'] == 'refined']
            # ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['theta_tt'], color=color, label=label)
            ax[(0, 1)].plot(results_refined_df['iter'],
                            results_refined_df['theta_tt'] / results_refined_df['theta_c'], color=color,
                            label=label)
            ax[(0, 1)].axvline(results_refined_df['objective'].argmin() + len(results_refined_df) + 1,
                               linestyle='dotted', color=color)
            ax[(1, 1)].axvline(results_refined_df['objective'].argmin() + len(results_refined_df) + 1,
                               linestyle='dotted', color=color)

        ax[(0, 1)].axhline(float(theta_true['tt']) / float(theta_true['c']), linestyle='dashed', color='black')
        ax[(0, 1)].tick_params(labelbottom=False)
        # ax[(0, 1)].yaxis.set_ticklabels([])
        plt.setp(ax[(0, 1)].get_yticklabels(), visible=False)
        # ax[(0,1)].yaxis.set_major_formatter(yfmt2)

        ax[(0, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1, int(
            np.ceil((results_norefined_df['iter'].max() - results_norefined_df['iter'].min()) / 10))))

        ax[(0, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))

        # ii) Objective function values

        # - No refined
        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_norefined_df = results_df[scenario][results_df[scenario]['stage'] == 'norefined']

            ax[(1, 0)].plot(results_norefined_df['iter'], results_norefined_df['objective'], color=color, label=label)

        # if results_norefined_df['objective'].min() != results_norefined_df['objective'].max():
        #     ax[(1,0)].set_ylim(results_norefined_df['objective'].min(), results_norefined_df['objective'].max())

        # ax[(1, 0)].set_ylabel(r"$ ||(x(\hat{\theta})-\bar{x}||_2^2 $")

        ax[(1, 0)].set_ylabel("objective function")
        ax[(1, 0)].set_xlabel("iterations (" + methods[0] + ")")
        # ax[(1, 0)].yaxis.set_major_formatter(yfmt3)

        # - Refined
        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_refined_df = results_df[scenario][results_df[scenario]['stage'] == 'refined']

            ax[(1, 1)].plot(results_refined_df['iter'], results_refined_df['objective'], color=color, label=label)

        ax[(1, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1, int(
            np.ceil((results_norefined_df['iter'].max() - results_norefined_df['iter'].min()) / 10))))

        ax[(1, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))

        # if results_refined_df['objective'].min() != results_refined_df['objective'].max():
        #     ax[(1,1)].set_ylim(results_refined_df['objective'].min(), results_refined_df['objective'].max())

        ax[(1, 1)].set_xlabel("iterations (" + methods[1] + ")")
        plt.setp(ax[(1, 1)].get_yticklabels(), visible=False)
        # ax[(1, 1)].yaxis.set_ticklabels([])
        # ax[(1, 1)].yaxis.set_major_formatter(yfmt4)

        for axi in [ax[(0, 0)], ax[(0, 1)], ax[(1, 0)], ax[(1, 1)]]:
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        for axi in [ax[(1, 0)], ax[(1, 1)]]:
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axi.yaxis.set_major_formatter(yfmt)

        # Legend
        lines, labels = ax[(1, 1)].get_legend_handles_labels()
        # ph = [plt.plot([], marker="", ls="")[0]]
        # lines = ph + lines
        # labels = ["travel time"] + labels
        # g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)
        fig.legend(lines, labels, loc='upper center', ncol=2, prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.52, -0.25]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax[(1, 1)].transAxes))

        fig.tight_layout()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        return fig

    def convergence_experiment_yang(self,
                                    results_df: Union[Dict, pd.DataFrame],
                                    colors,
                                    labels,
                                    filename: str,
                                    methods: [str],
                                    folder: str = None,
                                    theta_true=None):

        '''
     Plot convergence to the true ratio of theta and the reduction of the objective function over iterations

     :argument results

     '''

        if folder is None:
            folder = self.folder

        fig = plt.figure(figsize=(7, 6))
        # ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])
        ax = {}
        dim_subplots = (2, 2)

        ax[(0, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 1)
        ax[(0, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 2, sharey=ax[(0, 0)])

        ax[(1, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 3)
        ax[(1, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 4, sharey=ax[(1, 0)])

        # matplotlib.rcParams['text.usetex'] = True

        # i) Theta estimates in the two plots in the top

        # - No refined

        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_norefined_df = results_df[scenario][results_df[scenario]['stage'] == 'norefined']

            ax[(0, 0)].plot(results_norefined_df['iter'], results_norefined_df['theta_tt'], color=color, label=label)

        ax[(0, 0)].axhline(float(theta_true['tt']), linestyle='dashed', color='black')
        # ax[(0, 0)].set_ylabel(r'$\hat{\theta_t}$')
        ax[(0, 0)].set_ylabel("coefficient")
        # ax[(0, 0)].set_ticklabels([])

        ax[(0, 0)].tick_params(labelbottom=False)

        trans = transforms.blended_transform_factory(
            ax[(0, 0)].get_yticklabels()[0].get_transform(), ax[(0, 0)].transData)
        # ax[(0,0)].text(0, float(theta_true['tt']), "{:.0f}".format(float(theta_true['tt'])), transform=trans,ha="right", va="center", fontdict=None)

        # - Refined
        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_refined_df = results_df[scenario][results_df[scenario]['stage'] == 'refined']
            ax[(0, 1)].plot(results_refined_df['iter'], results_refined_df['theta_tt'], color=color, label=label)

        ax[(0, 1)].axhline(float(theta_true['tt']), linestyle='dashed', color='black')
        ax[(0, 1)].tick_params(labelbottom=False)
        # ax[(0, 1)].yaxis.set_ticklabels([])
        # plt.setp(ax[(0, 1)].get_yticklabels(), visible=False)

        # ax[(0,1)].yaxis.set_major_formatter(yfmt2)

        ax[(0, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1, int(
            np.ceil((results_norefined_df['iter'].max() - results_norefined_df['iter'].min()) / 10))))

        ax[(0, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))

        # ax[(0, 0)].set_yticks([-15, -10, -5, -1])

        # ii) Objective function values

        # - No refined
        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_norefined_df = results_df[scenario][results_df[scenario]['stage'] == 'norefined']

            ax[(1, 0)].plot(results_norefined_df['iter'], results_norefined_df['objective'], color=color, label=label)

        # if results_norefined_df['objective'].min() != results_norefined_df['objective'].max():
        #     ax[(1,0)].set_ylim(results_norefined_df['objective'].min(), results_norefined_df['objective'].max())

        ax[(1, 0)].axhline(0, linestyle='dashed', color='black')

        # ax[(1, 0)].set_ylabel(r"$ ||(x(\hat{\theta})-\bar{x}||_2^2 $")
        ax[(1, 0)].set_ylabel("objective function")
        ax[(1, 0)].set_xlabel("iterations (" + methods[0] + ")")
        # ax[(1, 0)].yaxis.set_major_formatter(yfmt3)

        ax[(1, 0)].set_xticks(np.arange(results_norefined_df['iter'].min(), results_norefined_df['iter'].max() + 1, int(
            np.ceil((results_norefined_df['iter'].max() - results_norefined_df['iter'].min()) / 10))))

        # - Refined
        for scenario, color, label in zip(results_df.keys(), colors, labels):
            results_refined_df = results_df[scenario][results_df[scenario]['stage'] == 'refined']

            ax[(1, 1)].plot(results_refined_df['iter'], results_refined_df['objective'], color=color, label=label)

        ax[(1, 1)].axhline(0, linestyle='dashed', color='black')

        ax[(1, 1)].set_xticks(np.arange(results_refined_df['iter'].min(), results_refined_df['iter'].max() + 1, int(
            np.ceil((results_refined_df['iter'].max() - results_refined_df['iter'].min()) / 10))))

        ax[(1, 1)].set_xlabel("iterations (" + methods[1] + ")")
        # plt.setp(ax[(1, 1)].get_yticklabels(), visible=False)
        # ax[(1, 1)].yaxis.set_ticklabels([])
        # ax[(1, 1)].yaxis.set_major_formatter(yfmt4)

        # Change color style to white and black in box plots
        for axi in [ax[(0, 0)], ax[(0, 1)], ax[(1, 0)], ax[(1, 1)]]:
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        for axi in [ax[(1, 0)], ax[(1, 1)]]:
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axi.yaxis.set_major_formatter(yfmt)

        # Legend
        lines, labels = ax[(1, 1)].get_legend_handles_labels()
        # g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)
        fig.legend(lines, labels, loc='upper center', ncol=2, prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.52, -0.25]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax[(1, 1)].transAxes))

        fig.tight_layout()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        return fig

    def monotonocity_traffic_count_functions(self,
                                             filename,
                                             traffic_count_links_df: pd.DataFrame,
                                             folder: str = None):

        if folder is None:
            folder = self.folder

        # matplotlib.rcParams['text.usetex'] = True

        fig, ax = plt.subplots(figsize=(5, 4))

        # Plot

        traffic_count_links_df_plot = traffic_count_links_df.pivot(index='theta', columns='link', values='count')

        # ax.plot(traffic_count_links_df_plot)

        traffic_count_links_df_plot.plot(ax=ax, legend=False)
        # ax.legend()

        # plt.show()

        # TODO: Add horizontal lines for the true traffic counts. Interpolate curves so they are smoother

        # ax.axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)

        # plt.setp(ax, xlabel=r"$\hat{\theta}$", ylabel=r"$\hat{x(\theta)}$")

        if matplotlib.rcParams['text.usetex']:
            x_label = r"$\hat{\theta}_t$"
            y_label = r"$\hat{x(\theta)}$"
        else:
            x_label = "coefficient"
            y_label = "traffic flow"

        # Set font sizes
        for axi in reversed(fig.get_axes()):
            axi.set_xlabel(x_label)
            axi.set_ylabel(y_label)
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)
            # axi.legend(prop=dict(size=18))

            # for label in (axi.get_xticklabels() + axi.get_yticklabels()):
            #     # label.set_fontname('Arial')
            #     label.set_fontsize(16)

        # Legend
        lines, labels = ax.get_legend_handles_labels()
        # g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)
        fig.legend(lines, labels, loc='upper center', ncol=5, prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.55, -0.15]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax.transAxes))

        # fig.legend()
        # plt.legend()
        fig.tight_layout()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        return fig

    def pseudoconvexity_objective_function_small_networks_lite(self,
                                                               filename: str,
                                                               f_vals: Dict,
                                                               x_range: List,
                                                               theta_true: Dict,
                                                               colors,
                                                               labels,
                                                               sign_first_derivative: bool = False,
                                                               folder: str = None):

        if folder is None:
            folder = self.folder

        # matplotlib.rcParams['text.usetex'] = True

        dim_subplots = (2, 2)

        fig = plt.figure(figsize=(7, 6))
        # ax = fig.subplots(nrows=self.dim_subplots[0], ncols=self.dim_subplots[1])
        ax = {}
        ax[(0, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 1)
        ax[(0, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 2)
        # ax[(0, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 2)

        ax[(1, 0)] = plt.subplot(dim_subplots[0], dim_subplots[1], 3)
        ax[(1, 1)] = plt.subplot(dim_subplots[0], dim_subplots[1], 4)

        for i, j, k, color, label in zip(f_vals.keys(), f_vals.values(), range(0, len(f_vals)), colors, labels):
            pos_plot = int(np.floor(k / dim_subplots[0])), int(k % dim_subplots[1])
            ax[pos_plot].plot(x_range, j, color=color, label=label)
            ax[pos_plot].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
            ax[pos_plot].axhline(y=0, color=color, linestyle='dashed', linewidth=0.5)
            # ax[(0, 0)].set_ylabel(r"$ ||x(\hat{\theta})-\bar{x}||_2^2$")

        ax[(0, 0)].set_xticklabels([])
        ax[(0, 1)].set_xticklabels([])

        ax[(1, 0)].set_xticks(np.arange(int(round(min(x_range))), int(round(max(x_range))) + 0.1, 5))
        ax[(1, 1)].set_xticks(np.arange(int(round(min(x_range))), int(round(max(x_range))) + 0.1, 5))

        # ax[(1, 0)].set_ylabel("objective function")
        # ax[(1, 0)].set_xlabel("iterations (" + methods[0] + ")")

        # Legend

        lines, labels = [], []
        for axi in reversed(fig.get_axes()):
            # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            if not sign_first_derivative:
                yfmt = ScalarFormatterForceFormat()
                yfmt.set_powerlimits((0, 0))
                axi.yaxis.set_major_formatter(yfmt)
            else:
                axi.set_yticks([-1, 0, 1])

            linei, labeli = axi.get_legend_handles_labels()
            lines = linei + lines
            labels = labeli + labels
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        if sign_first_derivative:

            plt.setp(ax[(0, 1)].get_yticklabels(), visible=False)
            plt.setp(ax[(1, 1)].get_yticklabels(), visible=False)

            # set labels
            if matplotlib.rcParams['text.usetex']:
                _label = r"$\hat{\theta}_t$"
                y_label = r"$\nabla_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2)$"
            else:
                x_label = "coefficient"
                y_label = "sign of first derivative"

        else:

            # set labels
            if matplotlib.rcParams['text.usetex']:
                x_label = r"$\hat{\theta}_t$"
                y_label = r"$||x(\hat{\theta})-\bar{x}||_2^2$"
            else:
                x_label = "coefficient"
                y_label = "objective function"

        plt.setp(ax[(1,0)], xlabel=x_label)
        plt.setp(ax[(1, 1)], xlabel=x_label)
        plt.setp(ax[(0, 0)], ylabel=y_label)
        plt.setp(ax[(1, 0)], ylabel=y_label)


        fig.legend(lines, labels, loc='upper center', ncol=4, prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.52, -0.25]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax[(1, 1)].transAxes))

        fig.tight_layout()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

    def pseudoconvexity_objective_function_small_networks(self,
                                                          filename,
                                                          f_vals,
                                                          grad_f_vals,
                                                          hessian_f_vals,
                                                          x_range,
                                                          theta_true,
                                                          alpha_bh=0, folder: str = None):

        if folder is None:
            folder = self.folder

        # matplotlib.rcParams['text.usetex'] = True

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))
        # fig.suptitle("Analysis of  (strict) quasiconvexity of L2 norm"
        #              "\n(theta_true = " + str(theta_true) + ")")

        # Plot objective function over an interval
        # ax[(0, 0)].set_title("\n\nObj. function (L2)")
        ax[(0, 0)].set_title("\n\n")
        y_vals = f_vals
        ax[(0, 0)].plot(x_range, y_vals, color='red')
        ax[(0, 0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 0)].set_xticklabels([])

        if matplotlib.rcParams['text.usetex']:
            y_label = r"$||x(\hat{\theta})-\bar{x}||_2^2$"
        else:
            y_label = "objective function"

        ax[(0, 0)].set_ylabel(y_label)

        # r"$\hat{\theta}$"

        # ax[(0, 1)].set_title("Gradient L2-norm")
        # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
        y_vals = grad_f_vals
        ax[(0, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 1)].plot(x_range, y_vals, color='red')
        ax[(0, 1)].set_xticklabels([])

        if matplotlib.rcParams['text.usetex']:
            y_label = r"$\nabla_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2)$"
        else:
            y_label = "First derivative"

        ax[(0, 1)].set_ylabel(y_label)

        # ax[(0, 2)].set_title("Sign Gradient L2-norm")
        # y_vals = [np.sign(np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1))) for x_val in x_range]

        # To avoid zero gradient (and sign) because numerical precision
        epsilon = 1e-12
        y_vals = np.sign(np.array(y_vals)+epsilon)
        ax[(1, 0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        ax[(1, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        ax[(1, 0)].plot(x_range, y_vals, color='red')
        # ax[(1, 0)].set_xticks(np.arange(int(round(min(x_range))), int(round(max(x_range))) + 0.1, 5))

        if matplotlib.rcParams['text.usetex']:
            y_label = r"$\textmd{sign} (\nabla_{\theta} ||x(\hat{\theta})-\bar{x}||_2^2 )$"
        else:
            y_label = "Sign of first derivative"

        ax[(1, 0)].set_ylabel(y_label)
        ax[(1, 0)].set_yticks([-1, 0, 1])
        # plt.setp(ax[(1, 0)].get_yticklabels(), visible=False)
        # plt.setp(ax[(1, 1)].get_yticklabels(), visible=False)

        # ax[(1, 0)].set_xticklabels([])

        # Hessian L2-norm
        # ax[(1, 3)].set_title("Hessian L2 norm")

        # J = gradients_l2norm(theta, deltatt, q, linkflow)
        # H = np.sum(q * hessian_sigmoid(theta=theta, deltatt=deltatt), axis=1)
        # R = np.sum(objective_function_sigmoids_system(theta, q, deltatt), axis=1) - linkflow.T

        # [np.sum(q * hessian_sigmoid(theta=x_val, deltatt=deltatt), axis=1) for x_val in x_range]

        # y_vals = hessian_f_vals
        # # y_vals = np.sign(y_vals)
        # ax[(1, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1, 1)].plot(x_range, y_vals, color='red', )
        # ax[(1, 1)].set_ylabel(r"$\nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2) $")

        # Sign Hessian L2-norm

        y_vals = np.sign(hessian_f_vals)
        # y_vals = hessian_f_vals
        # y_vals = np.sign(y_vals)
        ax[(1, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        ax[(1, 1)].plot(x_range, y_vals, color='red', )
        ax[(1, 1)].set_xticks(np.arange(int(round(min(x_range))), int(round(max(x_range))) + 0.1, 5))

        if matplotlib.rcParams['text.usetex']:
            y_label = r"$\textmd{sign} (\nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2))$"
        else:
            y_label = "Sign of second derivative"

        ax[(1, 1)].set_ylabel(y_label)
        ax[(1, 1)].set_yticks([-1, 0, 1])

        # ax[(0, 2)].set_title("Hessian L2-norm")
        # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
        # ax[(0,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(0,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(0,1)].plot(x_range, y_vals,color='red')

        # #Plot sigmoid system
        # ax[(1,0)].set_title("L1 norm")
        # y_vals = [np.mean(np.abs(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)) for x_val in x_range]
        # ax[(1,0)].plot(x_range, y_vals,color = 'red')
        # ax[(1,0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1,0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1,0)].set_title("Sigmoid system")
        # ax[(0, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(0, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(1, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(1, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        if matplotlib.rcParams['text.usetex']:
            x_label = r"$\hat{\theta}$"
        else:
            x_label = "coefficient"

        lines, labels = [], []
        for axi in fig.get_axes():
            axi.set_xlabel(x_label)
            # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            linei, labeli = axi.get_legend_handles_labels()
            lines = linei + lines
            labels = labeli + labels
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        for axi in [ax[(0, 0)],ax[(0, 1)]]:
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axi.yaxis.set_major_formatter(yfmt)

        # plt.setp(ax[-1, :], xlabel=x_label)
        # plt.setp(ax[:, 0], ylabel=r"$\theta_i$")

        fig.tight_layout()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

    def coordinatewise_pseudoconvexity_loss_function(self,
                                                     filename,
                                                     results_df,
                                                     x_range,
                                                     theta_true,
                                                     colors,
                                                     labels,
                                                     folder: str = None,
                                                     xticks=None):

        if folder is None:
            folder = self.folder

        # matplotlib.rcParams['text.usetex'] = True

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))
        # fig.suptitle("Analysis of  (strict) quasiconvexity of L2 norm"
        #              "\n(theta_true = " + str(theta_true) + ")")

        # Attributes
        attrs = list(results_df['attr'].unique())

        # Plot objective function over an interval
        # ax[(0, 0)].set_title("\n\nObj. function (L2)")
        ax[(0, 0)].set_title("\n\n")
        ax[(0, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 0)].set_xticklabels([])
        if matplotlib.rcParams['text.usetex']:
            y_label = r"$||x(\hat{\theta})-\bar{x}||_2^2$"
        else:
            y_label = "objective function"

        ax[(0, 0)].set_ylabel(y_label)

        for attr, color, label in zip(attrs, colors, labels):
            y_vals = results_df[results_df['attr'] == attr]['f_vals']
            ax[(0, 0)].plot(x_range, y_vals, color=color, label=label)
            ax[(0, 0)].axvline(x=theta_true[attr], color=color, linestyle='dashed', linewidth=0.5)

        # r"$\hat{\theta}$"

        # ax[(0, 1)].set_title("Gradient L2-norm")
        # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
        ax[(0, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 1)].set_xticklabels([])
        if matplotlib.rcParams['text.usetex']:
            y_label = r"$\nabla_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2)$"
        else:
            y_label = "first derivative"

        ax[(0, 1)].set_ylabel(y_label)

        for attr, color, label in zip(attrs, colors, labels):
            y_vals = results_df[results_df['attr'] == attr]['grad_f_vals']
            ax[(0, 1)].plot(x_range, y_vals, color=color, label=label)
            ax[(0, 1)].axvline(x=theta_true[attr], color=color, linestyle='dashed', linewidth=0.5)

        # ax[(0, 2)].set_title("Sign Gradient L2-norm")
        # y_vals = [np.sign(np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1))) for x_val in x_range]
        ax[(1, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)

        if matplotlib.rcParams['text.usetex']:
            y_label = r"$\textmd{sign} (\nabla_{\theta} ||x(\hat{\theta})-\bar{x}||_2^2 )$"
        else:
            y_label = "sign of first derivative"

        ax[(1, 0)].set_ylabel(y_label)
        ax[(1, 0)].set_yticks([-1, 0, 1])

        for attr, color, label in zip(attrs, colors, labels):
            y_vals = np.sign(results_df[results_df['attr'] == attr]['grad_f_vals'])
            ax[(1, 0)].plot(x_range, y_vals, color=color, label=label)
            ax[(1, 0)].axvline(x=theta_true[attr], color=color, linestyle='dashed', linewidth=0.5)

        # ax[(1, 0)].set_xticks(np.arange(int(min(x_range)), int(max(x_range)), 3))
        # ax[(1, 0)].set_xticklabels([])

        # Hessian L2-norm
        # ax[(1, 3)].set_title("Hessian L2 norm")

        # J = gradients_l2norm(theta, deltatt, q, linkflow)
        # H = np.sum(q * hessian_sigmoid(theta=theta, deltatt=deltatt), axis=1)
        # R = np.sum(objective_function_sigmoids_system(theta, q, deltatt), axis=1) - linkflow.T

        # [np.sum(q * hessian_sigmoid(theta=x_val, deltatt=deltatt), axis=1) for x_val in x_range]

        # y_vals = hessian_f_vals
        # # y_vals = np.sign(y_vals)
        # ax[(1, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1, 1)].plot(x_range, y_vals, color='red', )
        # ax[(1, 1)].set_ylabel(r"$\nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2) $")

        # Sign Hessian L2-norm

        # y_vals = hessian_f_vals
        # y_vals = np.sign(y_vals)
        ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)

        if matplotlib.rcParams['text.usetex']:
            y_label = r"$\textmd{sign} (\nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2))$"
        else:
            y_label = "sign of second derivative"

        ax[(1, 1)].set_ylabel(y_label)
        ax[(1, 1)].set_yticks([-1, 0, 1])

        for attr, color, label in zip(attrs, colors, labels):
            y_vals = np.sign(results_df[results_df['attr'] == attr]['hessian_f_vals'])
            ax[(1, 1)].plot(x_range, y_vals, color=color, label=label)
            ax[(1, 1)].axvline(x=theta_true[attr], color=color, linestyle='dashed', linewidth=0.5)

        # ax[(0, 2)].set_title("Hessian L2-norm")
        # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
        # ax[(0,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(0,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(0,1)].plot(x_range, y_vals,color='red')

        # #Plot sigmoid system
        # ax[(1,0)].set_title("L1 norm")
        # y_vals = [np.mean(np.abs(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)) for x_val in x_range]
        # ax[(1,0)].plot(x_range, y_vals,color = 'red')
        # ax[(1,0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1,0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1,0)].set_title("Sigmoid system")
        # ax[(0, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(0, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(1, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(1, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        if matplotlib.rcParams['text.usetex']:
            x_label = r"$\hat{\theta}$"
        else:
            x_label = "coefficient"

        lines, labels = [], []
        for axi in fig.get_axes():
            axi.set_xlabel(x_label)

            # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            if xticks is not None:
                axi.set_xticks(xticks)
            linei, labeli = axi.get_legend_handles_labels()
            lines = linei + lines
            labels = labeli + labels
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        for axi in [ax[(0, 0)],ax[(0, 1)]]:
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axi.yaxis.set_major_formatter(yfmt)


        # # Set font sizes
        # for axi in fig.get_axes():
        #
        #     axi.xaxis.get_label().set_fontsize(10)
        #     axi.yaxis.get_label().set_fontsize(10)
        #     # axi.legend(prop=dict(size=18))
        #
        #     for label in (axi.get_xticklabels() + axi.get_yticklabels()):
        #         # label.set_fontname('Arial')
        #         label.set_fontsize(12)

        # set labels
        # plt.setp(ax[-1, :], xlabel=r"$\hat{\theta}$")

        # plt.setp(ax[:, 0], ylabel=r"$\theta_i$")

        # Legend
        lines, labels = ax[(1, 1)].get_legend_handles_labels()
        # g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)
        fig.legend(lines, labels, loc='upper center', ncol=2, prop={'size': self.fontsize}
                   , bbox_to_anchor=[0.52, -0.25]
                   , bbox_transform=BlendedGenericTransform(fig.transFigure, ax[(1, 1)].transAxes))

        fig.tight_layout()

        plt.show()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

    def pseudoconvexity_loss_function(self,
                                      filename,
                                      f_vals,
                                      x_range,
                                      xticks,
                                      theta_true=0,
                                      folder: str = None,
                                      grad_f_vals=None,
                                      hessian_f_vals=None,
                                      color=None):

        if folder is None:
            folder = self.folder

        # matplotlib.rcParams['text.usetex'] = True

        if color is None:
            color = 'red'

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))
        # fig.suptitle("Analysis of  (strict) quasiconvexity of L2 norm"
        #              "\n(theta_true = " + str(theta_true) + ")")

        # Plot objective function over an interval
        # ax[(0, 0)].set_title("\n\nObj. function (L2)")
        ax[(0, 0)].set_title("\n\n")
        y_vals = f_vals
        ax[(0, 0)].plot(x_range, y_vals, color=color)
        ax[(0, 0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        ax[(0, 0)].set_xticklabels([])
        if matplotlib.rcParams['text.usetex']:
            y_label = r"$||x(\hat{\theta})-\bar{x}||_2^2$"
        else:
            y_label = "objective function"

        ax[(0, 0)].set_ylabel(y_label)

        # r"$\hat{\theta}$"

        if grad_f_vals is not None:
            # ax[(0, 1)].set_title("Gradient L2-norm")
            # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
            y_vals = grad_f_vals
            ax[(0, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
            ax[(0, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
            ax[(0, 1)].plot(x_range, y_vals, color=color)
            ax[(0, 1)].set_xticklabels([])
            if matplotlib.rcParams['text.usetex']:
                y_label = r"$\nabla_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2)$"
            else:
                y_label = "first derivative"
            ax[(0, 1)].set_ylabel(y_label)

            # ax[(0, 2)].set_title("Sign Gradient L2-norm")
            # y_vals = [np.sign(np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1))) for x_val in x_range]
            y_vals = np.sign(y_vals)
            ax[(1, 0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
            ax[(1, 0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
            ax[(1, 0)].plot(x_range, y_vals, color=color)
            if matplotlib.rcParams['text.usetex']:
                y_label = r"$\textmd{sign} (\nabla_{\theta} ||x(\hat{\theta})-\bar{x}||_2^2 )$"
            else:
                y_label = "sign of first derivative"

            ax[(1, 0)].set_ylabel(y_label)
            ax[(1, 0)].set_yticks([-1, 0, 1])

            # ax[(1, 0)].set_xticks(np.arange(int(min(x_range)), int(max(x_range)), 3))
            # ax[(1, 0)].set_xticklabels([])

            # Hessian L2-norm
            # ax[(1, 3)].set_title("Hessian L2 norm")

            # J = gradients_l2norm(theta, deltatt, q, linkflow)
            # H = np.sum(q * hessian_sigmoid(theta=theta, deltatt=deltatt), axis=1)
            # R = np.sum(objective_function_sigmoids_system(theta, q, deltatt), axis=1) - linkflow.T

            # [np.sum(q * hessian_sigmoid(theta=x_val, deltatt=deltatt), axis=1) for x_val in x_range]

            # y_vals = hessian_f_vals
            # # y_vals = np.sign(y_vals)
            # ax[(1, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
            # ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
            # ax[(1, 1)].plot(x_range, y_vals, color='red', )
            # ax[(1, 1)].set_ylabel(r"$\nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2) $")

            # Sign Hessian L2-norm

        if hessian_f_vals is not None:
            y_vals = np.sign(hessian_f_vals)
            # y_vals = hessian_f_vals
            # y_vals = np.sign(y_vals)
            ax[(1, 1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
            ax[(1, 1)].plot(x_range, y_vals, color=color, )
            ax[(1, 1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)

            if matplotlib.rcParams['text.usetex']:
                y_label = r"$\textmd{sign} (\nabla^2_{\theta} (||x(\hat{\theta})-\bar{x}||_2^2))$"
            else:
                y_label = "sign of second derivative"

            ax[(1, 1)].set_ylabel(y_label)
            ax[(1, 1)].set_yticks([-1, 0, 1])

        # ax[(0, 2)].set_title("Hessian L2-norm")
        # y_vals = [np.mean(2*(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)*np.sum(q*gradient_sigmoid(theta = x_val, deltatt = deltatt),axis = 1)) for x_val in x_range]
        # ax[(0,1)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(0,1)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(0,1)].plot(x_range, y_vals,color='red')

        # #Plot sigmoid system
        # ax[(1,0)].set_title("L1 norm")
        # y_vals = [np.mean(np.abs(np.sum(objective_function_sigmoids_system(x_val, q = q, deltatt = deltatt),axis = 1)-linkflow.T)) for x_val in x_range]
        # ax[(1,0)].plot(x_range, y_vals,color = 'red')
        # ax[(1,0)].axvline(x=theta_true, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1,0)].axhline(y=0, color='black', linestyle='dashed', linewidth=0.5)
        # ax[(1,0)].set_title("Sigmoid system")
        # ax[(0, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(0, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(1, 0)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        # ax[(1, 1)].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        lines, labels = [], []
        # set labels
        # plt.setp(ax[-1, :], xlabel=r"$\hat{\theta}$")
        # plt.setp(ax[:, 0], ylabel=r"$\theta_i$")

        if matplotlib.rcParams['text.usetex']:
            x_label = r"$\hat{\theta}$"
        else:
            x_label = "coefficient"

        for axi in fig.get_axes():
            axi.set_xlabel(x_label)
            # axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            yfmt = ScalarFormatterForceFormat()
            yfmt.set_powerlimits((0, 0))
            axi.set_xlim(min(x_range), max(x_range))
            axi.set_xticks(xticks)
            axi.set_xticklabels(xticks)
            axi.yaxis.set_major_formatter(yfmt)
            linei, labeli = axi.get_legend_handles_labels()
            lines = linei + lines
            labels = labeli + labels
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)

        fig.tight_layout()

        plt.show()

        fig.savefig(folder + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

    def consistency_experiment(self,
                               results_experiment,
                               alpha,
                               sd_x=None,
                               folder=None,
                               range_initial_values=None,
                               summary_only=True,
                               silent_mode=True):

        assert 'method' in results_experiment.keys()

        theta_true = pd.Series(data=results_experiment['theta_true'].values,
                               index=results_experiment['attr']).drop_duplicates()

        methods = results_experiment[['method']].drop_duplicates().values.flatten().tolist()

        if folder is None:
            folder = self.folder

        # Computation of False positives and negatives
        false_negatives, false_positives = False, False

        if 'fp' in list(set(results_experiment.f_type.values)):
            false_positives = True

        if 'fn' in list(set(results_experiment.f_type.values)):
            false_negatives = True

        # x) Summary plot (computation time, ci_width, average bias and proportion of false positives/negatives
        # matplotlib.rcParams['text.usetex'] = False

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))

        # 1) bias in estimates
        sns.boxplot(x="method", y="bias", data=results_experiment[results_experiment.attr != 'vot'], ax=ax[(0, 0)],
                    showfliers=False, color='white')
        # ax[(0, 0)].set_ylabel('bias ' + r"$(\hat{\theta})$")
        ax[(0, 0)].set_ylabel('bias in coefficients')
        if range_initial_values is not None:
            ax[(0, 0)].set(ylim=range_initial_values)

        # 2) Bias in value of time
        sns.boxplot(x="method", y="bias",
                    data=results_experiment[results_experiment.attr == 'vot'],
                    showfliers=False, ax=ax[(0, 1)], color='white')
        # ax[(0, 1)].set_ylabel('bias ' + r"$(\hat{\theta}_t/\hat{\theta}_c)$")
        ax[(0, 1)].set_ylabel('bias in value of time')
        # if range_initial_values is not None:
        #     ax[(0, 1)].set(ylim=range_initial_values)
        ax[(0, 1)].set(ylim=(-0.2, 0.2))

        if false_negatives:
            # 3) False negatives
            sns.barplot(x="method", y="fn",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fn')],
                        ax=ax[(1, 0)], color='white', edgecolor="black", linewidth=1.5, errwidth=1.5)

            ax[(1, 0)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
            ax[(1, 0)].axhline(0, linestyle='--', color='gray')
            ax[(1, 0)].set(ylim=(0, 1.05))
            ax[(1, 0)].set_ylabel('false negatives')

        if false_positives:

            # 4) False positives
            sns.barplot(x="method", y="fp",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fp')],
                        ax=ax[(1, 1)], color='white', edgecolor="black", linewidth=1.5, errwidth=1.5)
            # sns.catplot(x="level", y="fn", hue = 'kind', kind = 'point'
            #     , data=results_experiment[(results_experiment.attr != 'vot') & (results_experiment.f_type == 'fp')], ax=ax[(1, 1)], color = 'white',errcolor="black", edgecolor="black",linewidth=1.5, errwidth=1.5)
            ax[(1, 1)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
            ax[(1, 1)].set(ylim=(0, 1.05))
            ax[(1, 1)].set_ylabel('false positives')

        elif sd_x is not None:

            # NRMSE
            sns.barplot(x="method", y="nrmse", data=results_experiment, color="white",
                        errcolor="black", edgecolor="black", linewidth=1.5, errwidth=1.5, ax=ax[(1, 1)])
            ax[(1, 1)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
            ax[(1, 1)].set(ylim=(0, 1))
            ax[(1, 1)].set_ylabel("nrmse")


        else:
            # computation time
            sns.barplot(x="method", y="time", data=results_experiment, ax=ax[(1, 1)]
                        , color='white', edgecolor="black", linewidth=1.5, errwidth=1.5)
            ax[(1, 1)].set_ylabel('comp. time [s/rep.]')
            ax[(1, 1)].axhline(0, linestyle='--', color='gray')

        # Change color style to white and black in box plots
        for axi in [ax[(0, 0)], ax[(0, 1)]]:
            for i, box in enumerate(axi.patches):
                box.set_edgecolor('black')
                box.set_facecolor('white')

            plt.setp(axi.patches, edgecolor='k', facecolor='w')
            plt.setp(axi.lines, color='k')

            plt.setp(axi.artists, edgecolor='black', facecolor='white')

        ax[(0, 0)].axhline(0, linestyle='--', color='gray')
        ax[(0, 1)].axhline(0, linestyle='--', color='gray')

        if false_positives:
            ax[(1, 1)].axhline(alpha, linestyle='--', color='gray')
        elif sd_x is not None:
            ax[(1, 1)].axhline(sd_x, linestyle='--', color='gray')

        fig.tight_layout()
        # plt.show()

        self.save_fig(fig, folder, 'inference_summary')

        if summary_only:
            return

        # Plot of false positives and negatives

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

        if 'fp' in list(set(results_experiment.f_type.values)):
            sns.barplot(x="method", y="fp",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fp')]
                        , ax=ax[(1)])

        if 'fn' in list(set(results_experiment.f_type.values)):
            sns.barplot(x="method", y="fn",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fn')]
                        , ax=ax[(0)])

        for axi in fig.get_axes():
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_xlabel('')

        ax[0].set_ylabel('false negatives')
        ax[1].set_ylabel('false positives')

        ax[0].axhline(0, linestyle='--', color='gray')
        ax[1].axhline(0, linestyle='--', color='gray')

        ax[0].set(ylim=(-0.05, 1.05))
        ax[1].set(ylim=(-0.05, 1.05))

        fig.tight_layout()

        self.save_fig(fig, folder, 'falseposneg')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (i) Facet of box plots for distribution of  parameter estimates and among estimation methods

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='method', y='theta', palette="Set1",
            col='attr', kind='box', col_wrap=2, sharey=False, sharex=False, showfliers=False, height=2, color='#abc9ea',
        )

        def horizontal_theta_true_line(x, **kwargs):
            plt.axhline(theta_true[str(list(x)[0])], linestyle='--', color='gray')

        g.map(horizontal_theta_true_line, 'attr')

        titles = ['travel time', 'waiting time', 'cost', 'null parameter']
        for axi, title in zip(g.axes, titles):
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_title(title)
            # axi.tick_params(labelbottom=False)

        list(map(lambda x: x.tick_params(labelbottom=False), g.axes[0:len(g.axes) - 2]))

        g.set_axis_labels('', "estimate")

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'parameter_estimates')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # fig.savefig(self.folder + '/' + network_name + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        # (ii) Facet of box plots for distribution of bias in parameter estimates and estimation methods.

        # bias computation
        # results_experiment['theta_true'] = 0

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='attr', y='bias', palette="Set1",
            col='method', kind='box', col_wrap=2, sharey=True, sharex=False, showfliers=False, height=2, color='#abc9ea'
        )

        def horizontal_theta_true_line(x, **kwargs):
            plt.axhline(0, linestyle='--', color='gray')

        g.map(horizontal_theta_true_line, 'attr')

        g.set_axis_labels("attribute", "bias")

        # titles = ['no refined opt', 'refined opt', 'combined opt']
        # titles = ['ngd', 'gn', 'ngd+gn']
        # titles = methods
        # for ax, title in zip(g.axes.flatten(), titles):
        #     ax.set_title(title)
        # ax.tick_params(labelbottom=True)
        # plt.ylabel()

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'parameter_bias')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (iii) VOT distribution only

        # sns.boxplot(x="method", y="ttest", data = results_experiment_plot[results_experiment_plot.cols == 'ttest-norefined']
        #             , showfliers=False )
        # plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

        sns.boxplot(x="method", y="theta", data=results_experiment[results_experiment.attr == 'vot'], showfliers=False,
                    ax=ax[(0)])
        ax[0].set_ylabel('value of time')
        ax[0].axhline(theta_true['vot'], linestyle='--', color='gray')

        sns.boxplot(x="method", y="bias", data=results_experiment[results_experiment.attr == 'vot'], showfliers=False,
                    ax=ax[(1)])
        ax[1].set_ylabel('bias')
        ax[1].axhline(0, linestyle='--', color='gray')

        for axi in fig.get_axes():
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_xlabel('')

        fig.tight_layout()

        # g.set_axis_labels("state", "value of time")

        self.save_fig(fig, folder, 'vot_estimate_bias')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # iv) Distribution plot of t-tests

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

        stats_list = ['ttest', 'method']
        results_experiment_plot = pd.DataFrame(
            data=np.c_[results_experiment[['ttest', 'method']], results_experiment['attr']]
            , columns=stats_list + ['attr'])

        results_experiment_plot = results_experiment_plot[results_experiment_plot.attr != 'vot']

        # reference: https://stackoverflow.com/questions/46045750/python-distplot-with-multiple-distributions

        # recast into long format
        results_experiment_plot = results_experiment_plot.melt(['attr', 'method'], var_name='cols', value_name='vals')
        results_experiment_plot['cols'] = results_experiment_plot['method']

        g = sns.displot(results_experiment_plot, col='cols', hue="attr", x="vals", palette="Set1", kde=True, legend=True
                        , height=2, col_wrap=2, facet_kws={'sharey': False, 'sharex': False})

        # plt.show()

        def vertical_ttest_line(x, **kwargs):
            # TODO: replace 1.96 by t critical value
            plt.axvline(1.96 * np.sign(theta_true[str(list(x)[0])] + 1e-7), linestyle='--', color='gray')

        g.map(vertical_ttest_line, 'attr')

        # https://stackoverflow.com/questions/37815774/seaborn-pairplot-legend-how-to-control-position

        # handles = g.legend.legendHandles()
        # labels = g.legend.get_label()
        # g.fig.legend(handles=handles, labels=labels, loc='lower right', ncol=1)
        # g.fig.legend(labels=labels, loc='lower center', ncol=4)
        # g.fig.legend(labels=results_experiment_plot.attr.unique(), loc='lower center', ncol=4)

        g.set_axis_labels("t-value", "freq")

        # titles = ['no refined opt', 'refined opt', 'combined opt']
        titles = methods  # ['ngd', 'gn', 'ngd+gn']
        for ax, title in zip(g.axes.flatten(), titles):
            ax.set_title(title)
            # ax.tick_params(labelbottom=True)
        # plt.ylabel()

        g.fig.tight_layout()

        plt.show()

        # https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
        self.save_fig(g.fig, folder, 'distribution_ttest')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (v) Facet of box plots for distribution of t-test for each attribute and among estimation methods

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='method', y='ttest', palette="Set1",
            col='attr', kind='box', col_wrap=2, height=2, sharey=False, sharex=True, showfliers=False
        )

        def horizontal_ttest_line(x, **kwargs):
            # TODO: replace 1.96 by t critical value
            plt.axhline(1.96 * np.sign(theta_true[str(list(x)[0])] + 1e-7), linestyle='--', color='gray')

        g.map(horizontal_ttest_line, 'attr')

        titles = ['travel time', 'waiting time', 'cost', 'null parameter']
        for axi, title in zip(g.axes, titles):
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_title(title)

        list(map(lambda x: x.tick_params(labelbottom=False), g.axes[0:len(g.axes) - 2]))

        g.set_axis_labels('', "t-test")

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'distribution_ttest')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (vi) Distribution of pvalues

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='attr', y='pvalue', palette="Set1",
            col='method', kind='box', col_wrap=2, sharey=True, sharex=True, showfliers=False, height=2
        )

        def horizontal_theta_true_line(x, **kwargs):
            plt.axhline(0.05, linestyle='--', color='gray')

        g.map(horizontal_theta_true_line, 'attr')

        g.set_axis_labels("attribute", "p-value")

        # titles = ['no refined opt', 'refined opt', 'combined opt']
        # for ax,title in zip(g.axes.flatten(), titles ):
        #     ax.set_title(title)
        # ax.tick_params(labelbottom=True)
        # plt.ylabel()

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'distribution_pvalues')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # vii) False positives and negatives

        h0 = 0
        alpha = 0.05

        results_experiment['fn'] = 0
        results_experiment['fp'] = 0
        results_experiment['f_type'] = ''

        for i in results_experiment.index:

            if results_experiment.at[i, 'theta_true'] == h0:

                results_experiment.at[i, 'f_type'] = 'fp'

                if results_experiment.at[i, 'pvalue'] < alpha:
                    results_experiment.at[i, 'fp'] = 1

                    # print(row['fn'])

            else:

                results_experiment.at[i, 'f_type'] = 'fn'

                if results_experiment.at[i, 'pvalue'] > alpha:
                    results_experiment.at[i, 'fn'] = 1

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

        if 'fp' in list(set(results_experiment.f_type.values)):
            sns.barplot(x="method", y="fp",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fp')]
                        , ax=ax[(1)])

        if 'fn' in list(set(results_experiment.f_type.values)):
            sns.barplot(x="method", y="fn",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fn')]
                        , ax=ax[(0)])

        for axi in fig.get_axes():
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_xlabel('')

        ax[1].set_ylabel('false positives')
        ax[0].set_ylabel('false negatives')

        ax[0].axhline(0, linestyle='--', color='gray')
        ax[1].axhline(0, linestyle='--', color='gray')

        ax[0].set(ylim=(-0.05, 1.05))
        ax[1].set(ylim=(-0.05, 1.05))

        fig.tight_layout()

        self.save_fig(fig, folder, 'falseposneg')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (viii) Confidence intervals

        # Issue with underscores https://github.com/lmfit/lmfit-py/issues/373
        # matplotlib.rcParams['text.usetex'] = False
        # fig, ax = plt.subplots(figsize=(4, 4))
        #
        # sns.barplot(x="method", y="width_confint", data=results_experiment)
        # # sns.barplot(x="method", y="time", data= results_experiment, ax=ax[(1)])
        # ax.axhline(0, linestyle='--', color='gray')

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='attr', y='width_confint', palette="Set1",
            col='method', kind='bar', col_wrap=2, sharey=True, sharex=True, height=2
        )

        g.set_axis_labels("attribute", "CI width")

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'width_ci')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # matplotlib.rcParams['text.usetex'] = True

        # matplotlib.rcParams['text.usetex'] = True

    def levels_experiment(self,
                          results_experiment,
                          alpha,
                          sd_x,
                          folder: str = None,
                          range_initial_values=None,
                          summary_only=True,
                          silent_mode=True):

        assert 'level' in results_experiment.keys()

        theta_true = pd.Series(
            data=results_experiment['theta_true'].values,
            index=results_experiment['attr']).drop_duplicates()

        levels = results_experiment[['level']].drop_duplicates().values.flatten().tolist()

        if folder is None:
            folder = self.folder

        # x) Summary plot (computation time, ci_width, average bias and proportion of false positives/negatives

        # matplotlib.rcParams['text.usetex'] = False

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))

        # 1) bias in estimates
        sns.boxplot(x="level", y="bias", data=results_experiment[results_experiment.attr != 'vot'], color="white",
                    ax=ax[(0, 0)], showfliers=False)
        # ax[(0, 0)].set_ylabel('bias ' + r"$(\hat{\theta})$")
        ax[(0, 0)].set_ylabel('bias in coefficients')
        if range_initial_values is not None:
            ax[(0, 0)].set(ylim=range_initial_values)
        # ax[(1, 0)].set_ylabel('bias ' + r"$(\hat{\theta} - \theta)$")

        # # 2) Bias in value of time
        # if 'vot' in results_experiment.attr.values:
        #     sns.boxplot(x="level", y="bias", data=results_experiment[results_experiment.attr == 'vot'], showfliers=False,
        #                 ax=ax[(0, 1)])
        #     ax[(0, 1)].axhline(0, linestyle='--', color='gray')
        #     ax[(0, 1)].set_ylabel('bias' + r"$(\hat{\theta}_t/\hat{\theta}_c)$")
        #
        #     if range_initial_values is not None:
        #         # ax[(0, 1)].set(ylim=(-5e-1,5e-1))
        #         ax[(0, 1)].set(ylim=range_initial_values)

        # # Objective function
        #
        # sns.boxplot(x="level", y="objective", data=results_experiment, showfliers=False,
        #             ax=ax[(0, 1)])
        # ax[(0, 1)].axhline(0, linestyle='--', color='gray')
        # ax[(0, 1)].set_ylabel("Objective function")

        # 2) NRMSE

        sns.barplot(x="level", y="nrmse", data=results_experiment, color="white",
                    errcolor="black", edgecolor="black", linewidth=1.5, errwidth=1.5, ax=ax[(0, 1)])
        ax[(0, 1)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
        ax[(0, 1)].set(ylim=(0, 1))
        ax[(0, 1)].set_ylabel("nrmse")

        # 3) False negatives
        sns.pointplot(x="level", y="fn",
                      data=results_experiment[(results_experiment.attr != 'vot') & (results_experiment.f_type == 'fn')],
                      ax=ax[(1, 0)], color='black', errwidth=1.5,
                      scale=0.7, capsize=.2)

        ax[(1, 0)].axhline(0, linestyle='--', color='gray')
        ax[(1, 0)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
        ax[(1, 0)].set(ylim=(-0.05, 1.05))
        ax[(1, 0)].set_ylabel('false negatives')

        if 'fp' in list(set(results_experiment.f_type.values)):
            # 4) False positives
            sns.pointplot(x="level", y="fp",
                          data=results_experiment[
                              (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fp')],
                          ax=ax[(1, 1)], color='black',
                          errwidth=1.5,
                          scale=0.7, capsize=.2)

            ax[(1, 1)].set_yticks(np.arange(0, 1 + 0.1, 0.2))
            ax[(1, 1)].set(ylim=(-0.05, 1.05))
            ax[(1, 1)].set_ylabel('false positives')

        # Change color style to white and black in box plots
        for axi in [ax[(0, 0)]]:
            for i, box in enumerate(axi.patches):
                box.set_edgecolor("black")
                box.set_facecolor("white")

            plt.setp(axi.patches, edgecolor='black', facecolor='white')
            plt.setp(axi.lines, color='black')

            plt.setp(axi.artists, edgecolor='black', facecolor='white')

        ax[(0, 0)].axhline(0, linestyle='--', color='gray')
        ax[(0, 1)].axhline(sd_x, linestyle='--', color='gray')
        ax[(1, 0)].axhline(0, linestyle='--', color='gray')
        ax[(1, 1)].axhline(alpha, linestyle='--', color='gray')

        fig.tight_layout()

        self.save_fig(fig, folder, 'inference_summary')

        if summary_only:
            return

        # (i) Facet of box plots for distribution of  parameter estimates and among estimation levels

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='level', y='theta', palette="Set1",
            col='attr', kind='box', col_wrap=2, sharey=False, sharex=False, showfliers=False, height=2, color='#abc9ea',
        )

        def horizontal_theta_true_line(x, **kwargs):
            plt.axhline(theta_true[str(list(x)[0])], linestyle='--', color='gray')

        g.map(horizontal_theta_true_line, 'attr')

        titles = ['travel time', 'waiting time', 'cost', 'null parameter']
        for axi, title in zip(g.axes, titles):
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_title(title)
            # axi.tick_params(labelbottom=False)

        list(map(lambda x: x.tick_params(labelbottom=False), g.axes[0:len(g.axes) - 2]))

        g.set_axis_labels('', "estimate")

        g.fig.tight_layout()

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # fig.savefig(self.folder + '/' + network_name + '/' + filename + ".pdf", pad_inches=0.1, bbox_inches="tight")

        self.save_fig(g.fig, folder, 'parameter_estimates')

        # (ii) Facet of box plots for distribution of bias in parameter estimates and estimation levels.

        # bias computation
        results_experiment['theta_true'] = 0

        attr_list = list(set(results_experiment.attr))

        for attr in attr_list:
            results_experiment['theta_true'].loc[results_experiment.attr == attr] = theta_true[attr]

        results_experiment['bias'] = results_experiment['theta'] - results_experiment['theta_true']

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='attr', y='bias', palette="Set1",
            col='level', kind='box', col_wrap=2, sharey=True, sharex=False, showfliers=False, height=2, color='#abc9ea'
        )

        def horizontal_theta_true_line(x, **kwargs):
            plt.axhline(0, linestyle='--', color='gray')

        g.map(horizontal_theta_true_line, 'attr')

        g.set_axis_labels("attribute", "bias")

        # # titles = ['no refined opt', 'refined opt', 'combined opt']
        # titles = ['ngd', 'gn', 'ngd+gn']
        titles = list(set(results_experiment['level']))
        for ax, title in zip(g.axes.flatten(), titles):
            ax.set_title(title)
            # ax.tick_params(labelbottom=True)
        # plt.ylabel()

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'parameter_bias')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (iii) VOT distribution only

        # sns.boxplot(x="level", y="ttest", data = results_experiment_plot[results_experiment_plot.cols == 'ttest-norefined']
        #             , showfliers=False )
        # plt.show()

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

        sns.boxplot(x="level", y="theta", data=results_experiment[results_experiment.attr == 'vot'], showfliers=False,
                    ax=ax[(0)])
        ax[0].set_ylabel('value of time')
        ax[0].axhline(theta_true['vot'], linestyle='--', color='gray')

        sns.boxplot(x="level", y="bias", data=results_experiment[results_experiment.attr == 'vot'], showfliers=False,
                    ax=ax[(1)])
        ax[1].set_ylabel('bias')
        ax[1].axhline(0, linestyle='--', color='gray')

        for axi in fig.get_axes():
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_xlabel('')

        fig.tight_layout()

        # g.set_axis_labels("state", "value of time")

        self.save_fig(fig, folder, 'vot_estimate_bias')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # iv) Distribution plot of t-tests

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

        stats_list = ['ttest', 'level']
        results_experiment_plot = pd.DataFrame(
            data=np.c_[results_experiment[['ttest', 'level']], results_experiment['attr']]
            , columns=stats_list + ['attr'])

        # reference: https://stackoverflow.com/questions/46045750/python-distplot-with-multiple-distributions

        # recast into long format
        results_experiment_plot = results_experiment_plot.melt(['attr', 'level'], var_name='cols', value_name='vals')
        results_experiment_plot['cols'] = results_experiment_plot['level']

        g = sns.FacetGrid(results_experiment_plot, col='cols', hue="attr", palette="Set1"
                          , height=2, col_wrap=2, sharey=False, sharex=True, legend_out=True)
        g = (g.map(sns.distplot, "vals", hist=False))

        def vertical_ttest_line(x, **kwargs):
            # TODO: replace 1.96 by t critical value
            plt.axvline(1.96 * np.sign(theta_true[str(list(x)[0])] + 1e-7), linestyle='--', color='gray')

        g.map(vertical_ttest_line, 'attr')

        # https://stackoverflow.com/questions/37815774/seaborn-pairplot-legend-how-to-control-position

        handles = g._legend_data.values()
        labels = g._legend_data.keys()
        # g.fig.legend(handles=handles, labels=labels, loc='lower right', ncol=1)
        g.fig.legend(handles=handles, labels=labels, loc='lower center', ncol=4)

        g.set_axis_labels("t-value", "freq")

        # titles = ['no refined opt', 'refined opt', 'combined opt']
        titles = levels  # ['ngd', 'gn', 'ngd+gn']
        for ax, title in zip(g.axes.flatten(), titles):
            ax.set_title(title)
            # ax.tick_params(labelbottom=True)
        # plt.ylabel()

        g.fig.tight_layout()

        # https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot

        self.save_fig(g.fig, folder, 'distribution_ttest')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (v) Facet of box plots for distribution of t-test for each attribute and among estimation levels

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='level', y='ttest', palette="Set1",
            col='attr', kind='box', col_wrap=2, height=2, sharey=False, sharex=False, showfliers=False
        )

        def horizontal_ttest_line(x, **kwargs):
            # TODO: replace 1.96 by t critical value
            plt.axhline(1.96 * np.sign(theta_true[str(list(x)[0])] + 1e-7), linestyle='--', color='gray')

        g.map(horizontal_ttest_line, 'attr')

        titles = ['travel time', 'waiting time', 'cost', 'null parameter']
        for axi, title in zip(g.axes, titles):
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_title(title)

        list(map(lambda x: x.tick_params(labelbottom=False), g.axes[0:len(g.axes) - 2]))

        g.set_axis_labels('', "t-test")

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'distribution_ttest')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (vi) Distribution of pvalues

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='attr', y='pvalue', palette="Set1",
            col='level', kind='box', col_wrap=2, sharey=True, sharex=False, showfliers=False, height=2
        )

        def horizontal_theta_true_line(x, **kwargs):
            plt.axhline(0.05, linestyle='--', color='gray')

        g.map(horizontal_theta_true_line, 'attr')

        g.set_axis_labels("attribute", "p-value")

        # titles = ['no refined opt', 'refined opt', 'combined opt']
        # for ax,title in zip(g.axes.flatten(), titles ):
        #     ax.set_title(title)
        # ax.tick_params(labelbottom=True)
        # plt.ylabel()

        g.fig.tight_layout()

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        self.save_fig(g.fig, folder, 'distribution_pvalues')

        # vii) False positives and negatives

        h0 = 0
        alpha = 0.05

        results_experiment['fn'] = 0
        results_experiment['fp'] = 0
        results_experiment['f_type'] = ''

        for i in results_experiment.index:

            if results_experiment.at[i, 'theta_true'] == h0:

                results_experiment.at[i, 'f_type'] = 'fp'

                if results_experiment.at[i, 'pvalue'] < alpha:
                    results_experiment.at[i, 'fp'] = 1

                    # print(row['fn'])

            else:

                results_experiment.at[i, 'f_type'] = 'fn'

                if results_experiment.at[i, 'pvalue'] > alpha:
                    results_experiment.at[i, 'fn'] = 1

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

        if 'fp' in list(set(results_experiment.f_type.values)):
            sns.barplot(x="level", y="fp",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fp')]
                        , ax=ax[(1)])

        if 'fn' in list(set(results_experiment.f_type.values)):
            sns.barplot(x="level", y="fn",
                        data=results_experiment[
                            (results_experiment.attr != 'vot') & (results_experiment.f_type == 'fn')]
                        , ax=ax[(0)])

        for axi in fig.get_axes():
            # plt.setp(axi.xaxis.get_majorticklabels(), rotation=90)
            axi.set_xlabel('')

        ax[1].set_ylabel('false positives')
        ax[0].set_ylabel('false negatives')

        ax[0].axhline(0, linestyle='--', color='gray')
        ax[1].axhline(0, linestyle='--', color='gray')

        ax[0].set(ylim=(-0.05, 1.05))
        ax[1].set(ylim=(-0.05, 1.05))

        fig.tight_layout()

        self.save_fig(fig, folder, 'falseposneg')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

        # (viii) Confidence intervals

        # Issue with underscores https://github.com/lmfit/lmfit-py/issues/373
        # matplotlib.rcParams['text.usetex'] = False
        # fig, ax = plt.subplots(figsize=(4, 4))
        #
        # sns.barplot(x="level", y="width_confint", data=results_experiment)
        # # sns.barplot(x="level", y="time", data= results_experiment, ax=ax[(1)])
        # ax.axhline(0, linestyle='--', color='gray')

        g = sns.catplot(
            data=results_experiment[results_experiment.attr != 'vot'], x='attr', y='width_confint', palette="Set1",
            col='level', kind='bar', col_wrap=2, sharey=True, sharex=False, height=2
        )

        g.set_axis_labels("attribute", "CI width")

        g.fig.tight_layout()

        self.save_fig(g.fig, folder, 'width_ci')

        if not silent_mode:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

    def heatmap_demand(self,
                       Q: Matrix,
                       folderpath: str,
                       filename: str):

        rows, cols = Q.shape

        od_df = pd.DataFrame({'origin': pd.Series([], dtype=int)
                                 , 'destination': pd.Series([], dtype=int)
                                 , 'trips': pd.Series([], dtype=int)})

        counter = 0
        for origin in range(0, rows):
            for destination in range(0, cols):
                # od_df.loc[counter] = [(origin+1,destination+1), N['train'][current_network].Q[(origin,destination)]]
                od_df.loc[counter] = [int(origin + 1), int(destination + 1), Q[(origin, destination)]]
                counter += 1

        od_df.origin = od_df.origin.astype(int)
        od_df.destination = od_df.destination.astype(int)

        od_pivot_df = od_df.pivot_table(index='origin', columns='destination', values='trips')

        # uniform_data = np.random.rand(10, 12)
        fig, ax = plt.subplots()
        ax = sns.heatmap(od_pivot_df, linewidth=0.5, cmap="Blues")

        plt.show()

        fig.savefig( folderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight")

        plt.close(fig)

    def cumulative_demand(self,
                          Q: Matrix,
                          filename: str,
                          threshold: Proportion = None,
                          folderpath: str = None):

        # ods_sorted = list(np.dstack(np.unravel_index(np.argsort(-Q.ravel()), Q.shape)))[0]

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

        fig, axs = plt.subplots(nrows = 1, ncols = 2, tight_layout=True, figsize=(9, 4))

        #Cumulative absolute trips
        q = Q[Q.nonzero()]

        abs_x = sorted(q,reverse=True)
        abs_y = np.cumsum(abs_x).astype("float32")

        abs_y = np.hstack([0,abs_y])
        # abs_x = np.linspace(0, len(abs_x)+1, abs_y.size)
        abs_x = np.arange(0,len(abs_y),1)
        axs[0].plot(abs_x, abs_y)
        axs[0].set_xlabel('o-d pair (sorted by demand)')
        axs[0].set_ylabel('cumulative trips')

        #Cumulative proportion of trips

        rel_x = sorted(q,reverse=True)+[0]
        rel_y = np.cumsum(rel_x).astype("float32")
        rel_y /= rel_y.max()
        rel_y *= 100.
        rel_y = np.hstack([0, rel_y])
        # rel_x = np.arange(1,100+0.1,1)
        rel_x = np.linspace(0,100, rel_y.size)
        # nearest_perc = np.array([find_nearest(rel_x, i) for i in np.arange(0, 101, 10)])
        # plt.sca(axs[1])
        # plt.xticks(-nearest_perc,list(reversed(list(map(round,nearest_perc)))))
        # axs[1].plot(-rel_x, rel_y)
        axs[1].plot(rel_x, rel_y)
        axs[1].set_xlabel('percentile of o-d pairs')
        axs[1].set_ylabel('percentage of total trips')

        if threshold is not None:
            idx, x_vline = find_nearest(rel_x, threshold*100)
            axs[1].vlines(x=x_vline, ymin = -0.01, ymax = rel_y[idx],
                           color='black', linestyle='dashed', linewidth=0.5)
            axs[1].hlines(y=rel_y[idx], xmin=-0.01, xmax=x_vline,
                          color='black', linestyle='dashed', linewidth=0.5)
            axs[0].vlines(x=int(x_vline/100*len(abs_x)), ymin = -0.01, ymax = int(abs_y[idx]),
                           color='black', linestyle='dashed', linewidth=0.5)
            axs[0].hlines(y=int(abs_y[idx]), xmin=-0.01, xmax=int(x_vline/100*len(abs_x)),
                          color='black', linestyle='dashed', linewidth=0.5)


        for axi in axs:
            axi.tick_params(axis='y', labelsize=self.fontsize)
            axi.tick_params(axis='x', labelsize=self.fontsize)
            axi.xaxis.label.set_size(self.fontsize)
            axi.yaxis.label.set_size(self.fontsize)
            axi.margins(x=0.01, y =0.01)

        plt.show()

        fig.savefig(folderpath + '/' + filename, pad_inches=0.1, bbox_inches="tight")

        plt.close(fig)




def draw_networkx_digraph_edge_labels(G, pos,
                                      edge_labels=None,
                                      label_pos=0.5,
                                      font_size=10,
                                      font_color='k',
                                      font_family='sans-serif',
                                      font_weight='normal',
                                      alpha=None,
                                      bbox=None,
                                      ax=None,
                                      rotate=True,
                                      **kwds):
    """Modify networkX to Draw edge labels so it properly draw and put labels for DIgraph with bidrectional edges. Not working for multidigraphs

    Parameters
    ----------
    G : graph
       A networkx graph

    pos : dictionary
       A dictionary with nodes as keys and positions as values.
       Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
       Draw the graph in the specified Matplotlib axes.

    alpha : float or None
       The text transparency (default=None)

    edge_labels : dictionary
       Edge labels in a dictionary keyed by edge two-tuple of text
       labels (default=None). Only labels for the keys in the dictionary
       are drawn.

    label_pos : float
       Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int
       Font size for text labels (default=12)

    font_color : string
       Font color string (default='k' black)

    font_weight : string
       Font weight (default='normal')

    font_family : string
       Font family (default='sans-serif')

    bbox : Matplotlib bbox
       Specify text box shape and colors.

    clip_on : bool
       Turn on clipping at axis boundaries (default=True)

    Returns
    -------
    dict
        `dict` of labels keyed on the edges

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.github.io/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw()
    draw_networkx()
    draw_networkx_nodes()
    draw_networkx_edges()
    draw_networkx_labels()
    """

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    except RuntimeError:
        print("Matplotlib unable to open display")
        raise

    offset = kwds.pop("offset", 0.0)

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = nx.get_edge_attributes(G, 'rad')
        # labels = {u[0]: v[3] for u, v in zip(G.edges,G.edges(data=True))}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2, p), label in labels.items():

        label = np.round(label, 1)

        (x1, y1) = pos[n1]  # + (label['rad'],label['rad'])
        (x2, y2) = pos[n2]  # + (label['rad'],label['rad'])
        _norm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos),
                  y1 * label_pos + y2 * (1.0 - label_pos))

        # Modificatino to existing method#
        # source: https://math.stackexchange.com/questions/995659/given-two-points-find-another-point-a-perpendicular-distance-away-from-the-midp

        # Understand what radians means in terms of distance to locate more accurately the link label from the middle points between nodes
        d = label * _norm / np.pi
        # d = np.sqrt(label)
        (x, y) = (x - d * (y1 - y2) / _norm, y - d * (x2 - x1) / _norm)

        # x += offset * (y2 - y1) / _norm
        # y += offset * -(x2 - x1) / _norm

        # rotate = False
        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < - 90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(np.array((angle,)),
                                                        xy.reshape((1, 2)))[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle='round',
                        ec=(1.0, 1.0, 1.0),
                        fc=(1.0, 1.0, 1.0),
                        )
        # if not is_string_like(label):
        if not nx.utils.is_string_like(label):
            label = str(label)  # this makes "1" and 1 labeled the same

        # set optional alignment
        horizontalalignment = kwds.get('horizontalalignment', 'center')
        verticalalignment = kwds.get('verticalalignment', 'center')

        t = ax.text(x, y,
                    label,
                    size=font_size,
                    color=font_color,
                    family=font_family,
                    weight=font_weight,
                    alpha=alpha,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    rotation=trans_angle,
                    transform=ax.transData,
                    bbox=bbox,
                    zorder=1,
                    clip_on=True,
                    )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)

    return text_items
