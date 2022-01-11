import numpy as np
import matplotlib.pyplot as plt

from tick.base import TimeFunction
from tick.hawkes import SimuHawkesExpKernels
from tick.hawkes import SimuPoissonProcess
from tick.plot import plot_point_process
from tick.dataset import fetch_hawkes_bund_data
from tick.hawkes import HawkesConditionalLaw
from tick.hawkes import SimuHawkesSumExpKernels, HawkesSumExpKern
# from tick.plot import plot_point_process
from tick.plot import plot_hawkes_kernel_norms, plot_hawkes_kernels
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti, HawkesSumExpKern, HawkesKernelExp, SimuHawkes, HawkesExpKern


def Hawkes_sim():
    period_length = 100
    t_values = np.linspace(0, period_length)
    y_values = 0.2 * np.maximum(
        np.sin(t_values * (2 * np.pi) / period_length), 0.2)
    baselines = np.array(
        [TimeFunction((t_values, y_values), border_type=TimeFunction.Cyclic)])

    decay = 0.1
    adjacency = np.array([[0.5]])

    hawkes = SimuHawkesExpKernels(adjacency, decay, baseline=baselines, seed=2093,
                                  verbose=False)
    hawkes.track_intensity(0.1)
    hawkes.end_time = 6 * period_length
    hawkes.simulate()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    plot_point_process(hawkes, ax=ax)

    t_values = np.linspace(0, hawkes.end_time, 1000)
    ax.plot(t_values, hawkes.get_baseline_values(0, t_values), label='baseline',
            ls='--', lw=1)
    ax.set_ylabel("$\lambda(t)$", fontsize=18)
    ax.legend()

    plt.title("Intensity Hawkes process with exponential kernel and varying "
              "baseline")
    fig.tight_layout()
    plt.show()


def H_Poisson_sim():
    run_time = 50
    intensity = 5

    poi = SimuPoissonProcess(intensity, end_time=run_time, verbose=False)
    poi.simulate()
    plot_point_process(poi)
    plt.show()


def learn_hawkes():
    timestamps_list = fetch_hawkes_bund_data()

    kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))
    hawkes_learner = HawkesConditionalLaw(
        claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
        quad_method="log", n_quad=10, min_support=1e-4, max_support=1, n_threads=4)

    hawkes_learner.fit(timestamps_list)
    plot_point_process(hawkes_learner, plot_intensity=True)
    plt.show()

    # plot_hawkes_kernel_norms(hawkes_learner,
    #                          node_names=["P_u", "P_d", "T_a", "T_b"])


def hawkes_estimated_vs_truth():
    end_time = 1000

    # decays = [0.1, 0.5, 1.]
    # baseline = [0.12, 0.07]
    # adjacency = [[[0, .1, .4], [.2, 0., .2]], [[0, 0, 0], [.6, .3, 0]]]
    #
    # hawkes_exp_kernels = SimuHawkesSumExpKernels(
    #     adjacency=adjacency, decays=decays, baseline=baseline, end_time=end_time,
    #     verbose=False, seed=1039)

    intensity = 1.0
    decay = 0.1
    baseline = [0.12]
    exp_kernel = HawkesKernelExp(intensity, decay)
    hawkes_sim = SimuHawkes([[exp_kernel]], baseline, end_time=end_time, verbose=False, seed=1039, force_simulation=True)
    hawkes_sim.track_intensity(0.1)
    hawkes_sim.simulate()

    # hawkes_exp_kernels.track_intensity(0.1)
    # hawkes_exp_kernels.simulate()

    # learner = HawkesSumExpKern(decays, penalty='elasticnet', elastic_net_ratio=0.8)
    # learner.fit(hawkes_exp_kernels.timestamps)

    hawkes_learner = HawkesExpKern(decay, penalty='elasticnet', elastic_net_ratio=0.8)
    hawkes_learner.fit(hawkes_sim.timestamps)

    t_min = 0
    t_max = 200
    fig, ax_list = plt.subplots(1, 1, figsize=(10, 6))
    # learner.plot_estimated_intensity(hawkes_exp_kernels.timestamps, t_min=t_min,
    #                                  t_max=t_max, ax=ax_list)
    hawkes_learner.plot_estimated_intensity(hawkes_sim.timestamps, t_min=t_min,
                                     t_max=t_max, ax=ax_list)

    # plot_point_process(hawkes_exp_kernels, plot_intensity=True, t_min=t_min,
    #                    t_max=t_max, ax=ax_list)
    plot_point_process(hawkes_sim, plot_intensity=True, t_min=t_min,
                       t_max=t_max, ax=ax_list)
    ax_list.lines[0].set_label('estimated')
    ax_list.lines[1].set_label('original')

    # Change original intensity style
    ax_list.lines[1].set_linestyle('--')
    ax_list.lines[1].set_alpha(0.8)

    # avoid duplication of scatter plots of events
    ax_list.collections[1].set_alpha(0)

    ax_list.legend()

    # Enhance plot
    # for ax in ax_list:
    #     # Set labels to both plots
    #     ax.lines[0].set_label('estimated')
    #     ax.lines[1].set_label('original')
    #
    #     # Change original intensity style
    #     ax.lines[1].set_linestyle('--')
    #     ax.lines[1].set_alpha(0.8)
    #
    #     # avoid duplication of scatter plots of events
    #     ax.collections[1].set_alpha(0)
    #
    #     ax.legend()

    fig.tight_layout()
    plt.show()


def parametric_hawkes():
    end_time = 1000
    n_realizations = 10

    decays = [.5, 2., 6.]
    baseline = [0.12, 0.07]
    adjacency = [[[0, .1, .4], [.2, 0., .2]],
                 [[0, 0, 0], [.6, .3, 0]]]

    hawkes_exp_kernels = SimuHawkesSumExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, verbose=False, seed=1039)

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)

    multi.end_time = [(i + 1) / 10 * end_time for i in range(n_realizations)]
    multi.simulate()

    learner = HawkesSumExpKern(decays, penalty='elasticnet',
                               elastic_net_ratio=0.8)
    learner.fit(multi.timestamps)

    fig = plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels, show=False)

    for ax in fig.axes:
        ax.set_ylim([0., 1.])

    plt.show()


if __name__ == '__main__':
    hawkes_estimated_vs_truth()