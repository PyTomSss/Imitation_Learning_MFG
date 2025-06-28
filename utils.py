import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1) Pareto  (max ε_MPC  vs  NIG)
# ------------------------------------------------------------------
def plot_pareto_eps_vs_nig(model_cls,
                           param_grid,
                           horizon,          # H
                           extra_kwargs=None # p.ex. {"L":0.8} ou {"L1":0.8,"L2":1.2}
                           ):
    """
    model_cls      : la classe (AttractorMFG, CrowdAvoidanceMFG, TriStateCongestionMFG, …)
    param_grid     : liste d'itérables  ▸ Attractor : [α]          (float)
                                     ▸ Tri-State : [(α1, α2)] (tuple)
    horizon        : H
    extra_kwargs   : dict passé à l'instanciation (L ou L1,L2, etc.)

    Affiche un scatter (max ε_MPC, NIG).
    """
    extra_kwargs = extra_kwargs or {}
    eps_max, nig_values = [], []

    for p in param_grid:
        # ------------- instanciation du modèle -------------
        m = model_cls(H=horizon, **extra_kwargs)

        # ------------- appel méthodes -------------
        if isinstance(p, tuple):
            # cas multi-paramètres  (Tri-State)
            eps_BC, eps_van, eps_mpc = m.errors(*p)
            nig = m.NIG(*p)
        else:
            # cas simple  (Attractor, CrowdAvoidance)
            eps_BC, eps_van, eps_mpc = m.errors(p)
            nig = m.NIG(p)

        eps_max.append(np.max(eps_mpc))
        nig_values.append(nig)

    # ------------- scatter -------------
    plt.figure()
    plt.scatter(eps_max, nig_values)
    plt.xlabel("max ε_MPC")
    plt.ylabel("NIG")
    plt.title("Pareto : compromis erreur / performance")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------------
# 2) Courbes ε_BC / ε_vanilla / ε_MPC  au fil du temps
# ------------------------------------------------------------------
def plot_error_curves(model_cls,
                      alphas,              # float  ou  (α1, α2)
                      horizon,
                      extra_kwargs=None):
    """
    Trace ε_BC, ε_vanilla, ε_MPC pour un choix de paramètres.
    alphas        : α  (float)             ou  (α1, α2)  (tuple)
    """
    extra_kwargs = extra_kwargs or {}
    m = model_cls(H=horizon, **extra_kwargs)

    if isinstance(alphas, tuple):
        eps_BC, eps_van, eps_mpc = m.errors(*alphas)
    else:
        eps_BC, eps_van, eps_mpc = m.errors(alphas)

    t = np.arange(horizon + 1)

    plt.figure()
    plt.plot(t, eps_BC,  label="ε_BC")
    plt.plot(t, eps_van, label="ε_vanilla")
    plt.plot(t, eps_mpc, label="ε_MPC")
    plt.xlabel("t")
    plt.ylabel("erreur")
    plt.title("Évolution temporelle des erreurs")
    plt.legend()
    plt.grid(True)
    plt.show()
