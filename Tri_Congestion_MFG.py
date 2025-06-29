import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import matplotlib.pyplot as plt

@dataclass
class TriStateCongestionMFG:
    """
    MFG « Tri-State » : 3 états (s0 neutre, s1 congestionné, s2 désirable)  
    et 3 actions depuis s0 : stay / to_s1 / to_s2.

    Paramètres
    ----------
    H   : horizon en pas de temps
    L1  : coefficient de congestion pour l'accès à s1
    L2  : coefficient de congestion pour l'accès à s2
    rho0_s1, rho0_s2 : répartitions initiales (s0 = 1 − rho0_s1 − rho0_s2)

    Méthodes
    --------
    * run(alpha1, alpha2)              : trajectoires ρ_t^{Eπ} (extrême) et ρ_t^{π}
    * errors(alpha1, alpha2)           : ε_BC, ε_vanilla-ADV, ε_MPC-ADV  
    * NIG(alpha1, alpha2)              : Nash Imitation Gap  
    """
    H: int
    L1: float
    L2: float
    rho0_s1: float = 0.0
    rho0_s2: float = 0.0

    # historiques remplis par run()
    hist_E_pi: List[np.ndarray] = field(init=False, default_factory=list)  # (ρ0,ρ1,ρ2)
    hist_pi:   List[np.ndarray] = field(init=False, default_factory=list)

    # ---------- Policy ----------
    @staticmethod
    def policy_probs(alpha1: float, alpha2: float) -> Tuple[float, float]:
        """
        Renvoie (alpha1, alpha12) bornés et tels que alpha1+alpha2 ≤ 1.
        """
        a1 = float(np.clip(alpha1, 0.0, 1.0))
        a2 = float(np.clip(alpha2, 0.0, 1.0))
        if a1 + a2 > 1.0:
            
            total = a1 + a2
            a1, a2 = a1 / total, a2 / total
        return a1, a2

    # ---------- Dynamics ----------
    def _next_rho(self,
                  rho: np.ndarray,
                  alpha1: float,
                  alpha2: float) -> np.ndarray:
       
        rho0, rho1, rho2 = rho
        a1, a2 = self.policy_probs(alpha1, alpha2)  

        # probabilités de succès depuis s0
        p1 = a1 * max(0.0, 1.0 - self.L1 * rho1)
        p2 = a2 * rho1 * max(0.0, 1.0 - self.L2 * rho2)

        #update
        new_rho0 = rho0 * (1.0 - p1 - p2)
        new_rho1 = rho1 + rho0 * p1
        new_rho2 = rho2 + rho0 * p2

        return np.array([new_rho0, new_rho1, new_rho2])

    # ---------- Simulation ----------
    def run(self, alpha1: float, alpha2: float) -> None:
        """
        Calcule toutes les trajectoires pour t = 0…H.
        - hist_E_pi : politique extrême 
        - hist_pi   : politique donnée
        """
        rho_init = np.array([1.0 - self.rho0_s1 - self.rho0_s2,
                             self.rho0_s1,
                             self.rho0_s2])

        self.hist_E_pi = [rho_init]
        self.hist_pi   = [rho_init]

        for _ in range(self.H):

            self.hist_E_pi.append(
                self._next_rho(self.hist_E_pi[-1], alpha1, alpha2)
            )

            self.hist_pi.append(
                self._next_rho(self.hist_pi[-1], alpha1, alpha2)
            )

    # ---------- compute erros ----------
    def errors(self, alpha1: float, alpha2: float):
        """
        Renvoie trois ndarrays de taille H+1 :
          ε_BC, ε_vanilla-ADV, ε_MPC-ADV
        """
        self.run(alpha1, alpha2)

        alpha_sum = alpha1 + alpha2
        hist_E = np.array(self.hist_E_pi)       
        hist_P = np.array(self.hist_pi)

        crowd_E = hist_E[:, 1] + hist_E[:, 2]    
        crowd_P = hist_P[:, 1] + hist_P[:, 2]    

        bc  = np.full(self.H + 1, 2 * alpha_sum)
        van = 2 * (alpha_sum + (1.0 - alpha_sum) * crowd_E)
        mpc = 2 * (alpha_sum + (1.0 - alpha_sum) * crowd_P)

        return bc, van, mpc

    # ---------- Nash Imitation Gap ----------
    def NIG(self, alpha1: float, alpha2: float) -> float:
        """
        Somme sur t = 0…H-1 des proportions en (s1,s2) sous π.
        """
        self.run(alpha1, alpha2)
        hist = np.array(self.hist_pi)[:-1]          # t = 0 … H-1
        return np.sum(hist[:, 1] + hist[:, 2])
    
### Affichage d’une heat-map du NIG final dans l’espace des politiques 

def plot_nig_heatmap(
    H: int = 40,
    L1: float = 0.5,
    L2: float = 0.5,
    n_pts: int = 41,
    alpha1_bounds: tuple = (0.0, 1.0),
    alpha2_bounds: tuple = (0.0, 1.0),
    cmap: str = "viridis",
    model_cls=TriStateCongestionMFG,
    model_kwargs=None,
):
    """
    Affiche une *heat-map* 2D du NIG final dans l’espace des politiques (α₁, α₂).

    Paramètres
    ----------
    H, L1, L2   : hyper-paramètres du jeu (horizon et coefficients de congestion)
    n_pts       : nombre de points sur chaque axe (=> résolution n_pts × n_pts)
    alpha1_bounds, alpha2_bounds : bornes [min, max] pour chaque alpha
    cmap        : colormap matplotlib
    model_cls   : classe MFG (par défaut TriStateCongestionMFG)
    model_kwargs: dict optionnel pour passer d’autres kwargs au constructeur
    """
    model_kwargs = model_kwargs or {}

    # ------------------------------------------------------------------
    # Grille d’alphas
    # ------------------------------------------------------------------
    alpha1_grid = np.linspace(*alpha1_bounds, n_pts)
    alpha2_grid = np.linspace(*alpha2_bounds, n_pts)

    # Tableau où l’on stocke le NIG final
    nig_map = np.zeros((n_pts, n_pts))

    # ------------------------------------------------------------------
    # Calcul du NIG pour chaque couple (α₁, α₂)
    # ------------------------------------------------------------------
    for i, a1 in enumerate(alpha1_grid):
        for j, a2 in enumerate(alpha2_grid):
            game = model_cls(
                H=H,
                L1=L1,
                L2=L2,
                **model_kwargs,
            )
            nig_map[j, i] = game.NIG(a1, a2)  # j = axe vertical (α₂), i = horizontal (α₁)

    # ------------------------------------------------------------------
    # Affichage de la heat-map
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        nig_map,
        origin="lower",
        extent=[alpha1_bounds[0], alpha1_bounds[1], alpha2_bounds[0], alpha2_bounds[1]],
        aspect="auto",
        cmap=cmap,
    )
    cbar = plt.colorbar(im)
    cbar.set_label("NIG final", rotation=270, labelpad=15)

    plt.xlabel(r"$\alpha_1$ (go to $s_1$)")
    plt.ylabel(r"$\alpha_2$ (go to $s_2$)")
    plt.title(rf"NIG Heat-map  (H={H}, L₁={L1}, L₂={L2})")
    plt.tight_layout()
    plt.show()

